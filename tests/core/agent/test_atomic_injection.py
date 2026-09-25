from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xagent.core.agent.context import ContextManager
from xagent.core.agent.runner import (
    AgentRunner,
    UserMessageInjectionConflictError,
    UserMessageInjectionOutcome,
    UserMessageInjectionRejectedError,
)
from xagent.core.agent.runtime import ExecutionInterrupted, PatternRuntime


@pytest.fixture
def live():
    manager = ContextManager()
    manager._contexts.clear()
    context = manager.create_context("atomic")
    context.add_user_message("original")
    tracer = SimpleNamespace(
        load_latest_checkpoint=AsyncMock(return_value=None), checkpoint=AsyncMock()
    )
    runner = AgentRunner(
        SimpleNamespace(llm=None), tracer=tracer, context_manager=manager
    )
    yield runner, context, tracer
    manager._contexts.clear()


@pytest.mark.asyncio
async def test_candidate_is_invisible_until_persisted(live):
    runner, context, tracer = live
    entered, release = asyncio.Event(), asyncio.Event()

    async def write(**payload):
        assert [m["content"] for m in payload["context"]["messages"]] == [
            "original",
            "new",
        ]
        entered.set()
        await release.wait()

    tracer.checkpoint.side_effect = write
    task = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    await entered.wait()
    try:
        assert [m.content for m in context.messages] == ["original"]
        assert "_pending_user_message_trace_turn_id" not in context.metadata
    finally:
        release.set()
        await task
    assert [m.content for m in context.messages] == ["original", "new"]


@pytest.mark.asyncio
async def test_confirmed_failed_write_leaves_no_ghost(live):
    runner, context, tracer = live
    tracer.checkpoint.side_effect = RuntimeError("write failed")
    with pytest.raises(UserMessageInjectionRejectedError, match="write failed"):
        await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert [m.content for m in context.messages] == ["original"]
    tracer.checkpoint.side_effect = None
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert tracer.checkpoint.await_count == 2


@pytest.mark.asyncio
async def test_lost_ack_readback_confirms_message(live):
    runner, context, tracer = live

    async def write(**payload):
        tracer.load_latest_checkpoint.return_value = payload
        raise RuntimeError("lost acknowledgement")

    tracer.checkpoint.side_effect = write
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert [m.content for m in context.messages] == ["original", "new"]


@pytest.mark.asyncio
async def test_uncertain_write_blocks_stale_checkpoint(live):
    runner, context, tracer = live

    async def write(**payload):
        tracer.load_latest_checkpoint.side_effect = RuntimeError("read unavailable")
        raise RuntimeError("write outcome unknown")

    tracer.checkpoint.side_effect = write
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert [m.content for m in context.messages] == ["original"]
    runtime = PatternRuntime(execution_id="atomic", tracer=tracer)
    with pytest.raises(ExecutionInterrupted):
        await runtime.checkpoint("late", context=context, pattern=SimpleNamespace())
    assert tracer.checkpoint.await_count == 1


@pytest.mark.asyncio
async def test_ordinary_checkpoint_waits_for_injection_publication(live):
    runner, context, tracer = live
    entered, release = asyncio.Event(), asyncio.Event()
    snapshots = []

    async def write(**payload):
        if payload["label"] == "user_message_injected":
            entered.set()
            await release.wait()
        snapshots.append(payload)

    tracer.checkpoint.side_effect = write
    injection = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    await entered.wait()
    runtime = PatternRuntime(execution_id="atomic", tracer=tracer)
    ordinary = asyncio.create_task(
        runtime.checkpoint("ordinary", context=context, pattern=SimpleNamespace())
    )
    await asyncio.sleep(0)
    assert not ordinary.done()
    # The executing pattern can still append output while persistence awaits.
    context.add_assistant_message("concurrent output")
    release.set()
    await asyncio.gather(injection, ordinary)
    assert [m.content for m in context.messages] == [
        "original",
        "concurrent output",
        "new",
    ]
    assert snapshots[-1]["label"] == "ordinary"
    assert snapshots[-1]["context"]["messages"][-1]["content"] == "new"


@pytest.mark.asyncio
async def test_concurrent_duplicate_injection_writes_once(live):
    runner, context, tracer = live
    first, second = await asyncio.gather(
        runner.inject_user_message("atomic", "new", turn_id="turn"),
        runner.inject_user_message("atomic", "new", turn_id="turn"),
    )
    assert {first.outcome, second.outcome} == {
        UserMessageInjectionOutcome.POSTED_FRESH,
        UserMessageInjectionOutcome.POSTED_REPLAY,
    }
    assert tracer.checkpoint.await_count == 1
    assert len(context.messages) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["write", "readback"])
async def test_cancelled_uncertain_injection_blocks_stale_writes(live, cancel_at):
    runner, context, tracer = live
    entered = asyncio.Event()

    async def block():
        entered.set()
        await asyncio.Event().wait()

    async def write(**payload):
        await block()

    # AsyncMock's side_effect must itself await the suspended read.
    async def read(_):
        await block()

    async def fail_write(**payload):
        tracer.load_latest_checkpoint.side_effect = read
        raise RuntimeError("uncertain")

    tracer.checkpoint.side_effect = write if cancel_at == "write" else fail_write
    task = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [m.content for m in context.messages] == ["original"]
    runtime = PatternRuntime(execution_id="atomic", tracer=tracer)
    with pytest.raises(ExecutionInterrupted):
        await runtime.checkpoint("late", context=context, pattern=SimpleNamespace())


@pytest.mark.asyncio
async def test_concurrent_cold_injections_share_context(live):
    runner, context, tracer = live
    payload = {"context": context.to_dict()}
    runner.context_manager.remove_context("atomic")
    reads, both = 0, asyncio.Event()

    async def read(_):
        nonlocal reads
        reads += 1
        if reads == 2:
            both.set()
        await both.wait()
        return payload

    tracer.load_latest_checkpoint.side_effect = read
    a, b = await asyncio.gather(
        runner.inject_user_message("atomic", "one", turn_id="one"),
        runner.inject_user_message("atomic", "two", turn_id="two"),
    )
    assert a.context is b.context is runner.context_manager.get_context("atomic")
    assert {m.content for m in a.context.messages} == {"original", "one", "two"}
    assert {
        m["content"] for m in tracer.checkpoint.call_args.kwargs["context"]["messages"]
    } == {"original", "one", "two"}


@pytest.mark.asyncio
async def test_cancel_before_write_does_not_poison_context(live):
    from xagent.core.agent.context.execution import context_checkpoint_gate

    runner, context, tracer = live
    gate = context_checkpoint_gate(context)
    async with gate.shared():
        task = asyncio.create_task(
            runner.inject_user_message("atomic", "new", turn_id="turn")
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not gate.injection_uncertain
    tracer.checkpoint.assert_not_awaited()
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", ["malformed", {"context": {}}, {"context": {"messages": ["bad"]}}]
)
async def test_malformed_readback_cannot_prove_absence(live, payload):
    runner, context, tracer = live

    async def write(**_):
        tracer.load_latest_checkpoint.return_value = payload
        raise RuntimeError("unknown write")

    tracer.checkpoint.side_effect = write
    result = await runner.inject_user_message("atomic", "new", turn_id="turn")
    assert result.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert [m.content for m in context.messages] == ["original"]


@pytest.mark.asyncio
async def test_live_unknown_stops_run_and_explicit_resume_reloads_checkpoint(live):
    runner, context, tracer = live
    entered, release = asyncio.Event(), asyncio.Event()
    durable = {"context": context.to_dict()}

    class Pattern:
        async def run(self, *, context, runtime, **kwargs):
            entered.set()
            await release.wait()
            await runtime.checkpoint("after", context=context, pattern=self)
            return {"success": True, "output": "done"}

    runner.agent.patterns = [Pattern()]
    operation = asyncio.create_task(
        runner.run("original", execution_id="atomic", checkpoint=durable)
    )
    await entered.wait()

    async def ambiguous_write(**payload):
        nonlocal durable
        durable = payload
        tracer.load_latest_checkpoint.side_effect = RuntimeError("read unavailable")
        raise RuntimeError("ack lost")

    tracer.checkpoint.side_effect = ambiguous_write
    try:
        outcome = await runner.inject_user_message("atomic", "new", turn_id="turn")
        assert outcome.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
        release.set()
        result = await operation
        assert result["status"] == "interrupted"
        assert tracer.checkpoint.await_count == 1
        tracer.load_latest_checkpoint.side_effect = None
        tracer.load_latest_checkpoint.return_value = durable
        tracer.checkpoint.side_effect = None
        result = await runner.resume("atomic")
        assert result["success"]
        restored = runner.context_manager.get_context("atomic")
        assert [m.content for m in restored.messages].count("new") == 1
    finally:
        release.set()
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["callback", "watermark"])
async def test_post_commit_cancellation_still_interrupts_live_run(live, cancel_at):
    from xagent.core.agent.runner import ExecutionControl

    runner, context, tracer = live
    entered = asyncio.Event()

    async def callback(**kwargs):
        if cancel_at == "callback":
            entered.set()
            await asyncio.Event().wait()
        else:
            context.metadata["_user_message_trace_watermark"] = "turn"

    async def write(**payload):
        if payload["label"] == "user_message_trace_watermark":
            entered.set()
            await asyncio.Event().wait()

    runner.callbacks = [SimpleNamespace(on_user_message_posted=callback)]
    tracer.checkpoint.side_effect = write
    runtime = PatternRuntime(execution_id="atomic", tracer=tracer)
    runner._active_controls["atomic"] = ExecutionControl(runtime=runtime, task=None)
    operation = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert await runtime.should_interrupt()
        assert context.messages[-1].content == "new"
    finally:
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_run_completion_waits_for_in_flight_injection(live):
    runner, context, tracer = live
    running, finish, writing, drain = (asyncio.Event() for _ in range(4))

    class Pattern:
        async def run(self, **kwargs):
            running.set()
            await finish.wait()
            return {"success": True, "output": "prior output"}

    runner.agent.patterns = [Pattern()]
    operation = asyncio.create_task(
        runner.run(
            "original", execution_id="atomic", checkpoint={"context": context.to_dict()}
        )
    )
    await running.wait()

    async def write(**payload):
        writing.set()
        await drain.wait()
        tracer.load_latest_checkpoint.side_effect = RuntimeError("read unavailable")
        raise RuntimeError("lost acknowledgement")

    tracer.checkpoint.side_effect = write
    injection = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="turn")
    )
    try:
        await writing.wait()
        finish.set()
        await asyncio.sleep(0)
        assert not operation.done()
        drain.set()
        assert (await injection).outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
        result = await operation
        assert result.get("status") != "interrupted"
        assert result["injection_outcome_unknown"] is True
        assert result["success"]
        assert result["output"] == "prior output"
        assert [
            m.content for m in runner.context_manager.get_context("atomic").messages
        ].count("new") == 0
    finally:
        finish.set()
        drain.set()
        await asyncio.gather(operation, injection, return_exceptions=True)


@pytest.mark.asyncio
async def test_live_input_after_result_publication_defers_until_old_run_finishes(live):
    runner, context, tracer = live
    publishing, finish = asyncio.Event(), asyncio.Event()

    async def on_run_end(**kwargs):
        publishing.set()
        await finish.wait()

    runner.callbacks = [SimpleNamespace(on_run_end=on_run_end)]
    runner.agent.patterns = [
        SimpleNamespace(run=AsyncMock(return_value={"success": True, "output": "done"}))
    ]
    operation = asyncio.create_task(
        runner.run(
            "original", execution_id="atomic", checkpoint={"context": context.to_dict()}
        )
    )
    try:
        await publishing.wait()
        result = await runner.inject_user_message("atomic", "new", turn_id="turn")
        assert result.outcome is UserMessageInjectionOutcome.NOT_POSTED
        tracer.checkpoint.assert_not_awaited()
        finish.set()
        assert (await operation)["success"]
        # The existing deferred path posts without interrupting the finished run.
        result = await runner.inject_user_message(
            "atomic", "new", turn_id="turn", request_interrupt=False
        )
        assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    finally:
        finish.set()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_explicit_new_input_reloads_uncertain_idle_context(live):
    from xagent.core.agent.context.execution import context_checkpoint_gate

    runner, old_context, tracer = live
    durable = None

    async def write(**payload):
        nonlocal durable
        durable = payload
        tracer.load_latest_checkpoint.side_effect = RuntimeError("unavailable")
        raise RuntimeError("lost ack")

    tracer.checkpoint.side_effect = write
    first = await runner.inject_user_message(
        "atomic", "first", turn_id="first", request_interrupt=False
    )
    assert first.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    tracer.load_latest_checkpoint.side_effect = None
    tracer.load_latest_checkpoint.return_value = durable
    tracer.checkpoint.side_effect = None
    second = await runner.inject_user_message(
        "atomic", "second", turn_id="second", request_interrupt=False
    )
    assert second.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert second.context is not old_context
    assert [m.content for m in second.context.messages] == [
        "original",
        "first",
        "second",
    ]
    assert context_checkpoint_gate(old_context).injection_uncertain
    assert tracer.checkpoint.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("active_run", [False, True])
async def test_fenced_context_rejects_later_input_without_writing(live, active_run):
    from xagent.core.agent.context.execution import context_checkpoint_gate
    from xagent.core.agent.runner import track_user_message_injection

    runner, context, tracer = live

    async def write(**payload):
        tracer.load_latest_checkpoint.side_effect = RuntimeError("unavailable")
        raise RuntimeError("lost ack")

    tracer.checkpoint.side_effect = write
    first = await runner.inject_user_message("atomic", "first", turn_id="first")
    assert first.outcome is UserMessageInjectionOutcome.OUTCOME_UNKNOWN
    assert context_checkpoint_gate(context).injection_uncertain
    tracer.load_latest_checkpoint.side_effect = None
    tracer.checkpoint.side_effect = None
    if active_run:
        runner._active_controls["atomic"] = SimpleNamespace(
            runtime=SimpleNamespace(last_checkpoint=None)
        )

    with track_user_message_injection() as attempt:
        second = await runner.inject_user_message(
            "atomic",
            "second",
            turn_id="second",
            # A live interrupt is fenced even when no run is active; a deferred
            # input is fenced only while the old execution still runs.
            request_interrupt=not active_run,
        )

    assert second.outcome is UserMessageInjectionOutcome.REJECTED_RETRYABLE
    assert attempt.outcome is UserMessageInjectionOutcome.REJECTED_RETRYABLE
    # Truthy so that no caller mistakes it for a deferrable NOT_POSTED.
    assert second.outcome
    assert second.context is context
    assert runner.context_manager.get_context("atomic") is context
    assert [m.content for m in context.messages] == ["original"]
    assert context_checkpoint_gate(context).injection_uncertain
    assert tracer.checkpoint.await_count == 1


@pytest.mark.asyncio
async def test_confirmed_readback_does_not_leave_transient_unknown_result(live):
    from xagent.core.agent.context.execution import context_checkpoint_gate

    runner, context, tracer = live
    started, interrupt, reading, release = (asyncio.Event() for _ in range(4))
    payload = None

    class Pattern:
        async def run(self, **kwargs):
            started.set()
            await interrupt.wait()
            raise ExecutionInterrupted("user pause")

    runner.agent.patterns = [Pattern()]
    operation = asyncio.create_task(
        runner.run(
            "original", execution_id="atomic", checkpoint={"context": context.to_dict()}
        )
    )
    await started.wait()

    async def read(_):
        reading.set()
        await release.wait()
        return payload

    async def write(**candidate):
        nonlocal payload
        payload = candidate
        tracer.load_latest_checkpoint.side_effect = read
        raise RuntimeError("lost ack")

    tracer.checkpoint.side_effect = write
    injection = asyncio.create_task(
        runner.inject_user_message("atomic", "new", turn_id="new")
    )
    try:
        await reading.wait()
        interrupt.set()
        gate = context_checkpoint_gate(runner.context_manager.get_context("atomic"))

        async def finalizer_waiting():
            while not gate._waiters:
                await asyncio.sleep(0)

        await asyncio.wait_for(finalizer_waiting(), 2)
        release.set()
        assert (await injection).outcome is UserMessageInjectionOutcome.POSTED_FRESH
        assert (await operation)["injection_outcome_unknown"] is False
    finally:
        release.set()
        interrupt.set()
        await asyncio.gather(operation, injection, return_exceptions=True)


@pytest.mark.asyncio
async def test_exceptional_run_exit_closes_live_admission(live):
    from xagent.core.agent.checkpoint import CheckpointPersistenceError

    runner, context, tracer = live

    class Pattern:
        async def run(self, **kwargs):
            raise CheckpointPersistenceError("failed checkpoint")

    runner.agent.patterns = [Pattern()]
    with pytest.raises(CheckpointPersistenceError):
        await runner.run(
            "original", execution_id="atomic", checkpoint={"context": context.to_dict()}
        )
    posted = await runner.inject_user_message("atomic", "late", turn_id="late")
    assert posted.outcome is UserMessageInjectionOutcome.NOT_POSTED
    tracer.checkpoint.assert_not_awaited()


@pytest.mark.asyncio
async def test_acceptance_evidence_survives_postwrite_registry_error(live):
    from xagent.core.agent.registry import ExecutionRegistry
    from xagent.core.agent.runner import track_user_message_injection

    runner, context, tracer = live
    registry = ExecutionRegistry()
    registry.register("atomic", runner)
    token = registry.subscribe(lambda _: registry.unsubscribe(token))
    with track_user_message_injection() as attempt:
        with pytest.raises(RuntimeError, match="dictionary changed"):
            await registry.post_user_message("atomic", "new", turn_id="new")
    assert attempt.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    assert context.messages[-1].content == "new"


@pytest.mark.asyncio
@pytest.mark.parametrize("after_write", [False, True])
async def test_cancellation_evidence_comes_from_write_boundary(live, after_write):
    from xagent.core.agent.runner import track_user_message_injection

    runner, context, tracer = live
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    if after_write:
        tracer.checkpoint.side_effect = blocked
    else:
        tracer.load_latest_checkpoint.side_effect = blocked
    with track_user_message_injection() as attempt:
        operation = asyncio.create_task(
            runner.inject_user_message("atomic", "new", turn_id="new")
        )
        await entered.wait()
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
    assert attempt.outcome is (
        UserMessageInjectionOutcome.OUTCOME_UNKNOWN
        if after_write
        else UserMessageInjectionOutcome.NOT_POSTED
    )


@pytest.mark.asyncio
async def test_write_only_checkpoint_wrapper_cannot_claim_authoritative_absence(live):
    from xagent.core.agent.checkpoint import (
        CheckpointUnavailableError,
        TraceCheckpointStore,
    )

    runner, context, tracer = live
    backend = SimpleNamespace(checkpoint=AsyncMock())
    runner.tracer = TraceCheckpointStore(backend)
    with pytest.raises(CheckpointUnavailableError, match="no readable backend"):
        await runner.inject_user_message("atomic", "new", turn_id="new")
    backend.checkpoint.assert_not_awaited()


_O = UserMessageInjectionOutcome


@pytest.mark.parametrize(
    "attempt,posted,error,expected",
    [
        # Returned outcomes.
        (_O.POSTED_FRESH, _O.POSTED_FRESH, None, "ACCEPTED"),
        (_O.POSTED_REPLAY, _O.POSTED_REPLAY, None, "ACCEPTED"),
        (_O.OUTCOME_UNKNOWN, _O.OUTCOME_UNKNOWN, None, "UNKNOWN"),
        (_O.REJECTED_RETRYABLE, _O.REJECTED_RETRYABLE, None, "NOT_ACCEPTED_RETRYABLE"),
        (_O.NOT_POSTED, _O.NOT_POSTED, None, "DEFER"),
        # A service layer that forwards the value without recording evidence.
        (_O.NOT_POSTED, _O.POSTED_FRESH, None, "ACCEPTED"),
        (_O.NOT_POSTED, None, None, "DEFER"),
        # Recorded acceptance wins over any later error, including cancellation.
        (_O.POSTED_FRESH, None, asyncio.CancelledError(), "ACCEPTED"),
        (_O.POSTED_REPLAY, None, RuntimeError("registry event"), "ACCEPTED"),
        (_O.OUTCOME_UNKNOWN, None, asyncio.CancelledError(), "UNKNOWN"),
        (_O.OUTCOME_UNKNOWN, None, RuntimeError("late"), "UNKNOWN"),
        (
            _O.NOT_POSTED,
            None,
            UserMessageInjectionRejectedError("absent"),
            "NOT_ACCEPTED_RETRYABLE",
        ),
        (_O.REJECTED_RETRYABLE, None, RuntimeError("lease"), "NOT_ACCEPTED_RETRYABLE"),
        (_O.NOT_POSTED, None, RuntimeError("read failed"), "FAILED_BEFORE_WRITE"),
        (_O.NOT_POSTED, None, asyncio.CancelledError(), "FAILED_BEFORE_WRITE"),
        (
            _O.NOT_POSTED,
            None,
            UserMessageInjectionConflictError("conflict"),
            "FAILED_BEFORE_WRITE",
        ),
    ],
)
def test_classify_injection_rules(attempt, posted, error, expected):
    from xagent.core.agent.runner import InjectionDisposition, classify_injection

    assert classify_injection(attempt, posted=posted, error=error) is getattr(
        InjectionDisposition, expected
    )


def test_classify_injection_rejects_untyped_outcome():
    from xagent.core.agent.runner import classify_injection

    with pytest.raises(TypeError):
        classify_injection(_O.NOT_POSTED, posted=False)


@pytest.mark.asyncio
async def test_real_runner_outcomes_classify_at_the_write_boundary(live):
    """The classifier agrees with what the runner actually records."""
    from xagent.core.agent.runner import (
        InjectionDisposition,
        classify_injection,
        track_user_message_injection,
    )

    runner, context, tracer = live
    # Confirmed absence: the write failed and the read-back shows no turn.
    tracer.checkpoint.side_effect = RuntimeError("write failed")
    with track_user_message_injection() as attempt:
        with pytest.raises(UserMessageInjectionRejectedError) as rejected:
            await runner.inject_user_message("atomic", "new", turn_id="absent")
    assert (
        classify_injection(attempt.outcome, error=rejected.value)
        is InjectionDisposition.NOT_ACCEPTED_RETRYABLE
    )

    # Accepted, then cancelled while tracing the accepted turn.
    tracer.checkpoint.side_effect = None

    async def cancelled_callback(**kwargs):
        raise asyncio.CancelledError()

    runner.callbacks = [SimpleNamespace(on_user_message_posted=cancelled_callback)]
    with track_user_message_injection() as attempt:
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await runner.inject_user_message("atomic", "new", turn_id="accepted")
    assert context.messages[-1].content == "new"
    assert (
        classify_injection(attempt.outcome, error=cancelled.value)
        is InjectionDisposition.ACCEPTED
    )
