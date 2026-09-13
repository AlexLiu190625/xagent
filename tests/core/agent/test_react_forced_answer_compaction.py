"""The forced-answer turn keeps its evidence, and says so when it lost some.

A turn whose schema is already down to ``final_answer`` does not compact, so
the values it answers from are still there.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from tests.core.agent.forced_answer_harness import (
    OBSERVATION_MARKER,
    CompactingLLM,
    ScriptedLLM,
    build_context,
    prompt_text,
)
from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.context.execution import (
    TOOL_EVIDENCE_REMOVED_METADATA_KEY as KEY,
)
from xagent.core.model.chat.exceptions import LLMToolProtocolError

_KEEP_DEFAULT = object()


def tool_call(name: str, arguments: str = "{}") -> dict[str, Any]:
    call = {
        "id": f"c-{name}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    return {"content": "", "tool_calls": [call]}


def final_answer_call() -> dict[str, Any]:
    return tool_call("final_answer", '{"answer": "done", "outcome": "completed"}')


def empty_final_answer_call() -> dict[str, Any]:
    return tool_call("final_answer", '{"answer": "", "outcome": "completed"}')


async def run_one_turn(
    *,
    context: ExecutionContext,
    forced: bool,
    llm: Any | None = None,
    compact_llm: Any | None = _KEEP_DEFAULT,
    pattern: ReActPattern | None = None,
) -> tuple[ReActPattern, ScriptedLLM]:
    """Drive exactly one ReAct iteration, forced or ordinary."""
    llm = llm or ScriptedLLM([final_answer_call()])
    pattern = pattern or ReActPattern(max_iterations=1)
    pattern.force_final_answer_next = forced
    runtime = PatternRuntime(execution_id=context.execution_id)
    await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=runtime,
        compact_llm=CompactingLLM() if compact_llm is _KEEP_DEFAULT else compact_llm,
    )
    return pattern, llm


# --------------------------------------------------------------------------
# Invariant A -- a forced turn does not compact
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forced_turn_keeps_every_observation_and_does_not_compact() -> None:
    """T-A1: the raw observation text reaches the model on a forced turn."""
    context = build_context(observations=6, threshold=500)
    before = len(context.messages)

    _, llm = await run_one_turn(context=context, forced=True)

    sent = prompt_text(llm.calls[0]["messages"])
    assert OBSERVATION_MARKER.format(index=0) in sent
    assert OBSERVATION_MARKER.format(index=5) in sent
    assert len(context.messages) >= before  # nothing was taken away
    assert "Compacted conversation summary:" not in sent
    assert context.metadata[KEY] is False


@pytest.mark.asyncio
async def test_ordinary_turn_still_compacts_and_latches() -> None:
    """T-A2: the skip is confined to the forced turn."""
    context = build_context(observations=6, threshold=500)
    before = len(context.messages)

    await run_one_turn(context=context, forced=False)

    assert len(context.messages) < before
    assert context.metadata[KEY] is True


@pytest.mark.asyncio
async def test_route_and_call_prompts_match_only_on_the_forced_turn() -> None:
    """T-A3: routing and the real call see the same messages when nothing compacts.

    Both prompts are locals inside the loop, so ``_messages_for_llm`` is wrapped
    to record what it returned. Asserting only "called twice" would pass under
    every implementation, including the broken one.
    """
    built: list[list[dict[str, Any]]] = []

    def record(pattern: ReActPattern) -> ReActPattern:
        original = pattern._messages_for_llm

        def wrapper(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            built.append(original(*args, **kwargs))
            return built[-1]

        pattern._messages_for_llm = wrapper  # type: ignore[method-assign]
        return pattern

    for forced, same in ((True, True), (False, False)):
        built.clear()
        await run_one_turn(
            context=build_context(observations=6, threshold=500),
            forced=forced,
            pattern=record(ReActPattern(max_iterations=1)),
        )
        assert (built[0] == built[1]) is same


class ProtocolErrorOnFirstCallLLM(ScriptedLLM):
    """Replays queued responses, but raises a protocol error on the first call.

    One of the two exits that clear the forced flag is only reachable this
    way: the code that says the model called a tool the narrowed schema no
    longer offers arrives as an adapter exception, not inside a response body.
    """

    def __init__(self, responses: list[Any], *, code: str) -> None:
        super().__init__(responses)
        self.code = code

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise LLMToolProtocolError(
                provider="fake", code=self.code, message="tool is unavailable"
            )
        if not self.responses:
            return final_answer_call()
        return self.responses.pop(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # Built per run rather than held in this list: a scripted model is spent
    # once it has replayed its queue.
    "make_llm, compactions_expected",
    [
        (lambda: ScriptedLLM([final_answer_call()]), 0),
        (lambda: ScriptedLLM([{"content": "plain text answer", "tool_calls": []}]), 0),
        (
            lambda: ScriptedLLM([empty_final_answer_call(), empty_final_answer_call()]),
            0,
        ),
        (
            lambda: ScriptedLLM([tool_call("list_clients"), tool_call("list_clients")]),
            1,
        ),
        (
            lambda: ProtocolErrorOnFirstCallLLM(
                [tool_call("list_clients")], code="unavailable_tool_call"
            ),
            1,
        ),
    ],
    ids=[
        "final_answer",
        "plain_text",
        "empty_answer_then_repair_fails",
        "out_of_schema_tool",
        "unavailable_tool_call",
    ],
)
async def test_skip_never_spans_more_than_one_turn(
    caplog: pytest.LogCaptureFixture,
    make_llm: Any,
    compactions_expected: int,
) -> None:
    """T-A4: the skipped compaction covers one turn and never a second.

    Two turns are offered to every shape. What is asserted is how many turns
    skipped compaction -- counted off the production log line, one per skip --
    and how many turns compacted. Asserting ``force_final_answer_next is
    False`` instead would hold under every implementation, right or wrong:
    ``_finalize_success`` clears that field on any run that ends normally, so
    it cannot tell an exit that cleared the flag itself from a run that merely
    finished.

    The two exits that clear the flag are separate lines in separate handlers,
    so they get one shape each; deleting either one leaves the other green.
    """
    context = build_context(observations=6, threshold=500)
    runtime = PatternRuntime(execution_id=context.execution_id)
    compactions: list[Any] = []
    compact = runtime.compact_context_if_needed

    async def counted(**kwargs: Any) -> Any:
        compactions.append(kwargs.get("metadata"))
        return await compact(**kwargs)

    runtime.compact_context_if_needed = counted  # type: ignore[method-assign]
    pattern = ReActPattern(max_iterations=2)
    pattern.force_final_answer_next = True

    with caplog.at_level(logging.INFO, logger="xagent.core.agent.pattern.react.react"):
        await pattern.run(
            context=context,
            tools=[],
            llm=make_llm(),
            runtime=runtime,
            compact_llm=CompactingLLM(),
        )

    skipped = [r for r in caplog.records if "did not compact" in r.getMessage()]
    assert len(skipped) == 1
    assert len(compactions) == compactions_expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forced, threshold, expected",
    [
        (True, 500, "over_threshold=True"),
        (True, 10**6, "over_threshold=False"),
        (False, 500, None),
    ],
    ids=["forced_over", "forced_under", "ordinary_turn"],
)
async def test_the_skip_is_logged_with_numbers_only(
    caplog: pytest.LogCaptureFixture, forced: bool, threshold: int, expected: str | None
) -> None:
    """T-A5: one info line per skipped compaction, ids and numbers only.

    ``over_threshold`` is what separates a skip that mattered from one on a
    turn that was never going to compact anyway.
    """
    context = build_context(observations=6, threshold=threshold)
    with caplog.at_level(logging.INFO, logger="xagent.core.agent.pattern.react.react"):
        await run_one_turn(context=context, forced=forced)

    lines = [
        r.getMessage() for r in caplog.records if "did not compact" in r.getMessage()
    ]
    if expected is None:
        assert lines == []
        return
    assert len(lines) == 1
    assert f"execution_id={context.execution_id}" in lines[0]
    assert "iteration=0" in lines[0]
    assert f"threshold={threshold}" in lines[0]
    assert expected in lines[0]
    assert OBSERVATION_MARKER.format(index=0) not in lines[0]
    assert "list_clients" not in lines[0]
