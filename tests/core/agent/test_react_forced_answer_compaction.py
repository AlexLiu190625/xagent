"""The forced-answer turn keeps its evidence, and says so when it lost some.

A turn whose schema is already down to ``final_answer`` does not compact, so
the values it answers from are still there; and once any compaction on this
context removed an observation, every prompt that answers -- or decides to --
without tools says so instead of claiming the results accumulated.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import pytest

from tests.core.agent.forced_answer_harness import (
    OBSERVATION_MARKER,
    CompactingLLM,
    ScriptedLLM,
    WindowlessCompactingLLM,
    build_context,
    prompt_text,
)
from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.context import ContextManager
from xagent.core.agent.context.execution import (
    TOOL_EVIDENCE_REMOVED_METADATA_KEY as KEY,
)
from xagent.core.agent.context.execution import (
    CompactResult,
    note_compaction_evidence_loss,
    tool_evidence_removed,
)
from xagent.core.agent.grounding import EVIDENCE_REMOVED_FACTS, grounding_rule
from xagent.core.agent.pattern.auto.auto import (
    DECISION_TOOL_NAME,
    AutoAction,
    AutoPattern,
)
from xagent.core.agent.pattern.dag.dag import DAGPattern
from xagent.core.model.chat.exceptions import (
    LLMContextLengthError,
    LLMToolProtocolError,
)

_KEEP_DEFAULT = object()
FACTS_HEAD = EVIDENCE_REMOVED_FACTS.split(".")[0]


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
    """The raw observation text reaches the model on a forced turn."""
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
    """The skip is confined to the forced turn."""
    context = build_context(observations=6, threshold=500)
    before = len(context.messages)

    await run_one_turn(context=context, forced=False)

    assert len(context.messages) < before
    assert context.metadata[KEY] is True


@pytest.mark.asyncio
async def test_route_and_call_prompts_match_only_on_the_forced_turn() -> None:
    """Routing and the real call see the same messages when nothing compacts.

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
    """The skipped compaction covers one turn and never a second.

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
    """One info line per skipped compaction, ids and numbers only.

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


# --------------------------------------------------------------------------
# Invariant B -- latching, and what each compaction shape does to the marker
# --------------------------------------------------------------------------


def _result(**metadata: Any) -> CompactResult:
    return CompactResult(
        compacted=True,
        original_count=10,
        final_count=2,
        strategy="llm_summary",
        metadata=dict(metadata),
    )


@pytest.mark.parametrize(
    "result, expected",
    [
        (_result(dropped_tool_result_count=6), True),
        (_result(dropped_tool_result_count=1), True),
        (_result(removed_count=11, dropped_tool_result_count=0), False),
        (_result(removed_count=0), False),
        (CompactResult(False, 3, 3, "none", {}), False),
        (None, False),
        (_result(fallback_suppressed=True), False),
        (_result(dropped_context_ref_count=4, dropped_tool_result_count=0), False),
        (_result(dropped_tool_result_count=True), False),
        (_result(dropped_tool_result_count="6"), False),
        (_result(llm_compact_error="boom", dropped_tool_result_count=3), True),
        (_result(llm_summary_unusable=True, dropped_tool_result_count=3), True),
    ],
    ids="summary_dropped_six truncate_dropped_one assistant_text_only "
    "tail_window_kept_all under_threshold runtime_returned_none "
    "compact_window_unknown image_refs_only count_is_a_bool count_is_a_string "
    "summary_raised_then_truncated summary_unusable_then_truncated".split(),
)
def test_latch_reads_dropped_observations_not_compacted(
    result: Any, expected: bool
) -> None:
    """Only a removed observation latches the marker."""
    context = ContextManager().create_context(execution_id="latch")
    note_compaction_evidence_loss(context, result)
    assert context.metadata[KEY] is expected


def test_latch_is_monotonic() -> None:
    """A later lossless compaction does not clear an earlier loss."""
    context = ContextManager().create_context(execution_id="latch-monotonic")
    note_compaction_evidence_loss(context, _result(dropped_tool_result_count=3))
    note_compaction_evidence_loss(context, _result(dropped_tool_result_count=0))
    assert context.metadata[KEY] is True


def test_only_one_place_in_src_writes_the_marker_false() -> None:
    """Nothing resets the marker; the single False write is the stamp."""
    src = pathlib.Path(__file__).resolve().parents[3] / "src" / "xagent"
    writes = {
        str(path.relative_to(src))
        for path in src.rglob("*.py")
        for line in path.read_text().splitlines()
        if "TOOL_EVIDENCE_REMOVED_METADATA_KEY]" in line and "False" in line
    }
    assert writes == {"core/agent/context/manager.py"}


@pytest.mark.asyncio
async def test_truncate_path_latches_without_writing_a_notice() -> None:
    """The truncate path stays silent; the marker is what carries it.

    Reached with no summarizer at all, the only way to get it: the ReAct call
    site substitutes the main model when no compact model is set.
    """
    context = build_context(observations=6, threshold=500)
    runtime = PatternRuntime(execution_id=context.execution_id)
    result = await runtime.compact_context_if_needed(context=context, llm=None)
    note_compaction_evidence_loss(context, result)

    assert result.strategy == "truncate"
    assert context.metadata[KEY] is True
    body = "\n".join(message.content for message in context.messages)
    assert "Compacted conversation summary:" not in body
    assert "dropped by this compaction" not in body


@pytest.mark.asyncio
async def test_windowless_compact_model_does_not_latch() -> None:
    """The marker also latches when no window is declared, the one way that cell is reachable."""
    context = build_context(observations=6, threshold=500)
    await run_one_turn(
        context=context, forced=False, compact_llm=WindowlessCompactingLLM()
    )
    assert context.metadata[KEY] is False
    assert len(context.messages) > 2


@pytest.mark.asyncio
async def test_loss_in_an_earlier_turn_reaches_the_later_forced_turn() -> None:
    """The cross-turn cell -- compaction one turn, forced answer the next."""
    context = build_context(observations=6, threshold=500)
    await run_one_turn(context=context, forced=False)
    assert context.metadata[KEY] is True

    _, llm = await run_one_turn(context=context, forced=True)
    sent = prompt_text(llm.calls[0]["messages"])
    assert FACTS_HEAD in sent
    assert "accumulated" not in sent


class RoutingLLM(ScriptedLLM):
    """A routing model: declares a window, and always picks ``react``."""

    context_window = 200_000

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return tool_call(
            DECISION_TOOL_NAME,
            json.dumps(
                {
                    "action": "react",
                    "reason": "the request needs tools",
                    "requires_current_or_external_facts": True,
                    "existing_context_sufficient": False,
                }
            ),
        )


@pytest.mark.asyncio
async def test_auto_routing_loss_reaches_the_react_forced_turn() -> None:
    """The loss Auto's own compaction caused is stated downstream.

    Driven through ``AutoPattern._decide`` rather than by calling the compact
    runtime and the latch by hand: what is under test is that the routing
    stage latches at all, so the latch has to come from production code. A
    cell that latched inside its own body would stay green with that call
    site deleted.
    """
    context = build_context(observations=6, threshold=500)
    runtime = PatternRuntime(execution_id=context.execution_id)
    routing_llm = RoutingLLM()

    decision = await AutoPattern()._decide(
        context=context,
        tools=[],
        llm=routing_llm,
        compact_llm=CompactingLLM(),
        runtime=runtime,
    )

    assert decision.decision.action is AutoAction.REACT
    assert context.metadata[KEY] is True
    routed = prompt_text(routing_llm.calls[0]["messages"])
    assert FACTS_HEAD in routed
    assert "accumulated" not in routed

    # Auto hands the same context object to the pattern it routed to.
    _, llm = await run_one_turn(context=context, forced=True)
    assert FACTS_HEAD in prompt_text(llm.calls[0]["messages"])


# --------------------------------------------------------------------------
# Invariant C -- the four prompts that answer, or decide to answer, tool-less
# --------------------------------------------------------------------------


def _react_forced(context: ExecutionContext) -> str:
    return prompt_text(
        ReActPattern()._messages_for_llm(
            context,
            has_tools=True,
            force_final_answer=True,
            tool_names=["final_answer"],
        )
    )


def _react_decision(context: ExecutionContext) -> str:
    # The prompt this builder writes is the last message; the ones before it
    # are history, which gap 8 covers and this cell does not.
    return ReActPattern()._messages_for_repeated_tool_decision(
        context, {"tool_name": "list_clients", "consecutive_tool_calls": 6}
    )[-1]["content"]


def _dag_assessment(context: ExecutionContext) -> str:
    return DAGPattern(lambda **_: None)._completion_assessment_messages(context)[0][
        "content"
    ]


def _auto_decision(context: ExecutionContext) -> str:
    return AutoPattern()._decision_prompt(
        [], evidence_removed=tool_evidence_removed(context)
    )


# Main's wording at the spots this change splits into branches, quoted whole
# rather than probed for a few words: probing catches a deleted phrase but not
# a rewritten one, and "only when" turned into "whenever" inverts the rule
# while leaving every probe satisfied. The quote stops where main's own shared
# fragments begin -- grounding_rule and the file-reference and language rules
# are not this change's text, and pinning them here would redden this cell for
# an edit made somewhere else, whose cheapest repair is to edit this value.
MAIN_FORCED_ANSWER_OPENING = (
    "Produce the final user-facing answer by calling the final_answer "
    "control tool exactly once using the accumulated conversation and tool "
    "results. Do not call any other tool and do not output tool-call markup "
    "as plain text. Set outcome=completed only when every requested action "
    "or verification succeeded; otherwise set outcome=partial or "
    "outcome=blocked and say what remains. If a previous ask_user_question "
    "narrowed the request to a selected subset of items or resources, the "
    "final answer must cover only that subset — leave out anything outside "
    "it even if an earlier tool call already returned data about it. "
)
MAIN_DECISION_COMPLETION_CLAUSE = (
    "Use this as the controlling request when deciding whether the "
    "accumulated tool results have completed the user's requested work."
)
MAIN_DECISION_SUFFICIENCY_CLAUSE = (
    "Choose final_answer when the conversation and accumulated tool results "
    "are sufficient to answer the latest user request."
)


CONSUMERS = [
    pytest.param(
        _react_forced, (MAIN_FORCED_ANSWER_OPENING,), True, id="react_forced_answer"
    ),
    pytest.param(
        _react_decision,
        (MAIN_DECISION_COMPLETION_CLAUSE, MAIN_DECISION_SUFFICIENCY_CLAUSE),
        False,
        id="react_repeated_tool_decision",
    ),
    pytest.param(_dag_assessment, (), True, id="dag_completion_assessment"),
    pytest.param(_auto_decision, (), True, id="auto_routing_decision"),
]


@pytest.mark.parametrize("build, _main_fragments, _carries_grounding_rule", CONSUMERS)
def test_every_toolless_answer_prompt_states_the_loss(
    build: Any, _main_fragments: tuple[str, ...], _carries_grounding_rule: bool
) -> None:
    """Every tool-less answer prompt carries the facts, and none of them claims accumulated results.

    The scope of the second assertion is the prompt this builder writes for
    this turn, not the whole payload: a guidance line written into the context
    on an earlier turn can still carry main's wording, which is an acknowledged
    gap no per-turn builder can reach.
    """
    context = build_context(observations=2, threshold=500)
    context.metadata[KEY] = True
    assert FACTS_HEAD in build(context)
    assert "accumulated" not in build(context)


@pytest.mark.parametrize("build, _main_fragments, carries_grounding_rule", CONSUMERS)
def test_the_loss_is_stated_before_the_shared_grounding_rule(
    build: Any, _main_fragments: tuple[str, ...], carries_grounding_rule: bool
) -> None:
    """Which of the two comes first is a decision, so it is pinned at every site.

    The three prompts that carry both state what is no longer readable before
    the wider rule about answering from sources. The fourth builder carries no
    shared rule at all; that exclusion is asserted rather than left silent, so
    adding the rule there without picking an order reddens this cell.
    """
    context = build_context(observations=2, threshold=500)
    context.metadata[KEY] = True
    text = build(context)
    grounding_head = grounding_rule(can_call_tools=False).split(".")[0]
    assert FACTS_HEAD in text
    if not carries_grounding_rule:
        assert grounding_head not in text
        return
    assert text.index(FACTS_HEAD) < text.index(grounding_head)


@pytest.mark.parametrize("build, _main_fragments, _carries_grounding_rule", CONSUMERS)
def test_a_run_that_never_lost_anything_reads_exactly_like_main(
    build: Any, _main_fragments: tuple[str, ...], _carries_grounding_rule: bool
) -> None:
    """The marker-false branch is word-for-word main's wording.

    The context must come from ``ContextManager.create_context``. One built
    directly carries no key, reads as "evidence removed", and would make this
    cell fail for a reason that has nothing to do with the wording.
    """
    stamped = build_context(observations=2, threshold=500, via_manager=True)
    assert stamped.metadata[KEY] is False
    text = build(stamped)
    assert FACTS_HEAD not in text
    if build is _react_forced:
        assert MAIN_FORCED_ANSWER_OPENING in text
    if build is _react_decision:
        assert MAIN_DECISION_COMPLETION_CLAUSE in text
        assert MAIN_DECISION_SUFFICIENCY_CLAUSE in text


def test_a_directly_built_context_reads_as_evidence_removed() -> None:
    """Deliberate, and pinned so nobody later "fixes" it to False.

    Production stamps the key at its one brand-new-context site, so this shape
    is unreachable there.
    """
    assert tool_evidence_removed(ExecutionContext(system_prompt="s")) is True


def test_the_outcome_rule_stays_a_conditional() -> None:
    """A dropped result message is not a dropped action."""
    context = build_context(observations=2, threshold=500)
    context.metadata[KEY] = True
    text = _react_forced(context)
    assert "outcome=completed" in text
    assert "Do not rest an outcome=completed claim on an observation" in text
    assert "outcome=completed is unavailable" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("removed", [True, False], ids=["removed", "intact"])
async def test_the_guidance_written_into_the_context_matches_the_marker(
    removed: bool,
) -> None:
    """This guidance is history, so it must agree with the next prompt."""
    context = build_context(observations=2, threshold=500)
    context.metadata[KEY] = removed
    pattern = ReActPattern()
    pattern.repeated_tool_decision = {
        "tool_name": "list_clients",
        "consecutive_tool_calls": 6,
    }
    decision = tool_call(
        "react_decision",
        '{"action": "final_answer", "reason": "r", "missing_verification": ""}',
    )
    await pattern._run_repeated_tool_decision(
        context=context,
        llm=ScriptedLLM([decision]),
        runtime=PatternRuntime(execution_id=context.execution_id),
    )
    guidance = context.messages[-1].content
    assert guidance.startswith("Repeated tool decision completion guidance")
    if removed:
        assert "accumulated" not in guidance
        assert "Compaction has removed tool observations from this run" in guidance
        # End to end: the forced turn that reads this message back must not
        # find main's claim sitting in its own payload.
        _, llm = await run_one_turn(context=context, forced=True)
        assert "accumulated" not in prompt_text(llm.calls[0]["messages"])
    else:
        assert guidance.endswith(
            "The repeated-tool decision selected final_answer, so the next "
            "normal ReAct step must produce the final user-facing answer from "
            "the accumulated conversation and tool results. Do not call more "
            "tools in that final step. Do not send a progress update or "
            "promise future work as the final answer; if the accumulated "
            "results are insufficient or show the task is incomplete, say "
            "that directly."
        )


@pytest.mark.asyncio
async def test_protocol_repair_retry_inherits_the_same_wording() -> None:
    """The repair retry inside a forced turn carries the same wording.

    The retry is reached, not simulated: the first response calls
    final_answer with an empty answer field, which the pattern treats as a
    protocol violation and repairs once. The second call is identified by the
    repair instruction only that path writes, so a cell that merely rebuilt
    the ordinary forced prompt could not pass as this one.
    """
    context = build_context(observations=2, threshold=500)
    context.metadata[KEY] = True
    llm = ScriptedLLM([empty_final_answer_call(), final_answer_call()])
    pattern = ReActPattern(max_iterations=1)
    pattern.force_final_answer_next = True

    await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=PatternRuntime(execution_id=context.execution_id),
        compact_llm=CompactingLLM(),
    )

    assert len(llm.calls) == 2
    repair = prompt_text(llm.calls[1]["messages"])
    assert "called final_answer with an empty answer field" in repair
    assert FACTS_HEAD in repair
    assert "accumulated" not in repair


# --------------------------------------------------------------------------
# Failure path
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_skipped_compaction_that_overflows_fails_the_run_cleanly() -> None:
    """Overflow ends the run -- no retry, no fallback compaction, no answer."""

    class OverflowLLM(ScriptedLLM):
        async def chat(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            raise LLMContextLengthError(
                "This model's maximum context length is 8192 tokens"
            )

    context = build_context(observations=6, threshold=500)
    assistants_before = sum(1 for m in context.messages if m.role == "assistant")
    llm = OverflowLLM()
    pattern = ReActPattern(max_iterations=2)
    pattern.force_final_answer_next = True

    with pytest.raises(LLMContextLengthError):
        await pattern.run(
            context=context,
            tools=[],
            llm=llm,
            runtime=PatternRuntime(execution_id=context.execution_id),
            compact_llm=CompactingLLM(),
        )

    assert len(llm.calls) == 1
    assert sum(1 for m in context.messages if m.role == "assistant") == (
        assistants_before
    )
    assert context.metadata[KEY] is False
