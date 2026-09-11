"""A run remembers which tool observations compaction destroyed.

Compaction can destroy tool observations in the middle of a run, and the turn
that loses them is often not the turn that later has to answer without them.
This file tests that ``ReActPattern`` keeps one durable record of that loss --
``lost_tool_evidence`` -- on the pattern instance for the life of the run, that
the record accumulates regardless of whether the turn that lost the evidence
was itself a forced-answer turn, that it survives a checkpoint round trip, and
that it is cleared once the run actually finishes. How a later forced-answer
turn reads this record back and changes its own instruction is a separate
concern this file does not cover.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.pattern.react.react import (
    LOST_TOOL_EVIDENCE_GRANULARITY,
    LOST_TOOL_EVIDENCE_MAX_CALL_IDS,
    CompactionLoss,
    LostToolEvidence,
)


class _EmptyArgs(BaseModel):
    pass


class NamedTool:
    """Minimal work tool whose schema name the test chooses."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

        class Metadata:
            pass

        self.metadata = Metadata()
        self.metadata.name = name  # type: ignore[attr-defined]
        self.metadata.description = f"Read {name}."  # type: ignore[attr-defined]

    def args_type(self) -> type[BaseModel]:
        return _EmptyArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        return {"output": f"{self.name} result"}


class RecordingLLM:
    """Chat LLM that records every call and replays scripted responses.

    ``context_window`` is required for real compaction to pick the summary
    path at all: without it, ``compact_context_if_needed`` cannot bound the
    summary request and reports the request as unfittable rather than
    running it.
    """

    context_window = 128_000

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            return {"content": "fallback answer", "done": True}
        return self.responses.pop(0)


def final_answer_response(answer: str = "Here is what remains.") -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call_final",
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "arguments": (
                        '{"response_language":"English","outcome":"partial",'
                        f'"answer":"{answer}"}}'
                    ),
                },
            }
        ],
        "done": False,
    }


def empty_final_answer_response(call_id: str = "call_final") -> dict[str, Any]:
    """A ``final_answer`` call whose ``answer`` field is blank.

    ``_response_requires_tool_protocol_retry`` rejects this and routes it
    into the pattern's single repair retry, whatever the retry's own
    response turns out to be.
    """
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "arguments": (
                        '{"response_language":"English","outcome":"partial","answer":""}'
                    ),
                },
            }
        ],
        "done": False,
    }


def build_context(
    *,
    tool_name: str = "calculator",
    observations: int = 3,
    max_messages: int = 4,
    result: Any | None = None,
    execution_id: str = "forced-answer-compaction",
) -> ExecutionContext:
    """Context primed to compact on every turn.

    ``max_messages`` decides whether the message-dropping fallback actually
    removes anything: a window wider than the transcript keeps every message
    and reports zero dropped observations even though it reports
    ``compacted=True``.
    """
    context = ExecutionContext(execution_id=execution_id)
    context.compact_config.threshold = 1
    context.compact_config.max_messages = max_messages
    context.add_user_message(f"Use {tool_name} and report every value.")
    for index in range(observations):
        call_id = f"seed_{index}"
        context.add_assistant_message(
            "",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name},
                }
            ],
        )
        context.add_tool_result(
            tool_name,
            {"output": "x" * 400} if result is None else result,
            tool_call_id=call_id,
        )
    return context


def state_at(
    runtime: PatternRuntime, label: str, occurrence: int = 0
) -> dict[str, Any]:
    matching = [entry for entry in runtime.checkpoints if entry.get("label") == label]
    return dict(matching[occurrence].get("pattern_state") or {})


def compact_result(
    *,
    compacted: bool = True,
    count: Any = 1,
    by_name: Any = None,
    call_ids: Any = None,
    without_call_id: Any = None,
    strategy: str = "llm_summary",
) -> Any:
    """Build a ``CompactResult`` with only the metadata keys under test set.

    ``call_ids`` and ``without_call_id`` let a test script the two keys this
    branch writes for identifying individual dropped calls, including
    leaving them absent or malformed -- shapes real compaction is not known
    to produce, but that ``_dropped_tool_evidence`` must still not
    misinterpret.
    """
    from xagent.core.agent.context.execution import CompactResult

    metadata: dict[str, Any] = {}
    if count is not None:
        metadata["dropped_tool_result_count"] = count
    if by_name is not None:
        metadata["dropped_tool_results_by_name"] = by_name
    if call_ids is not None:
        metadata["dropped_tool_result_call_ids"] = call_ids
    if without_call_id is not None:
        metadata["dropped_tool_results_without_call_id"] = without_call_id
    return CompactResult(
        compacted=compacted,
        original_count=9,
        final_count=2,
        strategy=strategy,
        metadata=metadata,
    )


def tool_call_response(tool_name: str, call_id: str = "call_work") -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }
        ],
        "done": False,
    }


@pytest.mark.asyncio
async def test_real_compaction_reports_the_keys_the_gate_reads() -> None:
    """Anchor every scripted-metadata test in this file to what compaction
    actually writes.

    Every ``compact_result()`` call elsewhere in this file claims a metadata
    shape that this test proves the two real compacting paths -- the LLM
    summary and the message-dropping backstop -- actually produce, so a
    rename or a semantic change upstream in ``ExecutionContext`` cannot leave
    the rest of this file passing against a fiction.
    """
    summary_context = build_context(max_messages=40)
    summary_result = await PatternRuntime().compact_context_if_needed(
        context=summary_context,
        llm=RecordingLLM([{"content": "A summary of the run."}]),
    )
    assert summary_result.compacted is True
    assert summary_result.strategy == "llm_summary"
    assert summary_result.metadata["dropped_tool_result_count"] == 3
    assert summary_result.metadata["dropped_tool_results_by_name"] == {"calculator": 3}
    assert set(summary_result.metadata["dropped_tool_result_call_ids"]) == {
        "seed_0",
        "seed_1",
        "seed_2",
    }
    assert summary_result.metadata["dropped_tool_results_without_call_id"] == 0

    drop_context = build_context(max_messages=4)
    drop_result = await PatternRuntime().compact_context_if_needed(
        context=drop_context,
        llm=RecordingLLM([{"content": ""}]),
    )
    assert drop_result.compacted is True
    assert drop_result.strategy == "truncate"
    assert drop_result.metadata["dropped_tool_result_count"] == 1
    assert drop_result.metadata["dropped_tool_results_by_name"] == {"calculator": 1}
    assert set(drop_result.metadata["dropped_tool_result_call_ids"]) == {"seed_0"}
    assert drop_result.metadata["dropped_tool_results_without_call_id"] == 0
    # The tail window kept a later successful observation of the same tool.
    # "Is there still tool evidence in context" would answer yes here, which
    # is exactly why the record identifies losses by call id instead of by
    # tool name.
    assert any(message.role == "tool" for message in drop_context.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_summary", [True, False], ids=["llm_summary", "truncate"])
@pytest.mark.parametrize("forced", [True, False], ids=["forced", "unforced"])
async def test_an_unforced_turn_still_records_what_it_lost(
    use_summary: bool, forced: bool
) -> None:
    """A turn folds compaction's loss into the run's record either way.

    Checked at the ``after_llm`` checkpoint, which is written right after
    compaction folds the loss into ``pattern.lost_tool_evidence`` but before
    the ``final_answer`` control tool's own finalize step -- which
    deliberately clears the record once the run is over, and is exactly what
    ``test_a_finished_run_leaves_no_lost_evidence_behind`` checks below. A
    live post-run read would show an empty record for every cell here
    precisely because that clearing step ran, which is why this test reads
    the mid-run checkpoint instead.

    Mutation this test catches: if the fold-in call in
    ``_run_tool_calling_loop`` were moved inside
    ``if force_final_answer_now:``, the unforced cell's checkpointed
    ``call_ids`` would read back empty here; with the actual unconditional
    fold-in it does not.
    """
    max_messages = 40 if use_summary else 4
    scripted_summary_content = "A summary of the run." if use_summary else ""

    reference_context = build_context(max_messages=max_messages)
    reference_result = await PatternRuntime().compact_context_if_needed(
        context=reference_context,
        llm=RecordingLLM([{"content": scripted_summary_content}]),
    )
    expected_call_ids = sorted(
        reference_result.metadata["dropped_tool_result_call_ids"]
    )
    assert expected_call_ids

    context = build_context(max_messages=max_messages)
    tools: list[Any] = [NamedTool("calculator")]
    llm = RecordingLLM([final_answer_response()])
    compact_llm = RecordingLLM([{"content": scripted_summary_content}])
    pattern = ReActPattern(max_iterations=6)
    pattern.force_final_answer_next = forced
    runtime = PatternRuntime()

    await pattern.run(
        context=context,
        tools=tools,
        llm=llm,
        compact_llm=compact_llm,
        runtime=runtime,
    )

    recorded = state_at(runtime, "after_llm")["lost_tool_evidence"]
    assert recorded["call_ids"] == expected_call_ids
    assert recorded["call_ids"] != []


@pytest.mark.parametrize(
    (
        "source_call_ids",
        "source_unnameable",
        "expected_call_ids",
        "expected_unnameable",
    ),
    [
        pytest.param(set(), False, set(), False, id="empty"),
        pytest.param({"call_1"}, False, {"call_1"}, False, id="one_id"),
        pytest.param(
            {"call_1", "call_2", "call_3"},
            False,
            {"call_1", "call_2", "call_3"},
            False,
            id="several_ids",
        ),
        pytest.param(set(), True, set(), True, id="unnameable_alone"),
        pytest.param(
            {"call_1", "call_2"},
            True,
            {"call_1", "call_2"},
            True,
            id="ids_plus_unnameable",
        ),
        pytest.param(
            {
                f"call_{index:04d}"
                for index in range(LOST_TOOL_EVIDENCE_MAX_CALL_IDS + 5)
            },
            False,
            set(
                sorted(
                    f"call_{index:04d}"
                    for index in range(LOST_TOOL_EVIDENCE_MAX_CALL_IDS + 5)
                )[:LOST_TOOL_EVIDENCE_MAX_CALL_IDS]
            ),
            True,
            id="more_ids_than_cap",
        ),
    ],
)
def test_the_lost_evidence_record_round_trips_through_a_checkpoint(
    source_call_ids: set[str],
    source_unnameable: bool,
    expected_call_ids: set[str],
    expected_unnameable: bool,
) -> None:
    """``get_state()``/``load_state()`` must reproduce the record on the
    other side.

    Without a correct round trip, a run resumed from a checkpoint would
    silently start with a wrong lost-evidence record: either forgetting real
    losses (a missing-key regression) or inventing ones that were never
    there.

    The over-cap cell does not expect the exact same call ids back: writing
    more ids than a read-back accepts is a legitimate outcome of a long run's
    own accumulation over many compactions, and the cap's truncation-on-read
    is what the dedicated over-cap test below pins down in detail. Included
    here only to confirm that ``get_state()`` itself never truncates -- only
    ``from_state()`` does -- so the source's full set survives the write side
    of the round trip even though the read side then shortens it.
    """
    source = ReActPattern()
    source.lost_tool_evidence = LostToolEvidence(
        call_ids=set(source_call_ids), unnameable=source_unnameable
    )
    state = source.get_state()
    assert len(state["lost_tool_evidence"]["call_ids"]) == len(source_call_ids)

    destination = ReActPattern()
    destination.load_state(state)

    assert destination.lost_tool_evidence.call_ids == expected_call_ids
    assert destination.lost_tool_evidence.unnameable is expected_unnameable


@pytest.mark.parametrize(
    "raw",
    [
        [],
        "x",
        {"call_ids": "x"},
        {"granularity": LOST_TOOL_EVIDENCE_GRANULARITY, "call_ids": [1]},
        {"granularity": "tool_name", "call_ids": []},
        # Both of these carry a call id rather than an empty list, and the
        # second carries a falsy value. Without either, a build that kept no
        # guard at all and simply coerced the field would land on the same
        # answer these cells expect, and the cell would pass against both.
        {
            "granularity": LOST_TOOL_EVIDENCE_GRANULARITY,
            "call_ids": ["call_a"],
            "unnameable": "yes",
        },
        {
            "granularity": LOST_TOOL_EVIDENCE_GRANULARITY,
            "call_ids": ["call_a"],
            "unnameable": 0,
        },
    ],
    ids=[
        "not_a_dict_list",
        "not_a_dict_str",
        "missing_granularity",
        "non_string_call_id",
        "wrong_granularity",
        "unnameable_not_bool_str",
        # bool is a subclass of int in Python, so a guard written as
        # isinstance(raw, int) would wave this one through.
        "unnameable_not_bool_int",
    ],
)
def test_an_unreadable_record_reads_as_evidence_still_missing(raw: Any) -> None:
    """Three separate facts ``load_state`` must never confuse with one
    another: a key that was never written, a value that was written and
    reads back cleanly, and a value that was written but cannot be read.

    Mutation this test catches: collapsing the missing-key branch and the
    unreadable-value branch of ``load_state`` into one conditional expression
    makes the missing-key cell report ``unnameable=True``; with the two
    branches kept separate it correctly reports ``False``.
    """
    base_state = ReActPattern().get_state()

    missing_key_state = dict(base_state)
    missing_key_state.pop("lost_tool_evidence", None)
    missing_key_pattern = ReActPattern()
    missing_key_pattern.load_state(missing_key_state)
    assert missing_key_pattern.lost_tool_evidence.call_ids == set()
    assert missing_key_pattern.lost_tool_evidence.unnameable is False
    assert missing_key_pattern.lost_tool_evidence.still_missing() is False

    wellformed_state = dict(base_state)
    wellformed_state["lost_tool_evidence"] = {
        "granularity": LOST_TOOL_EVIDENCE_GRANULARITY,
        "call_ids": ["call_a", "call_b"],
        "unnameable": False,
    }
    wellformed_pattern = ReActPattern()
    wellformed_pattern.load_state(wellformed_state)
    assert wellformed_pattern.lost_tool_evidence.call_ids == {"call_a", "call_b"}
    assert wellformed_pattern.lost_tool_evidence.unnameable is False

    unreadable_state = dict(base_state)
    unreadable_state["lost_tool_evidence"] = raw
    unreadable_pattern = ReActPattern()
    unreadable_pattern.load_state(unreadable_state)
    assert unreadable_pattern.lost_tool_evidence.call_ids == set()
    assert unreadable_pattern.lost_tool_evidence.unnameable is True
    assert unreadable_pattern.lost_tool_evidence.still_missing() is True


def test_an_over_long_record_is_truncated_and_says_so() -> None:
    """A read-back over the cap is shortened, not silently accepted whole.

    Without the cap, a payload naming an unbounded number of ids -- whether
    from a run that genuinely lost that many observations over its lifetime,
    or from a build with a different limit -- would be accepted at whatever
    size it arrives, defeating the reason the cap exists.
    """
    ids = sorted(
        f"call_{index:04d}" for index in range(LOST_TOOL_EVIDENCE_MAX_CALL_IDS + 5)
    )
    raw = {
        "granularity": LOST_TOOL_EVIDENCE_GRANULARITY,
        "call_ids": ids,
        "unnameable": False,
    }

    evidence = LostToolEvidence.from_state(raw)

    assert len(evidence.call_ids) == LOST_TOOL_EVIDENCE_MAX_CALL_IDS
    assert evidence.call_ids == set(ids[:LOST_TOOL_EVIDENCE_MAX_CALL_IDS])
    assert evidence.unnameable is True


@pytest.mark.parametrize("times", [1, 3, 10])
def test_an_unnameable_loss_is_recorded_once_however_often_it_is_seen(
    times: int,
) -> None:
    """``unnameable`` is a flag, not a tally, so repeats must not change the
    resulting state.

    Without this, replaying the same turn after an interrupt -- which folds
    the same compaction loss into the record a second time -- would make the
    record depend on how many times a turn happened to be replayed rather
    than on what compaction actually destroyed.
    """
    loss = CompactionLoss(call_ids=(), unaccounted=2)

    repeated = ReActPattern()
    for _ in range(times):
        repeated._record_lost_tool_evidence(loss)

    once = ReActPattern()
    once._record_lost_tool_evidence(loss)

    assert (
        repeated.get_state()["lost_tool_evidence"]
        == once.get_state()["lost_tool_evidence"]
    )
    assert repeated.lost_tool_evidence.unnameable is True
    assert repeated.lost_tool_evidence.call_ids == set()


@pytest.mark.parametrize(
    ("call_ids_value", "without_call_id_value", "expected_call_ids"),
    [
        pytest.param(
            ["call_a", "call_b"],
            1,
            {"call_a", "call_b"},
            id="readable_ids_plus_uncounted",
        ),
        pytest.param(None, 0, set(), id="call_id_list_absent"),
        pytest.param("not-a-list", 0, set(), id="call_id_list_not_a_list"),
        pytest.param([1, 2], 0, set(), id="call_id_list_non_string_element"),
        pytest.param(["", "call_a"], 0, set(), id="call_id_list_empty_string_element"),
    ],
)
def test_an_observation_without_a_call_id_is_not_given_one(
    call_ids_value: Any,
    without_call_id_value: Any,
    expected_call_ids: set[str],
) -> None:
    """A destroyed observation this turn cannot name gets no invented
    identity.

    These call-id shapes -- an absent list, a non-list, a non-string
    element, an empty-string element -- have no known way to occur through a
    production entry point today: ``Message.tool_call_id`` is typed
    ``str | None`` and every wired writer fills it before a message reaches
    compaction. They are pinned here directly against
    ``ReActPattern._dropped_tool_evidence`` rather than forced through an
    end-to-end fake that would not occur in production.

    Without this behavior, a malformed or absent id list could be repaired
    by inventing a placeholder id -- letting a later answer believe one
    specific call's evidence came back when nobody ever identified it -- or
    by falling back to the tool's name, which would wrongly let any later
    call to that same tool count as bringing the lost observation back. Both
    are worse than admitting the loss cannot be named.
    """
    pattern = ReActPattern()
    result = compact_result(
        count=3, call_ids=call_ids_value, without_call_id=without_call_id_value
    )

    loss = pattern._dropped_tool_evidence(result)
    assert set(loss.call_ids) == expected_call_ids

    pattern._record_lost_tool_evidence(loss)
    assert pattern.lost_tool_evidence.call_ids == expected_call_ids
    assert pattern.lost_tool_evidence.unnameable is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("llm_responses", "max_iterations", "expected_label", "expected_success"),
    [
        # A final_answer call reaches _finalize_outcome (checkpoint "final")
        # through _execute_pending_tool_calls, whose caller then writes one
        # more checkpoint labeled after the control result's own status.
        pytest.param(
            [final_answer_response()],
            6,
            "completed",
            True,
            id="final_answer_tool_call",
        ),
        pytest.param(
            [{"content": "Here is the answer.", "tool_calls": [], "done": True}],
            6,
            "final",
            True,
            id="plain_assistant_text",
        ),
        # Same control-tool path as final_answer_tool_call above: the
        # retry's final_answer call also finishes through
        # _execute_pending_tool_calls.
        pytest.param(
            [empty_final_answer_response(), final_answer_response()],
            6,
            "completed",
            True,
            id="rejected_then_finishes",
        ),
        # A work-tool call, never a final_answer: with max_iterations=1 the
        # loop executes this one batch and then exhausts its range without
        # ever asking the model for an answer.
        pytest.param(
            [tool_call_response("calculator")],
            1,
            "max_iterations",
            False,
            id="max_iterations",
        ),
        pytest.param(
            [empty_final_answer_response(), empty_final_answer_response()],
            6,
            "invalid_tool_protocol",
            False,
            id="invalid_tool_protocol",
        ),
    ],
)
async def test_a_finished_run_leaves_no_lost_evidence_behind(
    llm_responses: list[Any],
    max_iterations: int,
    expected_label: str,
    expected_success: bool,
) -> None:
    """However a run ends, the checkpoint it leaves behind carries no
    leftover loss.

    A run can end five ways here: it answers on the first turn with a
    ``final_answer`` tool call, it answers with plain assistant text, its
    first answer is rejected for an empty ``answer`` field and the one
    repair retry then answers, it runs out of iterations without ever
    answering, or that one repair retry also comes back empty and the run is
    abandoned. The last two do not go through ``_finalize_outcome`` at all --
    they write a checkpoint labeled ``max_iterations`` or
    ``invalid_tool_protocol`` instead of ``final`` -- but a resume can load
    from either checkpoint just as it can load from ``final``, so a leftover
    record there is exactly as stale as one left behind on a normal finish.

    Every cell starts from the same context, whose three seeded tool
    observations this run's own first-turn compaction destroys, so the
    lost-evidence record genuinely holds something part-way through the run;
    without that, the assertions below would pass vacuously against a record
    nothing ever populated.

    Mutation this test catches: dropping
    ``_clear_lost_tool_evidence_at_run_end()`` from the ``max_iterations``
    path or from ``_invalid_tool_protocol_result`` leaves that one cell's
    ``call_ids`` still holding the seeded run's lost ids, while the three
    cells that finish through ``_finalize_outcome`` keep passing regardless,
    because that path clears the record on its own.
    """
    context = build_context(max_messages=4)
    pattern = ReActPattern(max_iterations=max_iterations)
    runtime = PatternRuntime()

    result = await pattern.run(
        context=context,
        tools=[NamedTool("calculator")],
        llm=RecordingLLM(llm_responses),
        compact_llm=RecordingLLM([{"content": ""}]),
        runtime=runtime,
    )

    assert result["success"] is expected_success
    last_checkpoint = runtime.checkpoints[-1]
    assert last_checkpoint.get("label") == expected_label
    final_state = dict(last_checkpoint.get("pattern_state") or {})
    assert final_state["lost_tool_evidence"]["call_ids"] == []
    assert final_state["lost_tool_evidence"]["unnameable"] is False
    assert pattern.lost_tool_evidence.call_ids == set()
    assert pattern.lost_tool_evidence.unnameable is False
