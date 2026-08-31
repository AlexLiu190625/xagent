"""Single-tool ``ask_user_question`` field-name deduplication.

The multi-tool waiting path (``_pause_for_tool_results`` in
``react.py``) has run a same-shape dedup loop across every waiting
tool's interactions for a while; the single-tool ``ask_user_question``
branch in ``_handle_control_tool`` did not, so two interactions the model
named the same field would both reach the write-side validator unchanged,
tripping its duplicate-field rule and costing the whole question. This
delivery gives the single-tool path the identical dedup shape, at a
narrower scope: ``used_fields`` spans only this one call's own
interactions, never a sibling tool's, because a single-tool call has no
sibling to collide with.

Rules pinned here, each aligned to the batch path's own four (the dedup
loop in ``_pause_for_tool_results``):

1. base name is ``str(item.get("field") or "response")``
2. the first occupant of a base keeps it; suffixing starts at ``_2``
3. a ``while`` loop, not a one-shot suffix append, so a third collision on
   an already-suffixed name does not reuse a taken name
4. ``used_fields`` is scoped to this one call only
"""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _ask_user_question_call(
    call_id: str, message: str, interactions: list[dict[str, Any]]
):
    import json

    return {
        "content": message,
        "tool_calls": [
            {
                "id": call_id,
                "function": {
                    "name": "ask_user_question",
                    "arguments": json.dumps(
                        {"message": message, "interactions": interactions}
                    ),
                },
            }
        ],
    }


async def _run_ask_user_question(
    interactions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    llm = FakeLLM(
        responses=[_ask_user_question_call("call_1", "Question", interactions)]
    )
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-dedup")
    context = ExecutionContext()
    context.add_user_message("Ask")

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)
    assert result["status"] == "waiting_for_user"
    return result["interactions"]


@pytest.mark.asyncio
async def test_dedup_cell_1_two_same_named_fields() -> None:
    out = await _run_ask_user_question(
        [
            {"type": "text_input", "field": "city", "label": "City A"},
            {"type": "text_input", "field": "city", "label": "City B"},
        ]
    )
    assert [item["field"] for item in out] == ["city", "city_2"]


@pytest.mark.asyncio
async def test_dedup_cell_2_three_same_named_fields() -> None:
    out = await _run_ask_user_question(
        [
            {"type": "text_input", "field": "city", "label": "A"},
            {"type": "text_input", "field": "city", "label": "B"},
            {"type": "text_input", "field": "city", "label": "C"},
        ]
    )
    assert [item["field"] for item in out] == ["city", "city_2", "city_3"]


@pytest.mark.asyncio
async def test_dedup_cell_3_second_collision_does_not_reuse_a_taken_name() -> None:
    """city, city_2, city -> city, city_2, city_3. The third one must not
    become city_2 again -- pins the `while`, not `if`, requirement."""

    out = await _run_ask_user_question(
        [
            {"type": "text_input", "field": "city", "label": "A"},
            {"type": "text_input", "field": "city_2", "label": "B"},
            {"type": "text_input", "field": "city", "label": "C"},
        ]
    )
    assert [item["field"] for item in out] == ["city", "city_2", "city_3"]


@pytest.mark.asyncio
async def test_dedup_cell_4_blank_field_names_never_reach_the_dedup_loop_blank() -> (
    None
):
    """_normalize_ask_user_interactions (called before this dedup loop, on
    both the single- and multi-tool paths) already disambiguates a blank
    field by index (``response_{index}``) before this loop ever sees it --
    verified directly against that function -- so ``item.get("field")``
    can never be empty by the time this loop runs, on either path. The
    dedup loop's own ``or "response"`` fallback is therefore unreachable
    from this pipeline, the same way the batch path's identical fallback
    already was; this test pins the real, observed output rather than a
    hypothetical collision that cannot occur through this call site."""

    out = await _run_ask_user_question(
        [
            {"type": "text_input", "field": "", "label": "A"},
            {"type": "text_input", "field": "", "label": "B"},
        ]
    )
    assert [item["field"] for item in out] == ["response_0", "response_1"]


@pytest.mark.asyncio
async def test_dedup_cell_5_no_collision_leaves_every_name_unchanged() -> None:
    out = await _run_ask_user_question(
        [
            {"type": "text_input", "field": "city", "label": "A"},
            {"type": "text_input", "field": "state", "label": "B"},
        ]
    )
    assert [item["field"] for item in out] == ["city", "state"]


# The batch-path regression cell (multi-tool dedup unchanged, byte for
# byte) is already covered by the existing, untouched
# test_pause_for_tool_results_deduplicates_normalized_fields
# (test_react.py) -- not duplicated here.


@pytest.mark.asyncio
async def test_dedup_cell_7_deduplicated_output_passes_the_write_side_validator() -> (
    None
):
    """The dedup loop's whole purpose: two same-named interactions must
    survive validate_v1_write_payload's duplicate-field rule after
    deduplication, where they would have tripped it before."""

    from xagent.core.tools.adapters.vibe.ask_user_tool import AskUserQuestionArgs
    from xagent.web.services.task_interaction_service import validate_v1_write_payload

    out = await _run_ask_user_question(
        [
            {"type": "text_input", "field": "city", "label": "A"},
            {"type": "text_input", "field": "city", "label": "B"},
        ]
    )
    parsed = AskUserQuestionArgs.model_validate(
        {"message": "Question", "interactions": out}
    )
    validate_v1_write_payload(parsed)  # must not raise


# ---------------------------------------------------------------------------
# Mutations (run manually against the implementation, recorded in the
# delivery report rather than encoded as a pytest xfail):
#   - suffix starting at _1 instead of _2 -> cell 1 fails (expects "city_2")
#   - `while` replaced with `if` -> cell 3 fails (expects "city_3", would
#     get "city_2" again)
#   - used_fields scope widened to span multiple calls -> cell 5 fails,
#     because a name untouched by this call could get suffixed by a
#     leftover from a previous call in the same process
# ---------------------------------------------------------------------------
