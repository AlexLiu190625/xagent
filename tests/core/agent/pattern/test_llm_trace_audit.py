"""Unit tests for the LLM-payload truncation helper used by the audit
trace infrastructure.

The v1 runtime (which used to host per-pattern audit emit sites
covered here) was removed in upstream PR #403
(``feat: [v2 part8] remove agent v1 runtime``); per-site audit tests
have been dropped along with their targets. This file keeps the
transport-agnostic infrastructure tests that still apply on v2.

The v2-runtime audit injection (centralized in
``agent/runtime.py:on_llm_start / on_llm_end``) is covered by a
follow-up PR.
"""

from typing import Any, Dict, List, Optional

import pytest


def test_truncate_for_trace_short_string_passthrough() -> None:
    from xagent.core.agent.trace import truncate_for_trace

    assert truncate_for_trace("hi", max_bytes=100) == "hi"


def test_truncate_for_trace_long_string_truncated() -> None:
    from xagent.core.agent.trace import truncate_for_trace

    out = truncate_for_trace("x" * 1000, max_bytes=100)
    assert isinstance(out, str)
    assert "[truncated" in out
    # Head of original preserved
    assert out.startswith("x" * 100)


def test_truncate_for_trace_walks_dict_and_list() -> None:
    """Per-leaf truncation: dict shape preserved, only oversized string
    leaves get the ``[truncated N chars]`` marker.

    Uses ``max_bytes=4000`` so the post-trim serialized payload stays
    inside the hard-cap envelope and the original dict shape survives.
    The over-budget collapse path is covered by the separate
    ``..._dict_total_bounded_by_max_bytes`` test below.
    """
    from xagent.core.agent.trace import truncate_for_trace

    payload = {
        "messages": [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "short"},
        ],
        "response": "y" * 5000,
        "model_name": "stub",
        "attempt": 1,
    }
    out = truncate_for_trace(payload, max_bytes=4000)
    # Dict survives: shape preserved, not collapsed to placeholder
    assert isinstance(out, dict)
    assert "__truncated__" not in out
    # Scalars unchanged
    assert out["model_name"] == "stub"
    assert out["attempt"] == 1
    # Large string truncated
    assert "[truncated" in out["response"]
    # Nested list element truncated
    assert "[truncated" in out["messages"][0]["content"]
    # Short nested element unchanged
    assert out["messages"][1]["content"] == "short"


def test_truncate_for_trace_walks_dict_per_field() -> None:
    """Multi-field dict: every oversized value gets the trim marker;
    shape is preserved.

    Per-field budget is ``max_bytes // N_fields``, so a single field can
    overshoot by its truncation-suffix overhead (~25 bytes). The overall
    cap is enforced one level up by
    :func:`normalize_llm_trace_payload`, which only routes the
    truncatable subset of fields through this helper.
    """
    from xagent.core.agent.trace import truncate_for_trace

    big = "z" * 5000
    payload = {"a": big, "b": big, "c": big, "d": big}
    out = truncate_for_trace(payload, max_bytes=200)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"a", "b", "c", "d"}
    for key in ("a", "b", "c", "d"):
        assert "[truncated" in out[key], (
            f"expected per-field trim marker on {key!r}, got {out[key]!r}"
        )


def test_truncate_for_trace_multibyte_head_no_replacement_chars() -> None:
    """Multi-byte UTF-8 truncation must not produce U+FFFD chars.

    Regression: decoding the byte-sliced head with ``errors="replace"``
    inserts a replacement char whenever the slice ends mid-codepoint,
    which inflates ``len(head)`` and makes the reported truncated
    count inaccurate (can go negative for small budgets).
    """
    from xagent.core.agent.trace import truncate_for_trace

    # 100 CJK chars = 300 UTF-8 bytes; slice at 50 lands mid-codepoint.
    value = "中" * 100
    out = truncate_for_trace(value, max_bytes=50)
    assert isinstance(out, str)
    assert "�" not in out, f"replacement char leaked into head: {out!r}"
    assert "[truncated" in out


def test_truncate_for_trace_zero_disables() -> None:
    from xagent.core.agent.trace import truncate_for_trace

    long = "z" * 10_000
    assert truncate_for_trace(long, max_bytes=0) == long


def test_truncate_for_trace_deep_nesting_collapses() -> None:
    """Pathologically nested structures must not hit Python's recursion limit.

    Builds a 100-deep dict (well above the 50-frame guard, well below
    Python's default 1000-frame limit). Without the guard, sufficiently
    deep + large payloads could still blow the stack since each level
    eats a frame for both the dict comprehension and the recursive call.
    """
    from xagent.core.agent.trace import truncate_for_trace

    deep: Any = "leaf"
    for _ in range(100):
        deep = {"nested": deep}

    out = truncate_for_trace(deep, max_bytes=10_000)

    cur: Any = out
    depth = 0
    while isinstance(cur, dict) and "nested" in cur:
        cur = cur["nested"]
        depth += 1
        if depth > 200:
            pytest.fail("recursion guard never collapsed deep payload")

    assert isinstance(cur, str)
    assert "depth exceeds" in cur, (
        f"expected depth-guard placeholder at leaf, got {cur!r}"
    )


def test_ws_handler_drops_audit_only_events() -> None:
    """Server-only audit traces with ``__audit_only__: True`` must be
    dropped before reaching WebSocket clients.

    This is a security-critical assertion: the audit pipeline persists
    raw LLM I/O (messages, response) via DatabaseTraceHandler, and the
    drop in WebSocketTraceHandler is the only barrier preventing that
    same payload from being broadcast to connected clients.
    """
    from xagent.core.agent.trace import ACTION_START_LLM, TraceEvent
    from xagent.web.api.ws_trace_handlers import WebSocketTraceHandler

    handler = WebSocketTraceHandler(task_id=1)

    audit_event = TraceEvent(
        event_type=ACTION_START_LLM,
        task_id="t1",
        step_id="dag_skill_selection",
        data={
            "__audit_only__": True,
            "messages": [{"role": "user", "content": "raw prompt body"}],
            "action": "LLM call started",
        },
    )

    result = handler._convert_trace_event_to_stream_event(audit_event)
    assert result is None, (
        "audit_only event must be dropped before WS broadcast; "
        "got non-None stream event"
    )


def test_ws_handler_passes_non_audit_events() -> None:
    """Regression: dropping ``__audit_only__`` must not affect normal events."""
    from xagent.core.agent.trace import ACTION_START_LLM, TraceEvent
    from xagent.web.api.ws_trace_handlers import WebSocketTraceHandler

    handler = WebSocketTraceHandler(task_id=1)

    event = TraceEvent(
        event_type=ACTION_START_LLM,
        task_id="t1",
        step_id="step1",
        data={"action": "LLM call started", "step_name": "test_step"},
    )

    result = handler._convert_trace_event_to_stream_event(event)
    assert result is not None, "non-audit event was incorrectly dropped"
    assert result.get("step_id") == "step1"


@pytest.mark.asyncio
async def test_trace_action_end_truncates_llm_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration: trace_action_end with category=LLM applies the cap."""
    from xagent.core.agent.trace import (
        TraceCategory,
        Tracer,
        trace_action_end,
    )

    captured: List[Dict[str, Any]] = []

    class _RecordingTracer(Tracer):
        async def trace_event(  # type: ignore[override]
            self,
            event_type: Any,
            task_id: Any = None,
            step_id: Any = None,
            data: Any = None,
            parent_id: Any = None,
        ) -> str:
            captured.append(data or {})
            return "evt"

    monkeypatch.setenv("XAGENT_MAX_TRACE_PAYLOAD_BYTES", "200")

    await trace_action_end(
        _RecordingTracer(),
        "t",
        "s",
        TraceCategory.LLM,
        data={"response": "x" * 1000, "model_name": "m"},
    )

    assert len(captured) == 1
    assert "[truncated" in captured[0]["response"]
    assert captured[0]["model_name"] == "m"


# ---------------------------------------------------------------------------
# normalize_llm_trace_payload — reserved-field preservation
# ---------------------------------------------------------------------------


def test_normalize_preserves_reserved_under_truncation() -> None:
    """Reserved control / routing / metrics fields must pass through
    untouched even when truncatable content fields are trimmed.

    Regression for the bug rogercloud flagged: hard-cap collapse used
    to drop ``__audit_only__`` and break WS visibility filtering.
    """
    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {
        # routing / metadata / metrics — must survive verbatim
        "__audit_only__": True,
        "model_name": "gpt-4",
        "task_type": "dag_skill_selection",
        "step_id": "step-1",
        "step_name": "skill_selection",
        "action": "LLM call completed",
        "attempt": 1,
        "json_mode_failed": False,
        "success": True,
        "usage": {"input_tokens": 12, "output_tokens": 34, "total_tokens": 46},
        "messages_count": 2,
        # bulky content — must be trimmed
        "messages": [{"role": "user", "content": "x" * 200_000}],
        "response": "y" * 200_000,
    }
    out = normalize_llm_trace_payload(payload, max_bytes=200)

    assert isinstance(out, dict)
    assert out["__audit_only__"] is True
    assert out["model_name"] == "gpt-4"
    assert out["task_type"] == "dag_skill_selection"
    assert out["step_id"] == "step-1"
    assert out["step_name"] == "skill_selection"
    assert out["action"] == "LLM call completed"
    assert out["attempt"] == 1
    assert out["json_mode_failed"] is False
    assert out["success"] is True
    assert out["usage"] == {
        "input_tokens": 12,
        "output_tokens": 34,
        "total_tokens": 46,
    }
    assert out["messages_count"] == 2
    assert "[truncated" in str(out["messages"])
    assert "[truncated" in out["response"]


def test_normalize_passthrough_when_no_content_fields() -> None:
    """All-reserved payload returns unchanged (no spurious trim).

    Important so ``_emit_trace_event`` calling normalize on every LLM
    event is cheap when the event only carries metadata.
    """
    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {"__audit_only__": True, "model_name": "gpt-4", "attempt": 2}
    out = normalize_llm_trace_payload(payload, max_bytes=100)
    assert out is payload or out == payload


def test_normalize_passes_through_non_dict() -> None:
    """Non-dict input returns as-is — defensive for unusual callers."""
    from xagent.core.agent.trace import normalize_llm_trace_payload

    assert normalize_llm_trace_payload("not a dict") == "not a dict"
    assert normalize_llm_trace_payload(None) is None


def test_normalize_zero_disables() -> None:
    """``max_bytes=0`` (XAGENT_MAX_TRACE_PAYLOAD_BYTES=0) disables truncation."""
    from xagent.core.agent.trace import normalize_llm_trace_payload

    long_response = "x" * 10_000
    payload = {"response": long_response, "model_name": "m"}
    out = normalize_llm_trace_payload(payload, max_bytes=0)
    assert out["response"] == long_response


def test_normalize_unknown_fields_pass_through() -> None:
    """Unknown fields (neither reserved nor truncatable) pass through.

    Future-proofs against silently truncating a new routing flag added
    by an audit emit that this list hasn't been updated for yet.
    """
    from xagent.core.agent.trace import normalize_llm_trace_payload

    payload = {
        "future_routing_flag": True,
        "future_metric": 42,
        "response": "y" * 10_000,  # known truncatable
    }
    out = normalize_llm_trace_payload(payload, max_bytes=200)
    assert out["future_routing_flag"] is True
    assert out["future_metric"] == 42
    assert "[truncated" in out["response"]


# ---------------------------------------------------------------------------
# PatternRuntime trace-boundary cap (Finding 3 regression)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v2_runtime_emit_trace_caps_llm_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: PatternRuntime.on_llm_end with a 100 KB response gets
    capped at the trace boundary. Previously the runtime bypassed
    truncate_for_trace entirely and emitted the raw payload to the
    tracer.
    """
    from xagent.core.agent.runtime import PatternRuntime

    events: List[Dict[str, Any]] = []

    class _CaptureTracer:
        async def trace_event(
            self,
            event_type: Any,
            task_id: Any = None,
            step_id: Any = None,
            data: Any = None,
            parent_id: Any = None,
        ) -> str:
            events.append(
                {
                    "event_type": getattr(event_type, "value", str(event_type)),
                    "task_id": task_id,
                    "step_id": step_id,
                    "data": dict(data or {}),
                }
            )
            return "evt"

    class _FakeContext:
        execution_id = "task-x"
        messages: List[Any] = []

        def record_llm_usage(self, **_: Any) -> None:
            pass

    monkeypatch.setenv("XAGENT_MAX_TRACE_PAYLOAD_BYTES", "1000")
    runtime = PatternRuntime(tracer=_CaptureTracer(), execution_id="task-x")

    await runtime.on_llm_end(context=_FakeContext(), response="x" * 100_000)

    assert events, "no trace event captured"
    data = events[-1]["data"]
    assert isinstance(data.get("response"), str)
    assert len(data["response"]) < 5_000, (
        f"response should be capped well under 100k, got {len(data['response'])}"
    )
    assert "[truncated" in data["response"]
    # Reserved control field survives the boundary cap
    assert data["success"] is True


@pytest.mark.asyncio
async def test_v2_runtime_emit_trace_does_not_cap_tool_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: only LLM-category events get normalized at the
    boundary. TOOL / DAG / REACT / COMPACT / GENERAL events must pass
    through their data unchanged so we don't silently truncate tool
    output, DAG plans, etc.
    """
    from xagent.core.agent.runtime import PatternRuntime
    from xagent.core.agent.trace import (
        TraceAction,
        TraceCategory,
        TraceEventType,
        TraceScope,
    )

    events: List[Dict[str, Any]] = []

    class _CaptureTracer:
        async def trace_event(
            self,
            event_type: Any,
            task_id: Any = None,
            step_id: Any = None,
            data: Any = None,
            parent_id: Any = None,
        ) -> str:
            events.append({"data": dict(data or {})})
            return "evt"

    monkeypatch.setenv("XAGENT_MAX_TRACE_PAYLOAD_BYTES", "1000")
    runtime = PatternRuntime(tracer=_CaptureTracer(), execution_id="task-x")

    tool_end_event = TraceEventType(
        TraceScope.ACTION, TraceAction.END, TraceCategory.TOOL
    )
    huge_tool_output = "z" * 100_000
    await runtime._emit_trace_event(
        tool_end_event,
        task_id="task-x",
        step_id="step-1",
        data={"tool_output": huge_tool_output, "tool_name": "noop"},
    )

    assert len(events[-1]["data"]["tool_output"]) == 100_000
    assert "[truncated" not in events[-1]["data"]["tool_output"]


# ---------------------------------------------------------------------------
# Skill selector audit emit coverage
# ---------------------------------------------------------------------------


class _RecordingTracer:
    """Capture-only tracer used by selector audit emit tests."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        task_id: Any = None,
        step_id: Any = None,
        data: Any = None,
        parent_id: Any = None,
    ) -> str:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "step_id": step_id,
                "data": dict(data or {}),
            }
        )
        return "evt"


class _FakeLLM:
    """Minimal stub matching the surface area used by SkillSelector.select."""

    model_name = "fake-model"

    def __init__(
        self,
        *,
        fail_json_mode: bool = False,
        fail_all: bool = False,
        response_payload: str = '{"selected": true, "skill_name": "skill_a", "reasoning": "fits"}',
    ) -> None:
        self.fail_json_mode = fail_json_mode
        self.fail_all = fail_all
        self.response_payload = response_payload
        self.calls: List[Dict[str, Any]] = []

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.calls.append({"messages": messages, "response_format": response_format})
        if self.fail_all:
            raise RuntimeError("LLM unavailable")
        if response_format is not None and self.fail_json_mode:
            raise RuntimeError("JSON mode unsupported")
        return self.response_payload


_FAKE_CANDIDATES = [
    {"name": "skill_a", "description": "do A"},
    {"name": "skill_b", "description": "do B"},
]


@pytest.mark.asyncio
async def test_selector_audit_emits_start_end_on_success() -> None:
    """Happy path: JSON mode works -> start + end emitted on attempt=1."""
    from xagent.skills.selector import SkillSelector

    tracer = _RecordingTracer()
    selector = SkillSelector(llm=_FakeLLM())

    result = await selector.select(
        task="do thing A",
        candidates=_FAKE_CANDIDATES,
        tracer=tracer,
        task_id="task-1",
    )

    assert result is not None
    assert result["name"] == "skill_a"

    audit_events = [e for e in tracer.events if e["data"].get("__audit_only__") is True]
    assert len(audit_events) == 2, (
        f"expected 1 start + 1 end on success, got {len(audit_events)}: "
        f"{[e['data'].get('action') for e in audit_events]}"
    )

    start, end = audit_events
    assert start["data"]["action"] == "LLM call started"
    assert start["data"]["attempt"] == 1
    assert start["data"]["json_mode_failed"] is False
    assert end["data"]["action"] == "LLM call completed"
    assert end["data"]["attempt"] == 1
    assert end["data"]["json_mode_failed"] is False
    assert end["data"]["step_id"] == "dag_skill_selection"


@pytest.mark.asyncio
async def test_selector_audit_emits_fallback_on_json_mode_failure() -> None:
    """attempt=1 fails JSON mode -> 4 events with json_mode_failed semantics."""
    from xagent.skills.selector import SkillSelector

    tracer = _RecordingTracer()
    selector = SkillSelector(llm=_FakeLLM(fail_json_mode=True))

    result = await selector.select(
        task="do thing A",
        candidates=_FAKE_CANDIDATES,
        tracer=tracer,
        task_id="task-2",
    )

    assert result is not None
    assert result["name"] == "skill_a"

    audit_events = [e for e in tracer.events if e["data"].get("__audit_only__") is True]
    assert len(audit_events) == 4, (
        f"expected start1+err1+start2+end2, got "
        f"{[e['data'].get('action') for e in audit_events]}"
    )

    actions = [(e["data"]["attempt"], e["data"]["action"]) for e in audit_events]
    assert actions == [
        (1, "LLM call started"),
        (1, "LLM call failed"),
        (2, "LLM call started"),
        (2, "LLM call completed"),
    ]

    # json_mode_failed: False only on attempt=1 start; True on the rest
    assert audit_events[0]["data"]["json_mode_failed"] is False
    for event in audit_events[1:]:
        assert event["data"]["json_mode_failed"] is True, (
            f"expected json_mode_failed=True on {event['data']['action']} "
            f"attempt={event['data']['attempt']}"
        )


@pytest.mark.asyncio
async def test_selector_audit_emits_failure_end_when_both_attempts_fail() -> None:
    """Both attempts blow -> emit 4 events then re-raise."""
    from xagent.skills.selector import SkillSelector

    tracer = _RecordingTracer()
    selector = SkillSelector(llm=_FakeLLM(fail_all=True))

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await selector.select(
            task="do thing A",
            candidates=_FAKE_CANDIDATES,
            tracer=tracer,
            task_id="task-3",
        )

    audit_events = [e for e in tracer.events if e["data"].get("__audit_only__") is True]
    actions = [(e["data"]["attempt"], e["data"]["action"]) for e in audit_events]
    assert actions == [
        (1, "LLM call started"),
        (1, "LLM call failed"),
        (2, "LLM call started"),
        (2, "LLM call failed"),
    ], f"unexpected emit sequence: {actions}"

    # Critical: every audit event must carry the server-only flag
    for event in audit_events:
        assert event["data"]["__audit_only__"] is True


@pytest.mark.asyncio
async def test_selector_audit_no_emit_when_tracer_is_none() -> None:
    """Defensive: passing tracer=None must not crash selector."""
    from xagent.skills.selector import SkillSelector

    selector = SkillSelector(llm=_FakeLLM())

    result = await selector.select(
        task="do thing A",
        candidates=_FAKE_CANDIDATES,
        tracer=None,
        task_id=None,
    )

    assert result is not None
    assert result["name"] == "skill_a"
