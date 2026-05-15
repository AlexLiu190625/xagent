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

from typing import Any, Dict, List

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


def test_truncate_for_trace_dict_total_bounded_by_max_bytes() -> None:
    """A multi-field dict must not fan out to N*max_bytes total size.

    Two ways the cap is honored:

      - Budget enough for per-field trim to fit: dict shape survives,
        every value carries the ``[truncated N chars]`` marker.
      - Budget too small for trimmed shape: the dict collapses to
        ``{"__truncated__": "..."}`` (container TYPE preserved so
        downstream ``data.keys()`` callers don't break).
    """
    import json

    from xagent.core.agent.trace import truncate_for_trace

    big = "z" * 5000
    payload = {"a": big, "b": big, "c": big, "d": big}
    out = truncate_for_trace(payload, max_bytes=200)

    serialized = json.dumps(out)
    assert len(serialized) < 800, (
        f"dict fan-out broke the cap; serialized={len(serialized)} bytes"
    )
    assert isinstance(out, dict)
    if "__truncated__" in out:
        # Hard-cap path: dict collapsed to placeholder marker.
        assert "[truncated" in out["__truncated__"]
    else:
        # Per-field trim path: every value got truncated.
        for key in ("a", "b", "c", "d"):
            assert "[truncated" in out[key]


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
