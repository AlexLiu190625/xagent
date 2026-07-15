from __future__ import annotations

from xagent.core.tools.adapters.vibe.connector_runtime import REDACTED_RUNTIME_SECRET
from xagent.web.api.public_trace_events import normalize_public_trace_event


def test_normalize_public_trace_event_redacts_tool_runtime_secrets() -> None:
    event_type, data = normalize_public_trace_event(
        "tool_execution_start",
        {
            "tool_name": "shiftcare",
            "tool_args": {
                "headers": {
                    "Authorization": "Bearer public-stream-token",
                    "X-Account": "6185",
                },
                "connector_runtime": {
                    "secrets": {"authorization": "Bearer nested-token"},
                    "auth_selector": {"resource_owner_key": "xagent:user:owner"},
                },
            },
        },
    )

    assert event_type == "tool_execution_start"
    assert "public-stream-token" not in repr(data)
    assert "nested-token" not in repr(data)
    assert "xagent:user:owner" not in repr(data)
    assert data["tool_args"]["headers"]["Authorization"] == REDACTED_RUNTIME_SECRET
    assert data["tool_args"]["headers"]["X-Account"] == "6185"
    assert (
        data["tool_args"]["connector_runtime"]["auth_selector"]["resource_owner_key"]
        == REDACTED_RUNTIME_SECRET
    )


def test_mcp_load_summary_audit_event_is_not_fanned_out() -> None:
    from xagent.core.agent.trace import SYSTEM_INFO, TraceEvent
    from xagent.web.api.ws_trace_handlers import WebSocketTraceHandler

    event = TraceEvent(
        event_type=SYSTEM_INFO,
        task_id="42",
        data={
            "__audit_only__": True,
            "event_type": "mcp_load_summary",
            "requested_servers": ["Gmail"],
            "loaded_servers": [],
            "failures": [{"server_name": "Gmail", "reason": "oauth_token_required"}],
            "successful_tool_count": 0,
        },
    )

    assert WebSocketTraceHandler(42)._convert_trace_event_to_stream_event(event) is None
