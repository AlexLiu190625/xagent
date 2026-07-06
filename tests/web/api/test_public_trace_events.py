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
