"""Stable OpenAI-compatible names for agent delegation tools."""

from typing import Any

AGENT_TOOL_NAME_PREFIX = "agent_"
LEGACY_AGENT_TOOL_NAME_PREFIX = "call_agent_"


def gen_agent_tool_name(agent_id: Any) -> str:
    """Return the canonical tool name for an agent id."""
    if isinstance(agent_id, bool):
        raise ValueError("agent_id must be an integer")
    try:
        normalized_agent_id = int(agent_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("agent_id must be an integer") from exc
    if normalized_agent_id <= 0:
        raise ValueError("agent_id must be positive")
    return f"{AGENT_TOOL_NAME_PREFIX}{normalized_agent_id}"


def parse_agent_tool_id(tool_name: Any) -> int | None:
    if not isinstance(tool_name, str):
        return None
    if not tool_name.startswith(AGENT_TOOL_NAME_PREFIX):
        return None
    suffix = tool_name.removeprefix(AGENT_TOOL_NAME_PREFIX)
    if not suffix.isdecimal():
        return None
    agent_id = int(suffix)
    return agent_id if agent_id > 0 else None


def is_agent_tool_name(tool_name: Any) -> bool:
    if parse_agent_tool_id(tool_name) is not None:
        return True
    return isinstance(tool_name, str) and tool_name.startswith(
        LEGACY_AGENT_TOOL_NAME_PREFIX
    )
