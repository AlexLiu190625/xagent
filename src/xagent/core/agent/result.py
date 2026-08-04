from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

TOOL_FAILURE_CODES = frozenset(
    {
        "oauth_token_required",
        "unsupported_nested_interaction",
        "missing_delegated_output",
    }
)

_OAUTH_FAILURE_CODES = frozenset({"oauth_token_required"})

# Placeholder outputs the execution layers substitute when a run produced no
# text of its own. They are recognized, not produced, by the delegation
# classifier: a child that returns one of these completed without answering.
NO_OUTPUT_PLACEHOLDER = "No output provided"
NO_RESPONSE_PLACEHOLDER = "No response generated"


def normalize_tool_failure_code(value: Any) -> str | None:
    """Return an exact public tool failure code when it is allowlisted."""

    return value if type(value) is str and value in TOOL_FAILURE_CODES else None


def normalize_oauth_failure_code(value: Any) -> str | None:
    """Return an exact OAuth failure code carried by ``ClassifiedToolFailure``."""

    return value if type(value) is str and value in _OAUTH_FAILURE_CODES else None


@dataclass(frozen=True)
class ClassifiedToolFailure:
    """Sentinel an OAuth token resolver returns when it cannot refresh a token.

    Its only current consumer is the delegated-authorization retry path in
    the MCP adapter, which treats any instance as an OAuth failure regardless
    of the carried code. ``failure_code`` is validated against the OAuth-only
    set, not the wider runtime allowlist: a new public failure code becomes
    surfaceable by the runtime without becoming carriable by this type.
    """

    failure_code: str

    def __post_init__(self) -> None:
        if normalize_oauth_failure_code(self.failure_code) is None:
            raise ValueError("invalid tool failure code")


def tool_result_succeeded(result: Any) -> bool:
    """Classify the supported structured tool-result failure shapes."""

    if not isinstance(result, dict):
        return True
    if result.get("success") is False or result.get("is_error") is True:
        return False
    status = result.get("status")
    return not (isinstance(status, str) and status.lower() == "error")


def extract_assistant_message(result: dict[str, Any]) -> str | None:
    """Return the assistant-facing output from a normalized pattern result."""

    for key in ("response", "answer", "output", "content", "message"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return unwrap_final_answer_content(value)
    return None


def unwrap_final_answer_content(content: str) -> str:
    """Unwrap textual legacy final_answer JSON into display-ready content."""
    parsed = _parse_json_like_text(content)
    if not isinstance(parsed, dict):
        return content

    action = str(parsed.get("action") or "").strip()
    if action == "final_answer":
        return _stringify_answer_value(
            parsed.get("action_input")
            if "action_input" in parsed
            else parsed.get("answer", content)
        )

    if "final_answer" in parsed:
        return _stringify_answer_value(parsed["final_answer"])

    return content


def _parse_json_like_text(content: str) -> Any | None:
    text = _strip_code_fence(content.strip())
    if not text or not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        closing = lines[-1].strip()
        if closing == "```" or closing.startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return text


def _stringify_answer_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
