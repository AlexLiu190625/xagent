from __future__ import annotations

import pytest

from xagent.core.agent.result import (
    normalize_tool_failure_code,
    tool_result_succeeded,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("oauth_token_required", "oauth_token_required"),
        ("other_valid_code", None),
        (" oauth_token_required", None),
        ("OAUTH_TOKEN_REQUIRED", None),
        (None, None),
        (123, None),
    ],
)
def test_normalize_tool_failure_code_uses_exact_allowlist(value, expected):
    assert normalize_tool_failure_code(value) == expected


@pytest.mark.parametrize(
    "result",
    [
        {"success": False},
        {"status": "error"},
        {"status": "ERROR"},
        {"is_error": True},
    ],
)
def test_tool_result_succeeded_recognizes_supported_failure_shapes(result):
    assert tool_result_succeeded(result) is False


@pytest.mark.parametrize(
    "result",
    [
        None,
        "ok",
        {},
        {"success": True},
        {"status": "success"},
        {"is_error": False},
        {"is_error": 1},
    ],
)
def test_tool_result_succeeded_preserves_non_failure_results(result):
    assert tool_result_succeeded(result) is True
