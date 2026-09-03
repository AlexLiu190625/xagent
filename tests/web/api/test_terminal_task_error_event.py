"""Frame-shape contracts for ``create_terminal_task_error_event``.

Pinned here: the four call sites that pass no ``code`` still get the same
six-key frame, and a ``code`` that survives validation is written onto the
frame under its own key with nothing else alongside it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from xagent.web.api.websocket import (
    _client_visible_error_codes,
    create_terminal_task_error_event,
)

BASE_FIELDS = {"type", "message", "task_id", "task", "error", "timestamp"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
    ],
    ids=["neither"],
)
def test_terminal_error_event_shape_unchanged(kwargs: dict[str, Any]) -> None:
    """A caller that passes no code gets the same six-key frame."""

    event = create_terminal_task_error_event(1, "x", **kwargs)

    assert set(event.keys()) == BASE_FIELDS


def test_terminal_error_event_carries_a_valid_code() -> None:
    event = create_terminal_task_error_event(1, "x", code="missing_runtime_context")

    assert set(event.keys()) == BASE_FIELDS | {"code"}
    assert event["code"] == "missing_runtime_context"
    assert "details" not in event


def test_unknown_code_is_dropped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``code`` passes the same closed set the /v1 surface pins against.

    ``ConnectorRuntimeError`` types its code as a bare ``str`` and stores it
    without validation, so an unlisted value reaching the wire is a question
    of what raise sites happen to exist today, not of what the code enforces.
    """

    with caplog.at_level(logging.ERROR):
        event = create_terminal_task_error_event(
            1,
            "x",
            code="not_a_listed_code",
        )

    assert set(event.keys()) == BASE_FIELDS
    assert "not_a_listed_code" not in json.dumps(event)

    dropped = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR and "dropped=code" in record.getMessage()
    ]
    assert len(dropped) == 1
    assert "not_a_listed_code" in dropped[0]


@pytest.mark.parametrize("code", [["not", "hashable"], 7, object()])
def test_a_non_string_code_is_dropped_without_raising(
    code: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The type gate runs before the membership test.

    ``ConnectorRuntimeError`` types its code as a bare ``str`` and stores it
    unvalidated, and this builder runs inside an ``except`` that only logs a
    failed broadcast -- so a non-string value must cost the argument, not the
    frame. Only the unhashable case actually needs the gate: without it a list
    raises inside the frozenset membership test, while a hashable non-string
    (an int, a bare object) is simply not a member and already takes the drop
    path. All three are pinned so the outcome is the same shape either way.
    """
    with caplog.at_level(logging.ERROR):
        event = create_terminal_task_error_event(1, "x", code=code)
    assert set(event.keys()) == BASE_FIELDS
    dropped = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR and "dropped=code" in record.getMessage()
    ]
    assert len(dropped) == 1


@pytest.mark.parametrize(
    "code",
    [
        "connector_not_found",
        "invalid_runtime_context",
        "missing_runtime_context",
        "runtime_context_immutable",
        "runtime_secret_not_allowed",
        "runtime_secret_unavailable",
        "scheduled_secret_unavailable",
        "connector_runtime_unavailable",
        "mcp_oauth_authorization_failed",
        "delegated_authorization_failed",
    ],
)
def test_every_connector_runtime_code_survives_the_closed_set(code: str) -> None:
    """All ten connector-runtime codes are members, so none is dropped."""

    event = create_terminal_task_error_event(1, "x", code=code)

    assert event["code"] == code


def test_the_closed_set_is_the_v1_one_not_a_copy() -> None:
    from xagent.web.api.v1.errors import V1ErrorCode

    assert _client_visible_error_codes() == {member.value for member in V1ErrorCode}
