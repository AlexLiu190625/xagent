"""Frame-shape contracts for ``create_terminal_task_error_event``.

Two things are pinned here: the four call sites that pass neither ``code`` nor
``details`` still get the same six-key frame, and ``details`` is accepted as
``PublicErrorDetails`` itself and nothing else -- not a dict, not a duck type,
not a subclass.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from xagent.web.api.websocket import (
    _client_visible_error_codes,
    create_terminal_task_error_event,
)
from xagent.web.services.client_error_messages import PublicErrorDetails

BASE_FIELDS = {"type", "message", "task_id", "task", "error", "timestamp"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"code": "missing_runtime_context"},
        {"details": PublicErrorDetails(reason="not_provided")},
    ],
    ids=["neither", "code-only", "details-only"],
)
def test_terminal_error_event_shape_unchanged(kwargs: dict[str, Any]) -> None:
    """Both new fields are written together or not at all."""

    event = create_terminal_task_error_event(1, "x", **kwargs)

    assert set(event.keys()) == BASE_FIELDS


def test_terminal_error_event_carries_both_new_fields_together() -> None:
    event = create_terminal_task_error_event(
        1,
        "x",
        code="missing_runtime_context",
        details=PublicErrorDetails(reason="missing_context.auth_token"),
    )

    assert set(event.keys()) == BASE_FIELDS | {"code", "details"}
    assert event["code"] == "missing_runtime_context"
    assert event["details"] == {"reason": "missing_context.auth_token"}


def test_terminal_error_event_keeps_an_emptied_details_object() -> None:
    """A dropped reason still leaves the code, which the client reads."""

    event = create_terminal_task_error_event(
        1,
        "x",
        code="missing_runtime_context",
        details=PublicErrorDetails(reason="not a listed value"),
    )

    assert event["code"] == "missing_runtime_context"
    assert event["details"] == {}


class _DuckDetails:
    def to_wire(self) -> dict[str, str]:
        return {"reason": "not a listed value"}


@dataclass(frozen=True)
class _SubclassDetails(PublicErrorDetails):
    raw: str = ""

    def to_wire(self) -> dict[str, str]:
        # Never reads self.reason, so __post_init__'s whitelist is bypassed.
        return {"reason": self.raw}


@pytest.mark.parametrize(
    "details",
    [
        {"reason": "not a listed value"},
        "not a listed value",
        _DuckDetails(),
        _SubclassDetails(reason=None, raw="not a listed value"),
    ],
    ids=["dict", "str", "duck-type", "subclass"],
)
def test_public_error_details_is_the_only_accepted_shape(
    details: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The annotation is not the door; the explicit check in the body is.

    The subclass case is why the check reads ``type(...) is`` rather than
    ``isinstance``: a frozen dataclass can be subclassed, and a subclass that
    overrides ``to_wire`` without reading ``self.reason`` satisfies both mypy
    and ``isinstance`` while writing an unlisted string into the frame.

    Rejection drops the argument, it does not raise. The frame is the last
    thing between the user and a silent failure, and the one caller that
    passes these arguments builds the frame inside an ``except Exception``
    that only logs -- so an exception here would cost the whole frame.
    """

    with caplog.at_level(logging.ERROR):
        event = create_terminal_task_error_event(
            1, "x", code="missing_runtime_context", details=details
        )

    # The unlisted string the bad shape wanted to smuggle in never appears.
    assert set(event.keys()) == BASE_FIELDS
    assert "not a listed value" not in json.dumps(event)

    dropped = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR and "dropped=details" in record.getMessage()
    ]
    assert len(dropped) == 1
    assert type(details).__name__ in dropped[0]


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
            details=PublicErrorDetails(reason="not_provided"),
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

    event = create_terminal_task_error_event(
        1, "x", code=code, details=PublicErrorDetails(reason="not_provided")
    )

    assert event["code"] == code


def test_the_closed_set_is_the_v1_one_not_a_copy() -> None:
    from xagent.web.api.v1.errors import V1ErrorCode

    assert _client_visible_error_codes() == {member.value for member in V1ErrorCode}
