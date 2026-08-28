"""Frame-shape contracts for ``create_terminal_task_error_event``.

Two things are pinned here: the four call sites that pass neither ``code`` nor
``details`` still get the same six-key frame, and ``details`` is accepted as
``PublicErrorDetails`` itself and nothing else -- not a dict, not a duck type,
not a subclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from xagent.web.api.websocket import create_terminal_task_error_event
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
def test_public_error_details_is_the_only_accepted_shape(details: Any) -> None:
    """The annotation is not the door; the explicit check in the body is.

    The subclass case is why the check reads ``type(...) is`` rather than
    ``isinstance``: a frozen dataclass can be subclassed, and a subclass that
    overrides ``to_wire`` without reading ``self.reason`` satisfies both mypy
    and ``isinstance`` while writing an unlisted string into the frame.
    """

    with pytest.raises(TypeError):
        create_terminal_task_error_event(1, "x", code="c", details=details)
