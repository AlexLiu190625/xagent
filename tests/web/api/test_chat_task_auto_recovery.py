"""``auto_recovery`` on the web UI's task detail and status.

``GET /api/chat/task/{id}`` and ``GET /api/chat/task/{id}/status`` report the
interruption that left a PAUSED task as it stands: its ``task_auto_recovery``
row, only while the row still names the task's current run and
``state_version``. The row is not cleared when the run is resumed, so a stale
row must read as ``null``; a FAILED run's row is not shown, and the
operator-only fields never leave the server. History replay carries the same
view, and its reasserted pause says why the run stopped.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from tests.shared.auth_database import auth_db_override
from xagent.web.api.chat import chat_router
from xagent.web.models.auth_database import get_auth_db
from xagent.web.models.database import get_db, release_db_connection_if_clean
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery
from xagent.web.models.user import User
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    set_cache_backend_for_testing,
)
from xagent.web.services.task_auto_recovery import current_auto_recovery_view

from .conftest import _admin_headers, _direct_db_session, _override_get_db

pytestmark = pytest.mark.usefixtures("_test_db")

_chat_app = FastAPI()
_chat_app.include_router(chat_router)
_chat_app.dependency_overrides[get_db] = _override_get_db
_chat_app.dependency_overrides[get_auth_db] = auth_db_override(_override_get_db)
client = TestClient(_chat_app, raise_server_exceptions=False)

ROUTES = pytest.mark.parametrize(
    "route_suffix", ["", "/status"], ids=["detail", "status"]
)

RUN_ID = "run-current"
STATE_VERSION = 7
INTERRUPTED_AT = datetime(2026, 10, 1, 8, 30, tzinfo=timezone.utc)


def _paused_task(
    *,
    row: dict | None,
    status: TaskStatus = TaskStatus.PAUSED,
    conversation_storage_version: int = 1,
) -> int:
    """A ``status`` (PAUSED) task at ``RUN_ID``/``STATE_VERSION``, with a
    recovery row built from ``row`` overrides (``None``: no row)."""
    db = _direct_db_session()
    try:
        admin_id = int(db.query(User).filter(User.username == "admin").one().id)
        task = Task(
            user_id=admin_id,
            title="Interrupted task",
            description="Interrupted task",
            status=status,
            control_state=status.value,
            run_id=RUN_ID,
            state_version=STATE_VERSION,
            conversation_storage_version=conversation_storage_version,
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
        if row is not None:
            db.add(
                TaskAutoRecovery(
                    task_id=task_id,
                    **{
                        "run_id": RUN_ID,
                        "reason": "lease_expired",
                        "state": "manual",
                        "state_detail": "operator-detail",
                        "paused_state_version": STATE_VERSION,
                        "interrupted_at": INTERRUPTED_AT,
                        "episode_started_at": INTERRUPTED_AT,
                        "total_resumes": 2,
                        "last_command_id": "command-1",
                        "last_error": "operator-only diagnostic",
                        **row,
                    },
                )
            )
            db.commit()
        return task_id
    finally:
        db.close()


def _auto_recovery(task_id: int, route_suffix: str) -> object:
    response = client.get(
        f"/api/chat/task/{task_id}{route_suffix}", headers=_admin_headers()
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "auto_recovery" in body
    return body["auto_recovery"]


@ROUTES
def test_current_interruption_is_shown_minimally(route_suffix: str) -> None:
    _admin_headers()
    task_id = _paused_task(row={})

    assert _auto_recovery(task_id, route_suffix) == {
        "reason": "lease_expired",
        "interrupted_at": "2026-10-01T08:30:00+00:00",
    }


@ROUTES
@pytest.mark.parametrize(
    "stale",
    [
        {"paused_state_version": STATE_VERSION - 1},
        {"run_id": "run-earlier"},
    ],
    ids=["state_version", "run_id"],
)
def test_stale_interruption_is_hidden(route_suffix: str, stale: dict) -> None:
    """A row an earlier pause left behind: the task has resumed since."""
    _admin_headers()
    task_id = _paused_task(row=stale)

    assert _auto_recovery(task_id, route_suffix) is None


@ROUTES
@pytest.mark.parametrize("state", ["manual", "disabled"])
def test_failed_run_row_is_not_shown(route_suffix: str, state: str) -> None:
    """Settlement records a FAILED run's interruption fenced at the FAILED
    ``state_version``; it is bookkeeping, not a pause to explain."""
    _admin_headers()
    task_id = _paused_task(row={"state": state}, status=TaskStatus.FAILED)

    assert _auto_recovery(task_id, route_suffix) is None


@ROUTES
def test_unreadable_row_reads_as_null(route_suffix: str, caplog) -> None:
    """The view is informational: a failed read must not fail the request."""
    _admin_headers()
    task_id = _paused_task(row={})
    db = _direct_db_session()
    try:
        db.execute(text("DROP TABLE task_auto_recovery"))
        db.commit()
    finally:
        db.close()

    with caplog.at_level(logging.WARNING):
        assert _auto_recovery(task_id, route_suffix) is None
    assert "Reading the auto-recovery row" in caplog.text


def test_failed_read_leaves_the_session_usable_and_clean(caplog) -> None:
    """The read runs in a SAVEPOINT: after it fails the caller's transaction
    still runs statements (PostgreSQL would otherwise refuse them), and a
    successful read leaves the session releasable as read-only."""
    _admin_headers()
    task_id = _paused_task(row={})
    db = _direct_db_session()
    try:
        task = db.get(Task, task_id)
        assert current_auto_recovery_view(db, task) is not None
        assert release_db_connection_if_clean(db)

        db.execute(text("DROP TABLE task_auto_recovery"))
        db.commit()
        db.expunge_all()
        task = db.get(Task, task_id)
        with caplog.at_level(logging.WARNING):
            assert current_auto_recovery_view(db, task) is None
        assert "Reading the auto-recovery row" in caplog.text
        assert db.query(Task.id).filter(Task.id == task_id).scalar() == task_id
    finally:
        db.close()


@ROUTES
def test_task_without_interruption_reports_null(route_suffix: str) -> None:
    _admin_headers()
    task_id = _paused_task(row=None)

    assert _auto_recovery(task_id, route_suffix) is None


@pytest.fixture
def _response_cache():
    set_cache_backend_for_testing(InMemoryTTLCache())
    yield
    set_cache_backend_for_testing(None)


@ROUTES
@pytest.mark.usefixtures("_response_cache")
def test_cached_response_reads_the_row_afresh(route_suffix: str) -> None:
    """The response cache is keyed by the task row; the recovery row is not
    part of it, so a cache hit still reports the row as it stands."""
    _admin_headers()
    task_id = _paused_task(row=None)
    assert _auto_recovery(task_id, route_suffix) is None

    db = _direct_db_session()
    try:
        db.add(
            TaskAutoRecovery(
                task_id=task_id,
                run_id=RUN_ID,
                reason="llm_unavailable",
                state="manual",
                paused_state_version=STATE_VERSION,
                interrupted_at=INTERRUPTED_AT,
                episode_started_at=INTERRUPTED_AT,
            )
        )
        db.commit()
    finally:
        db.close()

    assert _auto_recovery(task_id, route_suffix) == {
        "reason": "llm_unavailable",
        "interrupted_at": "2026-10-01T08:30:00+00:00",
    }


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {},
            {
                "reason": "lease_expired",
                "interrupted_at": "2026-10-01T08:30:00+00:00",
            },
        ),
        ({"paused_state_version": STATE_VERSION - 1}, None),
        (None, None),
    ],
    ids=["current", "stale", "none"],
)
def test_history_task_info_carries_the_same_view(row, expected) -> None:
    """The web UI loads a task from its history replay's ``task_info``, not
    from the detail endpoint, so that frame carries the same view."""
    from xagent.web.api.websocket import _history_task_info

    _admin_headers()
    task_id = _paused_task(row=row)
    db = _direct_db_session()
    try:
        info = _history_task_info(db, db.get(Task, task_id), task_id)
    finally:
        db.close()

    assert info["data"]["auto_recovery"] == expected


@pytest.mark.parametrize(
    "conversation_storage_version", [2, 1], ids=["event_history", "legacy_history"]
)
@pytest.mark.parametrize(
    ("row", "reason"),
    [({}, "lease_expired"), ({"run_id": "run-earlier"}, None), (None, None)],
    ids=["current", "stale", "none"],
)
def test_reasserted_history_pause_says_why(
    conversation_storage_version: int, row, reason
) -> None:
    """Replayed activity reads as running to the client and drops the pause's
    reason; the pause history reasserts last carries it again."""
    from xagent.web.api.websocket import _load_historical_stream_snapshot_sync

    _admin_headers()
    task_id = _paused_task(
        row=row, conversation_storage_version=conversation_storage_version
    )
    db = _direct_db_session()
    try:
        admin_id = int(db.query(User).filter(User.username == "admin").one().id)
    finally:
        db.close()

    snapshot = _load_historical_stream_snapshot_sync(
        task_id, actor_user_id=admin_id, actor_is_admin=True
    )

    assert snapshot is not None
    paused = [event for event in snapshot.events if event["type"] == "task_paused"]
    assert len(paused) == 1
    assert paused[0].get("interruption_reason") == reason
    assert paused[-1] is snapshot.events[-1]
