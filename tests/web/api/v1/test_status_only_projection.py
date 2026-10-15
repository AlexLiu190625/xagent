"""``/v1`` reads of an SDK task whose automatic resume has stopped.

The task row stays PAUSED; ``GET /v1/chat/tasks/{id}`` and the events stream
report ``failed`` with the text ``status_only_terminal_view`` returns. The
response cache holds the stored values only: the view comes from rows whose
writes do not touch ``tasks.updated_at``, so it is applied after every cache
read. The service-level cases run on SQLite and, when
``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL through the services
``engine`` fixture; the route-level cases use the ``/v1`` test application.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import pathlib
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT
from tests.shared.auth_database import auth_db_override
from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.test_status_only_append_supersede import (
    _accept,
    _hand_off_start,
)
from tests.web.services.test_task_auto_resume import PV, RUN, _seed, _user
from tests.web.services.test_task_auto_resume import factory as factory_fixture
from xagent.web.api.chat import chat_router
from xagent.web.api.v1 import _events_stream as es
from xagent.web.api.v1 import tasks as v1_tasks
from xagent.web.api.v1.deps import (
    AgentPrincipalSnapshot,
    ApiKeyPrincipal,
    RuntimeApiKeySnapshot,
)
from xagent.web.app import app
from xagent.web.models.agent import Agent
from xagent.web.models.auth_database import get_auth_db
from xagent.web.models.database import get_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery
from xagent.web.models.user import User
from xagent.web.services import task_auto_resume, task_event_bridge
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    cache_get,
    set_cache_backend_for_testing,
    task_snapshot_key,
)
from xagent.web.services.task_auto_recovery import current_auto_recovery_view
from xagent.web.services.task_auto_resume import run_auto_resume_tick
from xagent.web.services.task_lease_service import utc_now

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _override_get_db,
)
from .test_events_stream import (
    _create_agent_with_key,
    _create_task,
    _long_intervals,
    _principal_for,
    _set_task_status,
)

engine = engine_fixture
factory = factory_fixture

EXHAUSTED = (
    "Task failed after a system interruption; "
    "automatic resume stopped after repeated attempts."
)
NEW_TURN = "Task failed: the new message could not be started."

_SRC = pathlib.Path(v1_tasks.__file__).resolve().parents[4]


# --------------------------------------------------------------------------
# Service-level cases (SQLite and PostgreSQL)
# --------------------------------------------------------------------------


@pytest.fixture
def real_cache() -> Iterator[InMemoryTTLCache]:
    """A working cache backend: the default one stores nothing."""
    backend = InMemoryTTLCache()
    set_cache_backend_for_testing(backend)
    try:
        yield backend
    finally:
        set_cache_backend_for_testing(None)


@pytest.fixture
def shared_mode(monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_event_bridge, "get_task_event_bridge", lambda: Mock())
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: None
    )


def _sdk_task(factory, **row: Any) -> tuple[int, ApiKeyPrincipal]:
    """An SDK task of an agent-bound key, PAUSED on ``RUN`` with a due
    ``scheduled`` recovery row (``row`` overrides the row's seed)."""
    now = utc_now()
    with factory() as db:
        user = _user(db)
        agent = Agent(user_id=user.id, name="projection agent")
        db.add(agent)
        db.flush()
        tid = _seed(
            db, now, user=user, source="sdk", agent_id=agent.id, output="partial", **row
        )
        db.commit()
        principal = ApiKeyPrincipal(
            key=RuntimeApiKeySnapshot(key_prefix="pfx-projection"),
            agent=AgentPrincipalSnapshot(
                id=int(agent.id),
                user_id=int(user.id),
                execution_mode="balanced",
                status="published",
                origin="test",
            ),
        )
    return tid, principal


def _stored(factory, tid: int) -> tuple[Any, ...]:
    with factory() as db:
        task = db.get(Task, tid)
        return task.status, task.control_state, task.updated_at


def test_overlay_after_cache_read(factory, shared_mode, real_cache):
    """The cache keeps the stored status; the view is applied after every
    read, and an append (which does not move ``tasks.updated_at``) is never
    answered with a cached ``failed``."""
    tid, principal = _sdk_task(factory, total=20)

    waiting = v1_tasks._get_chat_task_sync(tid, principal)
    assert (waiting.status, waiting.output) == ("paused", "partial")
    assert cache_get(task_snapshot_key(tid))["response"]["status"] == "paused"
    stored_before_stop = _stored(factory, tid)

    # The sweeper stops the row: no ``tasks`` write.
    assert run_auto_resume_tick().gave_up == 1
    assert _stored(factory, tid) == stored_before_stop

    stopped = v1_tasks._get_chat_task_sync(tid, principal)
    assert (stopped.status, stopped.control_state) == ("failed", "failed")
    assert (stopped.output, stopped.error) == (None, EXHAUSTED)
    with factory() as db:
        assert stopped.completed_at == db.get(TaskAutoRecovery, tid).updated_at
    assert cache_get(task_snapshot_key(tid))["response"]["status"] == "paused"
    assert _stored(factory, tid)[:2] == (TaskStatus.PAUSED, "paused")

    # Read with nothing cached, so the entry is written while the task is
    # stopped: it must still hold the stored status.
    real_cache.delete(task_snapshot_key(tid))
    refilled = v1_tasks._get_chat_task_sync(tid, principal)
    assert refilled.status == "failed"
    assert cache_get(task_snapshot_key(tid))["response"]["status"] == "paused"

    # A shared-mode append reserves the START without touching updated_at.
    _accept(factory, tid)
    assert _stored(factory, tid) == stored_before_stop

    appended = v1_tasks._get_chat_task_sync(tid, principal)
    assert appended.status == "paused"
    assert cache_get(task_snapshot_key(tid))["response"]["status"] == "paused"
    assert _stored(factory, tid)[:2] == (TaskStatus.PAUSED, "paused")


@pytest.mark.parametrize("mode", ["combined", "shared"])
def test_append_to_projected_failure(factory, monkeypatch, mode):
    """A caller that appends after reading ``failed`` is not shown ``failed``
    again while its message is queued, and then sees the turn run."""
    monkeypatch.setattr(task_event_bridge, "get_task_event_bridge", lambda: Mock())
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: None
    )
    monkeypatch.setenv(
        "XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true" if mode == "shared" else "false"
    )
    tid, principal = _sdk_task(factory, total=20)
    assert run_auto_resume_tick().gave_up == 1
    assert v1_tasks._get_chat_task_sync(tid, principal).status == "failed"

    prepared = _accept(factory, tid)

    if mode == "combined":
        # The direct claim starts the turn at once.
        assert v1_tasks._get_chat_task_sync(tid, principal).status == "running"
        return
    queued = v1_tasks._get_chat_task_sync(tid, principal)
    assert queued.status == "paused"
    assert _stored(factory, tid)[:2] == (TaskStatus.PAUSED, "paused")

    _hand_off_start(factory, tid, prepared.command_db_id)

    assert v1_tasks._get_chat_task_sync(tid, principal).status == "running"


def test_the_counter_counts_each_computed_view_once(factory, real_cache, monkeypatch):
    seen: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        v1_tasks,
        "increment_counter",
        lambda metric, *, attributes=None, **_k: seen.append(
            (metric, dict(attributes or {}))
        ),
    )
    tid, principal = _sdk_task(factory, total=20)
    v1_tasks._get_chat_task_sync(tid, principal)
    assert seen == []

    run_auto_resume_tick()
    v1_tasks._get_chat_task_sync(tid, principal)

    assert seen == [("xagent.v1.task.projected_failed", {"state": "exhausted"})]


def test_effective_snapshot_overlays_only_the_view_fields(factory):
    tid, principal = _sdk_task(factory, total=20)
    run_auto_resume_tick()
    raw = v1_tasks._load_task_info_snapshot(tid, principal)
    assert raw.view is not None
    assert (raw.status, raw.control_state) == (TaskStatus.PAUSED, "paused")

    effective = v1_tasks.effective_task_snapshot(raw)

    assert (effective.status, effective.control_state) == (TaskStatus.FAILED, "failed")
    assert raw.output == "partial"
    assert (effective.output, effective.error) == (None, EXHAUSTED)
    assert (effective.run_id, effective.state_version, effective.updated_at) == (
        raw.run_id,
        raw.state_version,
        raw.updated_at,
    )


# --------------------------------------------------------------------------
# Who computes and who overlays; which routes derive
# --------------------------------------------------------------------------


def _calls(name: str) -> list[tuple[pathlib.Path, str | None]]:
    """Every call of ``name`` under ``src/xagent``: the file and the
    function that contains it."""
    found: list[tuple[pathlib.Path, str | None]] = []
    for path in sorted((_SRC / "xagent").rglob("*.py")):
        text = path.read_text()
        if name not in text:
            continue
        tree = ast.parse(text)

        def visit(node: ast.AST, function: str | None) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function = node.name
            if isinstance(node, ast.Call):
                func = node.func
                called = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if called == name:
                    found.append((path, function))
            for child in ast.iter_child_nodes(node):
                visit(child, function)

        visit(tree, None)
    return found


def test_view_call_sites():
    tasks_py = pathlib.Path(v1_tasks.__file__).resolve()
    defining = pathlib.Path(task_auto_resume.__file__).resolve()

    computed = [c for c in _calls("status_only_terminal_view") if c[0] != defining]
    assert computed == [(tasks_py, "_load_task_info_snapshot")]

    overlaid = _calls("effective_task_snapshot")
    assert overlaid, "the events endpoint must overlay its snapshots"
    assert {path for path, _ in overlaid} == {tasks_py}


DERIVE = {
    ("GET", "/v1/chat/tasks/{task_id}"),
    ("GET", "/v1/chat/tasks/{task_id}/events"),
}
ALLOWLIST = {
    ("POST", "/v1/chat/tasks"),
    ("POST", "/v1/chat/tasks/{task_id}/messages"),
    ("POST", "/v1/chat/tasks/{task_id}/reply"),
    ("POST", "/v1/workforces/{workforce_id}/runs"),
}
EXCLUDE = {
    ("POST", "/v1/chat/files"),
    ("GET", "/v1/chat/tasks/{task_id}/steps"),
    ("GET", "/v1/me"),
    ("GET", "/v1/templates"),
    ("GET", "/v1/templates/{template_id}"),
    ("GET", "/v1/agents"),
    ("POST", "/v1/agents"),
    ("POST", "/v1/agents/from-template"),
    ("POST", "/v1/agents/{agent_id}/api-key"),
}


def test_v1_routes_classified():
    routes = {
        (method, route.path)
        for route in app.routes
        if getattr(route, "path", "").startswith("/v1")
        for method in getattr(route, "methods", ())
    }
    assert routes == DERIVE | ALLOWLIST | EXCLUDE
    assert len(DERIVE) + len(ALLOWLIST) + len(EXCLUDE) == len(
        DERIVE | ALLOWLIST | EXCLUDE
    )
    # A route that reports task status must read it through the one loader.
    assert "_load_task_info_snapshot" in inspect.getsource(v1_tasks._get_chat_task_sync)
    assert "effective_task_snapshot" in inspect.getsource(
        v1_tasks.stream_chat_task_events
    )
    assert "_load_task_info_snapshot" in inspect.getsource(
        v1_tasks.stream_chat_task_events
    )


# --------------------------------------------------------------------------
# The events stream (route-level, SQLite)
# --------------------------------------------------------------------------


@pytest.fixture
def api_db(_test_db) -> Iterator[None]:
    with patch("xagent.web.services.task_orchestrator._schedule_bg", new=MagicMock()):
        es.reset_principal_stream_counts_for_testing()
        yield
        es.reset_principal_stream_counts_for_testing()


def _stop_in_db(task_id: int, *, state: str = "exhausted") -> None:
    """PAUSE the task and record that automatic resume stopped for good."""
    _set_task_status(
        task_id, TaskStatus.PAUSED, output="partial", control_state="paused"
    )
    db = _direct_db_session()
    try:
        task = db.get(Task, task_id)
        now = utc_now()
        db.add(
            TaskAutoRecovery(
                task_id=task_id,
                run_id=task.run_id,
                reason="lease_expired",
                state=state,
                paused_state_version=int(task.state_version),
                interrupted_at=now,
                episode_started_at=now,
            )
        )
        db.commit()
    finally:
        db.close()


def _frames(body: str) -> list[tuple[str, dict[str, Any]]]:
    result = []
    for block in (b for b in body.split("\n\n") if b.strip()):
        event = next(
            line.split(": ", 1)[1] for line in block.splitlines() if "event:" in line
        )
        data = json.loads(block.split("data: ", 1)[1])
        result.append((event, data))
    return result


def _fast_watchdog(monkeypatch) -> None:
    real = es.build_event_stream_response

    def build(**kwargs):
        return real(**kwargs, **_long_intervals(watchdog_interval_seconds=0.05))

    monkeypatch.setattr(es, "build_event_stream_response", build)


async def _until_completed(response) -> list[tuple[str, dict[str, Any]]]:
    seen: list[tuple[str, dict[str, Any]]] = []
    while True:
        chunk = await asyncio.wait_for(
            response.body_iterator.__anext__(), timeout=DB_PROGRESS_TIMEOUT
        )
        seen.extend(_frames(chunk))
        if any(event == "task.completed" for event, _ in seen):
            return seen


@pytest.mark.parametrize("path", ["attach_fast_path", "watchdog_after_stop"])
async def test_stream_terminal_on_view(api_db, monkeypatch, path):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    _fast_watchdog(monkeypatch)
    if path == "attach_fast_path":
        _stop_in_db(task_id)
    else:
        _set_task_status(
            task_id, TaskStatus.PAUSED, output="partial", control_state="paused"
        )

    response = await v1_tasks.stream_chat_task_events(task_id, principal)
    try:
        if path == "watchdog_after_stop":
            first = await asyncio.wait_for(
                response.body_iterator.__anext__(), timeout=DB_PROGRESS_TIMEOUT
            )
            assert _frames(first) == [("task.status", {"status": "paused"})]
            _stop_in_db(task_id)
        seen = await _until_completed(response)
    finally:
        await response.body_iterator.aclose()

    if path == "attach_fast_path":
        # The initial snapshot is already the view: the first frame says
        # failed and a paused one never goes out.
        assert seen[0] == ("task.status", {"status": "failed"})
        assert ("task.status", {"status": "paused"}) not in seen
    completed = [data for event, data in seen if event == "task.completed"]
    assert completed == [{"status": "failed", "output": None, "error": EXHAUSTED}]
    assert es.count_task_sinks(task_id) == 0


# --------------------------------------------------------------------------
# Other surfaces keep the stored status
# --------------------------------------------------------------------------

_chat_app = FastAPI()
_chat_app.include_router(chat_router)
_chat_app.dependency_overrides[get_db] = _override_get_db
_chat_app.dependency_overrides[get_auth_db] = auth_db_override(_override_get_db)
_chat_client = TestClient(_chat_app, raise_server_exceptions=False)


def test_web_reads_unchanged_for_sdk(api_db):
    """``/api/chat`` and the WebSocket's historical snapshot show a stopped
    SDK task as ``paused``, with the recovery notice they always carried."""
    from xagent.web.api.websocket import _history_task_info

    headers = _admin_headers()
    db = _direct_db_session()
    try:
        admin_id = int(db.query(User).filter(User.username == "admin").one().id)
        task = Task(
            user_id=admin_id,
            title="SDK task",
            description="SDK task",
            source="sdk",
            run_id=RUN,
            state_version=PV,
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
    finally:
        db.close()
    _stop_in_db(task_id)

    for suffix in ("", "/status"):
        response = _chat_client.get(
            f"/api/chat/task/{task_id}{suffix}", headers=headers
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "paused"
        assert body["auto_recovery"]["auto_resume"]["stop_reason"] == "limit_reached"

    db = _direct_db_session()
    try:
        task = db.get(Task, task_id)
        info = _history_task_info(
            db, task, task_id, auto_recovery=current_auto_recovery_view(db, task)
        )
    finally:
        db.close()
    assert info["data"]["status"] == "paused"
    assert info["data"]["auto_recovery"]["auto_resume"]["stop_reason"] == (
        "limit_reached"
    )
