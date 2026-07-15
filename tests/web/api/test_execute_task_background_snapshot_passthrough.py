"""Test that ``execute_task_background`` consumes the snapshot when
provided and falls back to its legacy Task SELECT when it isn't.

Background:
    Profiling measured the inline ``db.query(Task)`` at the top of
    ``execute_task_background`` at ~3.3s of synchronous DB read under
    contention (the same row had just been queried by
    ``_schedule_bg._runner``). The off-loop snapshot path plumbs a
    ``task_setup_snapshot`` parameter through ``_runner`` →
    ``execute_task_background`` → ``get_agent_for_task`` so the Task
    SELECT happens once, off-loop, in
    ``load_task_setup_snapshot_sync``.

What this test pins:

    * Snapshot path: ``db.query(Task)`` is **not** called on the
      request session. ``db.query(User)`` still fires once because
      ``get_user_tool_overrides`` is a hook (``Callable[[Session,
      Any], dict]``) that may read arbitrary ORM fields off the user
      object -- swapping in a primitive shim would be a quiet BC
      break.
    * Legacy / WS path (``task_setup_snapshot=None``):
      ``db.query(Task)`` fires once (the existence check) and
      ``db.query(User)`` fires once. This pins the WS-fallback
      behaviour so a future refactor that drops the fallback branch
      gets caught here, not in production logs.
"""

from __future__ import annotations

from collections import Counter
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api.websocket import execute_task_background
from xagent.web.models.agent import AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    TaskSetupSnapshot,
    _TaskFields,
)


def _make_task_orm() -> Task:
    """Fake ORM Task row used only in the legacy / WS fallback path."""
    t = Task(
        id=42,
        user_id=1,
        title="exec-bg test",
        description="x",
        status=TaskStatus.RUNNING,
        agent_id=7,
        agent_type="standard",
    )
    return t


def _make_user_orm() -> User:
    return User(id=1, username="exec-bg-user", password_hash="hash", is_admin=False)


def _make_snapshot(*, source: str | None = "trigger") -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=42,
            user_id=1,
            status=TaskStatus.RUNNING,
            source=source,
            agent_id=7,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
        ),
        task_pattern="single_call",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=AgentRuntimeFields(
            id=7,
            name="snap-agent",
            status=AgentStatus.PUBLISHED,
            instructions="be terse",
        ),
        agent_config={
            "llms": (None, None, None, None),
            "execution_mode": "flash",
            "instructions": "be terse",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": ["basic"],
        },
        excluded_agent_id=7,
    )


class _QueryCounter:
    """Wrap ``db.query`` so we can count invocations per model class.

    ``Session.query(Model)`` is the SQLAlchemy entry point; later
    ``.filter()`` / ``.first()`` chain calls don't re-enter
    ``Session.query``, so counting at the entry is enough to detect a
    Task or User SELECT.
    """

    def __init__(self) -> None:
        self.calls_by_model: Counter[type] = Counter()
        self._returns: dict[type, Any] = {}

    def set_first(self, model: type, value: Any) -> None:
        self._returns[model] = value

    def __call__(self, model: type) -> Any:
        self.calls_by_model[model] += 1
        result = MagicMock()
        result.filter = MagicMock(return_value=result)
        result.first = MagicMock(return_value=self._returns.get(model))
        result.all = MagicMock(return_value=[])
        result.order_by = MagicMock(return_value=result)
        return result


def _build_db_mock(*, task_row: Any, user_row: Any) -> tuple[MagicMock, _QueryCounter]:
    counter = _QueryCounter()
    counter.set_first(Task, task_row)
    counter.set_first(User, user_row)
    db = MagicMock()
    db.query = counter
    return db, counter


def _common_patches(db: Any, agent_service: Any) -> list[Any]:
    """Patch the rest of ``execute_task_background``'s body so the
    test focuses on the Task / User query counters at the top.
    Downstream layers (transcript load, agent run, status update,
    persist) are stubbed -- a failure there would mask the counter
    assertions which run before the call returns / raises.
    """

    def _fake_get_db():
        yield db

    # ``execute_task_background`` re-imports these names inside the
    # function body, so patches must target the source modules (the
    # local rebind inside the function picks up whatever the source
    # module's attribute resolves to at lookup time).
    return [
        patch(
            "xagent.web.models.database.get_db",
            return_value=_fake_get_db(),
        ),
        patch(
            "xagent.web.api.websocket.background_task_manager.wait_for_previous",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.websocket._register_uploaded_files_for_agent",
        ),
        patch(
            "xagent.web.api.websocket._normalize_file_outputs",
            return_value=([], {}),
        ),
        patch(
            "xagent.web.api.websocket._rewrite_file_links_to_file_id",
            side_effect=lambda s, _m: s,
        ),
        patch(
            "xagent.web.services.task_execution_context_service.load_task_execution_recovery_state",
            new=AsyncMock(return_value={"messages": [], "skill_context": None}),
        ),
        patch(
            "xagent.web.services.chat_history_service.persist_assistant_message",
        ),
        patch(
            "xagent.web.services.chat_history_service.load_task_transcript",
            return_value=[],
        ),
    ]


def _build_fake_agent_service() -> MagicMock:
    """Minimal stand-in for ``AgentService`` covering the methods
    ``execute_task_background`` calls after the queries we're counting.
    Returning a successful run keeps the downstream finalize path
    happy (status update + persist) without a real agent runtime.
    """
    svc = MagicMock()
    svc.set_outbound_message_handler = MagicMock()
    svc.set_conversation_history = MagicMock()
    svc.set_execution_context_messages = MagicMock()
    svc.set_recovered_skill_context = MagicMock()
    svc.execute_task = AsyncMock(
        return_value={"success": True, "output": "ok", "status": "completed"}
    )
    svc.workspace = None
    return svc


@pytest.mark.asyncio
async def test_snapshot_path_skips_task_query_keeps_user_query() -> None:
    """Snapshot provided → Task SELECT must not fire on the request
    session; User SELECT must still fire once (kept for hook compat).
    """
    db, counter = _build_db_mock(task_row=_make_task_orm(), user_row=_make_user_orm())
    snapshot = _make_snapshot()
    agent_service = _build_fake_agent_service()
    agent_manager = MagicMock(get_agent_for_task=AsyncMock(return_value=agent_service))

    with _Patches(_common_patches(db, agent_service)):
        try:
            await execute_task_background(
                task_id=42,
                user_message="hi",
                context={},
                agent_manager=agent_manager,
                task_owner_user_id=1,
                task_setup_snapshot=snapshot,
            )
        except Exception:
            # Downstream finalize stubs may raise; the query counts
            # are recorded before that point.
            pass

    assert counter.calls_by_model[Task] == 0, (
        f"Task queried {counter.calls_by_model[Task]} time(s) on the request "
        "session with snapshot provided -- expected 0. The snapshot "
        "passthrough exists to skip this re-read."
    )
    assert counter.calls_by_model[User] == 1, (
        f"User queried {counter.calls_by_model[User]} time(s) -- expected 1 "
        "(kept for get_user_tool_overrides hook compat)."
    )
    forwarded_snapshot = agent_manager.get_agent_for_task.await_args.kwargs[
        "task_setup_snapshot"
    ]
    assert forwarded_snapshot is snapshot
    assert forwarded_snapshot.task.source == "trigger"


@pytest.mark.asyncio
async def test_execute_task_background_accepts_missing_context() -> None:
    """``context=None`` means no turn metadata, not a setup failure."""

    db, _counter = _build_db_mock(task_row=_make_task_orm(), user_row=_make_user_orm())
    snapshot = _make_snapshot()
    agent_service = _build_fake_agent_service()
    agent_manager = MagicMock(get_agent_for_task=AsyncMock(return_value=agent_service))

    with _Patches(_common_patches(db, agent_service)):
        try:
            await execute_task_background(
                task_id=42,
                user_message="hi",
                context=None,
                agent_manager=agent_manager,
                task_owner_user_id=1,
                task_setup_snapshot=snapshot,
            )
        except Exception:
            # The test only covers setup argument shaping. Downstream finalize
            # stubs may raise after get_agent_for_task has received the args.
            pass

    agent_manager.get_agent_for_task.assert_awaited_once()
    assert (
        agent_manager.get_agent_for_task.await_args.kwargs["connector_runtime_turn_id"]
        is None
    )


@pytest.mark.asyncio
async def test_legacy_path_runs_task_and_user_queries() -> None:
    """No snapshot (WS path) → both legacy queries fire exactly once."""
    db, counter = _build_db_mock(task_row=_make_task_orm(), user_row=_make_user_orm())
    agent_service = _build_fake_agent_service()
    agent_manager = MagicMock(get_agent_for_task=AsyncMock(return_value=agent_service))

    with _Patches(_common_patches(db, agent_service)):
        try:
            await execute_task_background(
                task_id=42,
                user_message="hi",
                context={},
                agent_manager=agent_manager,
                task_owner_user_id=1,
                task_setup_snapshot=None,
            )
        except Exception:
            pass

    assert counter.calls_by_model[Task] == 1, (
        f"Task queried {counter.calls_by_model[Task]} time(s) on legacy path "
        "-- expected 1 (the existence check). WS fallback must not regress."
    )
    assert counter.calls_by_model[User] == 1


class _Patches:
    def __init__(self, patches: list[Any]) -> None:
        self._patches = patches

    def __enter__(self) -> None:
        for p in self._patches:
            p.start()

    def __exit__(self, *exc_info: Any) -> None:
        for p in reversed(self._patches):
            p.stop()
