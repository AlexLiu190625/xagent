"""Integration test for PR3 Step 3: ``get_agent_for_task`` consumes
``TaskSetupSnapshot`` produced on a worker thread.

What this test pins:

    1. ``get_agent_for_task`` calls
       ``load_task_setup_snapshot_sync`` via ``asyncio.to_thread`` --
       i.e. the loader runs on a separate thread, not on the loop.
       The test asserts the loader's ``threading.get_ident()`` differs
       from the loop thread's.

    2. While the loader is sleeping, the event loop is still able to
       drive other coroutines forward. We verify by kicking off a
       concurrent ``asyncio.sleep`` task and confirming it advances
       during the snapshot load window. This is the core invariant
       Step 3 exists to provide -- main-loop release during the
       synchronous DB block (issue #427).

    3. The snapshot's fields are observed by ``get_agent_for_task``
       on the loop thread without lazy-loading from the loader's
       session (which has already closed). Equivalent to the no-leak
       contract enforced unit-side in ``test_task_setup_snapshot``,
       restated here at the integration boundary.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.task import TaskStatus
from xagent.web.models.user import User
from xagent.web.services.task_setup_snapshot import (
    TaskSetupSnapshot,
    _TaskFields,
)


def _make_user() -> User:
    return User(id=1, username="snap-int-user", password_hash="hash", is_admin=False)


def _build_snapshot() -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=42,
            user_id=1,
            status=TaskStatus.PENDING,
            agent_id=None,
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
        agent=None,
        agent_config=None,
        excluded_agent_id=None,
    )


@pytest.mark.asyncio
async def test_snapshot_runs_off_loop_thread() -> None:
    """``asyncio.to_thread`` must hand the loader off to a worker
    thread -- otherwise we haven't released the loop and Step 3 is a
    no-op. Compare the loader's thread id to the loop thread's."""
    loop_thread_id = threading.get_ident()
    loader_thread_id: dict[str, int] = {}

    def fake_loader(task_id: int, user_id: int | None) -> TaskSetupSnapshot:
        loader_thread_id["id"] = threading.get_ident()
        return _build_snapshot()

    manager = AgentServiceManager()
    user = _make_user()

    db = MagicMock()
    # Pre-Step-3 existence check still uses the request db. Mock the
    # row presence so we go down the normal-creation path.
    task_row = MagicMock()
    task_row.status = TaskStatus.PENDING
    db.query.return_value.filter.return_value.first.return_value = task_row

    with (
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            side_effect=fake_loader,
        ),
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.web.api.chat.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ),
        patch(
            "xagent.web.sandbox_manager.get_sandbox_manager",
            return_value=None,
        ),
        patch("xagent.web.api.chat.AgentService"),
    ):
        try:
            await manager.get_agent_for_task(task_id=42, db=db, user=user)
        except Exception:
            # Downstream AgentService mock may raise during workspace
            # setup; the off-loop assertion runs before that point.
            pass

    assert "id" in loader_thread_id, (
        "Loader was never invoked -- patch path or call site changed."
    )
    assert loader_thread_id["id"] != loop_thread_id, (
        f"Loader ran on the loop thread (id={loop_thread_id}). "
        "Step 3's whole point is to push the synchronous DB block off "
        "the loop via ``asyncio.to_thread`` -- if this fails, either "
        "the to_thread wrapper was removed or the loader is being "
        "called inline."
    )


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_during_snapshot_load() -> None:
    """The other half of the Step 3 contract: while the snapshot
    loader is sleeping (simulating a slow DB read), other coroutines
    on the same loop must still be able to make progress.

    We block the loader for 0.3s and concurrently schedule a tight
    polling task that records its tick count. If ``to_thread`` works,
    the polling task progresses across many ticks during the loader's
    sleep. If Step 3 regressed back to an inline synchronous call,
    the polling task records at most one tick (no progress until the
    blocking sleep returns).
    """
    snapshot = _build_snapshot()
    ticks: list[float] = []
    loader_done = asyncio.Event()

    def slow_loader(task_id: int, user_id: int | None) -> TaskSetupSnapshot:
        # Synchronous sleep on the worker thread. If the call is
        # actually executed inline on the loop thread, this freezes
        # the entire loop and the poll task can't tick.
        import time

        time.sleep(0.3)
        return snapshot

    async def poll() -> None:
        loop = asyncio.get_running_loop()
        start = loop.time()
        while not loader_done.is_set():
            ticks.append(loop.time() - start)
            # Short sleep to yield, but the *loop* must run to come
            # back to us. If the loader is hogging the loop this
            # await never resumes until the sleep returns.
            await asyncio.sleep(0.02)

    manager = AgentServiceManager()
    user = _make_user()
    db = MagicMock()
    task_row = MagicMock()
    task_row.status = TaskStatus.PENDING
    db.query.return_value.filter.return_value.first.return_value = task_row

    async def driver() -> None:
        with (
            patch(
                "xagent.web.api.chat.load_task_setup_snapshot_sync",
                side_effect=slow_loader,
            ),
            patch.object(manager, "_load_persisted_conversation_history"),
            patch(
                "xagent.web.api.chat.create_task_tracer",
                return_value=MagicMock(),
            ),
            patch(
                "xagent.web.api.chat.create_default_tools",
                new=AsyncMock(return_value=([], MagicMock())),
            ),
            patch(
                "xagent.web.sandbox_manager.get_sandbox_manager",
                return_value=None,
            ),
            patch("xagent.web.api.chat.AgentService"),
        ):
            try:
                await manager.get_agent_for_task(task_id=42, db=db, user=user)
            except Exception:
                pass
        loader_done.set()

    await asyncio.gather(driver(), poll())

    # With a 0.3s blocking sleep on the worker thread and 0.02s
    # polling intervals on the loop, we expect on the order of ~10
    # ticks. Use a permissive floor of >= 5 to keep the test stable
    # under busy CI while still failing loudly if the loop genuinely
    # freezes (which would yield 0-1 ticks).
    assert len(ticks) >= 5, (
        f"Loop ticked only {len(ticks)} times during the 0.3s snapshot "
        "load -- the loader appears to be running inline on the loop "
        "thread (Step 3 regressed)."
    )


@pytest.mark.asyncio
async def test_loop_consumes_snapshot_after_session_close() -> None:
    """When the loader returns, the snapshot must be fully usable on
    the loop thread with the loader's session already closed. We
    simulate the post-close state by passing a snapshot whose dict
    contents were copied (not still backed by an ORM proxy), and
    confirm ``get_agent_for_task`` reaches the AgentService
    construction step using snapshot fields.

    A snapshot that secretly held an ORM ref would normally raise
    ``DetachedInstanceError`` on attribute access here -- but because
    the snapshot is a frozen dataclass holding primitives, the loop
    consumes it without further DB access. This test pins that
    expectation at the integration boundary.
    """
    snapshot = _build_snapshot()

    constructed: dict[str, Any] = {}

    class _FakeAgentService:
        def __init__(self, **kwargs: Any) -> None:
            constructed.update(kwargs)
            self.workspace = None

        def cleanup_workspace(self) -> None: ...

    manager = AgentServiceManager()
    user = _make_user()
    db = MagicMock()
    task_row = MagicMock()
    task_row.status = TaskStatus.PENDING
    db.query.return_value.filter.return_value.first.return_value = task_row

    with (
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ),
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.web.api.chat.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ),
        patch(
            "xagent.web.sandbox_manager.get_sandbox_manager",
            return_value=None,
        ),
        patch("xagent.web.api.chat.AgentService", new=_FakeAgentService),
    ):
        await manager.get_agent_for_task(task_id=42, db=db, user=user)

    # ``pattern`` and ``task_id`` should have flowed through from the
    # snapshot to the AgentService constructor. If they didn't, the
    # consumer was reading from a stale ORM ref and would have raised
    # before reaching this point.
    assert constructed.get("pattern") == "single_call"
    assert constructed.get("task_id") == "42"
