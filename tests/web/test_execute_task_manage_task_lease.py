"""Tests for ``AgentServiceManager.execute_task`` lease delegation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services.task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
)
from xagent.web.services.workforce_runtime import sync_workforce_run_status


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'execute_task_lease.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


class _FakeAgentService:
    async def execute_task(self, **_kwargs):
        return {"success": True}

    def set_interrupt_checker(self, _checker):
        # execute_task_background sets the mid-run quota checker after tracking
        # starts and clears it on completion; the double must accept both.
        pass


@pytest.mark.asyncio
async def test_execute_task_preflight_pool_wait_does_not_block_event_loop(
    tmp_path,
) -> None:
    """Quota/workforce preflight must wait for the pool off the event loop."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'execute-preflight-pool.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.5,
        connect_args={"check_same_thread": False},
    )
    User.__table__.create(bind=engine)
    Task.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with factory() as setup_db:
        user = User(username="pool-user", password_hash="hash", is_admin=False)
        setup_db.add(user)
        setup_db.flush()
        task = Task(
            user_id=user.id,
            title="pool test",
            description="test",
            status=TaskStatus.RUNNING,
            execution_mode="auto",
        )
        setup_db.add(task)
        setup_db.commit()
        task_id = int(task.id)

    held_connection = engine.connect()
    caller_db = factory()
    manager = AgentServiceManager()
    ticks = 0
    stop = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    try:
        with (
            patch(
                "xagent.web.models.database.get_session_local",
                return_value=factory,
            ),
            patch.object(
                manager, "_acquire_sandbox_task", new=AsyncMock(return_value=None)
            ),
            patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
            patch(
                "xagent.web.tracking.task_tracker.TaskTracker",
                side_effect=RuntimeError("skip tracking in pool-boundary test"),
            ),
        ):
            await asyncio.sleep(0.02)
            ticks_before_wait = ticks
            execute = asyncio.create_task(
                manager.execute_task(
                    agent_service=_FakeAgentService(),
                    task="hello",
                    tracking_task_id=str(task_id),
                    db_session=caller_db,
                    manage_task_lease=False,
                )
            )
            await asyncio.sleep(0.08)

            assert ticks - ticks_before_wait >= 4
            assert not execute.done()

            held_connection.close()
            result = await execute
            assert result["success"] is True
    finally:
        if not held_connection.closed:
            held_connection.close()
        stop.set()
        await ticker_task
        caller_db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_execute_task_acquires_and_releases_lease_when_manage_true(
    db_session,
) -> None:
    user = User(username="lease-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="lease test",
        description="test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.commit()

    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")
    manager = AgentServiceManager()

    with (
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=fake_lease,
        ) as mock_acquire,
        patch(
            "xagent.web.api.chat.release_task_lease_with_workforce_sync",
        ) as mock_release,
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch.object(
            manager, "_acquire_sandbox_task", new=AsyncMock(return_value=None)
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status",
            return_value=False,
        ) as mock_sync,
    ):
        result = await manager.execute_task(
            agent_service=_FakeAgentService(),
            task="hello",
            tracking_task_id=str(task.id),
            db_session=db_session,
            manage_task_lease=True,
        )

    assert result["success"] is True
    mock_acquire.assert_called_once()
    mock_release.assert_called_once()
    mock_sync.assert_called_once()


@pytest.mark.asyncio
async def test_execute_task_skips_lease_but_syncs_running_when_manage_false(
    db_session,
) -> None:
    user = User(username="lease-user2", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="lease test",
        description="test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.commit()

    manager = AgentServiceManager()

    with (
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
        ) as mock_acquire,
        patch(
            "xagent.web.api.chat.release_task_lease_with_workforce_sync",
        ) as mock_release,
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ) as mock_stop_hb,
        patch.object(
            manager, "_acquire_sandbox_task", new=AsyncMock(return_value=None)
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status",
            return_value=False,
        ) as mock_sync,
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            side_effect=RuntimeError("skip tracking in unit test"),
        ),
    ):
        result = await manager.execute_task(
            agent_service=_FakeAgentService(),
            task="hello",
            tracking_task_id=str(task.id),
            db_session=db_session,
            manage_task_lease=False,
        )

    assert result["success"] is True
    mock_acquire.assert_not_called()
    mock_release.assert_not_called()
    mock_sync.assert_called_once()
    mock_stop_hb.assert_awaited_once_with(None, None)


@pytest.mark.asyncio
async def test_execute_task_surfaces_mid_run_quota_reason(db_session) -> None:
    """When the mid-run quota gate trips, the run result is reshaped to a
    terminal quota_exceeded carrying the reason as output (mirroring the start
    gate) instead of the pattern-interrupt path's silent flip to PAUSED."""
    user = User(username="quota-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="quota test",
        description="test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.commit()

    manager = AgentServiceManager()
    reason = "Monthly ai_credits_per_month quota reached. Upgrade your plan."
    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.complete_tracking = AsyncMock()
    tracker.quota_interrupt_reason = reason  # the mid-run gate tripped

    agent_service = _FakeAgentService()
    # The pattern-interrupt path returns a silent "interrupted" result.
    agent_service.execute_task = AsyncMock(  # type: ignore[method-assign]
        return_value={"success": False, "status": "interrupted", "error": "interrupted"}
    )

    with (
        patch("xagent.web.api.chat.run_task_lease_heartbeat", new=AsyncMock()),
        patch("xagent.web.api.chat.stop_task_lease_heartbeat", new=AsyncMock()),
        patch.object(
            manager, "_acquire_sandbox_task", new=AsyncMock(return_value=None)
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
        patch("xagent.web.api.chat.sync_workforce_run_status", return_value=False),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
    ):
        result = await manager.execute_task(
            agent_service=agent_service,
            task="hello",
            tracking_task_id=str(task.id),
            db_session=db_session,
            manage_task_lease=False,
        )

    assert result["status"] == "quota_exceeded"
    assert result["success"] is False
    assert result["output"] == reason
    assert result["error"] == reason
    # A mid-run interrupt is always the quota checker, so the result carries the
    # code (matching the start gate) to drive the app-layer dialog.
    assert result["error_code"] == "quota_exceeded"


@pytest.mark.asyncio
async def test_execute_task_start_gate_forwards_structured_reason(db_session) -> None:
    """When the start gate returns a structured reason (mapping), the run result
    carries its message plus error_code/error_details so the client can localise
    and branch, instead of only a plain string."""
    user = User(username="quota-start-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="quota start test",
        description="test",
        status=TaskStatus.PENDING,
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.commit()

    manager = AgentServiceManager()
    block = {
        "code": "quota_exceeded",
        "metric": "runs_per_month",
        "limit": 0,
        "plan": "basic",
        "message": "Team quota exhausted for this billing period.",
    }

    # The gate short-circuits before lease/tracker/execution, so a patched
    # check_run_gate returning the structured block is enough.
    with patch("xagent.web.services.quota_hooks.check_run_gate", return_value=block):
        result = await manager.execute_task(
            agent_service=_FakeAgentService(),
            task="hello",
            tracking_task_id=str(task.id),
            db_session=db_session,
            manage_task_lease=False,
        )

    assert result["status"] == "quota_exceeded"
    assert result["success"] is False
    assert result["output"] == block["message"]
    assert result["error_code"] == "quota_exceeded"
    assert result["error_details"] == block


@pytest.mark.asyncio
async def test_execute_task_cleans_up_when_sandbox_acquire_raises(
    db_session,
) -> None:
    """A reclaimed-sandbox raise from ``_acquire_sandbox_task`` must still
    run the finally cleanup: heartbeat stop, lease release, and tracker
    completion (whose only call site is that finally block)."""
    user = User(username="lease-user3", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="lease test",
        description="test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.commit()

    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")
    manager = AgentServiceManager()
    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.complete_tracking = AsyncMock()
    agent_service = _FakeAgentService()
    agent_service.execute_task = AsyncMock()  # type: ignore[method-assign]
    agent_service.set_interrupt_checker = MagicMock()  # type: ignore[method-assign]

    with (
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.api.chat.release_task_lease_with_workforce_sync",
        ) as mock_release,
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ) as mock_stop_hb,
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(side_effect=RuntimeError("sandbox reclaimed")),
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()) as mock_sbx,
        patch(
            "xagent.web.api.chat.sync_workforce_run_status",
            return_value=False,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
    ):
        with pytest.raises(RuntimeError, match="sandbox reclaimed"):
            await manager.execute_task(
                agent_service=agent_service,
                task="hello",
                tracking_task_id=str(task.id),
                db_session=db_session,
                manage_task_lease=True,
            )

    agent_service.execute_task.assert_not_awaited()
    mock_stop_hb.assert_awaited_once()
    mock_release.assert_called_once()
    assert mock_release.call_args.kwargs["status"] == TaskStatus.FAILED
    tracker.complete_tracking.assert_awaited_once()
    mock_sbx.assert_awaited_once_with(None)
    # The mid-run quota checker must be cleared in the finally so a reused
    # agent_service can't keep calling this finished run's tracker.
    agent_service.set_interrupt_checker.assert_any_call(None)


@pytest.mark.asyncio
async def test_execute_task_persists_final_usage_before_releasing_lease() -> None:
    """The final usage snapshot belongs to the current run, so it must land
    while that run still owns its lease."""
    task_id = 101
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    events: list[str] = []

    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.quota_interrupt_reason = None

    async def complete_tracking() -> None:
        events.append("usage")

    tracker.complete_tracking = AsyncMock(side_effect=complete_tracking)

    async def stop_heartbeat(*_args) -> None:
        events.append("heartbeat")

    def release_lease(*_args, **_kwargs) -> bool:
        events.append("lease")
        return True

    async def release_sandbox(*_args) -> None:
        events.append("sandbox")

    with (
        patch(
            "xagent.web.api.chat._check_task_run_gate_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat._sync_task_workforce_running_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            side_effect=release_lease,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=stop_heartbeat,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ) as tracker_factory,
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value="user:101"),
        ),
        patch.object(manager, "_release_sandbox_task", new=release_sandbox),
    ):
        result = await manager.execute_task(
            agent_service=_FakeAgentService(),
            task="hello",
            tracking_task_id=str(task_id),
            manage_task_lease=True,
        )

    assert result["success"] is True
    tracker_factory.assert_called_once_with(
        task_id=task_id,
        expected_run_id="test-run",
        expected_runner_id="test-runner",
    )
    assert events == ["usage", "heartbeat", "lease", "sandbox"]


@pytest.mark.asyncio
async def test_execute_task_final_usage_pool_timeout_retains_lease() -> None:
    """One exhausted final checkout must not trigger a second lease checkout."""
    task_id = 103
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.complete_tracking = AsyncMock(
        side_effect=SQLAlchemyTimeoutError("pool exhausted")
    )
    tracker.quota_interrupt_reason = None

    release_lease = MagicMock()
    stop_heartbeat = AsyncMock()
    release_sandbox = AsyncMock()

    with (
        patch(
            "xagent.web.api.chat._check_task_run_gate_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat._sync_task_workforce_running_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            release_lease,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=stop_heartbeat,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value="user:103"),
        ),
        patch.object(manager, "_release_sandbox_task", new=release_sandbox),
    ):
        with pytest.raises(SQLAlchemyTimeoutError, match="pool exhausted"):
            await manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )

    tracker.complete_tracking.assert_awaited_once()
    stop_heartbeat.assert_awaited_once()
    release_lease.assert_not_called()
    release_sandbox.assert_awaited_once_with("user:103")


@pytest.mark.asyncio
async def test_execute_task_heartbeat_pool_timeout_retains_lease() -> None:
    """Heartbeat pool exhaustion must not be followed by lease release I/O."""
    task_id = 104
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.complete_tracking = AsyncMock()
    tracker.quota_interrupt_reason = None
    heartbeat_timeout = SQLAlchemyTimeoutError("heartbeat pool exhausted")
    release_lease = MagicMock()
    release_sandbox = AsyncMock()

    with (
        patch(
            "xagent.web.api.chat._check_task_run_gate_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat._sync_task_workforce_running_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            release_lease,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(
                return_value=TaskLeaseHeartbeatOutcome(pool_timeout=heartbeat_timeout)
            ),
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value="user:104"),
        ),
        patch.object(manager, "_release_sandbox_task", new=release_sandbox),
    ):
        with pytest.raises(SQLAlchemyTimeoutError, match="heartbeat pool exhausted"):
            await manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )

    tracker.complete_tracking.assert_awaited_once()
    release_lease.assert_not_called()
    release_sandbox.assert_awaited_once_with("user:104")


@pytest.mark.asyncio
async def test_execute_task_cancellation_during_heartbeat_stop_drains_cleanup() -> None:
    """Caller cancellation during heartbeat shutdown must not strand the
    lease or skip tracker and sandbox cleanup."""
    task_id = 102
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    events: list[str] = []
    heartbeat_stop_entered = asyncio.Event()
    allow_heartbeat_stop = asyncio.Event()

    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.quota_interrupt_reason = None

    async def complete_tracking() -> None:
        events.append("usage")

    tracker.complete_tracking = AsyncMock(side_effect=complete_tracking)

    async def stop_heartbeat(*_args) -> None:
        events.append("heartbeat-enter")
        heartbeat_stop_entered.set()
        await allow_heartbeat_stop.wait()
        events.append("heartbeat-finish")

    def release_lease(*_args, **_kwargs) -> bool:
        events.append("lease")
        return True

    async def release_sandbox(*_args) -> None:
        events.append("sandbox")

    with (
        patch(
            "xagent.web.api.chat._check_task_run_gate_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat._sync_task_workforce_running_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            side_effect=release_lease,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=stop_heartbeat,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value="user:102"),
        ),
        patch.object(manager, "_release_sandbox_task", new=release_sandbox),
    ):
        execution = asyncio.create_task(
            manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )
        )
        await asyncio.wait_for(heartbeat_stop_entered.wait(), timeout=1)
        execution.cancel()
        await asyncio.sleep(0)
        allow_heartbeat_stop.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(execution, timeout=1)

    assert events == [
        "usage",
        "heartbeat-enter",
        "heartbeat-finish",
        "lease",
        "sandbox",
    ]


def test_sync_workforce_run_status_running_is_idempotent(db_session) -> None:
    """Repeat RUNNING sync is a no-op when WorkforceRun is already running."""
    user = User(username="sync-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.flush()
    manager = Agent(
        user_id=user.id,
        name="Manager",
        description="desc",
        instructions="instr",
        execution_mode="balanced",
        models={"general": "test-model"},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        status=AgentStatus.PUBLISHED,
    )
    db_session.add(manager)
    db_session.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name="Team",
        description="desc",
        manager_agent_id=manager.id,
        status="active",
    )
    db_session.add(workforce)
    db_session.flush()
    task = Task(
        user_id=user.id,
        title="sync test",
        description="test",
        status=TaskStatus.RUNNING,
        agent_id=manager.id,
        agent_config={},
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="running",
        snapshot={"version": 1},
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = {"workforce_run_id": run.id}
    db_session.commit()

    assert sync_workforce_run_status(db_session, task, TaskStatus.RUNNING) is False
    assert sync_workforce_run_status(db_session, task, TaskStatus.RUNNING) is False
    db_session.refresh(run)
    assert run.status == "running"
    assert run.completed_at is None
