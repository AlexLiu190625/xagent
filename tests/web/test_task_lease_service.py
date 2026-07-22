"""Tests for task execution leases."""

import asyncio
import threading
from datetime import timedelta

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.core.agent.checkpoint import CHECKPOINT_TYPE, LEGACY_CHECKPOINT_TYPES
from xagent.web.models import database as database_module
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import ExecutionMode, Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services import task_lease_service
from xagent.web.services.db_runtime import is_database_pool_timeout
from xagent.web.services.task_lease_service import (
    TaskLease,
    TaskLeaseRefreshState,
    acquire_task_lease,
    acquire_task_lease_no_commit,
    mark_task_paused_if_stale,
    refresh_task_lease,
    release_task_lease,
    run_task_lease_heartbeat,
    stop_task_lease_heartbeat,
    utc_now,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'lease.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture()
def queue_pool_runtime_db(tmp_path):
    """A real one-slot QueuePool used to exercise checkout contention."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'lease-queue-pool.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.4,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    try:
        yield engine, SessionLocal
    finally:
        engine.dispose()


def _create_task(db, *, status=TaskStatus.PENDING) -> Task:
    user = User(username="lease-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    task = Task(
        user_id=user.id,
        title="Lease test",
        description="Lease test",
        status=status,
        execution_mode="auto",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def test_task_model_default_execution_mode_is_auto(db_session) -> None:
    user = User(username="default-mode-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    task = Task(
        user_id=user.id,
        title="Default mode",
        description="Default mode",
        status=TaskStatus.PENDING,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    assert task.execution_mode == "auto"
    assert task.execution_mode_enum == ExecutionMode.AUTO


def test_task_lease_acquire_refresh_and_release(db_session) -> None:
    task = _create_task(db_session)

    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")

    assert lease is not None
    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "runner-a"
    assert task.lease_expires_at is not None
    assert task.run_id == lease.run_id
    assert task.state_version == 1
    assert task.control_state == "running"

    assert acquire_task_lease(db_session, int(task.id), runner_id="runner-b") is None
    assert refresh_task_lease(db_session, lease) == TaskLeaseRefreshState.REFRESHED
    assert release_task_lease(db_session, lease, status=TaskStatus.COMPLETED) is True
    db_session.refresh(task)
    assert task.status == TaskStatus.COMPLETED
    assert task.state_version == 2
    assert task.control_state == "completed"
    assert task.runner_id is None
    assert task.lease_expires_at is None


def test_acquire_returns_run_id_from_update_without_followup_select(
    queue_pool_runtime_db,
) -> None:
    engine, SessionLocal = queue_pool_runtime_db
    with SessionLocal() as seed_db:
        task = _create_task(seed_db)
        task_id = int(task.id)

    statements: list[str] = []

    def record_statement(
        _conn,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record_statement)
    try:
        with SessionLocal() as db:
            lease = acquire_task_lease(db, task_id, runner_id="runner-returning")
    finally:
        event.remove(engine, "before_cursor_execute", record_statement)

    assert lease is not None
    assert lease.run_id
    assert len(statements) == 1
    assert statements[0].lstrip().upper().startswith("UPDATE")
    assert "RETURNING" in statements[0].upper()


def test_acquire_no_commit_leaves_transaction_owned_by_caller(db_session) -> None:
    task = _create_task(db_session)

    lease = acquire_task_lease_no_commit(
        db_session,
        int(task.id),
        runner_id="transaction-owner",
    )

    assert lease is not None
    db_session.rollback()
    db_session.refresh(task)
    assert task.status == TaskStatus.PENDING
    assert task.runner_id is None
    assert task.run_id is None


def test_fail_and_release_task_lease_rejects_superseded_owner(db_session) -> None:
    task = _create_task(db_session)
    stale_lease = acquire_task_lease(
        db_session,
        int(task.id),
        runner_id="old-runner",
    )
    assert stale_lease is not None

    task.runner_id = "new-runner"
    task.run_id = "new-run"
    task.error_message = None
    task.output = "new owner output"
    db_session.commit()

    changed = task_lease_service.fail_and_release_task_lease_no_commit(
        db_session,
        stale_lease,
        error_message="stale runner failed",
    )
    db_session.commit()

    assert changed is False
    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "new-runner"
    assert task.run_id == "new-run"
    assert task.error_message is None
    assert task.output == "new owner output"


def test_fail_and_release_task_lease_atomically_fails_current_owner(db_session) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert lease is not None
    task.output = "stale output"
    db_session.commit()
    state_version = int(task.state_version)

    changed = task_lease_service.fail_and_release_task_lease_no_commit(
        db_session,
        lease,
        error_message="setup failed",
    )
    db_session.commit()

    assert changed is True
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.control_state == "failed"
    assert task.state_version == state_version + 1
    assert task.error_message == "setup failed"
    assert task.output is None
    assert task.runner_id is None
    assert task.lease_expires_at is None


def test_release_task_lease_refuses_ownerless_running_state(db_session) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert lease is not None

    with pytest.raises(ValueError, match="RUNNING"):
        release_task_lease(db_session, lease, status=TaskStatus.RUNNING)

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "runner-a"
    assert task.lease_expires_at is not None


@pytest.mark.asyncio
async def test_lease_heartbeat_keeps_loop_responsive_during_pool_checkout(
    queue_pool_runtime_db,
    monkeypatch,
) -> None:
    engine, SessionLocal = queue_pool_runtime_db
    with SessionLocal() as seed_db:
        task = _create_task(seed_db, status=TaskStatus.RUNNING)
        task_id = int(task.id)
        task.runner_id = "runner-a"
        task.run_id = "run-a"
        seed_db.commit()

    def constrained_get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(
        task_lease_service,
        "get_db",
        constrained_get_db,
        raising=False,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    held_connection = engine.connect()
    stop_event = asyncio.Event()
    ticker_stop = asyncio.Event()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while not ticker_stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    heartbeat_task = asyncio.create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=task_id, runner_id="runner-a", run_id="run-a"),
            stop_event,
        )
    )
    ticker_task = asyncio.create_task(ticker())
    try:
        await asyncio.sleep(0.12)
        assert ticks >= 3, "QueuePool checkout blocked the asyncio event loop"
    finally:
        held_connection.close()
        stop_event.set()
        await asyncio.wait_for(heartbeat_task, timeout=1)
        ticker_stop.set()
        await ticker_task

    with SessionLocal() as verify_db:
        refreshed = verify_db.query(Task).filter(Task.id == task_id).one()
        assert refreshed.last_heartbeat_at is not None


@pytest.mark.asyncio
async def test_stop_heartbeat_drains_inflight_refresh(monkeypatch) -> None:
    refresh_started = threading.Event()
    allow_refresh_to_finish = threading.Event()

    def blocking_refresh(_lease: TaskLease) -> TaskLeaseRefreshState:
        refresh_started.set()
        assert allow_refresh_to_finish.wait(timeout=2)
        return TaskLeaseRefreshState.REFRESHED

    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_lease_isolated",
        blocking_refresh,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=1, runner_id="runner-a", run_id="run-a"),
            stop_event,
        )
    )
    await asyncio.wait_for(asyncio.to_thread(refresh_started.wait, 1), timeout=1)

    stopper = asyncio.create_task(stop_task_lease_heartbeat(heartbeat_task, stop_event))
    await asyncio.sleep(0.02)
    assert not stopper.done()

    allow_refresh_to_finish.set()
    await asyncio.wait_for(stopper, timeout=1)
    assert heartbeat_task.done()


@pytest.mark.asyncio
async def test_stop_heartbeat_propagates_cancellation_after_drain(monkeypatch) -> None:
    refresh_started = threading.Event()
    allow_refresh_to_finish = threading.Event()

    def blocking_refresh(_lease: TaskLease) -> TaskLeaseRefreshState:
        refresh_started.set()
        assert allow_refresh_to_finish.wait(timeout=2)
        return TaskLeaseRefreshState.REFRESHED

    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_lease_isolated",
        blocking_refresh,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=1, runner_id="runner-a", run_id="run-a"),
            stop_event,
        )
    )
    await asyncio.wait_for(asyncio.to_thread(refresh_started.wait, 1), timeout=1)

    stopper = asyncio.create_task(stop_task_lease_heartbeat(heartbeat_task, stop_event))
    await asyncio.sleep(0)
    stopper.cancel()
    await asyncio.sleep(0.02)
    assert not stopper.done()

    allow_refresh_to_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stopper, timeout=1)
    assert heartbeat_task.done()


@pytest.mark.asyncio
async def test_stop_heartbeat_reports_unresolved_pool_timeout(monkeypatch) -> None:
    refresh_attempted = threading.Event()

    def timed_out_refresh(_lease: TaskLease) -> TaskLeaseRefreshState:
        refresh_attempted.set()
        raise SQLAlchemyTimeoutError("pool checkout timed out")

    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_lease_isolated",
        timed_out_refresh,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=1, runner_id="runner-a", run_id="run-a"),
            stop_event,
        )
    )
    await asyncio.wait_for(asyncio.to_thread(refresh_attempted.wait, 1), timeout=1)

    outcome = await stop_task_lease_heartbeat(heartbeat_task, stop_event)

    assert outcome.pool_timeout is not None
    assert is_database_pool_timeout(outcome.pool_timeout)
    assert outcome.lease_lost is False


@pytest.mark.asyncio
async def test_stop_heartbeat_reports_lost_ownership(monkeypatch) -> None:
    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_lease_isolated",
        lambda _lease: TaskLeaseRefreshState.LOST,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=1, runner_id="runner-a", run_id="run-a"),
            stop_event,
        )
    )
    await asyncio.wait_for(heartbeat_task, timeout=1)

    outcome = await stop_task_lease_heartbeat(heartbeat_task, stop_event)

    assert outcome.lease_lost is True
    assert outcome.pool_timeout is None


@pytest.mark.asyncio
async def test_heartbeat_does_not_report_owned_terminal_task_as_lease_lost(
    db_session,
    monkeypatch,
) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert lease is not None

    task.status = TaskStatus.COMPLETED
    task.control_state = "completed"
    db_session.commit()

    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    outcome = await asyncio.wait_for(
        run_task_lease_heartbeat(lease, asyncio.Event()),
        timeout=1,
    )

    assert outcome.lease_lost is False
    assert outcome.requires_ttl_recovery is False


@pytest.mark.asyncio
async def test_successful_heartbeat_clears_prior_pool_timeout(monkeypatch) -> None:
    refresh_recovered = threading.Event()
    attempts = 0

    def recovering_refresh(_lease: TaskLease) -> TaskLeaseRefreshState:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SQLAlchemyTimeoutError("transient pool checkout timeout")
        refresh_recovered.set()
        return TaskLeaseRefreshState.REFRESHED

    monkeypatch.setattr(
        task_lease_service,
        "refresh_task_lease_isolated",
        recovering_refresh,
    )
    monkeypatch.setattr(
        task_lease_service,
        "get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )

    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        run_task_lease_heartbeat(
            TaskLease(task_id=1, runner_id="runner-a", run_id="run-a"),
            stop_event,
        )
    )
    assert await asyncio.to_thread(refresh_recovered.wait, 1)

    outcome = await stop_task_lease_heartbeat(heartbeat_task, stop_event)

    assert outcome.requires_ttl_recovery is False


@pytest.mark.asyncio
async def test_cancellation_safe_acquire_drains_and_cleans_returned_lease() -> None:
    acquire_started = threading.Event()
    allow_acquire_to_finish = threading.Event()
    cleanup_started = threading.Event()
    allow_cleanup_to_finish = threading.Event()
    expected_lease = TaskLease(task_id=9, runner_id="runner-a", run_id="run-a")
    cleaned_leases: list[TaskLease] = []

    def acquire() -> TaskLease:
        acquire_started.set()
        assert allow_acquire_to_finish.wait(timeout=2)
        return expected_lease

    def cleanup(lease: TaskLease) -> None:
        cleanup_started.set()
        assert allow_cleanup_to_finish.wait(timeout=2)
        cleaned_leases.append(lease)

    operation = asyncio.create_task(
        task_lease_service.acquire_task_lease_cancellation_safe(acquire, cleanup)
    )
    await asyncio.wait_for(asyncio.to_thread(acquire_started.wait, 1), timeout=1)
    operation.cancel()
    await asyncio.sleep(0.02)
    assert not operation.done()

    allow_acquire_to_finish.set()
    await asyncio.wait_for(asyncio.to_thread(cleanup_started.wait, 1), timeout=1)
    assert not operation.done()

    allow_cleanup_to_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=1)
    assert cleaned_leases == [expected_lease]


def test_new_run_lease_claim_rejects_a_second_claim(db_session) -> None:
    task = _create_task(db_session)

    first = acquire_task_lease(
        db_session,
        int(task.id),
        runner_id="runner-a",
        new_run=True,
    )
    second = acquire_task_lease(
        db_session,
        int(task.id),
        runner_id="runner-a",
        new_run=True,
    )

    assert first is not None
    assert first.run_id
    assert second is None
    db_session.refresh(task)
    assert task.run_id == first.run_id


def test_lease_acquire_rejects_a_superseded_run(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.run_id = "current-run"
    task.control_state = "running"
    task.state_version = 3
    db_session.commit()

    assert (
        acquire_task_lease(
            db_session,
            int(task.id),
            runner_id="runner-a",
            expected_run_id="old-run",
        )
        is None
    )
    db_session.refresh(task)
    assert task.run_id == "current-run"
    assert task.state_version == 3


def test_old_lease_cannot_refresh_or_release_a_new_run(db_session) -> None:
    task = _create_task(db_session)
    old_lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert old_lease is not None

    task.run_id = "new-run"
    task.status = TaskStatus.RUNNING
    task.control_state = "running"
    task.runner_id = "runner-a"
    db_session.commit()

    assert refresh_task_lease(db_session, old_lease) == TaskLeaseRefreshState.LOST
    assert release_task_lease(db_session, old_lease, status=TaskStatus.FAILED) is False
    db_session.refresh(task)
    assert task.run_id == "new-run"
    assert task.status == TaskStatus.RUNNING


def test_stale_running_task_with_checkpoint_becomes_paused(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.runner_id = "dead-runner"
    task.lease_expires_at = utc_now() - timedelta(seconds=1)
    db_session.add(
        TraceEvent(
            task_id=task.id,
            event_id="checkpoint-1",
            event_type="system_update_general",
            timestamp=utc_now(),
            step_id=None,
            parent_event_id=None,
            data={
                "checkpoint_type": CHECKPOINT_TYPE,
                "snapshot": {"type": "checkpoint"},
            },
        )
    )
    db_session.commit()

    assert mark_task_paused_if_stale(db_session, task) is True
    db_session.refresh(task)
    assert task.status == TaskStatus.PAUSED
    assert task.runner_id is None
    assert task.lease_expires_at is None


def test_stale_running_task_ignores_child_agent_checkpoint(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.runner_id = "dead-runner"
    task.lease_expires_at = utc_now() - timedelta(seconds=1)
    db_session.add(
        TraceEvent(
            task_id=task.id,
            build_id="agent_123_child",
            event_id="child-checkpoint-1",
            event_type="system_update_general",
            timestamp=utc_now(),
            step_id=None,
            parent_event_id=None,
            data={
                "checkpoint_type": CHECKPOINT_TYPE,
                "snapshot": {"type": "checkpoint"},
            },
        )
    )
    db_session.commit()

    assert mark_task_paused_if_stale(db_session, task) is True
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.runner_id is None
    assert task.lease_expires_at is None


def test_stale_running_task_with_legacy_checkpoint_becomes_paused(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.runner_id = "dead-runner"
    task.lease_expires_at = utc_now() - timedelta(seconds=1)
    db_session.add(
        TraceEvent(
            task_id=task.id,
            event_id="legacy-checkpoint-1",
            event_type="system_update_general",
            timestamp=utc_now(),
            step_id=None,
            parent_event_id=None,
            data={
                "checkpoint_type": next(iter(LEGACY_CHECKPOINT_TYPES)),
                "snapshot": {"type": "checkpoint"},
            },
        )
    )
    db_session.commit()

    assert mark_task_paused_if_stale(db_session, task) is True
    db_session.refresh(task)
    assert task.status == TaskStatus.PAUSED
    assert task.runner_id is None
    assert task.lease_expires_at is None
