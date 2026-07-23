from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.managed_task_lease import (
    ManagedTaskLease,
    claim_managed_task_lease,
    start_managed_task_lease,
)
from xagent.web.services.task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
    acquire_task_lease,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'managed-lease.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _create_task(db) -> Task:
    user = User(username="managed-lease-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    task = Task(
        user_id=user.id,
        title="Managed lease",
        description="Managed lease",
        status=TaskStatus.PENDING,
        execution_mode="auto",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@pytest.mark.asyncio
async def test_managed_lease_releases_terminal_task(db_session) -> None:
    task = _create_task(db_session)
    managed = claim_managed_task_lease(db_session, int(task.id))
    assert managed is not None
    assert claim_managed_task_lease(db_session, int(task.id)) is None
    task.status = TaskStatus.COMPLETED
    task.control_state = "completed"
    db_session.commit()

    assert await managed.close() is True
    assert await managed.close() is False
    db_session.refresh(task)
    assert task.status == TaskStatus.COMPLETED
    assert task.runner_id is None
    assert task.lease_expires_at is None


@pytest.mark.asyncio
async def test_managed_lease_fails_an_unfinished_task(db_session) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), new_run=True)
    assert lease is not None
    managed = start_managed_task_lease(lease)

    assert await managed.close() is True
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.control_state == "failed"
    assert task.runner_id is None


@pytest.mark.asyncio
async def test_managed_lease_heartbeat_timeout_skips_release_checkout() -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=7, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    timeout = SQLAlchemyTimeoutError("heartbeat pool timeout")

    with (
        patch(
            "xagent.web.services.managed_task_lease.stop_task_lease_heartbeat",
            new=AsyncMock(return_value=TaskLeaseHeartbeatOutcome(pool_timeout=timeout)),
        ),
        patch(
            "xagent.web.services.managed_task_lease._release_managed_task_lease_sync"
        ) as release,
    ):
        assert await managed.close() is False

    release.assert_not_called()


@pytest.mark.asyncio
async def test_managed_lease_close_drains_release_before_cancellation() -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=8, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    release_started = threading.Event()
    allow_release = threading.Event()

    def blocking_release(_lease: TaskLease) -> bool:
        release_started.set()
        assert allow_release.wait(timeout=2)
        return True

    with (
        patch(
            "xagent.web.services.managed_task_lease.stop_task_lease_heartbeat",
            new=AsyncMock(return_value=TaskLeaseHeartbeatOutcome()),
        ),
        patch(
            "xagent.web.services.managed_task_lease._release_managed_task_lease_sync",
            side_effect=blocking_release,
        ),
    ):
        closing = asyncio.create_task(managed.close())
        assert await asyncio.to_thread(release_started.wait, 1)
        closing.cancel()
        await asyncio.sleep(0.02)
        assert not closing.done()
        allow_release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing


@pytest.mark.asyncio
async def test_managed_lease_close_drains_heartbeat_before_cancellation() -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=9, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    heartbeat_stop_started = asyncio.Event()
    allow_heartbeat_stop = asyncio.Event()

    async def blocking_heartbeat_stop(*_args) -> TaskLeaseHeartbeatOutcome:
        heartbeat_stop_started.set()
        await allow_heartbeat_stop.wait()
        return TaskLeaseHeartbeatOutcome()

    with (
        patch(
            "xagent.web.services.managed_task_lease.stop_task_lease_heartbeat",
            side_effect=blocking_heartbeat_stop,
        ),
        patch(
            "xagent.web.services.managed_task_lease._release_managed_task_lease_sync",
            return_value=True,
        ) as release,
    ):
        closing = asyncio.create_task(managed.close())
        await heartbeat_stop_started.wait()
        closing.cancel()
        await asyncio.sleep(0.02)
        assert not closing.done()
        allow_heartbeat_stop.set()
        with pytest.raises(asyncio.CancelledError):
            await closing

    release.assert_called_once_with(managed.lease)
