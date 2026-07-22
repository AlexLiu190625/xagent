"""Task execution leases for multi-process agent runners."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, cast

from sqlalchemy import case, func, or_, update
from sqlalchemy.orm import Session

from ...config import (
    get_task_lease_heartbeat_seconds,
    get_task_lease_ttl_seconds,
)
from ...core.agent.checkpoint import READABLE_CHECKPOINT_TYPES
from ..models.task import Task, TaskStatus, TraceEvent
from .db_runtime import (
    await_task_settlement,
    is_database_pool_timeout,
    run_db_io_cancellation_safe,
)
from .task_execution_controller import control_state_for_status

logger = logging.getLogger(__name__)

_RUNNER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:12]}"


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


@dataclass(frozen=True)
class TaskLease:
    task_id: int
    runner_id: str
    run_id: str | None = None


@dataclass(frozen=True)
class TaskLeaseHeartbeatOutcome:
    """Lease state observed when a heartbeat loop stops.

    A pool timeout is retained only until a later successful refresh proves
    that ownership is healthy again. Callers use an unresolved timeout (or a
    definite ownership loss) to avoid an immediate settlement checkout.
    """

    lease_lost: bool = False
    pool_timeout: BaseException | None = None

    @property
    def requires_ttl_recovery(self) -> bool:
        return self.lease_lost or self.pool_timeout is not None


class TaskLeaseRefreshState(str, Enum):
    """Result of refreshing one exact task-run lease."""

    REFRESHED = "refreshed"
    SETTLEMENT_READY = "settlement_ready"
    LOST = "lost"


async def acquire_task_lease_cancellation_safe(
    acquire: Callable[[], TaskLease | None],
    cleanup: Callable[[TaskLease], Any],
) -> TaskLease | None:
    """Acquire a lease and clean up a late result before propagating cancel.

    The acquisition callback and cleanup callback each execute in their own
    worker thread. When cancellation arrives during acquisition, the acquire
    worker is drained first; if it committed and returned a lease, cleanup is
    then drained as well. Only after both operations settle is cancellation
    delivered to the caller.
    """
    worker = asyncio.get_running_loop().create_task(asyncio.to_thread(acquire))
    lease, cancellation = await await_task_settlement(worker)
    if cancellation is None:
        return lease

    if lease is not None:
        try:
            await run_db_io_cancellation_safe(lambda: cleanup(lease))
        except asyncio.CancelledError:
            # A repeated caller cancellation was recorded and propagated only
            # after cleanup settled. Preserve the original cancellation below.
            pass
        except Exception:
            logger.exception(
                "Failed to clean up task %s lease after cancelled acquisition",
                lease.task_id,
            )
    raise cancellation


def get_runner_id() -> str:
    """Return the current process runner id."""
    return _RUNNER_ID


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _expires_at(now: datetime | None = None) -> datetime:
    return (now or utc_now()) + timedelta(seconds=get_task_lease_ttl_seconds())


def has_agent_checkpoint(db: Session, task_id: int) -> bool:
    """Return whether the task has a persisted agent checkpoint."""
    rows = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.build_id.is_(None),
            TraceEvent.event_type == "system_update_general",
        )
        .order_by(TraceEvent.id.desc())
        .limit(100)
        .all()
    )
    for row in rows:
        data: dict[str, Any] = (
            cast(dict[str, Any], row.data) if isinstance(row.data, dict) else {}
        )
        if data.get("checkpoint_type") in READABLE_CHECKPOINT_TYPES:
            return True
    return False


def acquire_task_lease(
    db: Session,
    task_id: int,
    *,
    runner_id: str | None = None,
    expected_run_id: str | None = None,
    new_run: bool = False,
) -> TaskLease | None:
    """Acquire the task execution lease if no live runner owns it.

    ``new_run=True`` atomically requires a non-running task and assigns a new
    run id in the same UPDATE. This is the durable claim used by channel
    transports so another worker cannot rotate the run between a status check
    and lease acquisition.
    """
    lease = acquire_task_lease_no_commit(
        db,
        task_id,
        runner_id=runner_id,
        expected_run_id=expected_run_id,
        new_run=new_run,
    )
    db.commit()
    if lease is None:
        logger.info(
            "Task %s lease acquisition denied for runner %s",
            task_id,
            runner_id or get_runner_id(),
        )
        return None
    logger.info(
        "Task %s lease acquired by runner %s",
        task_id,
        lease.runner_id,
    )
    return lease


def acquire_task_lease_no_commit(
    db: Session,
    task_id: int,
    *,
    runner_id: str | None = None,
    expected_run_id: str | None = None,
    new_run: bool = False,
) -> TaskLease | None:
    """Stage one atomic lease claim; the caller owns commit/rollback."""
    runner = runner_id or get_runner_id()
    now = utc_now()
    expires_at = _expires_at(now)
    candidate_run_id = expected_run_id or str(uuid.uuid4())
    current_version = func.coalesce(Task.state_version, 0)
    running_control_state = control_state_for_status(TaskStatus.RUNNING).value
    stmt = (
        update(Task)
        .where(Task.id == task_id)
        .where(
            or_(
                Task.status != TaskStatus.RUNNING,
                Task.runner_id == runner,
                Task.runner_id.is_(None),
                Task.lease_expires_at.is_(None),
                Task.lease_expires_at < now,
            )
        )
        .values(
            status=TaskStatus.RUNNING,
            runner_id=runner,
            last_heartbeat_at=now,
            lease_expires_at=expires_at,
            run_id=(
                candidate_run_id
                if new_run
                else case(
                    (Task.status != TaskStatus.RUNNING, candidate_run_id),
                    else_=func.coalesce(Task.run_id, candidate_run_id),
                )
            ),
            control_state=running_control_state,
            state_version=case(
                (
                    or_(
                        Task.status != TaskStatus.RUNNING,
                        Task.control_state != running_control_state,
                    ),
                    current_version + 1,
                ),
                else_=current_version,
            ),
        )
    )
    if new_run:
        stmt = stmt.where(Task.status != TaskStatus.RUNNING)
    if expected_run_id is not None:
        stmt = stmt.where(Task.run_id == expected_run_id)
    result = db.execute(
        stmt.returning(Task.run_id).execution_options(synchronize_session=False)
    )
    stored_run_id = result.scalar_one_or_none()
    if stored_run_id is None:
        return None
    return TaskLease(
        task_id=task_id,
        runner_id=runner,
        run_id=str(stored_run_id) if stored_run_id is not None else None,
    )


def acquire_task_lease_isolated(
    task_id: int,
    *,
    runner_id: str | None = None,
    expected_run_id: str | None = None,
    new_run: bool = False,
) -> TaskLease | None:
    """Same semantics as :func:`acquire_task_lease` but opens, commits,
    and closes its own ``SessionLocal``.

    Safe to call from ``asyncio.to_thread`` -- the inline call in
    ``_runner`` measured 3.75s of synchronous DB write on the main
    event loop (issue #427). Wrapping the existing helper preserves
    every transactional detail (the conditional UPDATE + rowcount
    guard) while letting the loop continue.
    """
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        return acquire_task_lease(
            db,
            task_id,
            runner_id=runner_id,
            expected_run_id=expected_run_id,
            new_run=new_run,
        )
    finally:
        db.close()


def refresh_task_lease(db: Session, lease: TaskLease) -> TaskLeaseRefreshState:
    """Refresh one exact lease or classify why it no longer needs refresh.

    A task finalizer commits its terminal status before post-result broadcasts
    complete, while the scheduler still owns and later releases the lease.  A
    heartbeat in that window must distinguish the same terminal run from a
    lease that another runner or run actually replaced.
    """
    now = utc_now()
    expires_at = _expires_at(now)
    stmt = (
        update(Task)
        .where(Task.id == lease.task_id)
        .where(Task.runner_id == lease.runner_id)
        .where(Task.status == TaskStatus.RUNNING)
        .values(last_heartbeat_at=now, lease_expires_at=expires_at)
    )
    if lease.run_id is not None:
        stmt = stmt.where(Task.run_id == lease.run_id)
    result = db.execute(stmt.execution_options(synchronize_session=False))
    if _rowcount(result) == 1:
        db.commit()
        return TaskLeaseRefreshState.REFRESHED

    owned_query = db.query(Task.status).filter(
        Task.id == lease.task_id,
        Task.runner_id == lease.runner_id,
    )
    if lease.run_id is not None:
        owned_query = owned_query.filter(Task.run_id == lease.run_id)
    owned_status = owned_query.scalar()
    db.commit()
    if owned_status is not None and owned_status != TaskStatus.RUNNING:
        return TaskLeaseRefreshState.SETTLEMENT_READY
    return TaskLeaseRefreshState.LOST


def refresh_task_lease_isolated(lease: TaskLease) -> TaskLeaseRefreshState:
    """Refresh ``lease`` in a short-lived session owned by this thread."""
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        return refresh_task_lease(db, lease)


def release_task_lease(
    db: Session,
    lease: TaskLease | None,
    *,
    status: TaskStatus,
) -> bool:
    """Release a task lease and set its final visible status."""
    released = release_task_lease_no_commit(db, lease, status=status)
    if lease is None:
        return False
    db.commit()
    return released


def release_task_lease_no_commit(
    db: Session,
    lease: TaskLease | None,
    *,
    status: TaskStatus,
) -> bool:
    """Stage release of one exact lease; the caller owns commit/rollback."""
    if status == TaskStatus.RUNNING:
        raise ValueError("Cannot release a task lease with RUNNING status")
    if lease is None:
        return False
    control_state = control_state_for_status(status).value
    current_version = func.coalesce(Task.state_version, 0)
    stmt = (
        update(Task)
        .where(Task.id == lease.task_id)
        .where(Task.runner_id == lease.runner_id)
        .values(
            status=status,
            runner_id=None,
            lease_expires_at=None,
            last_heartbeat_at=utc_now(),
            control_state=control_state,
            state_version=case(
                (
                    or_(Task.status != status, Task.control_state != control_state),
                    current_version + 1,
                ),
                else_=current_version,
            ),
        )
    )
    if lease.run_id is not None:
        stmt = stmt.where(Task.run_id == lease.run_id)
    result = db.execute(stmt.execution_options(synchronize_session=False))
    return _rowcount(result) == 1


def fail_and_release_task_lease_no_commit(
    db: Session,
    lease: TaskLease,
    *,
    error_message: str,
) -> bool:
    """Atomically fail and release the exact live lease without committing.

    The caller owns the transaction so related lifecycle projections can be
    synchronized before one final commit. A lease without a run id is not a
    sufficient ownership fence and is therefore never allowed to mutate the
    task row.
    """
    if lease.run_id is None:
        return False

    failed_control_state = control_state_for_status(TaskStatus.FAILED).value
    stmt = (
        update(Task)
        .where(Task.id == lease.task_id)
        .where(Task.runner_id == lease.runner_id)
        .where(Task.run_id == lease.run_id)
        .where(Task.status == TaskStatus.RUNNING)
        .values(
            status=TaskStatus.FAILED,
            runner_id=None,
            lease_expires_at=None,
            last_heartbeat_at=utc_now(),
            control_state=failed_control_state,
            state_version=func.coalesce(Task.state_version, 0) + 1,
            error_message=error_message,
            output=None,
        )
    )
    result = db.execute(stmt.execution_options(synchronize_session=False))
    return _rowcount(result) == 1


def release_current_runner_task_lease(
    db: Session,
    task_id: int,
    *,
    status: TaskStatus,
    runner_id: str | None = None,
    expected_run_id: str | None = None,
) -> bool:
    """Release the current runner's lease for a task."""
    if status == TaskStatus.RUNNING:
        raise ValueError("Cannot release a task lease with RUNNING status")
    runner = runner_id or get_runner_id()
    control_state = control_state_for_status(status).value
    current_version = func.coalesce(Task.state_version, 0)
    stmt = (
        update(Task)
        .where(Task.id == task_id)
        .where(Task.runner_id == runner)
        .values(
            status=status,
            runner_id=None,
            lease_expires_at=None,
            last_heartbeat_at=utc_now(),
            control_state=control_state,
            state_version=case(
                (
                    or_(Task.status != status, Task.control_state != control_state),
                    current_version + 1,
                ),
                else_=current_version,
            ),
        )
    )
    if expected_run_id is not None:
        stmt = stmt.where(Task.run_id == expected_run_id)
    result = db.execute(stmt.execution_options(synchronize_session=False))
    db.commit()
    return _rowcount(result) == 1


def mark_task_paused_if_stale(db: Session, task: Task) -> bool:
    """Convert a stale RUNNING task into a recoverable terminal state."""
    if task.status != TaskStatus.RUNNING:
        return False

    now = utc_now()
    lease_expires_at = task.lease_expires_at
    if lease_expires_at is not None and lease_expires_at.tzinfo is None:
        lease_expires_at = lease_expires_at.replace(tzinfo=timezone.utc)

    if lease_expires_at is not None and lease_expires_at >= now:
        return False

    from .task_execution_controller import (
        TaskControlState,
        apply_task_control_transition,
    )

    next_status = (
        TaskStatus.PAUSED
        if has_agent_checkpoint(db, int(task.id))
        else TaskStatus.FAILED
    )
    apply_task_control_transition(
        task,
        (
            TaskControlState.PAUSED
            if next_status == TaskStatus.PAUSED
            else TaskControlState.FAILED
        ),
        status=next_status,
    )
    setattr(task, "runner_id", None)
    setattr(task, "lease_expires_at", None)
    setattr(task, "last_heartbeat_at", now)
    db.commit()
    logger.info("Marked stale task %s as %s", task.id, task.status.value)
    return True


async def run_task_lease_heartbeat(
    lease: TaskLease,
    stop_event: asyncio.Event,
) -> TaskLeaseHeartbeatOutcome:
    """Keep a task lease alive until the execution finishes."""
    interval = get_task_lease_heartbeat_seconds()
    unresolved_pool_timeout: BaseException | None = None
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass

        try:
            refresh_state = await run_db_io_cancellation_safe(
                lambda: refresh_task_lease_isolated(lease)
            )
            if refresh_state == TaskLeaseRefreshState.SETTLEMENT_READY:
                logger.debug(
                    "Task %s lease heartbeat observed owned terminal state; "
                    "handing lease to settlement",
                    lease.task_id,
                )
                return TaskLeaseHeartbeatOutcome()
            if refresh_state == TaskLeaseRefreshState.LOST:
                logger.warning(
                    "Task %s lease heartbeat lost for runner %s",
                    lease.task_id,
                    lease.runner_id,
                )
                return TaskLeaseHeartbeatOutcome(lease_lost=True)
            unresolved_pool_timeout = None
        except Exception as e:
            if is_database_pool_timeout(e):
                unresolved_pool_timeout = e
            logger.warning(
                "Task %s lease heartbeat failed for runner %s: %s",
                lease.task_id,
                lease.runner_id,
                e,
            )
    return TaskLeaseHeartbeatOutcome(pool_timeout=unresolved_pool_timeout)


async def stop_task_lease_heartbeat(
    task: asyncio.Task[Any] | None,
    stop_event: asyncio.Event | None,
) -> TaskLeaseHeartbeatOutcome:
    if stop_event is not None:
        stop_event.set()
    if task is None:
        return TaskLeaseHeartbeatOutcome()

    try:
        outcome, cancellation = await await_task_settlement(task)
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling() > 0:
            raise
        # The heartbeat task itself had already been cancelled. Its worker I/O
        # is cancellation-safe, so there is nothing left to drain here.
        return TaskLeaseHeartbeatOutcome()
    if cancellation is not None:
        raise cancellation
    if isinstance(outcome, TaskLeaseHeartbeatOutcome):
        return outcome
    # Compatibility with externally supplied/mocked heartbeat tasks that
    # predate the structured result.
    return TaskLeaseHeartbeatOutcome()
