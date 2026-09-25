"""Durable capacity waiting, separate from command failure/retry accounting.

Hosts classify commands at acceptance; workers enforce the persisted bucket.
Disjoint buckets can reserve capacity for different traffic classes. A ticket
holds capacity through the task owner's actual execution cleanup, including
after a terminal business status has already been published. Expiry alone is
never permission to reclaim capacity: the existing owner recovery must fence
out that acquisition first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from sqlalchemy import and_, delete, exists, func, or_, select, update
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import ScalarSelect

from ...config import get_shared_task_execution_enabled
from ..models.task import Task, TaskStatus, task_status_predicate
from ..models.task_admission import TaskAdmissionBucket, TaskAdmissionTicket
from ..models.task_command import TaskExecutionCommand
from .task_coordinator_service import TaskLease

if TYPE_CHECKING:
    from .task_command_transport import ClaimedTaskCommand, SettledTaskCommand

_EXECUTION_KINDS = frozenset({"start", "resume", "resume_input", "message"})
_TERMINAL = ("completed", "failed")
_STOPPED_BEFORE_START = "Task stopped before execution started."


@dataclass(frozen=True)
class AdmissionPolicy:
    """One server-owned budget; all hosts must use the same configuration."""

    bucket: str
    capacity: int
    max_pending: int

    def __post_init__(self) -> None:
        if not self.bucket or len(self.bucket) > 255:
            raise ValueError("Admission bucket must contain 1-255 characters")
        if self.capacity < 1 or self.max_pending < 1:
            raise ValueError("Admission capacity and pending budget must be positive")


class AdmissionQueueFull(RuntimeError):
    """The host must translate this retryable refusal at its ingress boundary."""


AdmissionHook = Callable[[Session, TaskExecutionCommand], AdmissionPolicy | None]
_hook: AdmissionHook | None = None


def set_task_admission_hook(hook: AdmissionHook | None) -> None:
    """Install the trusted host classifier in both ingress and execution hosts."""
    global _hook
    _hook = hook


def stage_task_admission(db: Session, command: TaskExecutionCommand) -> None:
    """Stage a new command's ticket in its acceptance transaction, exactly once."""
    if _hook is None or command.kind not in _EXECUTION_KINDS:
        return
    policy = _hook(db, command)
    if policy is None:
        return
    if not get_shared_task_execution_enabled():
        raise RuntimeError("Execution admission requires shared task execution")
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert  # type: ignore[assignment]
    else:
        raise RuntimeError("Execution admission requires PostgreSQL or SQLite")
    db.execute(
        insert(TaskAdmissionBucket)
        .values(
            key=policy.bucket, capacity=policy.capacity, max_pending=policy.max_pending
        )
        .on_conflict_do_nothing(index_elements=["key"])
    )
    bucket = _lock_bucket(db, policy.bucket)
    if (bucket.capacity, bucket.max_pending) != (policy.capacity, policy.max_pending):
        raise ValueError("Drain the admission bucket before changing its policy")
    if _pending_count(db, policy.bucket) >= policy.max_pending:
        raise AdmissionQueueFull("Execution queue is full")
    db.add(
        TaskAdmissionTicket(
            command_id=command.id, task_id=command.task_id, bucket_key=policy.bucket
        )
    )
    db.flush()


def _pending_count(db: Session, key: str) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(TaskAdmissionTicket)
            .join(
                TaskExecutionCommand,
                TaskExecutionCommand.id == TaskAdmissionTicket.command_id,
            )
            .where(
                TaskAdmissionTicket.bucket_key == key,
                ~_held_ticket(TaskAdmissionTicket),
                TaskExecutionCommand.status.notin_(_TERMINAL),
            )
        )
        or 0
    )


def prepare_task_admission_retry(db: Session, command_id: int) -> None:
    """Recheck pending capacity while preserving the original classification."""
    ticket = db.get(TaskAdmissionTicket, command_id)
    if ticket is None:
        return
    # Keep the task-before-bucket lock order used by acceptance and dispatch.
    db.execute(select(Task.id).where(Task.id == ticket.task_id).with_for_update())
    bucket = _lock_bucket(db, str(ticket.bucket_key))
    command = db.get(TaskExecutionCommand, command_id, populate_existing=True)
    if command is None or command.status != "failed":
        return
    if _pending_count(db, str(bucket.key)) >= int(bucket.max_pending):
        raise AdmissionQueueFull("Execution queue is full")


def _lock_bucket(db: Session, key: str) -> TaskAdmissionBucket:
    # A write also serializes SQLite transactions; SELECT FOR UPDATE alone does not.
    db.execute(
        update(TaskAdmissionBucket)
        .where(TaskAdmissionBucket.key == key)
        .values(capacity=TaskAdmissionBucket.capacity)
    )
    return db.execute(
        select(TaskAdmissionBucket)
        .where(TaskAdmissionBucket.key == key)
        .execution_options(populate_existing=True)
    ).scalar_one()


def _live_count(bucket_key: object) -> ScalarSelect[int]:
    ticket, task = aliased(TaskAdmissionTicket), aliased(Task)
    return (
        select(func.count())
        .select_from(ticket)
        .join(task, task.id == ticket.task_id)
        .where(
            ticket.bucket_key == bucket_key,
            ticket.owner_attempt_id == task.lease_attempt_id,
            ticket.runner_id == task.runner_id,
        )
        .correlate(TaskAdmissionTicket)
        .scalar_subquery()
    )


def _held_ticket(ticket: type[TaskAdmissionTicket]) -> ColumnElement[bool]:
    task = aliased(Task)
    return exists(
        select(1)
        .where(
            task.id == ticket.task_id,
            task.lease_attempt_id == ticket.owner_attempt_id,
            task.runner_id == ticket.runner_id,
        )
        .correlate(ticket)
    )


def admission_eligible() -> ColumnElement[bool]:
    """Advisory queue scan; the transaction below rechecks after locking."""
    blocked = exists(
        select(1)
        .select_from(TaskAdmissionTicket)
        .join(
            TaskAdmissionBucket,
            TaskAdmissionBucket.key == TaskAdmissionTicket.bucket_key,
        )
        .where(
            TaskAdmissionTicket.command_id == TaskExecutionCommand.id,
            # A resumed claim under the same owner already holds its slot.
            or_(
                TaskAdmissionTicket.owner_attempt_id.is_(None),
                Task.lease_attempt_id.is_(None),
                TaskAdmissionTicket.owner_attempt_id != Task.lease_attempt_id,
                TaskAdmissionTicket.runner_id != Task.runner_id,
            ),
            or_(
                _live_count(TaskAdmissionTicket.bucket_key)
                >= TaskAdmissionBucket.capacity,
                _older_waiter(),
            ),
        )
        .correlate(TaskExecutionCommand, Task)
    )
    return or_(~blocked, _joins_running_execution())


def _joins_running_execution() -> ColumnElement[bool]:
    ticket, incoming = aliased(TaskAdmissionTicket), aliased(TaskAdmissionTicket)
    return and_(
        TaskExecutionCommand.kind == "message",
        task_status_predicate.eq(TaskStatus.RUNNING),
        Task.control_state == "running",
        exists(
            select(1)
            .select_from(ticket)
            .join(incoming, incoming.bucket_key == ticket.bucket_key)
            .where(
                incoming.command_id == TaskExecutionCommand.id,
                ticket.task_id == Task.id,
                ticket.runner_id == Task.runner_id,
                ticket.owner_attempt_id == Task.lease_attempt_id,
            )
            .correlate(Task, TaskExecutionCommand)
        ),
    )


def _older_waiter() -> ColumnElement[bool]:
    ticket, command = aliased(TaskAdmissionTicket), aliased(TaskExecutionCommand)
    return exists(
        select(1)
        .select_from(ticket)
        .join(command, command.id == ticket.command_id)
        .where(
            ticket.bucket_key == TaskAdmissionTicket.bucket_key,
            ticket.command_id < TaskAdmissionTicket.command_id,
            ~_held_ticket(ticket),
            command.status.notin_(_TERMINAL),
        )
        .correlate(TaskAdmissionTicket)
    )


def reserve_task_admission(db: Session, command_id: int, lease: TaskLease) -> bool:
    """Reserve with the command claim; caller already holds the exact task owner."""
    ticket = db.get(TaskAdmissionTicket, command_id)
    if ticket is None:
        return True
    bucket = _lock_bucket(db, str(ticket.bucket_key))
    if ticket.task_id != lease.task_id:
        raise ValueError("Admission ticket belongs to another task")
    if db.scalar(
        select(_joins_running_execution())
        .select_from(TaskExecutionCommand)
        .join(Task, Task.id == TaskExecutionCommand.task_id)
        .where(TaskExecutionCommand.id == command_id)
    ):
        return True
    if (ticket.runner_id, ticket.owner_attempt_id) == (
        lease.runner_id,
        lease.attempt_id,
    ):
        return True
    if db.scalar(
        select(_older_waiter()).where(TaskAdmissionTicket.command_id == command_id)
    ):
        return False
    active = db.scalar(select(_live_count(ticket.bucket_key)))
    if active is not None and active >= bucket.capacity:
        return False
    setattr(ticket, "runner_id", lease.runner_id)
    setattr(ticket, "owner_attempt_id", lease.attempt_id)
    db.flush()
    return True


def release_task_admissions(db: Session, lease: TaskLease) -> None:
    """Release only after the owner drained execution, never merely on status flip."""
    owned = (
        TaskAdmissionTicket.task_id == lease.task_id,
        TaskAdmissionTicket.runner_id == lease.runner_id,
        TaskAdmissionTicket.owner_attempt_id == lease.attempt_id,
    )
    db.execute(
        delete(TaskAdmissionTicket).where(
            *owned,
            TaskAdmissionTicket.command_id.in_(
                select(TaskExecutionCommand.id).where(
                    TaskExecutionCommand.status == "completed"
                )
            ),
        )
    )
    db.execute(
        update(TaskAdmissionTicket)
        .where(*owned)
        .values(runner_id=None, owner_attempt_id=None)
    )


def waiting_admission(command_id: object) -> ColumnElement[bool]:
    """Only unclaimed capacity-waiting commands may be overtaken by controls."""
    return exists(
        select(1).where(
            TaskAdmissionTicket.command_id == command_id,
            TaskAdmissionTicket.owner_attempt_id.is_(None),
        )
    )


def settle_queued_start_for_pause(
    command: ClaimedTaskCommand,
) -> SettledTaskCommand | None:
    """Atomically stop the exact unreserved START targeted by a claimed PAUSE.

    The transaction proves the START, ticket, PAUSE claim, and live owner.
    """
    from ..models.database import get_session_local
    from ..models.user import User
    from .task_command_terminal_events import (
        TerminalTaskEventDraft,
        stage_terminal_event,
    )
    from .task_command_transport import (
        SettledTaskCommand,
        command_identity_matches_task,
        command_processing_predicates,
        finish_task_command_no_commit,
    )
    from .task_coordinator_runtime import current_task_coordinator
    from .task_start_consumer import settle_failed_start_no_commit
    from .task_start_protocol import TaskStartPayload

    if command.kind.value != "pause" or command.target_run_id is None:
        return None
    coordinator = current_task_coordinator(command.task_id)
    lease = coordinator.lease if coordinator is not None else None
    if lease is None or lease.task_id != command.task_id:
        return None

    with get_session_local()() as db, db.begin():
        ownership = command_processing_predicates(
            db,
            command.id,
            lease.runner_id,
            expected_attempt_count=command.attempt_count,
            require_live_claim=True,
            owner_lease=lease,
        )
        pause = (
            db.query(TaskExecutionCommand)
            .filter(
                *ownership,
                TaskExecutionCommand.task_id == command.task_id,
                TaskExecutionCommand.actor_user_id == command.actor_user_id,
                TaskExecutionCommand.command_id == command.command_id,
                TaskExecutionCommand.kind == "pause",
                TaskExecutionCommand.target_run_id == command.target_run_id,
            )
            .with_for_update()
            .one_or_none()
        )
        task = db.get(Task, command.task_id, populate_existing=True)
        if task is None:
            return None
        if pause is None or not command_identity_matches_task(db, task, pause):
            return None
        actor = db.get(User, pause.actor_user_id)
        if actor is None or not (actor.id == task.user_id or actor.is_admin):
            return None
        if (
            pause.target_state_version != task.state_version
            or task.run_id == command.target_run_id
        ):
            return None

        candidates = (
            db.query(TaskExecutionCommand, TaskAdmissionTicket)
            .join(
                TaskAdmissionTicket,
                TaskAdmissionTicket.command_id == TaskExecutionCommand.id,
            )
            .filter(
                TaskExecutionCommand.task_id == task.id,
                TaskExecutionCommand.id < pause.id,
                TaskExecutionCommand.kind == "start",
                TaskExecutionCommand.status == "pending",
                TaskExecutionCommand.target_run_id == command.target_run_id,
                TaskExecutionCommand.target_state_version == task.state_version,
                TaskAdmissionTicket.task_id == task.id,
                TaskAdmissionTicket.runner_id.is_(None),
                TaskAdmissionTicket.owner_attempt_id.is_(None),
            )
            .order_by(TaskExecutionCommand.id.desc())
            .limit(2)
            .with_for_update()
            .all()
        )
        if len(candidates) != 1:
            return None
        start_row, _ticket = candidates[0]
        if not command_identity_matches_task(db, task, start_row):
            return None
        try:
            start = TaskStartPayload.model_validate(start_row.payload)
        except ValueError:
            return None
        if (
            start.run_id != command.target_run_id
            or start.run_id != start_row.target_run_id
            or start.turn_id != start_row.command_id
            or start.expected_run_id != task.run_id
            or start.state_version != task.state_version
        ):
            return None

        cancelled = db.scalar(
            update(TaskExecutionCommand)
            .where(
                TaskExecutionCommand.id == start_row.id,
                TaskExecutionCommand.status == "pending",
                waiting_admission(TaskExecutionCommand.id),
            )
            .values(
                status="failed",
                error=_STOPPED_BEFORE_START,
                completed_at=func.now(),
                result={"rejection_reason": "cancelled_before_admission"},
            )
            .returning(TaskExecutionCommand.id)
        )
        if cancelled is None:
            return None
        db.refresh(start_row)
        settle_failed_start_no_commit(db, start_row)
        stage_terminal_event(
            db,
            command_db_id=int(start_row.id),
            draft=TerminalTaskEventDraft(message_code=None, resend_safe=False),
        )
        result = {
            "task_id": command.task_id,
            "command_id": command.command_id,
            "kind": command.kind.value,
        }
        if not finish_task_command_no_commit(
            db,
            command.id,
            lease.runner_id,
            result=result,
            expected_attempt_count=command.attempt_count,
            require_live_claim=True,
            owner_lease=lease,
        ):
            raise RuntimeError("PAUSE claim changed while settling queued START")
        return SettledTaskCommand(result)


def settle_cancelled_admissions(db: Session, command_id: int) -> None:
    """Settle unreserved completions and commands invalidated by a control."""
    from .task_command_terminal_events import (
        TerminalTaskEventDraft,
        stage_terminal_event,
    )
    from .task_start_consumer import settle_failed_start_no_commit

    db.execute(
        delete(TaskAdmissionTicket).where(
            TaskAdmissionTicket.command_id == command_id,
            TaskAdmissionTicket.owner_attempt_id.is_(None),
        )
    )
    control = db.get(TaskExecutionCommand, command_id)
    if control is None or control.kind not in {"cancel", "pause"}:
        return
    task = db.get(Task, control.task_id, populate_existing=True)
    if task is None:
        return
    cancelled = db.scalars(
        update(TaskExecutionCommand)
        .where(
            TaskExecutionCommand.task_id == task.id,
            TaskExecutionCommand.id < command_id,
            TaskExecutionCommand.status == "pending",
            TaskExecutionCommand.target_state_version != task.state_version,
            waiting_admission(TaskExecutionCommand.id),
        )
        .values(
            status="failed",
            error="Task command stopped by a control request.",
            completed_at=func.now(),
            result={"rejection_reason": "cancelled_before_admission"},
        )
        .returning(TaskExecutionCommand.id)
    ).all()
    for cancelled_id in cancelled:
        command = db.get(TaskExecutionCommand, cancelled_id, populate_existing=True)
        assert command is not None
        scope = (
            command.payload.get("scope") if isinstance(command.payload, dict) else None
        )
        if command.kind == "start":
            settle_failed_start_no_commit(db, command)
        stage_terminal_event(
            db,
            command_db_id=cancelled_id,
            draft=TerminalTaskEventDraft(
                message_code=None,
                resend_safe=False,
                include_command_identity=scope != "external",
            ),
        )
