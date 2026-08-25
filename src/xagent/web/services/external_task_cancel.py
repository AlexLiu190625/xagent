"""Cancel execution of one external-source task turn.

The durable command channel already carries A2A cancels; its execution core
(``a2a.py``) loads its target with ``Task.source == "a2a"``, so an external
task cannot travel through it. This module is that core's external-scope
counterpart: the same load / hard-cancel / fenced-finalize shape against
``Task.source == "external"`` rows, with three differences the external
surface needs.

  - The visitor has no other channel to learn the turn ended, so the core
    broadcasts the terminal event itself once its finalize commits. The
    settlement path in ``task_orchestrator`` broadcasts only for setup/run
    errors, never for a cancellation.
  - The wait for the running coroutine is longer than the A2A one, because
    the external turn's finalize races the settlement that the cancelled
    coroutine is about to perform. See ``EXTERNAL_CANCEL_WAIT_SECONDS``.
  - Every expected failure leaves as ``TaskCommandRejected``. A rejected
    command is terminal without a client-visible error frame, which is the
    outcome an anonymous visitor should get for a stop that no longer has a
    target.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, update
from sqlalchemy.orm import Session

from ..models.chat_message import TaskChatMessage
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from .chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_PENDING,
    mark_user_message_delivery,
)
from .db_runtime import run_db_io_cancellation_safe
from .task_command_transport import TaskCommandRejected
from .task_execution_controller import TaskControlState

logger = logging.getLogger(__name__)

EXTERNAL_TASK_SOURCE = "external"

# Command payload scope that routes a CANCEL command to this core. A command
# without it keeps the A2A execution path it has always had.
EXTERNAL_COMMAND_SCOPE = "external"

_TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED}

# Cancellation waits for the run coroutine to unwind, per handle, so one
# external cancel occupies a dispatcher slot for up to twice this value
# (the running handle plus the resume coordinator). It is deliberately far
# above the A2A path's 0.5s: whichever writer lands first owns the terminal
# row, and letting the cancelled coroutine settle first keeps its lease and
# delivery reconciliation in the hands of the coroutine that ran the turn.
# Raising it raises the worst-case queueing delay for every other command on
# this process; lowering it makes the overlap window more likely.
EXTERNAL_CANCEL_WAIT_SECONDS = 5.0

# A cancelled turn and a worker shutting down both cut the response short,
# so the text every consumer of the interruption sees states the outcome
# without claiming a cause.
EXTERNAL_TURN_INTERRUPTED_MESSAGE = "This response was interrupted."

# Written to the durable row only when a cancel command actually applied,
# which is what keeps the two causes apart for whoever reads the row later.
EXTERNAL_CANCEL_ERROR_MESSAGE = "Stopped by the visitor."


def _task_run_id(task: Task) -> str | None:
    run_id = getattr(task, "run_id", None)
    return str(run_id) if run_id is not None else None


def _load_external_task(db: Session, *, task_id: int, agent_id: int) -> Task:
    task = (
        db.query(Task)
        .filter(
            Task.id == task_id,
            Task.agent_id == agent_id,
            Task.source == EXTERNAL_TASK_SOURCE,
        )
        .first()
    )
    if task is None:
        raise TaskCommandRejected(
            f"task {task_id} is not an external task of agent {agent_id}",
            reason="task_not_found",
        )
    return task


def _is_settled_external_cancel_target(
    task: Task,
    *,
    expected_run_id: str | None,
    expected_state_version: int,
) -> bool:
    """Report whether this exact target already reached its cancel outcome.

    The judgement is the durable state tuple, not a marker column: the same
    command replayed after its own finalize, and the settlement of the run
    this command cancelled, both leave the exact target run FAILED at the
    command's own state version or one past it. Either way the turn this
    command targeted is over, which is what the command asked for, so the
    replay is answered without writing anything.
    """

    return (
        task.status == TaskStatus.FAILED
        and _task_run_id(task) == expected_run_id
        and int(task.state_version or 0)
        in {expected_state_version, expected_state_version + 1}
    )


def _assert_external_cancel_target(
    task: Task,
    *,
    expected_run_id: str | None,
    expected_state_version: int,
) -> None:
    """Reject a cancel command whose immutable task-state target is stale."""

    current_run_id = _task_run_id(task)
    current_state_version = int(task.state_version or 0)
    if (
        current_run_id != expected_run_id
        or current_state_version != expected_state_version
        or task.status in _TERMINAL_STATUSES
    ):
        raise TaskCommandRejected(
            f"task {task.id} changed from run/version "
            f"{expected_run_id}/{expected_state_version} to "
            f"{current_run_id}/{current_state_version}/{task.status.value}",
            reason="stale_run",
        )


def _load_cancelable_external_task_sync(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
) -> bool:
    """Return whether the exact cancel target is already settled."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _load_external_task(db, task_id=task_id, agent_id=agent_id)
        if _is_settled_external_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        ):
            return True
        _assert_external_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        )
        return False


def _mark_cancelled_turn_delivery_dispatched(db: Session, task_id: int) -> None:
    """Close the cancelled turn's delivery row without committing ``db``.

    A delivery row is closed by whichever coroutine settles the turn it
    belongs to. When this finalize wins that race the settlement is fenced
    out, so nothing else ever moves the row off ``pending`` and every later
    resend of the same client message id is refused forever.
    ``dispatched`` rather than ``failed`` is the
    honest target: the run had already started, so the message may have been
    consumed and a retry invitation could double-execute it.

    The turn is identified as the newest pending user row on the task, since
    the command carries a task-state target and no turn id. A cancel applies
    to the turn that is running, which is the turn that owns that row.
    """

    row = (
        db.query(TaskChatMessage.turn_id)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "user",
            TaskChatMessage.delivery_status == DELIVERY_PENDING,
        )
        .order_by(TaskChatMessage.id.desc())
        .first()
    )
    if row is None or row[0] is None:
        return
    mark_user_message_delivery(
        db,
        task_id=task_id,
        turn_id=str(row[0]),
        status=DELIVERY_DISPATCHED,
    )


def _finalize_external_cancel_sync(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
) -> None:
    """Atomically persist cancellation for one exact task-state target."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _load_external_task(db, task_id=task_id, agent_id=agent_id)
        if _is_settled_external_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        ):
            return
        _assert_external_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        )

        statement = (
            update(Task)
            .where(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == EXTERNAL_TASK_SOURCE,
                func.coalesce(Task.state_version, 0) == expected_state_version,
                Task.status.notin_(_TERMINAL_STATUSES),
            )
            .values(
                status=TaskStatus.FAILED,
                control_state=TaskControlState.FAILED.value,
                state_version=expected_state_version + 1,
                runner_id=None,
                lease_attempt_id=None,
                lease_expires_at=None,
                last_heartbeat_at=None,
                error_message=EXTERNAL_CANCEL_ERROR_MESSAGE,
            )
        )
        if expected_run_id is None:
            statement = statement.where(Task.run_id.is_(None))
        else:
            statement = statement.where(Task.run_id == expected_run_id)

        updated = db.execute(
            statement.returning(Task).execution_options(synchronize_session=False)
        ).scalar_one_or_none()
        if updated is None:
            db.rollback()
            raise TaskCommandRejected(
                f"task {task_id} changed while finalizing cancel target "
                f"{expected_run_id}/{expected_state_version}",
                reason="stale_run",
            )
        _mark_cancelled_turn_delivery_dispatched(db, task_id)
        db.commit()


async def _broadcast_external_cancel_terminal_event(task_id: int) -> None:
    from ..api.websocket import create_terminal_task_error_event
    from ..api.websocket import manager as websocket_manager

    try:
        await websocket_manager.broadcast_to_task(
            create_terminal_task_error_event(
                task_id,
                EXTERNAL_TURN_INTERRUPTED_MESSAGE,
            ),
            task_id,
        )
    except Exception:
        # The terminal row is already committed; a failed notification must
        # not turn that durable success into a retried command.
        logger.warning(
            "task %s cancellation was committed but its terminal broadcast failed",
            task_id,
            exc_info=True,
        )


async def cancel_external_task_unserialized(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
) -> None:
    """Cancel one exact durable-command target while the caller owns its gate."""

    already_settled = await run_db_io_cancellation_safe(
        lambda: _load_cancelable_external_task_sync(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        )
    )
    if not already_settled:
        from ..api.websocket import background_task_manager

        await background_task_manager.cancel_task(
            task_id,
            timeout_seconds=EXTERNAL_CANCEL_WAIT_SECONDS,
        )
        await run_db_io_cancellation_safe(
            lambda: _finalize_external_cancel_sync(
                task_id=task_id,
                agent_id=agent_id,
                expected_run_id=expected_run_id,
                expected_state_version=expected_state_version,
            )
        )
    await _broadcast_external_cancel_terminal_event(task_id)
