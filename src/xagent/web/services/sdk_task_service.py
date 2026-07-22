"""Worker-owned persistence boundary for SDK task routes.

The public ``/v1/chat/tasks`` handlers are async orchestration code.  Every
synchronous SQLAlchemy operation owned by those handlers lives here, opens and
closes its own Session in the worker thread, and returns detached snapshots.
No ORM object or Session crosses back into the event loop.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Collection, Iterable, Sequence

from sqlalchemy import func

from ..models.database import get_session_local
from ..models.task import Task, TaskStatus, TraceEvent
from ..models.user import User
from .connector_runtime import (
    bind_create_connector_runtime_plan,
    persist_create_connector_runtime_context,
    prepare_append_connector_runtime,
    prepare_create_connector_runtime,
)
from .file_turn import (
    bind_turn_files,
    load_turn_file_lookups,
    materialize_turn_file_lookups,
)
from .hot_path_cache import invalidate_task_cache
from .task_execution_controller import (
    TaskControlState,
)

logger = logging.getLogger(__name__)


class SdkTaskNotFoundError(RuntimeError):
    """The API key cannot address the requested SDK task."""


class SdkAgentMismatchError(RuntimeError):
    """The request body agent does not match the authenticated agent."""


class SdkTurnFilesMissingError(RuntimeError):
    """One or more requested files are unavailable to this task owner."""

    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = tuple(missing)
        super().__init__(", ".join(self.missing))


def resolve_sdk_upload_owner_sync(
    *,
    task_id: int | None,
    authenticated_agent_id: int,
    default_user_id: int,
) -> int | None:
    """Resolve an SDK upload owner in a short worker-owned Session.

    Task-aware uploads deliberately use the task's persisted owner rather
    than the agent's current owner.  The task lookup is scoped by both the
    authenticated agent and ``source='sdk'`` so inaccessible tasks remain
    indistinguishable from missing tasks.  ``None`` means the resolved owner
    row no longer exists and lets the API preserve its internal-error mapping.
    """

    session_local = get_session_local()
    with session_local() as db:
        owner_user_id = default_user_id
        if task_id is not None:
            owner_user_id = (
                db.query(Task.user_id)
                .filter(
                    Task.id == task_id,
                    Task.agent_id == authenticated_agent_id,
                    Task.source == "sdk",
                )
                .scalar()
            )
            if owner_user_id is None:
                raise SdkTaskNotFoundError(task_id)

        existing_user_id = (
            db.query(User.id).filter(User.id == int(owner_user_id)).scalar()
        )
        return int(existing_user_id) if existing_user_id is not None else None


@dataclass(frozen=True)
class SdkCreateTaskPrepared:
    task_id: int
    agent_id: int
    task_owner_user_id: int
    actor_user_id: int
    created_at: datetime
    file_infos: tuple[dict[str, Any], ...]
    ephemeral_by_ref: dict[Any, dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class SdkAppendTaskPrepared:
    task_id: int
    agent_id: int
    task_owner_user_id: int
    actor_user_id: int
    file_infos: tuple[dict[str, Any], ...]
    ephemeral_by_ref: dict[Any, dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class SdkTaskSnapshot:
    task_id: int
    agent_id: int
    status: TaskStatus
    run_id: str | None
    state_version: int
    control_state: str
    input: str | None
    output: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class SdkTaskStepsVersion:
    task_id: int
    agent_id: int
    max_event_id: int


@dataclass(frozen=True)
class TraceEventSnapshot:
    id: int
    task_id: int
    event_id: str
    event_type: str
    timestamp: datetime
    step_id: str | None
    data: dict[str, Any]


@dataclass(frozen=True)
class SdkTaskStepsSnapshot:
    task_id: int
    agent_id: int
    max_event_id: int
    events: tuple[TraceEventSnapshot, ...]


def _task_status(value: Any) -> TaskStatus:
    if isinstance(value, TaskStatus):
        return value
    return TaskStatus(str(value))


def _materialize_files_for_create(
    *,
    file_ids: Sequence[str],
    owner_user_id: int,
) -> tuple[dict[str, Any], ...]:
    """Query file metadata, close the Session, then perform durable I/O."""

    if not file_ids:
        return ()
    session_local = get_session_local()
    with session_local() as db:
        lookups = load_turn_file_lookups(
            file_ids=list(file_ids),
            owner_user_id=owner_user_id,
            db=db,
            task_id=None,
        )
    file_infos, missing = materialize_turn_file_lookups(lookups)
    if missing:
        raise SdkTurnFilesMissingError(missing)
    return tuple(deepcopy(file_infos))


def prepare_create_sdk_task_sync(
    *,
    agent_id: int,
    task_owner_user_id: int,
    actor_user_id: int,
    tool_categories: Collection[str] | None,
    content: str,
    file_ids: Sequence[str],
    connector_runtime_context: Iterable[Any] | None,
) -> SdkCreateTaskPrepared:
    """Validate and commit a new PENDING SDK task in worker-owned Sessions.

    File lookup/materialization intentionally precedes connector validation,
    matching the public route's established error precedence.  The Task and
    connector context share one transaction whose commit is the last database
    operation, so an exception return cannot hide a committed PENDING row.
    """

    file_infos = _materialize_files_for_create(
        file_ids=file_ids,
        owner_user_id=task_owner_user_id,
    )
    task_source = "sdk"
    session_local = get_session_local()
    with session_local() as db:
        task = Task(
            user_id=task_owner_user_id,
            title=content[:50] or "SDK task",
            description=content,
            status=TaskStatus.PENDING,
            agent_id=agent_id,
            input=content,
            source=task_source,
            is_visible=False,
        )
        try:
            runtime_plan = prepare_create_connector_runtime(
                db=db,
                tool_categories=tool_categories,
                task_source=task_source,
                connector_user_id=task_owner_user_id,
                payload_items=connector_runtime_context,
            )
            bind_create_connector_runtime_plan(task=task, plan=runtime_plan)
            db.add(task)
            db.flush()
            task_id = int(task.id)
            persist_create_connector_runtime_context(
                db=db,
                task_id=task_id,
                plan=runtime_plan,
            )
            created_at = db.query(Task.created_at).filter(Task.id == task_id).scalar()
            if not isinstance(created_at, datetime):
                raise RuntimeError("New SDK task has no creation timestamp")
            prepared = SdkCreateTaskPrepared(
                task_id=task_id,
                agent_id=agent_id,
                task_owner_user_id=task_owner_user_id,
                actor_user_id=actor_user_id,
                created_at=created_at,
                file_infos=file_infos,
                ephemeral_by_ref=deepcopy(runtime_plan.ephemeral_by_ref),
            )
            db.commit()
        except BaseException:
            db.rollback()
            raise

    return prepared


def prepare_append_sdk_task_sync(
    *,
    task_id: int,
    authenticated_agent_id: int,
    actor_user_id: int,
    requested_agent_id: int,
    file_ids: Sequence[str],
    connector_runtime_context: Iterable[Any] | None,
) -> SdkAppendTaskPrepared:
    """Authorize and snapshot an append without retaining its Session.

    Error order is part of the SDK contract: task scope, body agent, connector
    context, then files.  Durable file materialization happens only after the
    query Session has closed.
    """

    session_local = get_session_local()
    with session_local() as db:
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == authenticated_agent_id,
                Task.source == "sdk",
            )
            .first()
        )
        if task is None:
            raise SdkTaskNotFoundError(task_id)
        if requested_agent_id != authenticated_agent_id:
            raise SdkAgentMismatchError(requested_agent_id)

        task_owner_user_id = int(task.user_id)
        runtime_plan = prepare_append_connector_runtime(
            db=db,
            task=task,
            connector_user_id=task_owner_user_id,
            payload_items=connector_runtime_context,
        )
        lookups = load_turn_file_lookups(
            file_ids=list(file_ids),
            owner_user_id=task_owner_user_id,
            db=db,
            task_id=task_id,
        )

    file_infos, missing = materialize_turn_file_lookups(lookups)
    if missing:
        raise SdkTurnFilesMissingError(missing)
    return SdkAppendTaskPrepared(
        task_id=task_id,
        agent_id=authenticated_agent_id,
        task_owner_user_id=task_owner_user_id,
        actor_user_id=actor_user_id,
        file_infos=tuple(deepcopy(file_infos)),
        ephemeral_by_ref=deepcopy(runtime_plan.ephemeral_by_ref),
    )


def bind_sdk_turn_files_sync(
    *,
    file_ids: Sequence[str],
    task_id: int,
    owner_user_id: int,
) -> None:
    """Bind already-validated files in a short worker-owned transaction."""

    if not file_ids:
        return
    session_local = get_session_local()
    with session_local() as db:
        bind_turn_files(
            file_ids=list(file_ids),
            task_id=task_id,
            owner_user_id=owner_user_id,
            db=db,
        )


def mark_pending_sdk_task_failed_sync(
    task_id: int,
    error_message: str,
    *,
    expected_agent_id: int,
    expected_owner_user_id: int,
) -> bool:
    """Best-effort compensation for a committed create that cannot start.

    The eligibility predicate and FAILED transition are one SQL statement.
    If the orchestrator claims the task first, the ``status=PENDING`` guard
    makes this update a no-op so RUNNING remains authoritative.
    """

    session_local = get_session_local()
    try:
        with session_local() as db:
            updated = (
                db.query(Task)
                .filter(
                    Task.id == task_id,
                    Task.agent_id == expected_agent_id,
                    Task.user_id == expected_owner_user_id,
                    Task.source == "sdk",
                    Task.status == TaskStatus.PENDING,
                )
                .update(
                    {
                        Task.status: TaskStatus.FAILED,
                        Task.control_state: TaskControlState.FAILED.value,
                        Task.state_version: func.coalesce(Task.state_version, 0) + 1,
                        Task.error_message: error_message,
                    },
                    synchronize_session=False,
                )
            )
            db.commit()
            if int(updated or 0) != 1:
                return False
    except Exception:
        logger.warning(
            "Failed to mark pending SDK task %s as failed",
            task_id,
            exc_info=True,
        )
        return False
    try:
        invalidate_task_cache(task_id)
    except Exception:
        logger.warning(
            "Failed to invalidate cache for compensated SDK task %s",
            task_id,
            exc_info=True,
        )
    return True


def load_sdk_task_snapshot_sync(
    task_id: int,
    authenticated_agent_id: int,
) -> SdkTaskSnapshot | None:
    """Return a detached scalar snapshot for SDK task polling."""

    session_local = get_session_local()
    with session_local() as db:
        row = (
            db.query(
                Task.id,
                Task.agent_id,
                Task.status,
                Task.run_id,
                Task.state_version,
                Task.control_state,
                Task.input,
                Task.output,
                Task.error_message,
                Task.created_at,
                Task.updated_at,
            )
            .filter(
                Task.id == task_id,
                Task.agent_id == authenticated_agent_id,
                Task.source == "sdk",
            )
            .first()
        )
        if row is None:
            return None
        if not isinstance(row.created_at, datetime) or not isinstance(
            row.updated_at, datetime
        ):
            raise RuntimeError(f"SDK task {task_id} has invalid timestamps")
        return SdkTaskSnapshot(
            task_id=int(row.id),
            agent_id=int(row.agent_id),
            status=_task_status(row.status),
            run_id=str(row.run_id) if row.run_id is not None else None,
            state_version=int(row.state_version or 0),
            control_state=str(row.control_state or "idle"),
            input=str(row.input) if row.input is not None else None,
            output=str(row.output) if row.output is not None else None,
            error_message=(
                str(row.error_message) if row.error_message is not None else None
            ),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


def load_sdk_task_steps_version_sync(
    task_id: int,
    authenticated_agent_id: int,
) -> SdkTaskStepsVersion | None:
    """Authorize a task and return the cache version for its public trace."""

    session_local = get_session_local()
    with session_local() as db:
        task_row = (
            db.query(Task.id, Task.agent_id)
            .filter(
                Task.id == task_id,
                Task.agent_id == authenticated_agent_id,
                Task.source == "sdk",
            )
            .first()
        )
        if task_row is None:
            return None
        max_event_id = (
            db.query(func.max(TraceEvent.id))
            .filter(
                TraceEvent.task_id == task_id,
                TraceEvent.build_id.is_(None),
            )
            .scalar()
            or 0
        )
        return SdkTaskStepsVersion(
            task_id=int(task_row.id),
            agent_id=int(task_row.agent_id),
            max_event_id=int(max_event_id),
        )


def load_sdk_task_steps_snapshot_sync(
    task_id: int,
    authenticated_agent_id: int,
) -> SdkTaskStepsSnapshot | None:
    """Return detached, ordered trace snapshots for the public step mapper."""

    session_local = get_session_local()
    with session_local() as db:
        task_row = (
            db.query(Task.id, Task.agent_id)
            .filter(
                Task.id == task_id,
                Task.agent_id == authenticated_agent_id,
                Task.source == "sdk",
            )
            .first()
        )
        if task_row is None:
            return None
        rows = (
            db.query(
                TraceEvent.id,
                TraceEvent.task_id,
                TraceEvent.event_id,
                TraceEvent.event_type,
                TraceEvent.timestamp,
                TraceEvent.step_id,
                TraceEvent.data,
            )
            .filter(
                TraceEvent.task_id == task_id,
                TraceEvent.build_id.is_(None),
            )
            .order_by(TraceEvent.id.asc())
            .all()
        )
        resolved_task_id = int(task_row.id)
        resolved_agent_id = int(task_row.agent_id)

    events = tuple(
        TraceEventSnapshot(
            id=int(row.id),
            task_id=int(row.task_id),
            event_id=str(row.event_id),
            event_type=str(row.event_type),
            timestamp=row.timestamp,
            step_id=str(row.step_id) if row.step_id is not None else None,
            data=deepcopy(row.data) if isinstance(row.data, dict) else {},
        )
        for row in rows
    )
    return SdkTaskStepsSnapshot(
        task_id=resolved_task_id,
        agent_id=resolved_agent_id,
        max_event_id=events[-1].id if events else 0,
        events=events,
    )
