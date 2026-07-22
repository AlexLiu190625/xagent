from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from time import monotonic
from types import MappingProxyType
from typing import Any, Mapping

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import String, and_, cast, false, or_
from sqlalchemy.orm import Session

from ...core.execution_scope import resolve_execution_scope
from ..models.agent import Agent, AgentStatus
from ..models.database import get_db, get_session_local
from ..models.task import Task, TaskStatus
from ..services.a2a_protocol import (
    A2A_VERSION,
    ALL_TASK_STATES,
    a2a_error,
    a2a_json_response,
    build_agent_card,
    extract_message_text,
    is_published_agent,
    message_context_id,
    message_task_id,
    new_context_id,
    sse_task_artifacts,
    sse_task_snapshot,
    sse_task_update,
    task_context_id,
    task_state,
    task_to_a2a,
)
from ..services.api_keys import AgentApiIdentity
from ..services.db_runtime import (
    drain_async_task_cancellation_safe,
    is_database_pool_timeout,
    run_db_io_cancellation_safe,
)
from ..services.task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    TaskCommandKind,
    dispatch_one_task_command,
    enqueue_task_command,
    load_task_command,
    retry_failed_task_command,
)
from ..services.task_execution_controller import (
    TaskControlState,
    apply_task_control_transition,
    task_execution_controller,
)
from ..services.task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
    acquire_task_lease_no_commit,
    release_task_lease_no_commit,
    run_task_lease_heartbeat,
    stop_task_lease_heartbeat,
)
from ..services.task_orchestrator import (
    TaskCreationSpec,
    TaskTurnError,
    TaskTurnNotFoundError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
)
from ..services.task_setup_snapshot import load_task_setup_snapshot_sync
from .v1.deps import get_agent_from_api_key, record_key_usage
from .v1.errors import V1ApiError

router = APIRouter(prefix="/api/a2a", tags=["a2a"])
logger = logging.getLogger(__name__)
_bearer = HTTPBearer(auto_error=False)
_TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED}
_STREAM_END_STATUSES = _TERMINAL_STATUSES | {
    TaskStatus.PAUSED,
    TaskStatus.WAITING_FOR_USER,
}
A2A_BLOCKING_WAIT_TIMEOUT_SECONDS = 60.0
A2A_STREAM_MAX_DURATION_SECONDS = 60.0 * 60.0
_A2A_OVERRIDE_STATES = (
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
    "TASK_STATE_AUTH_REQUIRED",
)
_A2A_TASK_STATUS_MAP: dict[str, tuple[TaskStatus, ...]] = {
    "TASK_STATE_SUBMITTED": (TaskStatus.PENDING,),
    "TASK_STATE_WORKING": (TaskStatus.RUNNING,),
    "TASK_STATE_INPUT_REQUIRED": (TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER),
    "TASK_STATE_COMPLETED": (TaskStatus.COMPLETED,),
    "TASK_STATE_FAILED": (TaskStatus.FAILED,),
}


@dataclass(frozen=True)
class _A2ATaskSnapshot:
    """Session-independent fields consumed by A2A response/poll helpers."""

    id: int
    status: TaskStatus
    updated_at: datetime | None
    output: str | None
    error_message: str | None
    agent_config: Mapping[str, Any]
    run_id: str | None


@dataclass(frozen=True)
class _A2ATurnPreparation:
    task: _A2ATaskSnapshot
    waiting_for_user: bool


@dataclass(frozen=True)
class _A2ACancelPreparation:
    """Detached command identity produced by the cancel DB transaction."""

    command_db_id: int
    command_identity: str


def _request_identity(
    path_agent_id: int,
    identity: AgentApiIdentity,
) -> AgentApiIdentity:
    _require_bound_agent(path_agent_id, identity)
    return identity


def _task_snapshot(task: Task) -> _A2ATaskSnapshot:
    raw_status = task.status
    status = (
        raw_status if isinstance(raw_status, TaskStatus) else TaskStatus(raw_status)
    )
    raw_config: Mapping[str, Any] = (
        task.agent_config if isinstance(task.agent_config, Mapping) else {}
    )
    raw_updated_at = getattr(task, "updated_at", None)
    return _A2ATaskSnapshot(
        id=int(task.id),
        status=status,
        updated_at=(raw_updated_at if isinstance(raw_updated_at, datetime) else None),
        output=str(task.output) if task.output is not None else None,
        error_message=(
            str(task.error_message) if task.error_message is not None else None
        ),
        agent_config=MappingProxyType(deepcopy(dict(raw_config))),
        run_id=str(task.run_id) if task.run_id is not None else None,
    )


async def _get_a2a_agent_from_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AgentApiIdentity:
    _validate_a2a_version(request)
    try:
        return await get_agent_from_api_key(credentials)
    except V1ApiError as exc:
        raise a2a_error(
            "invalid_api_key",
            exc.message,
            status_code=exc.http_status,
        ) from exc


def _resolve_published_agent(db: Session, agent_id: int) -> Agent:
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if agent is None or not is_published_agent(agent):
        raise a2a_error("agent_not_found", "Agent not found.", status_code=404)
    return agent


def _resolve_a2a_task(db: Session, task_id: int, agent: Agent) -> Task:
    return _resolve_a2a_task_for_agent_id(db, task_id, int(agent.id))


def _resolve_a2a_task_for_agent_id(
    db: Session,
    task_id: int,
    agent_id: int,
) -> Task:
    task = (
        db.query(Task)
        .filter(
            Task.id == task_id,
            Task.agent_id == agent_id,
            Task.source == "a2a",
        )
        .first()
    )
    if task is None:
        raise a2a_error("task_not_found", "Task not found.", status_code=404)
    return task


def _task_run_id(task: Task) -> str | None:
    run_id = getattr(task, "run_id", None)
    return str(run_id) if run_id is not None else None


def _require_bound_agent(path_agent_id: int, identity: AgentApiIdentity) -> None:
    if (
        identity.agent_id != int(path_agent_id)
        or identity.status != AgentStatus.PUBLISHED.value
    ):
        raise a2a_error("agent_not_found", "Agent not found.", status_code=404)


def _schedule_waiting_a2a_resume(
    *,
    task_id: int,
    agent_service: Any,
    task_owner_user_id: int,
    run_id: str | None,
    resolved_execution_scope: Any,
) -> None:
    from .websocket import background_task_manager, execute_resume_background

    if not background_task_manager.reserve_resume(task_id):
        raise RuntimeError(f"Task {task_id} already has a resume in progress")
    previous_task = background_task_manager.running_tasks.get(task_id)
    bg_task: asyncio.Task[None] | None = None
    try:
        bg_task = asyncio.create_task(
            execute_resume_background(
                task_id=task_id,
                agent_service=agent_service,
                task_owner_user_id=task_owner_user_id,
                expected_run_id=run_id,
                previous_task=previous_task,
                resolved_execution_scope=resolved_execution_scope,
            )
        )
        background_task_manager.register_reserved_resume(task_id, bg_task)
    except BaseException:
        if bg_task is not None:
            bg_task.cancel()
        background_task_manager.release_resume_reservation(task_id)
        raise


def _restore_waiting_resume_claim(
    task_id: int,
    agent_id: int,
    lease: TaskLease,
) -> bool:
    """Release exactly one failed A2A resume claim back to WAITING."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        claimed = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
                Task.status == TaskStatus.RUNNING,
                Task.runner_id == lease.runner_id,
                Task.run_id == lease.run_id,
            )
            .with_for_update()
            .first()
        )
        if claimed is None:
            db.rollback()
            return False
        released = release_task_lease_no_commit(
            db,
            lease,
            status=TaskStatus.WAITING_FOR_USER,
        )
        if not released:
            db.rollback()
            return False
        db.commit()
        return True


def _claim_waiting_a2a_resume(task_id: int, agent_id: int) -> TaskLease:
    """Atomically claim one WAITING task with a fenced, expiring lease."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
            )
            .with_for_update()
            .first()
        )
        if task is None:
            raise a2a_error("task_not_found", "Task not found.", status_code=404)
        if task.status != TaskStatus.WAITING_FOR_USER:
            raise a2a_error(
                "unsupported_operation",
                "Task is currently running and cannot accept a new message.",
                status_code=400,
                details={"taskId": task_id},
            )
        lease = acquire_task_lease_no_commit(
            db,
            task_id,
            expected_run_id=_task_run_id(task),
        )
        if lease is None:
            db.rollback()
            raise a2a_error(
                "unsupported_operation",
                "Task is currently running and cannot accept a new message.",
                status_code=400,
                details={"taskId": task_id},
            )
        db.commit()
        return lease


def _finalize_waiting_a2a_resume(
    *,
    task_id: int,
    agent_id: int,
    text: str,
    posted: bool,
    lease: TaskLease,
) -> _A2ATaskSnapshot:
    """Persist the checkpoint result without reusing the request Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
                Task.status == TaskStatus.RUNNING,
                Task.runner_id == lease.runner_id,
                Task.run_id == lease.run_id,
            )
            .with_for_update()
            .first()
        )
        if task is None:
            raise RuntimeError(
                f"Task {task_id} A2A resume claim is no longer owned by this run"
            )
        if not posted:
            # A WAITING_FOR_USER checkpoint should normally be durable. If it is
            # unavailable, retain the previous restart-safe behavior by starting a
            # new turn from transcript history instead of leaving the task stuck.
            released = release_task_lease_no_commit(
                db,
                lease,
                status=TaskStatus.PAUSED,
            )
            if not released:
                db.rollback()
                raise RuntimeError(
                    f"Task {task_id} A2A resume lease changed before fallback"
                )
        else:
            setattr(task, "input", text)
            setattr(task, "output", None)
            setattr(task, "error_message", None)
        db.commit()
        db.refresh(task)
        return _task_snapshot(task)


async def _resume_waiting_a2a_task(
    *,
    task_id: int,
    agent_id: int,
    task_owner_user_id: int,
    text: str,
    message_id: str,
) -> tuple[bool, _A2ATaskSnapshot]:
    scope = await run_db_io_cancellation_safe(lambda: resolve_execution_scope(task_id))
    setup_snapshot = await run_db_io_cancellation_safe(
        lambda: load_task_setup_snapshot_sync(task_id, task_owner_user_id)
    )
    if setup_snapshot is None:
        raise a2a_error("task_not_found", "Task not found.", status_code=404)

    from .chat import get_agent_manager

    agent_service = await get_agent_manager().get_agent_for_task(
        task_id,
        None,
        user=setup_snapshot.runtime_user,
        task_setup_snapshot=setup_snapshot,
        task_owner_user_id=task_owner_user_id,
        resolved_execution_scope=scope,
    )

    async def claim_finalize_and_schedule() -> tuple[bool, _A2ATaskSnapshot]:
        lease = await run_db_io_cancellation_safe(
            lambda: _claim_waiting_a2a_resume(task_id, agent_id)
        )
        heartbeat_stop_event = asyncio.Event()
        heartbeat_task: asyncio.Task[TaskLeaseHeartbeatOutcome] | None = (
            asyncio.create_task(run_task_lease_heartbeat(lease, heartbeat_stop_event))
        )

        async def stop_claim_heartbeat() -> TaskLeaseHeartbeatOutcome:
            nonlocal heartbeat_task
            task = heartbeat_task
            if task is None:
                return TaskLeaseHeartbeatOutcome()
            outcome = await stop_task_lease_heartbeat(task, heartbeat_stop_event)
            heartbeat_task = None
            return outcome

        def raise_if_heartbeat_unhealthy(
            outcome: TaskLeaseHeartbeatOutcome,
            *,
            component: str,
        ) -> None:
            if not outcome.requires_ttl_recovery:
                return
            logger.error(
                "task_id=%s component=%s A2A resume lease heartbeat is "
                "unhealthy; retaining the exact claim for TTL recovery "
                "(lost=%s, pool_timeout=%s)",
                task_id,
                component,
                outcome.lease_lost,
                outcome.pool_timeout is not None,
            )
            if outcome.pool_timeout is not None:
                raise outcome.pool_timeout
            raise RuntimeError(
                f"Task {task_id} A2A resume lease heartbeat lost ownership"
            )

        async def restore_after(error: BaseException, component: str) -> None:
            if is_database_pool_timeout(error):
                logger.error(
                    "task_id=%s component=%s database pool checkout timed out; "
                    "retaining A2A resume lease for TTL recovery: %s",
                    task_id,
                    component,
                    error,
                    exc_info=True,
                )
                return
            await run_db_io_cancellation_safe(
                lambda: _restore_waiting_resume_claim(
                    task_id,
                    agent_id,
                    lease,
                )
            )

        try:
            # Let the heartbeat coroutine enter its wait before yielding to the
            # external checkpoint post. The lease must remain renewable for the
            # whole await; otherwise a second runner can take over the same run
            # while this runner still injects the user message.
            await asyncio.sleep(0)
            if heartbeat_task is not None and heartbeat_task.done():
                raise_if_heartbeat_unhealthy(
                    heartbeat_task.result(),
                    component="a2a-resume-heartbeat",
                )
            try:
                posted = await agent_service.post_user_message(
                    str(task_id),
                    execution_message=text,
                    display_message=text,
                    turn_id=f"a2a:{task_id}:{message_id}",
                    request_interrupt=False,
                    reason="A2A input-required response",
                )
            except BaseException as exc:
                heartbeat_outcome = await stop_claim_heartbeat()
                if heartbeat_outcome.requires_ttl_recovery:
                    raise_if_heartbeat_unhealthy(
                        heartbeat_outcome,
                        component="a2a-resume-heartbeat",
                    )
                await restore_after(exc, "a2a-resume-post")
                raise

            heartbeat_outcome = await stop_claim_heartbeat()
            raise_if_heartbeat_unhealthy(
                heartbeat_outcome,
                component="a2a-resume-heartbeat",
            )

            try:
                finalized = await run_db_io_cancellation_safe(
                    lambda: _finalize_waiting_a2a_resume(
                        task_id=task_id,
                        agent_id=agent_id,
                        text=text,
                        posted=posted,
                        lease=lease,
                    )
                )
            except BaseException as exc:
                await restore_after(exc, "a2a-resume-finalize")
                raise

            if not posted:
                return False, finalized

            try:
                _schedule_waiting_a2a_resume(
                    task_id=task_id,
                    agent_service=agent_service,
                    task_owner_user_id=task_owner_user_id,
                    run_id=lease.run_id,
                    resolved_execution_scope=scope,
                )
            except BaseException as exc:
                await restore_after(exc, "a2a-resume-schedule")
                raise
            return True, finalized
        finally:
            if heartbeat_task is not None:
                try:
                    outcome = await stop_claim_heartbeat()
                    if outcome.requires_ttl_recovery:
                        logger.error(
                            "task_id=%s component=a2a-resume-heartbeat "
                            "cleanup retained the exact claim for TTL recovery "
                            "(lost=%s, pool_timeout=%s)",
                            task_id,
                            outcome.lease_lost,
                            outcome.pool_timeout is not None,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Failed to stop A2A resume claim heartbeat for task %s",
                        task_id,
                    )

    workflow = asyncio.create_task(claim_finalize_and_schedule())
    return await drain_async_task_cancellation_safe(workflow)


def _validate_a2a_version(request: Request) -> None:
    requested = request.headers.get("A2A-Version")
    if requested is None:
        requested = request.query_params.get("A2A-Version")
    if requested is None or not requested.strip():
        raise a2a_error(
            "version_not_supported",
            "A2A-Version header or query parameter is required.",
            status_code=400,
            details={"supportedVersions": A2A_VERSION},
        )
    requested = requested.strip()
    version_parts = requested.split(".")
    compatible = (
        len(version_parts) in {2, 3}
        and all(part.isdecimal() for part in version_parts)
        and version_parts[0] == A2A_VERSION.split(".", maxsplit=1)[0]
    )
    if not compatible:
        raise a2a_error(
            "version_not_supported",
            f"A2A protocol version {requested!r} is not supported.",
            status_code=400,
            details={"supportedVersions": A2A_VERSION},
        )


def _validate_send_configuration(body: Mapping[str, Any]) -> bool:
    configuration = body.get("configuration")
    if configuration is None:
        return False
    if not isinstance(configuration, Mapping):
        raise a2a_error(
            "invalid_argument",
            "configuration must be a JSON object.",
            status_code=400,
            details={"field": "configuration"},
        )
    if configuration.get("taskPushNotificationConfig") is not None:
        raise a2a_error(
            "push_notification_not_supported",
            "This agent does not support A2A push notifications.",
            status_code=400,
        )
    accepted_modes = configuration.get("acceptedOutputModes")
    if accepted_modes is not None:
        if not isinstance(accepted_modes, list) or not all(
            isinstance(mode, str) for mode in accepted_modes
        ):
            raise a2a_error(
                "invalid_argument",
                "acceptedOutputModes must be an array of media types.",
                status_code=400,
                details={"field": "configuration.acceptedOutputModes"},
            )
        if accepted_modes and "text/plain" not in accepted_modes:
            raise a2a_error(
                "content_type_not_supported",
                "This agent currently returns text/plain output only.",
                status_code=400,
                details={"supportedMediaType": "text/plain"},
            )
    return_immediately = configuration.get("returnImmediately", False)
    if not isinstance(return_immediately, bool):
        raise a2a_error(
            "invalid_argument",
            "returnImmediately must be a boolean.",
            status_code=400,
            details={"field": "configuration.returnImmediately"},
        )
    return return_immediately


async def _start_a2a_turn(
    *,
    agent_id: int,
    task_owner_user_id: int,
    execution_mode: str | None,
    text: str,
    message_id: str,
    context_id: str | None,
    task_id: int | None,
) -> _A2ATaskSnapshot:
    async def start_unserialized() -> _A2ATaskSnapshot:
        return await _start_a2a_turn_unserialized(
            agent_id=agent_id,
            task_owner_user_id=task_owner_user_id,
            execution_mode=execution_mode,
            text=text,
            message_id=message_id,
            context_id=context_id,
            task_id=task_id,
        )

    if task_id is not None:
        async with task_execution_controller.command(task_id):
            return await start_unserialized()
    return await start_unserialized()


async def _start_a2a_turn_unserialized(
    *,
    agent_id: int,
    task_owner_user_id: int,
    execution_mode: str | None,
    text: str,
    message_id: str,
    context_id: str | None,
    task_id: int | None,
) -> _A2ATaskSnapshot:
    payload = TaskTurnPayload(transcript_message=text)
    if task_id is None:
        resolved_context_id = context_id or new_context_id()
        try:
            started = await TaskTurnOrchestrator.create_and_begin_turn(
                creation=TaskCreationSpec(
                    task_owner_user_id=task_owner_user_id,
                    title=(text[:50] or "A2A task"),
                    description=text,
                    agent_id=agent_id,
                    execution_mode=execution_mode,
                    source="a2a",
                    is_visible=False,
                    agent_config={"a2a_context_id": resolved_context_id},
                ),
                payload=payload,
                actor_user_id=task_owner_user_id,
            )
        except TaskTurnNotFoundError as exc:
            raise a2a_error(
                "task_not_found", "Task not found.", status_code=404
            ) from exc
        except TaskTurnError as exc:
            raise a2a_error(
                "unsupported_operation",
                "Task is currently running and cannot accept a new message.",
                status_code=400,
            ) from exc
        return _A2ATaskSnapshot(
            id=started.task_id,
            status=started.status,
            updated_at=started.updated_at,
            output=None,
            error_message=None,
            agent_config=MappingProxyType({"a2a_context_id": resolved_context_id}),
            run_id=started.run_id or None,
        )

    prepared = await run_db_io_cancellation_safe(
        lambda: _prepare_existing_a2a_turn(
            agent_id=agent_id,
            context_id=context_id,
            task_id=task_id,
        )
    )
    task = prepared.task
    normalized_waiting_task = False
    if prepared.waiting_for_user:
        # Resume the trace-backed checkpoint so DAG/React step state is
        # preserved across workers and restarts. Only a missing checkpoint
        # falls back to the durable APPEND/replan path.
        resumed, task = await _resume_waiting_a2a_task(
            task_id=task.id,
            agent_id=agent_id,
            task_owner_user_id=task_owner_user_id,
            text=text,
            message_id=message_id,
        )
        if resumed:
            return task
        normalized_waiting_task = True

    try:
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=task.id,
            task_owner_user_id=task_owner_user_id,
            actor_user_id=task_owner_user_id,
            payload=payload,
            kind=TurnKind.APPEND,
            force_fresh=False,
        )
    except TaskTurnNotFoundError as exc:
        await run_db_io_cancellation_safe(
            lambda: _recover_failed_turn_start(
                task.id,
                restore_waiting=normalized_waiting_task,
            )
        )
        raise a2a_error("task_not_found", "Task not found.", status_code=404) from exc
    except TaskTurnError as exc:
        await run_db_io_cancellation_safe(
            lambda: _recover_failed_turn_start(
                task.id,
                restore_waiting=normalized_waiting_task,
            )
        )
        raise a2a_error(
            "unsupported_operation",
            "Task is currently running and cannot accept a new message.",
            status_code=400,
            details={"taskId": task.id},
        ) from exc
    except Exception as exc:
        if not is_database_pool_timeout(exc):
            await run_db_io_cancellation_safe(
                lambda: _recover_failed_turn_start(
                    task.id,
                    restore_waiting=normalized_waiting_task,
                )
            )
        else:
            logger.error(
                "task_id=%s component=a2a-turn-start database pool checkout "
                "timed out; skipping immediate recovery checkout",
                task.id,
            )
        raise

    return replace(
        task,
        status=started.status,
        updated_at=started.updated_at,
        output=None,
        error_message=None,
        run_id=started.run_id or None,
    )


def _prepare_existing_a2a_turn(
    *,
    agent_id: int,
    context_id: str | None,
    task_id: int,
) -> _A2ATurnPreparation:
    """Validate an existing A2A turn using one worker-owned Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_a2a_task_for_agent_id(db, task_id, agent_id)
        if task.status in _TERMINAL_STATUSES:
            raise a2a_error(
                "unsupported_operation",
                "Messages cannot be appended to a terminal A2A task.",
                status_code=400,
                details={"taskId": task.id},
            )
        stored_context_id = task_context_id(task)
        if context_id is not None and context_id != stored_context_id:
            raise a2a_error(
                "invalid_argument",
                "The supplied contextId does not match the referenced task.",
                status_code=400,
                details={"taskId": task.id, "contextId": context_id},
            )
        agent_config: dict[str, Any] = (
            dict(task.agent_config) if isinstance(task.agent_config, Mapping) else {}
        )
        if not agent_config.get("a2a_context_id"):
            agent_config["a2a_context_id"] = stored_context_id
            setattr(task, "agent_config", agent_config)
            db.commit()
            db.refresh(task)
        return _A2ATurnPreparation(
            task=_task_snapshot(task),
            waiting_for_user=task.status == TaskStatus.WAITING_FOR_USER,
        )


def _recover_failed_turn_start(
    task_id: int,
    *,
    restore_waiting: bool,
) -> None:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None:
            return
        if restore_waiting and task.status == TaskStatus.PAUSED:
            apply_task_control_transition(
                task,
                TaskControlState.WAITING_FOR_USER,
                status=TaskStatus.WAITING_FOR_USER,
                expected_run_id=_task_run_id(task),
            )
            db.commit()


async def _json_body(request: Request) -> Mapping[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise a2a_error(
            "invalid_request",
            "Request body must be valid JSON.",
            status_code=400,
        ) from exc
    if not isinstance(body, Mapping):
        raise a2a_error(
            "invalid_argument", "Request body must be a JSON object.", status_code=400
        )
    return body


def _message_payload(body: Mapping[str, Any]) -> Mapping[str, Any]:
    message = body.get("message")
    if not isinstance(message, Mapping):
        raise a2a_error(
            "invalid_argument",
            "Request body must include a message object.",
            status_code=400,
        )
    return message


def _message_id(message: Mapping[str, Any]) -> str:
    value = message.get("messageId")
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise a2a_error(
        "invalid_argument",
        "message.messageId must be a non-empty string.",
        status_code=400,
        details={"field": "message.messageId"},
    )


def _fetch_fresh_a2a_task_sync(
    agent_id: int,
    task_id: int,
) -> _A2ATaskSnapshot | None:
    session_local = get_session_local()
    with session_local() as db:
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
            )
            .first()
        )
        return _task_snapshot(task) if task is not None else None


async def _fetch_fresh_a2a_task(
    agent_id: int,
    task_id: int,
) -> _A2ATaskSnapshot | None:
    return await run_db_io_cancellation_safe(
        lambda: _fetch_fresh_a2a_task_sync(agent_id, task_id)
    )


def _task_stream_response(
    agent_id: int,
    task: _A2ATaskSnapshot,
) -> StreamingResponse:
    started_task_id = task.id

    async def _events() -> Any:
        deadline = monotonic() + A2A_STREAM_MAX_DURATION_SECONDS
        yield sse_task_snapshot(task)
        if task.status in _STREAM_END_STATUSES:
            return
        previous_state = task_state(task)
        previous_output = str(task.output or "")
        previous_error = str(task.error_message or "")
        artifact_finalized = (
            bool(previous_output) and task.status in _STREAM_END_STATUSES
        )
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.5, remaining))
            fresh = await _fetch_fresh_a2a_task(agent_id, started_task_id)
            if fresh is None:
                return
            fresh_output = str(fresh.output or "")
            fresh_state = task_state(fresh)
            fresh_error = str(fresh.error_message or "")
            stream_ended = fresh.status in _STREAM_END_STATUSES
            if fresh_output and fresh_output != previous_output:
                append = bool(previous_output) and fresh_output.startswith(
                    previous_output
                )
                chunk = fresh_output[len(previous_output) :] if append else fresh_output
                artifacts = sse_task_artifacts(
                    fresh,
                    text=chunk,
                    append=append,
                    last_chunk=stream_ended,
                )
                if artifacts:
                    yield artifacts
                artifact_finalized = stream_ended
            elif stream_ended and fresh_output and not artifact_finalized:
                artifacts = sse_task_artifacts(
                    fresh,
                    text=fresh_output,
                    append=False,
                    last_chunk=True,
                )
                if artifacts:
                    yield artifacts
                artifact_finalized = True
            if fresh_state != previous_state or fresh_error != previous_error:
                yield sse_task_update(fresh)
            previous_state = fresh_state
            previous_output = fresh_output
            previous_error = fresh_error
            if stream_ended:
                return

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"A2A-Version": A2A_VERSION},
    )


async def _wait_for_task(
    agent_id: int,
    task: _A2ATaskSnapshot,
) -> _A2ATaskSnapshot:
    if task.status in _STREAM_END_STATUSES:
        return task
    task_id = int(task.id)
    deadline = monotonic() + A2A_BLOCKING_WAIT_TIMEOUT_SECONDS
    fresh = task
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            return fresh
        await asyncio.sleep(min(0.25, remaining))
        fetched = await _fetch_fresh_a2a_task(agent_id, task_id)
        if fetched is None:
            raise a2a_error("task_not_found", "Task not found.", status_code=404)
        fresh = fetched
        if fresh.status in _STREAM_END_STATUSES:
            return fresh


def _page_offset(page_token: str | None) -> int:
    if page_token is None or page_token == "":
        return 0
    if page_token.isdecimal():
        return int(page_token)
    raise a2a_error(
        "invalid_argument",
        "pageToken is invalid.",
        status_code=400,
        details={"field": "pageToken"},
    )


@router.get("/agents/{agent_id}/.well-known/agent-card.json")
async def get_agent_card_well_known(
    agent_id: int,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    agent = _resolve_published_agent(db, agent_id)
    return a2a_json_response(build_agent_card(agent, request))


@router.post("/agents/{agent_id}/message:send")
async def send_message(
    agent_id: int,
    request: Request,
    authed: AgentApiIdentity = Depends(_get_a2a_agent_from_api_key),
) -> Any:
    identity = _request_identity(agent_id, authed)
    body = await _json_body(request)
    return_immediately = _validate_send_configuration(body)
    message = _message_payload(body)
    text = extract_message_text(message)
    message_id = _message_id(message)
    context_id = message_context_id(message, body)
    task_id = message_task_id(message, body)
    task = await _start_a2a_turn(
        agent_id=identity.agent_id,
        task_owner_user_id=identity.user_id,
        execution_mode=identity.execution_mode,
        text=text,
        message_id=message_id,
        context_id=context_id,
        task_id=task_id,
    )
    await record_key_usage(identity.key_prefix)
    if not return_immediately:
        task = await _wait_for_task(identity.agent_id, task)
    return a2a_json_response({"task": task_to_a2a(task)})


@router.post("/agents/{agent_id}/message:stream")
async def stream_message(
    agent_id: int,
    request: Request,
    authed: AgentApiIdentity = Depends(_get_a2a_agent_from_api_key),
) -> StreamingResponse:
    identity = _request_identity(agent_id, authed)
    body = await _json_body(request)
    _validate_send_configuration(body)
    message = _message_payload(body)
    text = extract_message_text(message)
    message_id = _message_id(message)
    context_id = message_context_id(message, body)
    task_id = message_task_id(message, body)
    task = await _start_a2a_turn(
        agent_id=identity.agent_id,
        task_owner_user_id=identity.user_id,
        execution_mode=identity.execution_mode,
        text=text,
        message_id=message_id,
        context_id=context_id,
        task_id=task_id,
    )
    await record_key_usage(identity.key_prefix)
    return _task_stream_response(identity.agent_id, task)


@router.get("/agents/{agent_id}/tasks/{task_id}")
async def get_task(
    agent_id: int,
    task_id: int,
    authed: AgentApiIdentity = Depends(_get_a2a_agent_from_api_key),
) -> Any:
    _require_bound_agent(agent_id, authed)
    resolved_agent_id = authed.agent_id
    task = await _fetch_fresh_a2a_task(resolved_agent_id, task_id)
    if task is None:
        raise a2a_error("task_not_found", "Task not found.", status_code=404)
    return a2a_json_response(task_to_a2a(task))


@router.get("/agents/{agent_id}/tasks")
async def list_tasks(
    agent_id: int,
    context_id: str | None = Query(default=None, alias="contextId"),
    status: str | None = Query(default=None),
    page_size: int = Query(default=50, ge=1, le=100, alias="pageSize"),
    page_token: str | None = Query(default=None, alias="pageToken"),
    include_artifacts: bool = Query(default=False, alias="includeArtifacts"),
    status_timestamp_after: datetime | None = Query(
        default=None,
        alias="statusTimestampAfter",
        description=(
            "Filter by the timestamp exposed in each A2A task status; "
            "this is backed by Task.updated_at."
        ),
    ),
    authed: AgentApiIdentity = Depends(_get_a2a_agent_from_api_key),
    db: Session = Depends(get_db),
) -> Any:
    _require_bound_agent(agent_id, authed)
    query = db.query(Task).filter(
        Task.agent_id == authed.agent_id,
        Task.source == "a2a",
    )
    if context_id is not None:
        stored_context_id = Task.agent_config["a2a_context_id"].as_string()
        query = query.filter(
            or_(
                stored_context_id == context_id,
                and_(
                    stored_context_id.is_(None),
                    cast(Task.id, String) == context_id,
                ),
            )
        )
    if status is not None:
        if status not in ALL_TASK_STATES:
            raise a2a_error(
                "invalid_argument",
                f"Unknown A2A task status: {status}",
                status_code=400,
                details={"field": "status"},
            )
        stored_a2a_state = Task.agent_config["a2a_state"].as_string()
        state_filters: list[Any] = []
        if status in _A2A_OVERRIDE_STATES:
            state_filters.append(stored_a2a_state == status)
        task_statuses = _A2A_TASK_STATUS_MAP.get(status)
        if task_statuses:
            state_filters.append(
                and_(
                    or_(
                        stored_a2a_state.is_(None),
                        ~stored_a2a_state.in_(_A2A_OVERRIDE_STATES),
                    ),
                    Task.status.in_(task_statuses),
                )
            )
        query = query.filter(or_(*state_filters) if state_filters else false())
    if status_timestamp_after is not None:
        query = query.filter(Task.updated_at > status_timestamp_after)

    offset = _page_offset(page_token)
    total_size = query.count()
    page = query.order_by(Task.id.desc()).offset(offset).limit(page_size).all()
    next_offset = offset + len(page)
    next_page_token = str(next_offset) if next_offset < total_size else ""
    return a2a_json_response(
        {
            "tasks": [
                task_to_a2a(task, include_artifacts=include_artifacts) for task in page
            ],
            "nextPageToken": next_page_token,
            "pageSize": page_size,
            "totalSize": total_size,
        }
    )


@router.api_route(
    "/agents/{agent_id}/tasks/{task_id}:subscribe", methods=["GET", "POST"]
)
async def subscribe_task(
    agent_id: int,
    task_id: int,
    authed: AgentApiIdentity = Depends(_get_a2a_agent_from_api_key),
) -> StreamingResponse:
    _require_bound_agent(agent_id, authed)
    resolved_agent_id = authed.agent_id
    task = await _fetch_fresh_a2a_task(resolved_agent_id, task_id)
    if task is None:
        raise a2a_error("task_not_found", "Task not found.", status_code=404)
    if task.status in _TERMINAL_STATUSES:
        raise a2a_error(
            "unsupported_operation",
            "A terminal task cannot be subscribed to.",
            status_code=400,
            details={"taskId": task.id},
        )
    return _task_stream_response(resolved_agent_id, task)


@router.post("/agents/{agent_id}/tasks/{task_id}:cancel")
async def cancel_task(
    agent_id: int,
    task_id: int,
    authed: AgentApiIdentity = Depends(_get_a2a_agent_from_api_key),
) -> Any:
    _require_bound_agent(agent_id, authed)
    resolved_agent_id = authed.agent_id
    actor_user_id = authed.user_id
    prepared = await run_db_io_cancellation_safe(
        lambda: _prepare_a2a_cancel_command_sync(
            task_id=task_id,
            agent_id=resolved_agent_id,
            actor_user_id=actor_user_id,
        )
    )

    from .websocket import execute_durable_task_command

    # Apply immediately when this process owns the target run. If another
    # worker owns it, that worker's dispatcher observes the durable row and
    # completes it; polling here preserves the synchronous A2A cancel contract.
    await dispatch_one_task_command(
        execute_durable_task_command,
        command_db_id=prepared.command_db_id,
    )
    deadline = monotonic() + 10.0
    while True:
        stored = await run_db_io_cancellation_safe(
            lambda: load_task_command(prepared.command_db_id)
        )
        if stored is not None and stored.status == COMMAND_COMPLETED:
            task = await _fetch_fresh_a2a_task(resolved_agent_id, task_id)
            if task is None:
                raise a2a_error("task_not_found", "Task not found.", status_code=404)
            return a2a_json_response(task_to_a2a(task))
        if stored is not None and stored.status == COMMAND_FAILED:
            if stored.rejection_reason == "stale_run":
                raise a2a_error(
                    "invalid_request",
                    "Task run changed before cancellation was applied; retry the request.",
                    status_code=409,
                    details={
                        "taskId": task_id,
                        "commandId": prepared.command_identity,
                    },
                )
            raise a2a_error(
                "internal_error",
                str(stored.error or "Task cancellation failed."),
                status_code=500,
            )
        if monotonic() >= deadline:
            raise a2a_error(
                "temporarily_unavailable",
                "Task cancellation was accepted but is still being applied.",
                status_code=503,
                details={
                    "taskId": task_id,
                    "commandId": prepared.command_identity,
                },
            )
        await asyncio.sleep(0.05)


def _prepare_a2a_cancel_command_sync(
    *,
    task_id: int,
    agent_id: int,
    actor_user_id: int,
) -> _A2ACancelPreparation:
    """Validate and persist one cancel command in a worker-owned Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_a2a_task_for_agent_id(db, task_id, agent_id)
        command_identity = f"cancel:{task_id}:{task.run_id or task.state_version}"
        enqueued = enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=actor_user_id,
            command_id=command_identity,
            kind=TaskCommandKind.CANCEL,
            payload={"agent_id": agent_id},
        )
        if not enqueued.payload_matches:
            raise a2a_error(
                "invalid_request",
                "Cancel command identity conflicts with a different request.",
                status_code=409,
            )
        if enqueued.status == COMMAND_FAILED:
            retry_failed_task_command(
                db,
                enqueued.command_id,
                target_run_id=_task_run_id(task),
                target_runner_id=(
                    str(task.runner_id)
                    if task.status == TaskStatus.RUNNING and task.runner_id is not None
                    else None
                ),
            )
        return _A2ACancelPreparation(
            command_db_id=enqueued.command_id,
            command_identity=command_identity,
        )


async def _cancel_task_unserialized(
    *,
    task_id: int,
    agent_id: int,
) -> Any:
    prepared = await run_db_io_cancellation_safe(
        lambda: _prepare_a2a_cancel_sync(task_id, agent_id)
    )
    if prepared is not None:
        return a2a_json_response(prepared)

    from .websocket import background_task_manager

    await background_task_manager.cancel_task(task_id)
    finalized = await run_db_io_cancellation_safe(
        lambda: _finalize_a2a_cancel_sync(task_id, agent_id)
    )
    return a2a_json_response(finalized)


def _prepare_a2a_cancel_sync(
    task_id: int,
    agent_id: int,
) -> dict[str, Any] | None:
    """Validate cancellation in a short worker-owned database session."""
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_a2a_task_for_agent_id(db, task_id, agent_id)
        agent_config: dict[str, Any] = (
            dict(task.agent_config) if isinstance(task.agent_config, dict) else {}
        )
        if agent_config.get("a2a_state") == "TASK_STATE_CANCELED":
            return task_to_a2a(task)
        if task.status in _TERMINAL_STATUSES:
            raise a2a_error(
                "task_not_cancelable",
                "Task is not in a cancelable state.",
                status_code=400,
                details={"taskId": task.id},
            )
        return None


def _finalize_a2a_cancel_sync(task_id: int, agent_id: int) -> dict[str, Any]:
    """Apply cancellation to the fresh task row after runtime shutdown."""
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_a2a_task_for_agent_id(db, task_id, agent_id)
        agent_config: dict[str, Any] = (
            dict(task.agent_config) if isinstance(task.agent_config, dict) else {}
        )
        if agent_config.get("a2a_state") == "TASK_STATE_CANCELED":
            return task_to_a2a(task)
        if task.status in _TERMINAL_STATUSES:
            return task_to_a2a(task)
        agent_config["a2a_state"] = "TASK_STATE_CANCELED"
        setattr(task, "agent_config", agent_config)
        apply_task_control_transition(
            task,
            TaskControlState.FAILED,
            status=TaskStatus.FAILED,
            expected_run_id=_task_run_id(task),
            expected_state_version=int(task.state_version or 0),
        )
        setattr(task, "output", None)
        setattr(task, "error_message", "Task canceled by A2A client.")
        db.commit()
        db.refresh(task)
        return task_to_a2a(task)
