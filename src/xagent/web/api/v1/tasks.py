"""SDK task endpoints: ``/v1/chat/tasks/*`` family.

Phase 1 surface this module owns:

  - POST /v1/chat/tasks
  - POST /v1/chat/tasks/{id}/messages
  - GET  /v1/chat/tasks/{id}
  - GET  /v1/chat/tasks/{id}/steps

All endpoints authenticate via ``get_agent_from_api_key`` and use the
stable ``V1ApiError`` envelope. Task turn lifecycle (claim RUNNING,
persist messages, schedule bg, sync output) is delegated to
``services.task_orchestrator.TaskTurnOrchestrator``, which is also used
by the WebSocket UI path so both transports share one state machine.
"""

import asyncio
import logging
from typing import Any, NoReturn, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from ....core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from ...config import is_allowed_file
from ...models.task import TaskStatus
from ...schemas.v1 import (
    AppendMessageRequest,
    AppendMessageResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    PublicStep,
    StepsResponse,
    TaskInfoResponse,
    UploadedFileInfo,
    UploadFilesResponse,
)
from ...services.api_keys import AgentApiIdentity
from ...services.connector_runtime import (
    pop_ephemeral_runtime_values,
    store_ephemeral_runtime_values,
)
from ...services.db_runtime import (
    drain_async_task_cancellation_safe,
    run_db_io_cancellation_safe,
)
from ...services.file_turn import (
    append_uploaded_files_context,
    build_uploaded_files_context,
    normalize_attachments_for_persistence,
)
from ...services.hot_path_cache import (
    cache_get,
    cache_set,
    cache_version_token,
    task_cache_ttl_seconds,
    task_snapshot_key,
    task_steps_key,
)
from ...services.managed_file_ref import DurableStorageOperationError
from ...services.sdk_task_service import (
    SdkAgentMismatchError,
    SdkTaskNotFoundError,
    SdkTurnFilesMissingError,
    bind_sdk_turn_files_sync,
    load_sdk_task_snapshot_sync,
    load_sdk_task_steps_snapshot_sync,
    load_sdk_task_steps_version_sync,
    mark_pending_sdk_task_failed_sync,
    prepare_append_sdk_task_sync,
    prepare_create_sdk_task_sync,
    resolve_sdk_upload_owner_sync,
)
from ...services.task_orchestrator import (
    TaskTurnError,
    TaskTurnNotFoundError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
)
from ._step_mapping import map_trace_events_to_public_steps
from .deps import get_agent_from_api_key, record_key_usage
from .errors import V1ApiError, V1ErrorCode

router = APIRouter()
logger = logging.getLogger(__name__)

_CONNECTOR_RUNTIME_SETUP_FAILED_MESSAGE = "Connector runtime setup failed."
_TASK_TURN_START_FAILED_MESSAGE = "Task turn start failed."


@router.post("/chat/files", response_model=UploadFilesResponse)
async def upload_task_files(
    files: list[UploadFile] = File(...),
    task_id: Optional[int] = Query(
        default=None,
        gt=0,
        description=(
            "Existing SDK task whose persisted runtime owner should own "
            "the uploaded files."
        ),
    ),
    authed: AgentApiIdentity = Depends(get_agent_from_api_key),
) -> UploadFilesResponse:
    """Store files for later attachment to a task turn.

    API-key-gated counterpart to the JWT-only ``POST /api/files/upload``.
    Files are stored unbound (``UploadedFile.task_id`` NULL); the returned
    ``file_id`` values are passed back in ``message.files`` on
    ``POST /v1/chat/tasks`` (or ``.../messages``), where they get bound to
    the task and exposed to the agent.

    When ``task_id`` is omitted, the upload is owned by the agent's current
    user for a future create request. When ``task_id`` is provided, the task
    is authorized through the key-bound agent and its persisted ``Task.user_id``
    owns the upload. This keeps historical tasks usable after agent ownership
    changes without transferring file ownership during append.
    """
    from ..files import store_v1_uploaded_files

    try:
        upload_owner_user_id = await run_db_io_cancellation_safe(
            lambda: resolve_sdk_upload_owner_sync(
                task_id=task_id,
                authenticated_agent_id=authed.agent_id,
                default_user_id=authed.user_id,
            )
        )
    except SdkTaskNotFoundError as exc:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404) from exc
    if upload_owner_user_id is None:
        raise V1ApiError(V1ErrorCode.INTERNAL_ERROR, 500)

    # Reject unsupported types up front with a clean v1 400. The storage
    # workflow otherwise raises a bare HTTPException that would bypass the v1
    # envelope and leak its internal ``task_type`` wording.
    for uploaded in files:
        if not is_allowed_file(uploaded.filename or "", "general"):
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT,
                400,
                message=f"Unsupported file type: {uploaded.filename}",
            )

    try:
        result = await store_v1_uploaded_files(
            upload_items=list(files),
            task_type="general",
            folder=None,
            owner_user_id=upload_owner_user_id,
            single_file_mode=False,
        )
    except HTTPException as exc:
        # Translate storage-layer HTTPExceptions to the stable v1 envelope.
        # 503 remains retryable; 413 remains 413; other client errors become 400.
        if exc.status_code == 503:
            raise V1ApiError(
                V1ErrorCode.INTERNAL_ERROR,
                503,
                message="File storage is temporarily unavailable.",
            ) from exc
        if 400 <= exc.status_code < 500:
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT,
                413 if exc.status_code == 413 else 400,
                message="File upload rejected.",
            ) from exc
        raise V1ApiError(V1ErrorCode.INTERNAL_ERROR, 500) from exc
    return UploadFilesResponse(
        files=[
            UploadedFileInfo(
                file_id=f["file_id"],
                filename=f["filename"],
                file_size=f["file_size"],
                mime_type=f.get("mime_type"),
            )
            for f in result.get("files", [])
        ]
    )


def _turn_payload(
    content: str, file_infos: list[dict[str, Any]] | tuple[dict[str, Any], ...]
) -> TaskTurnPayload:
    """Build a :class:`TaskTurnPayload`, file-enriching the execution channel.

    Consolidates the payload construction shared by create and append so the
    transcript-vs-execution split can't drift between the two entry points.
    """
    if not file_infos:
        return TaskTurnPayload(transcript_message=content)
    normalized_file_infos = list(file_infos)
    context = build_uploaded_files_context(normalized_file_infos)
    return TaskTurnPayload(
        transcript_message=content,
        execution_message=append_uploaded_files_context(content, context),
        attachments=(
            normalize_attachments_for_persistence(normalized_file_infos) or None
        ),
    )


def _raise_v1_connector_runtime_error(exc: ConnectorRuntimeError) -> NoReturn:
    try:
        code = V1ErrorCode(exc.code)
    except ValueError:
        code = V1ErrorCode.INVALID_RUNTIME_CONTEXT
    raise V1ApiError(
        code,
        exc.status_code,
        message=exc.safe_message,
        details=exc.to_public_error().get("details"),
    ) from exc


async def _store_connector_runtime_values_or_fail(
    *,
    task_id: int,
    agent_id: int,
    owner_user_id: int,
    turn_id: str,
    values_by_ref: dict,
    mark_task_failed: bool,
) -> None:
    try:
        store_ephemeral_runtime_values(turn_id, values_by_ref)
    except Exception as exc:
        pop_ephemeral_runtime_values(turn_id)
        if mark_task_failed:
            await run_db_io_cancellation_safe(
                lambda: mark_pending_sdk_task_failed_sync(
                    task_id,
                    _CONNECTOR_RUNTIME_SETUP_FAILED_MESSAGE,
                    expected_agent_id=agent_id,
                    expected_owner_user_id=owner_user_id,
                )
            )
        logger.warning(
            "Connector runtime setup failed for task %s turn %s",
            task_id,
            turn_id,
        )
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR,
            500,
            message=_CONNECTOR_RUNTIME_SETUP_FAILED_MESSAGE,
        ) from exc


@router.post(
    "/chat/tasks",
    status_code=202,
    response_model=CreateTaskResponse,
)
async def create_chat_task(
    request: CreateTaskRequest,
    authed: AgentApiIdentity = Depends(get_agent_from_api_key),
) -> CreateTaskResponse:
    """Create a new SDK-driven task and kick off its first turn.

    Single endpoint does three things atomically from the caller's
    perspective:

      1. Verifies the body's ``agent_id`` matches the agent bound to
         the presented API key. Mismatch -> 404 ``agent_not_found``
         (404 not 403, so the existence of unrelated agents isn't
         leaked via error code).
      2. Persists a new :class:`Task` row owned by the agent's user,
         with ``source='sdk'``, ``is_visible=False``, and ``input`` set
         to the user message. Also persists the first user message to
         ``task_chat_messages`` so the existing background execution
         path can consume it without special-casing this entry point.
      3. Schedules background execution via
         ``start_task_in_background`` (which uses the same coroutine
         the WebSocket handler does). Returns 202 immediately --
         callers poll ``GET /v1/chat/tasks/{task_id}`` to observe the
         eventual ``completed`` / ``failed`` status.

    Args:
        request: Validated :class:`CreateTaskRequest`. ``message.content``
            is guaranteed non-empty by Pydantic; ``agent_id`` is the
            target agent the SDK caller wants to invoke.
        authed: Detached key-bound agent identity resolved by the auth
            dependency; the single source of truth for what this caller
            may touch.

    Returns:
        :class:`CreateTaskResponse` with the new ``task_id``,
        ``agent_id``, ``status='running'`` (the atomic claim inside
        the handler flips the row from PENDING to RUNNING before the
        response is sent), and ``created_at`` for the caller to
        start polling from.

    Raises:
        V1ApiError 401: missing/invalid/revoked key (raised inside
            ``get_agent_from_api_key``; envelope is uniform with
            other auth failures).
        V1ApiError 404: ``request.agent_id != authed_agent.id``.
        500 (V1 envelope): any unexpected exception -- the global
            handler in ``web/app.py`` translates to
            ``{"error": {"code": "internal_error", ...}}`` and the raw
            exception message stays out of the response.
    """
    # Server-side agent_id consistency check. The key already binds an
    # agent; ``body.agent_id`` is required by the SDK contract for
    # forward-compat (and Python/TS SDK symmetry), but the bound
    # agent is the only authority. Mismatch is a 404 -- never a 403
    # -- so the existence of agent_id=N elsewhere in the system isn't
    # observable to this caller.
    if request.agent_id != authed.agent_id:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)

    operation = asyncio.create_task(_create_chat_task_workflow(request, authed))
    try:
        return await drain_async_task_cancellation_safe(operation)
    except SdkTurnFilesMissingError as exc:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            message="These file ids are not accessible: " + ", ".join(exc.missing),
        ) from exc
    except DurableStorageOperationError as exc:
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR,
            503,
            message="File storage is temporarily unavailable.",
        ) from exc
    except ConnectorRuntimeError as exc:
        _raise_v1_connector_runtime_error(exc)
    except TaskTurnNotFoundError as exc:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404) from exc
    except TaskTurnError as exc:
        raise V1ApiError(V1ErrorCode.TASK_BUSY, 409) from exc


async def _create_chat_task_workflow(
    request: CreateTaskRequest,
    authed: AgentApiIdentity,
) -> CreateTaskResponse:
    """Own a create from first durable write through turn scheduling.

    The route drains this child on caller cancellation.  That prevents a
    committed PENDING task from being abandoned between worker preparation and
    ``begin_turn``.
    """

    prepared = await run_db_io_cancellation_safe(
        lambda: prepare_create_sdk_task_sync(
            agent_id=authed.agent_id,
            task_owner_user_id=authed.user_id,
            actor_user_id=authed.user_id,
            tool_categories=authed.tool_categories,
            content=request.message.content,
            file_ids=tuple(request.message.files or ()),
            connector_runtime_context=tuple(
                item.model_dump() for item in (request.connector_runtime_context or ())
            ),
        )
    )
    payload = _turn_payload(request.message.content, prepared.file_infos)
    await _store_connector_runtime_values_or_fail(
        task_id=prepared.task_id,
        agent_id=prepared.agent_id,
        owner_user_id=prepared.task_owner_user_id,
        turn_id=payload.turn_id,
        values_by_ref=prepared.ephemeral_by_ref,
        mark_task_failed=True,
    )
    try:
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=prepared.task_id,
            task_owner_user_id=prepared.task_owner_user_id,
            actor_user_id=prepared.actor_user_id,
            payload=payload,
            kind=TurnKind.CREATE,
            force_fresh=False,
        )
    except (Exception, asyncio.CancelledError):
        pop_ephemeral_runtime_values(payload.turn_id)
        await run_db_io_cancellation_safe(
            lambda: mark_pending_sdk_task_failed_sync(
                prepared.task_id,
                _TASK_TURN_START_FAILED_MESSAGE,
                expected_agent_id=prepared.agent_id,
                expected_owner_user_id=prepared.task_owner_user_id,
            )
        )
        raise

    await run_db_io_cancellation_safe(
        lambda: bind_sdk_turn_files_sync(
            file_ids=tuple(info["file_id"] for info in prepared.file_infos),
            task_id=prepared.task_id,
            owner_user_id=prepared.task_owner_user_id,
        )
    )
    await record_key_usage(authed.key_prefix)
    return CreateTaskResponse(
        task_id=prepared.task_id,
        agent_id=prepared.agent_id,
        status=started.status.value,
        created_at=prepared.created_at,
        run_id=started.run_id,
        state_version=started.state_version,
        control_state=started.control_state,
    )


# Terminal task statuses for ``completed_at`` derivation in GET task.
# A task in any of these states is no longer running; ``updated_at``
# is the last DB write and thus the closest proxy to "when did the
# task end". For non-terminal states we return ``None`` so SDK
# clients can disambiguate "still running" from "ended at <time>".
_TERMINAL_STATUSES = (TaskStatus.COMPLETED, TaskStatus.FAILED)


@router.post(
    "/chat/tasks/{task_id}/messages",
    status_code=202,
    response_model=AppendMessageResponse,
)
async def append_message_to_task(
    task_id: int,
    request: AppendMessageRequest,
    authed: AgentApiIdentity = Depends(get_agent_from_api_key),
) -> AppendMessageResponse:
    """Append the next user message to an existing task and kick off its next turn.

    Phase 1 multi-turn model is task-centric: subsequent user inputs
    extend the same ``task_id`` rather than creating a new task or a
    new ``conversation_id``. This endpoint:

      1. Validates the path ``task_id`` exists and belongs to the
         key-bound agent (404 ``task_not_found`` otherwise).
      2. Validates ``body.agent_id`` matches the key-bound agent
         (404 ``agent_not_found`` otherwise).
      3. Rejects the call with 409 ``task_busy`` if the task is
         currently ``RUNNING`` -- the SDK client should poll
         ``GET /v1/chat/tasks/{id}`` until status leaves RUNNING and
         retry.
      4. Otherwise persists the new user message to
         ``task_chat_messages``, updates ``task.input`` to record
         this turn's input, and kicks off the next background turn
         via the same helper POST uses.

    Args:
        task_id: Path parameter; the target task's primary key.
        request: Validated :class:`AppendMessageRequest`. ``message.content``
            is guaranteed non-empty by Pydantic.
        authed: Detached key-bound agent identity from the auth dependency.

    Returns:
        :class:`AppendMessageResponse` with the task identity and an
        ``accepted_at`` timestamp.

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task not found OR not owned by the agent OR
            body.agent_id doesn't match the bound agent.
        V1ApiError 409: ``task_busy`` -- task currently RUNNING.
        500: any other unexpected error (V1 envelope via global handler).
    """
    operation = asyncio.create_task(
        _append_message_to_task_workflow(task_id, request, authed)
    )
    try:
        return await drain_async_task_cancellation_safe(operation)
    except SdkTaskNotFoundError as exc:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404) from exc
    except SdkAgentMismatchError as exc:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404) from exc
    except SdkTurnFilesMissingError as exc:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            message="These file ids are not accessible: " + ", ".join(exc.missing),
        ) from exc
    except DurableStorageOperationError as exc:
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR,
            503,
            message="File storage is temporarily unavailable.",
        ) from exc
    except ConnectorRuntimeError as exc:
        _raise_v1_connector_runtime_error(exc)
    except TaskTurnNotFoundError as exc:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404) from exc
    except TaskTurnError as exc:
        raise V1ApiError(V1ErrorCode.TASK_BUSY, 409) from exc


async def _append_message_to_task_workflow(
    task_id: int,
    request: AppendMessageRequest,
    authed: AgentApiIdentity,
) -> AppendMessageResponse:
    """Own append preparation and claim through caller cancellation."""

    prepared = await run_db_io_cancellation_safe(
        lambda: prepare_append_sdk_task_sync(
            task_id=task_id,
            authenticated_agent_id=authed.agent_id,
            actor_user_id=authed.user_id,
            requested_agent_id=request.agent_id,
            file_ids=tuple(request.message.files or ()),
            connector_runtime_context=tuple(
                item.model_dump() for item in (request.connector_runtime_context or ())
            ),
        )
    )
    payload = _turn_payload(request.message.content, prepared.file_infos)
    await _store_connector_runtime_values_or_fail(
        task_id=task_id,
        agent_id=prepared.agent_id,
        owner_user_id=prepared.task_owner_user_id,
        turn_id=payload.turn_id,
        values_by_ref=prepared.ephemeral_by_ref,
        mark_task_failed=False,
    )
    try:
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=task_id,
            task_owner_user_id=prepared.task_owner_user_id,
            actor_user_id=prepared.actor_user_id,
            payload=payload,
            kind=TurnKind.APPEND,
            force_fresh=False,
        )
    except (Exception, asyncio.CancelledError):
        pop_ephemeral_runtime_values(payload.turn_id)
        raise

    await run_db_io_cancellation_safe(
        lambda: bind_sdk_turn_files_sync(
            file_ids=tuple(info["file_id"] for info in prepared.file_infos),
            task_id=task_id,
            owner_user_id=prepared.task_owner_user_id,
        )
    )
    await record_key_usage(authed.key_prefix)
    return AppendMessageResponse(
        task_id=task_id,
        agent_id=prepared.agent_id,
        status=started.status.value,
        accepted_at=started.updated_at,
        run_id=started.run_id,
        state_version=started.state_version,
        control_state=started.control_state,
    )


@router.get("/chat/tasks/{task_id}", response_model=TaskInfoResponse)
async def get_chat_task(
    task_id: int,
    authed: AgentApiIdentity = Depends(get_agent_from_api_key),
) -> TaskInfoResponse:
    """Return a snapshot of one task's current state.

    SDK clients call this to poll a previously-submitted task for
    its status, latest output, or failure reason. The shape is
    deliberately flat -- detailed step-by-step execution data lives
    behind ``GET /v1/chat/tasks/{task_id}/steps`` (commit F).

    Args:
        task_id: Path parameter; the target task's primary key.
        authed: Detached key-bound agent identity.

    Returns:
        :class:`TaskInfoResponse` with ``task_id``, ``agent_id``,
        ``status``, latest-turn ``input`` / ``output`` / ``error``,
        ``created_at``, and ``completed_at`` (set only when the task
        has reached a terminal state).

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task missing or not owned by the calling agent.
    """
    task = await run_db_io_cancellation_safe(
        lambda: load_sdk_task_snapshot_sync(task_id, authed.agent_id)
    )
    if task is None:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)

    # completed_at is derived from updated_at when the task is in a
    # terminal state. Pre-terminal states return None so SDK clients
    # don't mis-interpret an in-flight task's last write timestamp as
    # a completion time.
    completed_at = task.updated_at if task.status in _TERMINAL_STATUSES else None
    cache_key = task_snapshot_key(task_id)
    task_updated_at = cache_version_token(task.updated_at)
    cached = cache_get(cache_key)
    if isinstance(cached, dict) and cached.get("updated_at") == task_updated_at:
        return TaskInfoResponse.model_validate(cached["response"])

    response = TaskInfoResponse(
        task_id=task.task_id,
        agent_id=task.agent_id,
        status=task.status.value,
        run_id=task.run_id,
        state_version=int(task.state_version or 0),
        control_state=str(task.control_state or "idle"),
        input=task.input,
        output=task.output,
        error=task.error_message,
        created_at=task.created_at,
        completed_at=completed_at,
    )
    cache_set(
        cache_key,
        {
            "updated_at": task_updated_at,
            "response": response.model_dump(mode="json"),
        },
        ttl_seconds=task_cache_ttl_seconds(),
    )
    return response


@router.get("/chat/tasks/{task_id}/steps", response_model=StepsResponse)
async def get_chat_task_steps(
    task_id: int,
    authed: AgentApiIdentity = Depends(get_agent_from_api_key),
) -> StepsResponse:
    """Return the public-timeline steps for a task.

    Pulls all :class:`TraceEvent` rows for the task in DB order, then
    collapses them via :func:`map_trace_events_to_public_steps` into
    the 4 stable public step types: ``thinking``, ``tool_call``,
    ``agent_delegation``, ``message``.

    The internal trace event taxonomy has ~32 ``event_type`` strings
    today; SDK callers see only the 4 types listed above. Internal
    events not on the public allow-list (LLM calls, memory ops,
    visualization ticks, DAG bookkeeping) are silently dropped --
    intentionally, so internal trace evolution doesn't break the SDK
    contract.

    Args:
        task_id: Path parameter; the target task's primary key.
        authed: Detached key-bound agent identity resolved by the auth
            dependency.

    Returns:
        :class:`StepsResponse` with ``task_id``, ``agent_id``, and the
        steps array in ``started_at`` ascending order. In-flight steps
        appear with ``status='running'`` and ``completed_at=null`` so
        SDK clients can poll this endpoint and observe progress.

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task missing or not owned by the calling agent.
    """
    version = await run_db_io_cancellation_safe(
        lambda: load_sdk_task_steps_version_sync(task_id, authed.agent_id)
    )
    if version is None:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)
    cache_key = task_steps_key(task_id)
    cached = cache_get(cache_key)
    if isinstance(cached, dict) and cached.get("max_event_id") == version.max_event_id:
        return StepsResponse.model_validate(cached["response"])

    snapshot = await run_db_io_cancellation_safe(
        lambda: load_sdk_task_steps_snapshot_sync(task_id, authed.agent_id)
    )
    if snapshot is None:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)

    # Pure mapping -- testable in isolation via
    # tests/web/api/v1/test_steps_mapping.py without spinning up a
    # FastAPI app or DB session.
    public_steps_data = map_trace_events_to_public_steps(list(snapshot.events))

    response = StepsResponse(
        task_id=snapshot.task_id,
        agent_id=snapshot.agent_id,
        steps=[PublicStep(**step) for step in public_steps_data],
    )
    cache_set(
        cache_key,
        {
            "max_event_id": snapshot.max_event_id,
            "response": response.model_dump(mode="json"),
        },
        ttl_seconds=task_cache_ttl_seconds(),
    )
    return response
