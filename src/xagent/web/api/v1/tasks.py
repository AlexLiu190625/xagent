"""SDK task endpoints: ``/v1/chat/tasks/*`` family.

Phase 1 surface (this module owns):

  - POST /v1/chat/tasks                (commit D, this commit)
  - POST /v1/chat/tasks/{id}/messages  (commit E)
  - GET  /v1/chat/tasks/{id}           (commit E)
  - GET  /v1/chat/tasks/{id}/steps     (commit F)

All endpoints authenticate via ``get_agent_from_api_key`` and use the
stable ``V1ApiError`` envelope. Background execution is started via
``web/services/task_execution.start_task_in_background``, which wraps
the same coroutine the WebSocket handler uses without modifying the
WS path.
"""

from datetime import datetime, timezone
from typing import Tuple

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...models.agent import Agent
from ...models.agent_api_key import AgentApiKey
from ...models.database import get_db
from ...models.task import Task, TaskStatus
from ...schemas.v1 import (
    AppendMessageRequest,
    AppendMessageResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    TaskInfoResponse,
)
from ...services.chat_history_service import persist_user_message
from ...services.task_execution import start_task_in_background
from .deps import get_agent_from_api_key
from .errors import V1ApiError, V1ErrorCode

router = APIRouter()


@router.post(
    "/chat/tasks",
    status_code=202,
    response_model=CreateTaskResponse,
)
async def create_chat_task(
    request: CreateTaskRequest,
    authed: Tuple[Agent, AgentApiKey] = Depends(get_agent_from_api_key),
    db: Session = Depends(get_db),
) -> CreateTaskResponse:
    """Create a new SDK-driven task and kick off its first turn.

    Single endpoint does three things atomically from the caller's
    perspective:

      1. Verifies the body's ``agent_id`` matches the agent bound to
         the presented API key. Mismatch -> 404 ``agent_not_found``
         (404 not 403, so the existence of unrelated agents isn't
         leaked via error code).
      2. Persists a new :class:`Task` row owned by the agent's user,
         with ``source='sdk'`` and ``input`` set to the user message.
         Also persists the first user message to
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
        authed: ``(Agent, AgentApiKey)`` tuple resolved by the auth
            dependency. The agent here is the *key-bound* agent, the
            single source of truth for what this caller may touch.
        db: SQLAlchemy session.

    Returns:
        :class:`CreateTaskResponse` with the new ``task_id``,
        ``agent_id``, ``status='pending'``, and ``created_at`` for the
        caller to start polling from.

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
    agent, _key = authed

    # Server-side agent_id consistency check. The key already binds an
    # agent; ``body.agent_id`` is required by the SDK contract for
    # forward-compat (and Python/TS SDK symmetry), but the bound
    # agent is the only authority. Mismatch is a 404 -- never a 403
    # -- so the existence of agent_id=N elsewhere in the system isn't
    # observable to this caller.
    if request.agent_id != agent.id:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)

    # title is what the web UI shows in its task list. Truncate to
    # 50 chars (matches the WS handler convention) so very long
    # user inputs don't fill the sidebar with a one-line wall of
    # text. The full message is preserved in ``description`` /
    # ``input`` / ``task_chat_messages``.
    title = request.message.content[:50] or "SDK task"

    # Single transaction: create the Task row with SDK-specific
    # fields populated. ``source='sdk'`` lets adoption metrics
    # queries split SDK traffic from web/widget; ``input`` records
    # this turn's user message so the GET endpoint can return it
    # without going through ``task_chat_messages``.
    task = Task(
        user_id=agent.user_id,
        title=title,
        description=request.message.content,
        status=TaskStatus.PENDING,
        agent_id=agent.id,
        input=request.message.content,
        source="sdk",
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # Persist the first message to ``task_chat_messages`` so:
    #   1. Background execution can build the conversation context
    #      the same way the WS path does.
    #   2. Future ``GET /v1/chat/tasks/{id}/steps`` includes it as a
    #      ``message`` (role='user') step.
    persist_user_message(
        db=db,
        task_id=int(task.id),
        user_id=int(agent.user_id),
        content=request.message.content,
    )

    # Kick off the background coroutine. The helper handles
    # registration with ``background_task_manager`` so a follow-up
    # POST on the same task waits for this turn to finish.
    await start_task_in_background(
        task=task,
        user_message=request.message.content,
        user=task.user,
        db=db,
    )

    return CreateTaskResponse(
        task_id=int(task.id),
        agent_id=int(agent.id),
        status="pending",
        created_at=task.created_at,
    )


# Terminal task statuses for ``completed_at`` derivation in GET task.
# A task in any of these states is no longer running; ``updated_at``
# is the last DB write and thus the closest proxy to "when did the
# task end". For non-terminal states we return ``None`` so SDK
# clients can disambiguate "still running" from "ended at <time>".
_TERMINAL_STATUSES = (TaskStatus.COMPLETED, TaskStatus.FAILED)


def _resolve_task_or_404(task_id: int, agent: Agent, db: Session) -> Task:
    """Resolve a task_id against the calling agent's ownership.

    Returns the :class:`Task` row when the task exists AND belongs to
    ``agent``. Any other case (missing row, row belongs to a different
    agent) raises :class:`V1ApiError` with ``task_not_found`` -- 404
    not 403, so the existence of tasks under other agents isn't
    observable through error code.

    Args:
        task_id: Path parameter from the route.
        agent: The key-bound agent resolved by
            ``get_agent_from_api_key``.
        db: SQLAlchemy session.

    Raises:
        V1ApiError(TASK_NOT_FOUND, 404): task missing OR not owned by
            the calling agent.
    """
    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None or task.agent_id != agent.id:
        raise V1ApiError(V1ErrorCode.TASK_NOT_FOUND, 404)
    return task


@router.post(
    "/chat/tasks/{task_id}/messages",
    status_code=202,
    response_model=AppendMessageResponse,
)
async def append_message_to_task(
    task_id: int,
    request: AppendMessageRequest,
    authed: Tuple[Agent, AgentApiKey] = Depends(get_agent_from_api_key),
    db: Session = Depends(get_db),
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
        authed: ``(Agent, AgentApiKey)`` from the auth dependency.
        db: SQLAlchemy session.

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
    agent, _key = authed

    # Resolve task first so cross-agent leak protection (404 instead
    # of 403 for "not yours") fires before any body-level checks.
    task = _resolve_task_or_404(task_id, agent, db)

    # body.agent_id mismatch is also a 404 -- but agent_not_found,
    # not task_not_found, because that's the field the caller got
    # wrong. Choosing AGENT_NOT_FOUND keeps it consistent with the
    # POST /v1/chat/tasks behavior for the same condition.
    if request.agent_id != agent.id:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)

    # Concurrency guard: appending a message while the prior turn is
    # still running would race the background coroutine's read of
    # ``task_chat_messages``. Return 409 so the SDK can poll status
    # and retry once we're past RUNNING.
    if task.status == TaskStatus.RUNNING:
        raise V1ApiError(V1ErrorCode.TASK_BUSY, 409)

    # Persist the new user message + update task.input to reflect the
    # latest turn (matches POST /v1/chat/tasks' write to task.input).
    persist_user_message(
        db=db,
        task_id=int(task.id),
        user_id=int(agent.user_id),
        content=request.message.content,
    )
    task.input = request.message.content  # type: ignore[assignment]
    db.commit()
    db.refresh(task)

    # Kick off the next background turn. Uses the same helper as
    # POST /v1/chat/tasks; background_task_manager ensures we wait
    # for any in-flight task before the new one starts.
    await start_task_in_background(
        task=task,
        user_message=request.message.content,
        user=task.user,
        db=db,
    )

    return AppendMessageResponse(
        task_id=int(task.id),
        agent_id=int(agent.id),
        status="pending",
        accepted_at=datetime.now(timezone.utc),
    )


@router.get("/chat/tasks/{task_id}", response_model=TaskInfoResponse)
async def get_chat_task(
    task_id: int,
    authed: Tuple[Agent, AgentApiKey] = Depends(get_agent_from_api_key),
    db: Session = Depends(get_db),
) -> TaskInfoResponse:
    """Return a snapshot of one task's current state.

    SDK clients call this to poll a previously-submitted task for
    its status, latest output, or failure reason. The shape is
    deliberately flat -- detailed step-by-step execution data lives
    behind ``GET /v1/chat/tasks/{task_id}/steps`` (commit F).

    Args:
        task_id: Path parameter; the target task's primary key.
        authed: ``(Agent, AgentApiKey)`` tuple.
        db: SQLAlchemy session.

    Returns:
        :class:`TaskInfoResponse` with ``task_id``, ``agent_id``,
        ``status``, latest-turn ``input`` / ``output`` / ``error``,
        ``created_at``, and ``completed_at`` (set only when the task
        has reached a terminal state).

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task missing or not owned by the calling agent.
    """
    agent, _key = authed
    task = _resolve_task_or_404(task_id, agent, db)

    # completed_at is derived from updated_at when the task is in a
    # terminal state. Pre-terminal states return None so SDK clients
    # don't mis-interpret an in-flight task's last write timestamp as
    # a completion time.
    completed_at = task.updated_at if task.status in _TERMINAL_STATUSES else None

    return TaskInfoResponse(
        task_id=int(task.id),
        agent_id=int(task.agent_id),
        status=task.status.value,
        input=task.input,
        output=task.output,
        error=task.error_message,
        created_at=task.created_at,
        completed_at=completed_at,
    )
