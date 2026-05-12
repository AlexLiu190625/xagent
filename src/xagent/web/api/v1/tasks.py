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

from typing import Tuple

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...models.agent import Agent
from ...models.agent_api_key import AgentApiKey
from ...models.database import get_db
from ...models.task import Task, TaskStatus
from ...schemas.v1 import CreateTaskRequest, CreateTaskResponse
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
