"""Pydantic request/response models for the ``/v1/chat/tasks/*`` SDK endpoints.

Kept in one module because the shapes are small, cross-referential
(CreateTask / AppendMessage both nest the same ``MessageBody``), and
will all be regenerated as TypeScript / Python SDK types from the
OpenAPI schema together.

Design notes:

  - ``MessageBody.role`` is currently fixed to ``"user"`` (SDK callers
    only push user input on this surface). We accept it as a field
    rather than hard-coding for forward-compatibility with future
    system / function message roles, but reject anything else at
    validation time.

  - ``metadata`` is a free-form passthrough dict the SaaS caller can
    use to round-trip its own correlation IDs (trace_id, request_id,
    etc). We don't interpret it server-side in Phase 1 but persist
    enough of the SDK call shape to support future debugging.

  - Timestamps are tz-aware ``datetime`` so SDK clients deserialize
    into proper datetimes (``datetime.fromisoformat`` works on both
    PG ``timestamptz`` and the ISO 8601 the FastAPI default
    serializer emits).
"""

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class MessageBody(BaseModel):
    """One chat message in the SDK request body.

    Currently the SDK surface only accepts ``role='user'`` -- the SDK
    is for SaaS clients pushing user input, not for replaying
    transcripts. Future-proofed as a string so we don't have to break
    the wire shape when adding ``system`` / ``function`` later.
    """

    role: Literal["user"] = Field(
        default="user",
        description=(
            "Currently must be 'user'. Reserved as a field for future "
            "expansion (system / function roles) without breaking the "
            "wire shape."
        ),
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The user's message text. Must be non-empty.",
    )


class CreateTaskRequest(BaseModel):
    """Body for ``POST /v1/chat/tasks``.

    ``agent_id`` is required and must match the agent bound to the
    presented API key; the server enforces ``body.agent_id ==
    authed.agent.id`` and returns 404 ``agent_not_found`` on mismatch.
    """

    agent_id: int = Field(
        ...,
        description=(
            "Target agent's primary key. Must match the agent the "
            "presented API key is bound to."
        ),
    )
    message: MessageBody = Field(..., description="First user message of the task.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Free-form correlation data the SDK caller can pass through "
            "(trace_id, request_id, etc). Not interpreted server-side "
            "in Phase 1."
        ),
    )


class CreateTaskResponse(BaseModel):
    """``POST /v1/chat/tasks`` -> 202 Accepted response.

    The task has been persisted and queued for background execution;
    callers poll ``GET /v1/chat/tasks/{task_id}`` to observe the
    transition pending -> running -> completed/failed.
    """

    task_id: int = Field(..., description="Newly created task primary key.")
    agent_id: int = Field(..., description="Agent the task is bound to.")
    status: str = Field(
        ...,
        description=(
            "Initial status, always 'pending' in the 202 response. "
            "Use GET /v1/chat/tasks/{task_id} to observe later transitions."
        ),
    )
    created_at: datetime = Field(..., description="UTC creation timestamp.")


class AppendMessageRequest(BaseModel):
    """Body for ``POST /v1/chat/tasks/{task_id}/messages``.

    Same shape as :class:`CreateTaskRequest` minus the lack of a
    ``metadata`` field by default -- callers append a new user
    message to an existing task. ``agent_id`` is required again
    (consistent with the SDK contract: every write carries the
    agent_id explicitly for forward-compat with multi-agent keys).
    """

    agent_id: int = Field(
        ...,
        description=(
            "Target agent's primary key. Must match the agent the "
            "presented API key is bound to and the task's agent_id."
        ),
    )
    message: MessageBody = Field(..., description="Next user message in the task.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Free-form correlation data passed through unchanged.",
    )


class AppendMessageResponse(BaseModel):
    """``POST /v1/chat/tasks/{task_id}/messages`` -> 202 Accepted response.

    The new user message has been persisted and the next turn queued
    for background execution; callers poll the same way they would
    after the initial POST /v1/chat/tasks.
    """

    task_id: int = Field(..., description="Existing task primary key.")
    agent_id: int = Field(..., description="Agent the task is bound to.")
    status: str = Field(
        ...,
        description="Initial status of the new turn, always 'pending'.",
    )
    accepted_at: datetime = Field(
        ...,
        description=(
            "UTC timestamp when the server accepted the message and "
            "scheduled background execution. Not the message's stored "
            "created_at (which may differ slightly due to DB clock)."
        ),
    )


class TaskInfoResponse(BaseModel):
    """``GET /v1/chat/tasks/{task_id}`` response.

    Returns a snapshot of the task's current state from the ``tasks``
    row. ``input`` / ``output`` / ``error`` reflect the **latest** turn
    only -- full transcript history is queryable via the ``/steps``
    endpoint's ``message`` type steps.
    """

    task_id: int
    agent_id: int
    status: str = Field(
        ...,
        description="One of: pending / running / paused / completed / failed.",
    )
    input: Optional[str] = Field(
        None,
        description="Latest-turn user input. Null if no message yet recorded.",
    )
    output: Optional[str] = Field(
        None,
        description=(
            "Latest-turn assistant output. Populated when status reaches "
            "'completed'; null while running or pending."
        ),
    )
    error: Optional[str] = Field(
        None,
        description="Last failure reason when status='failed'.",
    )
    created_at: datetime
    completed_at: Optional[datetime] = Field(
        None,
        description=(
            "UTC timestamp when the task reached a terminal state "
            "(completed or failed). Null while still running."
        ),
    )
