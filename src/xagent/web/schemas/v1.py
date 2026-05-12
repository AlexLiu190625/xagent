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
