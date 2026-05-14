"""Stable error schema for the ``/v1/*`` SDK surface.

Every error response on the SDK surface is shaped like:

    {
        "error": {
            "code": "<stable enum>",
            "message": "<human-readable, may change>"
        }
    }

The ``code`` is the contract SDK clients pin against; ``message`` is
free text that may be updated for clarity without breaking clients.

Why a separate error envelope from ``/api/*``:
    ``/api/*`` uses FastAPI's default ``{"detail": "..."}`` shape, which
    is fine for an in-house web UI but a poor SDK surface -- clients
    would have to string-match against ``detail`` to distinguish "wrong
    key" from "no such task" from "rate-limited". Coding the cause in a
    stable enum lets each SDK language map server errors to typed
    exceptions deterministically.
"""

from enum import Enum

from fastapi import Request
from fastapi.responses import JSONResponse


class V1ErrorCode(str, Enum):
    """Stable error codes for ``/v1/*`` responses.

    Values are the on-the-wire strings; SDK clients pin against these.
    Adding new codes is allowed; renaming or removing existing ones is
    a breaking change.
    """

    # Authentication failures (401). One opaque code covers
    # "missing header", "format wrong", "prefix not found", "secret
    # wrong", and "key revoked" so attackers cannot distinguish them
    # by error code (we already keep the timing constant via
    # verify_dummy).
    INVALID_API_KEY = "invalid_api_key"

    # Caller is authenticated but the agent_id they asked for is not
    # bound to their key. Returned as 404 (not 403) so the existence
    # of other tenants' agents is not leaked.
    AGENT_NOT_FOUND = "agent_not_found"

    # Caller is authenticated and agent_id is right, but the task_id
    # they asked for doesn't exist or doesn't belong to their agent.
    # Also 404 for the same leak-prevention reason.
    TASK_NOT_FOUND = "task_not_found"

    # The task is currently running and cannot accept new input.
    # 409 Conflict. Client should retry after polling task status.
    TASK_BUSY = "task_busy"

    # Request body failed Pydantic validation (empty content, wrong role,
    # missing required field, etc.). 422. The default ``{"detail": [...]}``
    # FastAPI shape is rewritten to this enum on the /v1/* path so SDK
    # clients always switch on ``body.error.code``.
    INVALID_INPUT = "invalid_input"

    # Phase 2 feature; reserved here so SDK clients can already encode
    # the mapping. Phase 1 never emits this.
    RATE_LIMITED = "rate_limited"

    # Server-side bug. Detail is sanitized; the raw exception stays in
    # the server log.
    INTERNAL_ERROR = "internal_error"


# Default human-readable text per code. Endpoints may override the
# message when more specific context is available (e.g. surfacing
# ``"task is running"`` vs the default ``"task is busy"``).
_DEFAULT_MESSAGES: dict[V1ErrorCode, str] = {
    V1ErrorCode.INVALID_API_KEY: "Invalid or revoked API key.",
    V1ErrorCode.AGENT_NOT_FOUND: "Agent not found or not accessible with this key.",
    V1ErrorCode.TASK_NOT_FOUND: "Task not found or not accessible with this key.",
    V1ErrorCode.TASK_BUSY: "Task is currently running; retry after it completes.",
    V1ErrorCode.INVALID_INPUT: "Request body failed validation.",
    V1ErrorCode.RATE_LIMITED: "Rate limit exceeded; retry later.",
    V1ErrorCode.INTERNAL_ERROR: "Internal server error.",
}


class V1ApiError(Exception):
    """Raise from any ``/v1/*`` handler to return a stable error response.

    Use this instead of ``HTTPException`` so the response envelope is
    consistent across the SDK surface. The exception handler in
    ``web/app.py`` translates this to:

        HTTP <http_status>
        {"error": {"code": <code>, "message": <message>}}

    Args:
        code: One of the :class:`V1ErrorCode` enum values.
        http_status: HTTP status code to send (401 / 404 / 409 / 429 / 500
            per code, but the endpoint chooses, not the code).
        message: Optional override of the default human-readable text.
    """

    def __init__(
        self,
        code: V1ErrorCode,
        http_status: int,
        message: str | None = None,
    ) -> None:
        super().__init__(message or _DEFAULT_MESSAGES[code])
        self.code = code
        self.http_status = http_status
        self.message = message or _DEFAULT_MESSAGES[code]


async def v1_api_error_handler(_request: Request, exc: V1ApiError) -> JSONResponse:
    """FastAPI exception handler -- mount via ``app.add_exception_handler``."""
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": {"code": exc.code.value, "message": exc.message}},
    )
