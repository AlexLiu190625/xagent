"""Authentication dependencies for ``/v1/*`` and authenticated A2A routes.

Synchronous SQLAlchemy pool waits, queries, and bcrypt checks run as one
cancellation-safe worker operation. The worker owns its Session from creation
through close, closes it before bcrypt begins, and returns only immutable
detached identity values.
"""

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ...services.api_keys import (
    AgentApiIdentity,
    InvalidApiKeyError,
    PersonalApiIdentity,
    authenticate_agent_api_key,
    authenticate_personal_api_key,
    record_agent_api_key_usage,
)
from ...services.db_runtime import run_db_io_cancellation_safe
from .errors import V1ApiError, V1ErrorCode

# ``auto_error=False`` lets every failure use the stable v1 error envelope.
_bearer = HTTPBearer(auto_error=False)


def _raw_credentials(
    credentials: HTTPAuthorizationCredentials | None,
) -> str | None:
    return credentials.credentials if credentials is not None else None


async def get_agent_from_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AgentApiIdentity:
    """Resolve an agent runtime key to a detached identity snapshot."""

    raw = _raw_credentials(credentials)
    try:
        return await run_db_io_cancellation_safe(
            lambda: authenticate_agent_api_key(raw)
        )
    except InvalidApiKeyError as exc:
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401) from exc


async def get_user_from_personal_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> PersonalApiIdentity:
    """Resolve a personal management key to a detached identity snapshot."""

    raw = _raw_credentials(credentials)
    try:
        return await run_db_io_cancellation_safe(
            lambda: authenticate_personal_api_key(raw)
        )
    except InvalidApiKeyError as exc:
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401) from exc


async def record_key_usage(key_prefix: str) -> None:
    """Best-effort usage tracking without blocking the event loop.

    This remains an explicit endpoint action rather than part of auth so
    read-only status polling is not counted as an invocation.
    """

    await run_db_io_cancellation_safe(lambda: record_agent_api_key_usage(key_prefix))
