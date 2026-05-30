"""Auth dependency for ``/v1/*`` endpoints.

``get_agent_from_api_key`` resolves the ``Authorization: Bearer
xag_<prefix>_<secret>`` header to the bound :class:`Agent` row,
returning ``(Agent, AgentApiKey)`` on success and raising
:class:`V1ApiError` ``invalid_api_key`` (HTTP 401) on every failure
path.

Failure paths intentionally share the same response code and burn the
same ~100ms of bcrypt work (via :func:`verify_dummy`) regardless of
which check failed:

  - Missing / malformed header
  - Prefix not in DB
  - Secret doesn't match the stored bcrypt hash
  - Bound agent row missing (orphan key, shouldn't happen but defended)

Without this symmetry, an attacker could enumerate live prefixes by
timing the response (slow = bcrypt ran = prefix exists; fast = index
miss = prefix doesn't exist). See SDK design doc §7 (key format and
auth flow) and §10 (security considerations).
"""

from datetime import datetime, timezone
from typing import Tuple

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session, joinedload

from ....core.utils.api_key import (
    ApiKeyKind,
    parse_api_key,
    verify_api_key,
    verify_dummy,
)
from ...models.agent import Agent, is_workforce_generated_manager_agent
from ...models.agent_api_key import AgentApiKey
from ...models.database import get_db
from ...models.user import User
from ...models.user_api_key import UserApiKey
from ...utils.db_timezone import normalize_datetime_from_db
from .errors import V1ApiError, V1ErrorCode

# ``auto_error=False`` so we can raise our own V1ApiError envelope
# instead of FastAPI's default 403 ``{"detail": "Not authenticated"}``
# when the header is missing -- the SDK contract is one error shape.
_bearer = HTTPBearer(auto_error=False)


async def get_agent_from_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Tuple[Agent, AgentApiKey]:
    """Resolve an SDK API key to ``(Agent, AgentApiKey)``.

    All failure paths spend ~100ms of bcrypt work and return the same
    ``invalid_api_key`` code, so an attacker cannot enumerate live
    prefixes by either error code or response timing.

    Returns:
        Tuple ``(agent, key_row)`` where ``agent`` is the bound Agent
        ORM row and ``key_row`` is the matching active AgentApiKey row.
        Use the key_row for downstream operations that need to know
        which prefix was presented (e.g. ``/v1/me``).

    Raises:
        V1ApiError(INVALID_API_KEY, 401): on any auth failure -- one
            opaque code so caller cannot distinguish *which* check
            failed.
    """
    # Missing header. Still burn bcrypt time so a curl-with-no-header
    # response can't be timed apart from a curl-with-bad-secret one.
    if credentials is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    raw = credentials.credentials

    # Format check. parse_api_key returns None for anything not shaped
    # like ``xag_<6 alnum>_<32 alnum>``.
    parsed = parse_api_key(raw)
    if parsed is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    if parsed.kind != ApiKeyKind.AGENT:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    prefix = parsed.prefix

    # Index lookup is O(1) on ix_agent_api_keys_key_prefix and excludes
    # revoked rows via the partial unique index path. ``joinedload``
    # pulls the bound Agent row in the same SELECT so we don't pay a
    # second round-trip on the success path (the relationship defaults
    # to lazy='select', which would otherwise emit a separate query
    # when we access ``key_row.agent`` below).
    key_row = (
        db.query(AgentApiKey)
        .options(joinedload(AgentApiKey.agent))
        .filter(
            AgentApiKey.key_prefix == prefix,
            AgentApiKey.revoked_at.is_(None),
        )
        .first()
    )

    # Prefix missing or revoked. verify_dummy to keep timing
    # indistinguishable from the "secret wrong" branch below.
    if key_row is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    # Real bcrypt check on the full key. Constant time within the cost
    # factor; ``verify_api_key`` returns False (not raise) on malformed
    # hash inputs.
    if not verify_api_key(raw, key_row.key_hash):  # type: ignore[arg-type]
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    # Bound agent. Should always exist (CASCADE on agents -> api_keys),
    # but defend against an out-of-band DELETE that somehow leaves a
    # dangling key row. The relationship was eagerly loaded above, so
    # this is a memory access, not another query. Treat as
    # invalid_api_key rather than 404 so the client retries with a
    # fresh key instead of investigating an imaginary agent.
    agent = key_row.agent
    if agent is None or is_workforce_generated_manager_agent(agent):
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    return agent, key_row


async def get_user_from_personal_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Tuple[User, UserApiKey]:
    """Resolve a personal management API key to ``(User, UserApiKey)``."""
    if credentials is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    raw = credentials.credentials
    parsed = parse_api_key(raw)
    if parsed is None or parsed.kind != ApiKeyKind.PERSONAL:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    key_row = (
        db.query(UserApiKey)
        .options(joinedload(UserApiKey.user))
        .filter(
            UserApiKey.key_prefix == parsed.prefix,
            UserApiKey.revoked_at.is_(None),
        )
        .first()
    )
    if key_row is None:
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    now = datetime.now(timezone.utc)
    expires_at = key_row.expires_at
    # ``DateTime(timezone=True)`` reads back naive on SQLite; normalize
    # to aware UTC before comparing so an expired key yields 401, not a
    # 500 from comparing naive vs aware datetimes.
    if (
        expires_at is not None and normalize_datetime_from_db(expires_at) <= now  # type: ignore[arg-type]
    ):
        verify_dummy()
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    if not verify_api_key(raw, key_row.key_hash):  # type: ignore[arg-type]
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    user = key_row.user
    if user is None:
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)

    return user, key_row
