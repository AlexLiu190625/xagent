"""Authentication dependencies for user segregation."""

from enum import Enum, auto
from typing import NoReturn, Optional, assert_never

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt
from sqlalchemy.orm import Session

from .auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from .models.database import get_db
from .models.user import User

# JWT Bearer token authentication
security = HTTPBearer()

_SQLITE_INTEGER_MIN = -(2**63)
_SQLITE_INTEGER_MAX = 2**63 - 1
_POSTGRESQL_INTEGER_MIN = -(2**31)
_POSTGRESQL_INTEGER_MAX = 2**31 - 1


class _AccessTokenRejectionReason(Enum):
    """Credential rejection categories owned by access-token authentication."""

    EXPIRED = auto()
    INVALID = auto()
    WRONG_TYPE = auto()
    INVALID_CLAIMS = auto()
    USER_NOT_FOUND = auto()


class _AccessTokenRejected(Exception):
    """Expected access-token rejection without implementation details."""

    def __init__(self, reason: _AccessTokenRejectionReason) -> None:
        super().__init__()
        self.reason = reason


def _has_matching_temporal_claim_conversion_failure(
    token: str, original_error: TypeError | OverflowError
) -> bool:
    """Whether unverified temporal data proves the decoder's exact failure."""
    try:
        claims = jwt.get_unverified_claims(token)
    except JWTError:
        return False

    for claim_name in ("exp", "nbf", "iat"):
        if claim_name not in claims:
            continue
        try:
            int(claims[claim_name])
        except ValueError:
            continue
        except (TypeError, OverflowError) as reproduced_error:
            if type(reproduced_error) is type(original_error):
                return True
    return False


def _validate_access_token_claim_bindability(
    username: object, user_id: object, db: Session
) -> tuple[str, int]:
    """Validate claim values against the bound database dialect's parameters."""
    if type(username) is not str or type(user_id) is not int:
        raise _AccessTokenRejected(_AccessTokenRejectionReason.INVALID_CLAIMS)
    try:
        username.encode("utf-8", "strict")
    except UnicodeEncodeError:
        raise _AccessTokenRejected(_AccessTokenRejectionReason.INVALID_CLAIMS) from None

    dialect_name = db.get_bind().dialect.name
    if dialect_name == "sqlite":
        if not _SQLITE_INTEGER_MIN <= user_id <= _SQLITE_INTEGER_MAX:
            raise _AccessTokenRejected(_AccessTokenRejectionReason.INVALID_CLAIMS)
    elif dialect_name == "postgresql":
        if "\x00" in username or not (
            _POSTGRESQL_INTEGER_MIN <= user_id <= _POSTGRESQL_INTEGER_MAX
        ):
            raise _AccessTokenRejected(_AccessTokenRejectionReason.INVALID_CLAIMS)

    return username, user_id


def _resolve_access_token_user(token: str, db: Session) -> User:
    """Return the matching user or raise a typed credential rejection."""
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"verify_sub": False},
        )
    except ExpiredSignatureError:
        raise _AccessTokenRejected(_AccessTokenRejectionReason.EXPIRED) from None
    except JWTError:
        raise _AccessTokenRejected(_AccessTokenRejectionReason.INVALID) from None
    except (TypeError, OverflowError) as error:
        if _has_matching_temporal_claim_conversion_failure(token, error):
            raise _AccessTokenRejected(
                _AccessTokenRejectionReason.INVALID_CLAIMS
            ) from None
        raise

    if payload.get("type") != "access":
        raise _AccessTokenRejected(_AccessTokenRejectionReason.WRONG_TYPE)

    username, user_id = _validate_access_token_claim_bindability(
        payload.get("sub"), payload.get("user_id"), db
    )

    user = db.query(User).filter(User.username == username, User.id == user_id).first()
    if user is None:
        raise _AccessTokenRejected(_AccessTokenRejectionReason.USER_NOT_FOUND)
    return user


def _required_http_rejection(reason: _AccessTokenRejectionReason) -> NoReturn:
    """Raise the established HTTP response for a typed credential rejection."""
    match reason:
        case _AccessTokenRejectionReason.EXPIRED:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer", "Error-Type": "TokenExpired"},
            )
        case _AccessTokenRejectionReason.INVALID:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer", "Error-Type": "InvalidToken"},
            )
        case _AccessTokenRejectionReason.WRONG_TYPE:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        case _AccessTokenRejectionReason.INVALID_CLAIMS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        case _AccessTokenRejectionReason.USER_NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        case _:
            assert_never(reason)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Get current authenticated user from a JWT token.

    Args:
        credentials: Bearer token credentials
        db: Database session

    Returns:
        User: Current authenticated user

    Raises:
        HTTPException: If the token is rejected as invalid credentials.
        Exception: Database and unexpected operational failures propagate unchanged.
    """
    try:
        return _resolve_access_token_user(credentials.credentials, db)
    except _AccessTokenRejected as rejected:
        _required_http_rejection(rejected.reason)


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Get current authenticated user if a token is provided.

    Args:
        credentials: Optional Bearer token credentials
        db: Database session

    Returns:
        Optional[User]: The matching user, or None for absent or rejected credentials.

    Raises:
        Exception: Database and unexpected operational failures propagate unchanged.
    """
    if not credentials:
        return None

    try:
        return _resolve_access_token_user(credentials.credentials, db)
    except _AccessTokenRejected:
        return None


def get_user_from_token(token: str, db: Session) -> Optional[User]:
    """
    Get user from an authentication token.

    Args:
        token: Authentication token
        db: Database session

    Returns:
        Optional[User]: User if credentials are valid, None if they are rejected.

    Raises:
        Exception: Database and unexpected operational failures propagate unchanged.
    """
    if token.startswith("Bearer "):
        token = token[7:]

    try:
        return _resolve_access_token_user(token, db)
    except _AccessTokenRejected:
        return None


def require_user(user: User = Depends(get_current_user)) -> User:
    """
    Require authenticated user (alias for get_current_user)

    Args:
        user: Current authenticated user

    Returns:
        User: Current authenticated user
    """
    return user


def is_admin_user(user: User) -> bool:
    """Whether the user has platform-admin privileges.

    Centralizes the ``is_admin`` check so cross-user authorization gates read
    consistently instead of re-deriving ``getattr(user, "is_admin", False)``.
    """
    return bool(getattr(user, "is_admin", False))


def get_user_from_websocket_token(token: str, db: Session) -> Optional[User]:
    """
    Get user from a WebSocket authentication token.

    Args:
        token: Authentication token from WebSocket
        db: Database session

    Returns:
        Optional[User]: User if credentials are valid, None if they are rejected.

    Raises:
        Exception: Database and unexpected operational failures propagate unchanged.
    """
    return get_user_from_token(token, db)
