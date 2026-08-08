"""Authentication dependency contract tests."""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt
from sqlalchemy import create_engine, event
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.web import auth_dependencies
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.auth_dependencies import (
    get_current_user,
    get_current_user_optional,
    get_user_from_token,
    get_user_from_websocket_token,
)
from xagent.web.models.database import Base
from xagent.web.models.user import User


@pytest.fixture
def db_session() -> Session:
    """Provide a SQLite session with one matching user."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(
        User(
            id=1,
            username="existing-user",
            email="existing@example.com",
            password_hash="not-used-by-auth-dependencies",
        )
    )
    session.commit()
    session.info["user_query_count"] = 0

    def count_user_queries(*_args: object) -> None:
        session.info["user_query_count"] += 1

    event.listen(engine, "before_cursor_execute", count_user_queries)
    try:
        yield session
    finally:
        event.remove(engine, "before_cursor_execute", count_user_queries)
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def _access_token(**claims: object) -> str:
    payload: dict[str, object] = {
        "type": "access",
        "sub": "existing-user",
        "user_id": 1,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    payload.update(claims)
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


@pytest.mark.parametrize(
    ("token", "detail", "headers"),
    [
        (
            _access_token(exp=datetime.now(timezone.utc) - timedelta(minutes=5)),
            "Token expired",
            {"WWW-Authenticate": "Bearer", "Error-Type": "TokenExpired"},
        ),
        (
            jwt.encode(
                {
                    "type": "access",
                    "sub": "existing-user",
                    "user_id": 1,
                    "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
                },
                "different-signing-secret",
                algorithm=JWT_ALGORITHM,
            ),
            "Invalid token",
            {"WWW-Authenticate": "Bearer", "Error-Type": "InvalidToken"},
        ),
        (
            _access_token(type="refresh"),
            "Invalid token type",
            {"WWW-Authenticate": "Bearer"},
        ),
        (
            _access_token(sub=None),
            "Invalid token payload",
            {"WWW-Authenticate": "Bearer"},
        ),
        (
            _access_token(sub=123),
            "Invalid token payload",
            {"WWW-Authenticate": "Bearer"},
        ),
        (
            _access_token(user_id=None),
            "Invalid token payload",
            {"WWW-Authenticate": "Bearer"},
        ),
        (
            _access_token(user_id="1"),
            "Invalid token payload",
            {"WWW-Authenticate": "Bearer"},
        ),
        (
            _access_token(user_id=True),
            "Invalid token payload",
            {"WWW-Authenticate": "Bearer"},
        ),
        (
            _access_token(sub="missing-user"),
            "User not found",
            {"WWW-Authenticate": "Bearer"},
        ),
    ],
    ids=(
        "expired",
        "invalid-signature",
        "wrong-type",
        "missing-sub",
        "wrong-type-sub",
        "missing-user-id",
        "wrong-type-user-id",
        "bool-user-id",
        "missing-user",
    ),
)
def test_access_token_rejection_matrix_preserves_required_http_contract(
    db_session: Session, token: str, detail: str, headers: dict[str, str]
) -> None:
    """Required HTTP auth exposes its established reason-specific rejection."""
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db_session)

    assert raised.value.status_code == 401
    assert raised.value.detail == detail
    assert raised.value.headers == headers


@pytest.mark.parametrize("claim", ("exp", "nbf", "iat"))
@pytest.mark.parametrize(
    "value",
    ([], {}, None, float("inf"), float("-inf")),
    ids=("list", "dict", "null", "positive-infinity", "negative-infinity"),
)
def test_malformed_temporal_claims_are_rejected_before_any_user_query(
    db_session: Session, claim: str, value: object
) -> None:
    """Dependency conversion failures in signed temporal claims are credentials."""
    token = _access_token(**{claim: value})
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db_session)

    assert raised.value.status_code == 401
    assert raised.value.detail == "Invalid token payload"
    assert raised.value.headers == {"WWW-Authenticate": "Bearer"}
    assert db_session.info["user_query_count"] == 0


@pytest.mark.parametrize(
    "claims",
    (
        {"iat": [], "exp": float("inf")},
        {"iat": float("inf"), "exp": []},
    ),
    ids=("type-error-before-overflow-error", "overflow-error-before-type-error"),
)
def test_mixed_malformed_temporal_claims_are_rejected_before_any_user_query(
    db_session: Session, claims: dict[str, object]
) -> None:
    """A later mixed temporal claim cannot defeat the original error proof."""
    token = _access_token(**claims)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db_session)

    assert raised.value.status_code == 401
    assert raised.value.detail == "Invalid token payload"
    assert raised.value.headers == {"WWW-Authenticate": "Bearer"}
    assert db_session.info["user_query_count"] == 0


@pytest.mark.parametrize("claim", ("exp", "nbf", "iat"))
def test_numeric_temporal_claims_keep_the_library_accepted_behavior(
    db_session: Session, claim: str
) -> None:
    """Integer NumericDate values continue through verified token validation."""
    now = int(datetime.now(timezone.utc).timestamp())
    value = now + 300 if claim == "exp" else now - 300
    token = _access_token(**{claim: value})
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    user = get_current_user(credentials, db_session)

    assert user.username == "existing-user"
    assert db_session.info["user_query_count"] == 1


@pytest.mark.parametrize("exception_type", (TypeError, OverflowError))
def test_unrelated_decode_type_errors_propagate_by_identity(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[Exception],
) -> None:
    """Only proven temporal-claim conversion failures are credential rejections."""
    original = exception_type("unrelated decode failure")

    def raise_original(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise original

    monkeypatch.setattr(auth_dependencies.jwt, "decode", raise_original)
    monkeypatch.setattr(
        auth_dependencies.jwt, "get_unverified_claims", lambda _token: {"aud": "x"}
    )
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

    with pytest.raises(exception_type) as raised:
        get_current_user(credentials, db_session)

    assert raised.value is original
    assert db_session.info["user_query_count"] == 0


@pytest.mark.parametrize(
    ("username", "user_id"),
    (
        ("", 0),
        ("long-" * 20, -1),
        ("sqlite\x00nul", 2**31),
        ("中文-😀", 2**63 - 1),
        ("signed-64-minimum", -(2**63)),
    ),
    ids=("empty", "long", "nul", "unicode-and-signed-64-max", "signed-64-min"),
)
def test_sqlite_bindable_claims_reach_matching_persisted_users(
    db_session: Session, username: str, user_id: int
) -> None:
    """SQLite-compatible claims preserve real persisted user identities."""
    db_session.add(
        User(
            id=user_id,
            username=username,
            email=f"sqlite-{user_id}@example.com",
            password_hash="not-used-by-auth-dependencies",
        )
    )
    db_session.commit()
    db_session.info["user_query_count"] = 0
    token = _access_token(sub=username, user_id=user_id)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    user = get_current_user(credentials, db_session)

    assert user.id == user_id
    assert user.username == username
    assert db_session.info["user_query_count"] == 1


@pytest.mark.parametrize(
    "claims",
    (
        {"user_id": 2**63},
        {"user_id": -(2**63) - 1},
        {"sub": "\ud800"},
    ),
    ids=("above-signed-64", "below-signed-64", "lone-surrogate"),
)
def test_sqlite_unbindable_claims_are_rejected_without_a_user_query(
    db_session: Session, claims: dict[str, object]
) -> None:
    """SQLite claim values outside the driver's bindability contract are rejected."""
    token = _access_token(**claims)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db_session)

    assert raised.value.detail == "Invalid token payload"
    assert db_session.info["user_query_count"] == 0


class _DialectSession:
    """A query-counting Session shape with an already-bound dialect."""

    def __init__(self, dialect_name: str, user: User | None = None) -> None:
        self._bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))
        self._user = user
        self.query_count = 0

    def get_bind(self) -> SimpleNamespace:
        return self._bind

    def query(self, _model: type[User]) -> "_DialectSession":
        self.query_count += 1
        return self

    def filter(self, *_conditions: object) -> "_DialectSession":
        return self

    def first(self) -> User | None:
        return self._user


@pytest.mark.parametrize(
    "claims",
    (
        {"user_id": 2**31},
        {"user_id": -(2**31) - 1},
        {"sub": "postgresql\x00nul"},
        {"sub": "\udfff"},
    ),
    ids=("above-signed-32", "below-signed-32", "nul", "lone-surrogate"),
)
def test_postgresql_unbindable_claims_are_rejected_without_a_user_query(
    claims: dict[str, object],
) -> None:
    """PostgreSQL-specific unbindable claims do not enter the User query."""
    db = _DialectSession("postgresql")
    token = _access_token(**claims)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db)  # type: ignore[arg-type]

    assert raised.value.detail == "Invalid token payload"
    assert db.query_count == 0


@pytest.mark.parametrize("username", ("", "long-" * 20))
def test_postgresql_empty_and_long_claims_remain_queryable(username: str) -> None:
    """PostgreSQL does not impose unsupported empty or length claim rules."""
    expected = User(
        id=2**31 - 1,
        username=username,
        email="postgresql@example.com",
        password_hash="not-used-by-auth-dependencies",
    )
    db = _DialectSession("postgresql", expected)
    token = _access_token(sub=username, user_id=2**31 - 1)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    assert get_current_user(credentials, db) is expected  # type: ignore[arg-type]
    assert db.query_count == 1


def test_unrecognized_dialect_runs_the_query_and_propagates_backend_failure() -> None:
    """Unknown dialects retain operational behavior instead of invented limits."""
    original = RuntimeError("backend-specific failure")

    class FailingDialectSession(_DialectSession):
        def query(self, _model: type[User]) -> "_DialectSession":
            self.query_count += 1
            raise original

    db = FailingDialectSession("unrecognized")
    token = _access_token(sub="\x00still-queryable", user_id=2**63)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(RuntimeError) as raised:
        get_current_user(credentials, db)  # type: ignore[arg-type]

    assert raised.value is original
    assert db.query_count == 1


@pytest.mark.parametrize(
    "adapter",
    (
        lambda credentials, db: get_current_user(credentials, db),
        lambda credentials, db: get_current_user_optional(credentials, db),
        lambda credentials, db: get_user_from_token(credentials.credentials, db),
        lambda credentials, db: get_user_from_websocket_token(
            credentials.credentials, db
        ),
    ),
    ids=("required", "optional", "token", "websocket-token-alias"),
)
def test_auth_adapters_propagate_database_pool_timeout_by_identity(
    adapter: object,
) -> None:
    """Operational database failures never become an authentication absence."""
    original = SQLAlchemyTimeoutError("auth pool timeout", None, None)

    class TimeoutSession(_DialectSession):
        def query(self, _model: type[User]) -> "_DialectSession":
            self.query_count += 1
            raise original

    db = TimeoutSession("sqlite")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=_access_token()
    )

    with pytest.raises(SQLAlchemyTimeoutError) as raised:
        adapter(credentials, db)  # type: ignore[operator,arg-type]

    assert raised.value is original
    assert db.query_count == 1


def test_token_adapter_propagates_real_queue_pool_checkout_timeout() -> None:
    """A real exhausted pool stays on the operational exception channel."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.01,
    )
    held_connection = engine.connect()
    session = sessionmaker(bind=engine)()
    try:
        with pytest.raises(SQLAlchemyTimeoutError):
            get_user_from_token(_access_token(), session)
    finally:
        session.close()
        held_connection.close()
        engine.dispose()


def test_optional_direct_none_returns_none(db_session: Session) -> None:
    """The optional helper preserves its direct no-credential contract."""
    assert get_current_user_optional(None, db_session) is None


@pytest.mark.parametrize(
    "adapter",
    (
        lambda credentials, db: get_current_user(credentials, db),
        lambda credentials, db: get_current_user_optional(credentials, db),
        lambda credentials, db: get_user_from_token(
            f"Bearer {credentials.credentials}", db
        ),
        lambda credentials, db: get_user_from_websocket_token(
            credentials.credentials, db
        ),
    ),
    ids=("required", "optional", "token-bearer-prefix", "websocket-token-alias"),
)
def test_valid_access_token_returns_matching_user_across_adapters(
    db_session: Session, adapter: object
) -> None:
    """All public access-token adapters share the same successful resolution."""
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=_access_token()
    )

    user = adapter(credentials, db_session)  # type: ignore[operator]

    assert user is not None
    assert user.id == 1
    assert user.username == "existing-user"


def test_invalid_claims_read_the_bound_dialect_without_pool_checkout() -> None:
    """Pre-query rejection inspects Session binding without acquiring a connection."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.01,
    )
    held_connection = engine.connect()
    session = sessionmaker(bind=engine)()
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=_access_token(user_id=2**63)
    )
    try:
        with pytest.raises(HTTPException) as raised:
            get_current_user(credentials, session)
    finally:
        session.close()
        held_connection.close()
        engine.dispose()

    assert raised.value.detail == "Invalid token payload"


def test_auth_dependencies_have_no_broad_exception_handler() -> None:
    """The shared authentication owner cannot collapse operational failures."""
    source = Path(auth_dependencies.__file__).read_text(encoding="utf-8")
    handlers = [
        handler
        for handler in ast.walk(ast.parse(source))
        if isinstance(handler, ast.ExceptHandler)
    ]

    def catches_exception(handler: ast.ExceptHandler) -> bool:
        if isinstance(handler.type, ast.Name):
            return handler.type.id == "Exception"
        if isinstance(handler.type, ast.Tuple):
            return any(
                isinstance(element, ast.Name) and element.id == "Exception"
                for element in handler.type.elts
            )
        return False

    assert not any(catches_exception(handler) for handler in handlers)
