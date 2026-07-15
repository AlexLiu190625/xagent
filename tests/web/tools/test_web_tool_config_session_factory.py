import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.tools.adapters.vibe.config import MCPConfigLoadError
from xagent.core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ConnectorRuntimeError,
)
from xagent.web.tools.config import WebToolConfig


def _factory():
    engine = create_engine("sqlite://")  # in-memory, fresh
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


class _Chain:
    """Minimal chainable query stub: filter/join return self, terminals empty."""

    def filter(self, *a, **k):
        return self

    def join(self, *a, **k):
        return self

    def all(self):
        return []

    def first(self):
        return None


class _ListChain:
    """Minimal chainable query stub with a fixed ``all()`` result."""

    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *a, **k):
        return self

    def join(self, *a, **k):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _StaticRowsSession:
    def __init__(self, rows):
        self._rows = list(rows)

    def query(self, *a, **k):
        return _ListChain(self._rows)


class _TrackingSession:
    """Records whether ``.query`` was driven (i.e. the session was used)."""

    def __init__(self):
        self.query_calls = 0
        self.closed = False

    def query(self, *a, **k):
        self.query_calls += 1
        return _Chain()

    def close(self):
        self.closed = True


class _FailingQuerySession:
    def __init__(self):
        self.query_calls = 0

    def query(self, *args, **kwargs):
        self.query_calls += 1
        raise RuntimeError("database-secret")


def test_get_session_factory_prefers_injected_factory():
    factory = _factory()
    cfg = WebToolConfig(db=None, request=None, db_factory=factory)
    assert cfg.get_session_factory() is factory


def test_factory_built_get_db_is_lazy_and_closed_by_close():
    factory = _factory()
    cfg = WebToolConfig(db=None, request=None, db_factory=factory)
    db1 = cfg.get_db()
    db2 = cfg.get_db()
    assert db1 is db2  # cached, single construction-time session
    cfg.close()
    # closing twice is safe
    cfg.close()


def test_live_db_path_unchanged():
    sentinel = object()
    cfg = WebToolConfig(db=sentinel, request=None)
    assert cfg.get_db() is sentinel
    cfg.close()  # must not raise; caller owns the request session


def test_legacy_oauth_session_uses_engine_when_caller_is_connection_bound():
    engine = create_engine("sqlite://")
    connection = engine.connect()
    caller_db = Session(bind=connection)
    cfg = WebToolConfig(db=caller_db, request=None, user_id=1)

    oauth_db = cfg._new_legacy_oauth_session()
    try:
        assert caller_db.get_bind() is connection
        assert oauth_db.get_bind() is engine
    finally:
        oauth_db.close()
        caller_db.close()
        connection.close()
        engine.dispose()


def test_custom_api_loader_uses_factory_session():
    # Factory-only (nested child) config: the loader must mint/reuse the lazy
    # factory session via get_db(), not read the None live ``self.db`` and
    # silently swallow ``None.query`` into an empty tool list.
    sess = _TrackingSession()
    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: sess, user_id=1)
    cfg.get_custom_api_configs()
    assert sess.query_calls >= 1


def test_mcp_loader_uses_factory_session():
    sess = _TrackingSession()
    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: sess, user_id=1)
    asyncio.run(cfg._load_mcp_server_configs())
    assert sess.query_calls >= 1


def test_mcp_config_scan_failure_raises_safe_typed_error():
    cfg = WebToolConfig(
        db=_FailingQuerySession(),
        request=None,
        user_id=1,
        include_mcp_tools=True,
    )

    with pytest.raises(MCPConfigLoadError) as exc_info:
        asyncio.run(cfg._load_mcp_server_configs())

    assert exc_info.value.summaries[0].server_name == "MCP server"
    assert exc_info.value.summaries[0].reason == "config_load_failed"
    assert "database-secret" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_failed_mcp_config_refresh_never_reuses_stale_cache():
    session = _FailingQuerySession()
    cfg = WebToolConfig(
        db=session,
        request=None,
        user_id=1,
        include_mcp_tools=True,
    )
    cfg._cached_mcp_configs = [{"name": "stale", "config": {"token": "secret"}}]
    cfg._mcp_hook_generation_at_load = -1

    for _ in range(2):
        with pytest.raises(MCPConfigLoadError):
            asyncio.run(cfg.get_mcp_server_configs())

    assert session.query_calls == 2


def test_connector_runtime_turn_switch_invalidates_runtime_caches():
    cfg = WebToolConfig(
        db=None,
        request=None,
        connector_runtime_turn_id="turn-1",
    )
    cfg._connector_runtime_view = {"custom_api:1": {"secrets": {"token": "old"}}}
    cfg._cached_mcp_configs = [{"id": 1, "connector_runtime": {"context": {}}}]

    assert cfg.set_connector_runtime_turn_id("turn-1") is False
    assert cfg._connector_runtime_view is not None
    assert cfg._cached_mcp_configs is not None

    assert cfg.set_connector_runtime_turn_id("turn-2") is True
    assert cfg._connector_runtime_turn_id == "turn-2"
    assert cfg._connector_runtime_view is None
    assert cfg._cached_mcp_configs is None


def test_connector_runtime_view_resolution_errors_fail_closed(monkeypatch):
    def _raise_runtime_lookup_error(**_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        "xagent.web.services.connector_runtime.load_connector_runtime_view",
        _raise_runtime_lookup_error,
    )
    cfg = WebToolConfig(
        db=object(),
        request=None,
        task_id="web_task_123",
        user_id=1,
        connector_runtime_turn_id="turn-1",
    )

    try:
        with pytest.raises(ConnectorRuntimeError) as exc_info:
            cfg._load_connector_runtime_view()
        assert exc_info.value.code == ERROR_CONNECTOR_RUNTIME_UNAVAILABLE
        assert exc_info.value.status_code == 503
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == "database unavailable"
        assert cfg._connector_runtime_view is None
    finally:
        cfg.close()


def test_mcp_config_loader_propagates_runtime_view_resolution_error(monkeypatch):
    def _raise_runtime_lookup_error(**_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        "xagent.web.services.connector_runtime.load_connector_runtime_view",
        _raise_runtime_lookup_error,
    )
    for name in (
        "load_user_env_overrides",
        "load_shared_env_overrides",
        "load_user_env_sources",
    ):
        monkeypatch.setattr(
            f"xagent.web.services.mcp_runtime.{name}", lambda *_a, **_k: {}
        )

    server = SimpleNamespace(
        id=7,
        name="ShiftCare",
        transport="streamable_http",
        description="runtime connector",
        runtime_bindings=[],
        allow_delegated_authorization=False,
        runtime_input_schema=None,
    )
    cfg = WebToolConfig(
        db=_StaticRowsSession([server]),
        request=None,
        task_id="web_task_123",
        user_id=1,
        connector_runtime_turn_id="turn-1",
        include_mcp_tools=True,
    )

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        asyncio.run(cfg._load_mcp_server_configs())

    assert exc_info.value.code == ERROR_CONNECTOR_RUNTIME_UNAVAILABLE
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_custom_api_config_loader_propagates_runtime_view_resolution_error(monkeypatch):
    def _raise_runtime_lookup_error(**_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        "xagent.web.services.connector_runtime.load_connector_runtime_view",
        _raise_runtime_lookup_error,
    )
    api = SimpleNamespace(
        id=11,
        name="ShiftCare",
        description="runtime API",
        url="https://api.example.test",
        method="GET",
        headers={},
        body=None,
        env={},
        runtime_input_schema=None,
        runtime_bindings=[],
        allow_delegated_authorization=False,
    )
    cfg = WebToolConfig(
        db=_StaticRowsSession([SimpleNamespace(custom_api=api)]),
        request=None,
        task_id="web_task_123",
        user_id=1,
        connector_runtime_turn_id="turn-1",
    )

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        cfg.get_custom_api_configs()

    assert exc_info.value.code == ERROR_CONNECTOR_RUNTIME_UNAVAILABLE
    assert isinstance(exc_info.value.__cause__, RuntimeError)
