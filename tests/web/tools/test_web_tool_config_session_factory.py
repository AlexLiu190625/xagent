import asyncio

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
        try:
            cfg._load_connector_runtime_view()
        except RuntimeError as exc:
            assert str(exc) == "database unavailable"
        else:
            raise AssertionError("runtime view lookup error was swallowed")
        assert cfg._connector_runtime_view is None
    finally:
        cfg.close()
