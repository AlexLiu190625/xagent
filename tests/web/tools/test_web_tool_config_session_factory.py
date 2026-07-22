import asyncio
import threading
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.core.tools.adapters.vibe.config import MCPConfigLoadError
from xagent.core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ConnectorRuntimeError,
)
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec
from xagent.web.models.tool_config import ToolConfig
from xagent.web.models.user import User
from xagent.web.services.tool_credentials import (
    set_user_tool_allowlist_hook,
    set_user_tool_overrides_hook,
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

    def connection(self):
        return object()

    def rollback(self):
        return None


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


def _saturated_tool_config(
    tmp_path, *, pool_timeout: float
) -> tuple[object, object, WebToolConfig]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'tool-factory.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=pool_timeout,
        connect_args={"check_same_thread": False},
    )
    ToolConfig.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    held_connection = engine.connect()
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=factory,
        user_id=1,
        workspace_config={"task_id": "_mock_"},
        task_id="_mock_",
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=["basic"]),
    )
    return engine, held_connection, cfg


@pytest.mark.asyncio
async def test_tool_factory_credential_prefetch_waits_off_event_loop(tmp_path):
    """Credential checkout must not freeze unrelated async work."""
    engine, held_connection, cfg = _saturated_tool_config(tmp_path, pool_timeout=0.5)
    ticks = 0
    stop = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    async def build_tools() -> list:
        return await ToolFactory.create_all_tools(cfg)

    ticker_task = asyncio.create_task(ticker())
    build_task = asyncio.create_task(build_tools())
    try:
        await asyncio.sleep(0.08)
        assert ticks >= 4
        assert not build_task.done()

        held_connection.close()
        await build_task
    finally:
        if not held_connection.closed:
            held_connection.close()
        if not build_task.done():
            build_task.cancel()
            await asyncio.gather(build_task, return_exceptions=True)
        stop.set()
        await ticker_task
        cfg.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_tool_factory_prefetch_propagates_pool_timeout(tmp_path):
    """A build-time checkout timeout must stop the build and reach its owner."""
    engine, held_connection, cfg = _saturated_tool_config(tmp_path, pool_timeout=0.05)
    try:
        with pytest.raises(SQLAlchemyTimeoutError):
            await ToolFactory.create_all_tools(cfg)
    finally:
        held_connection.close()
        cfg.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_factory_runtime_snapshot_is_rebuilt_for_each_build(monkeypatch):
    sessions: list[_TrackingSession] = []

    def session_factory() -> _TrackingSession:
        session = _TrackingSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(
        "xagent.web.tools.config.resolve_tool_credential",
        lambda *_args: None,
    )
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=session_factory,
        user_id=1,
        workspace_config={"task_id": "_mock_"},
        task_id="_mock_",
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=["basic"]),
    )

    await ToolFactory.create_all_tools(cfg)
    await ToolFactory.create_all_tools(cfg)

    assert len(sessions) == 2
    assert all(session.closed for session in sessions)
    assert cfg._factory_runtime_snapshot is None


@pytest.mark.asyncio
async def test_refreshed_factory_runtime_is_consumed_by_next_build(monkeypatch):
    sessions: list[_TrackingSession] = []

    def session_factory() -> _TrackingSession:
        session = _TrackingSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(
        "xagent.web.tools.config.resolve_tool_credential",
        lambda *_args: None,
    )
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=session_factory,
        user_id=1,
        workspace_config={"task_id": "_mock_"},
        task_id="_mock_",
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=["basic"]),
    )

    await cfg.refresh_runtime_policy()
    assert len(sessions) == 1
    assert cfg._factory_runtime_snapshot is not None

    await ToolFactory.create_all_tools(cfg)

    assert len(sessions) == 1
    assert sessions[0].closed
    assert cfg._factory_runtime_snapshot is None


@pytest.mark.asyncio
async def test_factory_runtime_snapshot_is_released_when_build_raises(monkeypatch):
    sessions: list[_TrackingSession] = []

    def session_factory() -> _TrackingSession:
        session = _TrackingSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(
        "xagent.web.tools.config.resolve_tool_credential",
        lambda *_args: None,
    )

    async def fail_build(_cls, _config):
        raise RuntimeError("registered tool build failed")

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        classmethod(fail_build),
    )
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=session_factory,
        user_id=1,
        workspace_config={"task_id": "_mock_"},
        task_id="_mock_",
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=["basic"]),
    )

    with pytest.raises(RuntimeError, match="registered tool build failed"):
        await ToolFactory.create_all_tools(cfg)

    assert len(sessions) == 1
    assert sessions[0].closed
    assert cfg._factory_runtime_snapshot is None


@pytest.mark.asyncio
async def test_runtime_refresh_snapshots_selected_sync_factory_inputs(
    monkeypatch,
):
    """After refresh, selected synchronous getters read only cached values."""
    main_thread_id = threading.get_ident()
    loader_thread_ids: list[int] = []
    session = _TrackingSession()

    def record(value):
        loader_thread_ids.append(threading.get_ident())
        return value

    monkeypatch.setattr(
        "xagent.web.tools.config.resolve_tool_credential",
        lambda *_args: record("credential"),
    )
    monkeypatch.setattr(
        "xagent.web.tools.config.get_sql_connection_map",
        lambda *_args: record({"WAREHOUSE": "sqlite:///warehouse.db"}),
    )

    model_values = {
        "get_default_vision_model": object(),
        "get_image_models": {"image": object()},
        "get_default_image_generate_model": object(),
        "get_default_image_edit_model": object(),
        "get_video_models": {"video": object()},
        "get_default_video_model": object(),
        "get_asr_models": {"asr": object()},
        "get_default_asr_model": object(),
        "get_tts_models": {"tts": object()},
        "get_default_tts_model": object(),
        "get_sound_effect_models": {"sound": object()},
        "get_default_sound_effect_model": object(),
        "get_music_models": {"music": object()},
        "get_default_music_model": object(),
    }
    for name, value in model_values.items():
        monkeypatch.setattr(
            f"xagent.web.services.model_service.{name}",
            lambda *_args, _value=value, **_kwargs: record(_value),
        )

    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=lambda: session,
        user_id=1,
        task_id="_mock_",
        workspace_config={"task_id": "_mock_"},
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(
            tool_categories=[
                "basic",
                "database",
                "image",
                "video",
                "audio",
                "vision",
                "mcp:custom-api",
            ]
        ),
    )

    await cfg.refresh_runtime_policy()

    def fail_factory():
        raise AssertionError("factory getter attempted a second database checkout")

    cfg._db_factory = fail_factory
    assert cfg.get_tool_credential("web_search", "api_key") == "credential"
    assert cfg.get_sql_connections() == {"WAREHOUSE": "sqlite:///warehouse.db"}
    assert cfg.get_custom_api_configs() == []
    assert cfg.get_vision_model() is model_values["get_default_vision_model"]
    assert cfg.get_image_models() is model_values["get_image_models"]
    assert (
        cfg.get_image_generate_model()
        is model_values["get_default_image_generate_model"]
    )
    assert cfg.get_image_edit_model() is model_values["get_default_image_edit_model"]
    assert cfg.get_video_models() is model_values["get_video_models"]
    assert cfg.get_video_model() is model_values["get_default_video_model"]
    assert cfg.get_asr_models() is model_values["get_asr_models"]
    assert cfg.get_asr_model() is model_values["get_default_asr_model"]
    assert cfg.get_tts_models() is model_values["get_tts_models"]
    assert cfg.get_tts_model() is model_values["get_default_tts_model"]
    assert cfg.get_sound_effect_models() is model_values["get_sound_effect_models"]
    assert (
        cfg.get_sound_effect_model() is model_values["get_default_sound_effect_model"]
    )
    assert cfg.get_music_models() is model_values["get_music_models"]
    assert cfg.get_music_model() is model_values["get_default_music_model"]
    assert loader_thread_ids
    assert all(thread_id != main_thread_id for thread_id in loader_thread_ids)


@pytest.mark.asyncio
async def test_default_model_prefetch_returns_every_pool_checkout(
    monkeypatch,
    tmp_path,
):
    from xagent.web.models import database
    from xagent.web.models.database import Base
    from xagent.web.services import model_service

    engine = create_engine(
        f"sqlite:///{tmp_path / 'default-models.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.1,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(database, "_SessionLocal", factory)

    checkouts = 0
    checkins = 0

    def record_checkout(*_args) -> None:
        nonlocal checkouts
        checkouts += 1

    def record_checkin(*_args) -> None:
        nonlocal checkins
        checkins += 1

    event.listen(engine, "checkout", record_checkout)
    event.listen(engine, "checkin", record_checkin)  # codespell:ignore checkin

    collection_getters = (
        "get_image_models",
        "get_video_models",
        "get_asr_models",
        "get_tts_models",
        "get_sound_effect_models",
        "get_music_models",
    )
    for getter_name in collection_getters:
        monkeypatch.setattr(
            f"xagent.web.services.model_service.{getter_name}",
            lambda *_args: {"configured": object()},
        )

    default_getters = (
        "get_default_vision_model",
        "get_default_image_generate_model",
        "get_default_image_edit_model",
        "get_default_video_model",
        "get_default_asr_model",
        "get_default_tts_model",
        "get_default_sound_effect_model",
        "get_default_music_model",
    )
    default_calls: list[str] = []
    for getter_name in default_getters:
        real_getter = getattr(model_service, getter_name)

        def record_default_call(
            *args,
            _getter_name=getter_name,
            _real_getter=real_getter,
            **kwargs,
        ):
            default_calls.append(_getter_name)
            return _real_getter(*args, **kwargs)

        monkeypatch.setattr(model_service, getter_name, record_default_call)

    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=factory,
        user_id=1,
        task_id="_mock_",
        workspace_config={"task_id": "_mock_"},
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(
            tool_categories=["vision", "image", "video", "audio"]
        ),
    )
    try:
        await cfg.prepare_factory_runtime()

        assert default_calls == list(default_getters)
        assert checkouts == checkins == 1
        assert engine.pool.checkedout() == 0
    finally:
        cfg.close()
        engine.dispose()


def test_legacy_default_model_resolvers_close_owned_pool_connections(
    monkeypatch,
    tmp_path,
):
    from xagent.web.models import database
    from xagent.web.models.database import Base
    from xagent.web.services import model_service

    engine = create_engine(
        f"sqlite:///{tmp_path / 'legacy-default-models.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.1,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(database, "_SessionLocal", factory)

    checkouts = 0
    checkins = 0

    def record_checkout(*_args) -> None:
        nonlocal checkouts
        checkouts += 1

    def record_checkin(*_args) -> None:
        nonlocal checkins
        checkins += 1

    event.listen(engine, "checkout", record_checkout)
    event.listen(engine, "checkin", record_checkin)  # codespell:ignore checkin

    default_getters = (
        model_service.get_default_vision_model,
        model_service.get_default_image_generate_model,
        model_service.get_default_image_edit_model,
        model_service.get_default_video_model,
        model_service.get_default_asr_model,
        model_service.get_default_tts_model,
        model_service.get_default_sound_effect_model,
        model_service.get_default_music_model,
    )
    try:
        for getter in default_getters:
            assert getter() is None
            assert engine.pool.checkedout() == 0

        assert checkouts == checkins == len(default_getters)
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_runtime_policy_refresh_waits_for_pool_off_event_loop(tmp_path):
    """A saturated policy-query pool must not freeze unrelated coroutines."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'tool-policy.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.5,
        connect_args={"check_same_thread": False},
    )
    User.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with factory() as db:
        user = User(username="policy-user", password_hash="hash", is_admin=False)
        db.add(user)
        db.commit()
        user_id = int(user.id)

    def policy_hook(db, user):
        assert db.query(User.id).filter(User.id == user.id).scalar() == user_id
        return {"calculator": {"enabled": False}}

    set_user_tool_overrides_hook(policy_hook)
    set_user_tool_allowlist_hook(lambda _db, _user: ["file"])
    held_connection = engine.connect()
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=factory,
        user_id=user_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )
    ticks = 0
    stop = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    try:
        await asyncio.sleep(0.02)
        ticks_before_wait = ticks
        refresh_task = asyncio.create_task(cfg.refresh_runtime_policy())
        await asyncio.sleep(0.08)
        assert ticks - ticks_before_wait >= 4
        assert not refresh_task.done()

        held_connection.close()
        await refresh_task
        assert cfg.get_user_tool_overrides() == {"calculator": {"enabled": False}}
        assert cfg.get_user_tool_allowlist() == ["file"]
    finally:
        if not held_connection.closed:
            held_connection.close()
        stop.set()
        await ticker_task
        cfg.close()
        set_user_tool_overrides_hook(None)
        set_user_tool_allowlist_hook(None)
        engine.dispose()


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
