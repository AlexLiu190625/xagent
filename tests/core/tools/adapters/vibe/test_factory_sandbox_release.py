"""Regression tests for ToolFactory release boundaries.

Issue #889 requires the config's DB connection to be released again before
sandbox workspace setup because override/allowlist reads may reopen it.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry


class _FakeSandbox:
    pass


class _FakeConfig:
    def __init__(self, calls):
        self._calls = calls

    def get_tool_selection_spec(self):
        return None

    def get_allowed_tools(self):
        return None

    def get_user_tool_overrides(self):
        self._calls.append("load_overrides")
        return {}

    def get_user_tool_allowlist(self):
        self._calls.append("load_allowlist")
        return None

    def release_db_connection(self):
        self._calls.append("release_db")

    def get_sandbox(self):
        return _FakeSandbox()

    def get_workspace_config(self):
        # ``_mock_`` selects MockWorkspace: no on-disk directories.
        return {"task_id": "_mock_", "base_dir": "/tmp"}

    def get_max_output_length(self):
        return 10000

    def get_max_field_count(self):
        return 100

    def get_max_recursion_depth(self):
        return 5


class _FailingPrepareConfig:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def prepare_factory_runtime(self) -> None:
        self._calls.append("prepare")
        raise RuntimeError("prepare failed")

    def release_prepared_factory_runtime(self) -> None:
        self._calls.append("release")


class _FailingPrepareAndReleaseConfig(_FailingPrepareConfig):
    def release_prepared_factory_runtime(self) -> None:
        super().release_prepared_factory_runtime()
        raise ValueError("release failed")


class _FailingReleaseConfig:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def release_prepared_factory_runtime(self) -> None:
        self._calls.append("release")
        raise ValueError("release failed")


class _FailingVerifiedHandoffConfig:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def handoff_factory_runtime(self) -> None:
        self._calls.append("handoff")
        raise ValueError("handoff failed")


@pytest.mark.asyncio
async def test_release_prepared_runtime_when_prepare_fails():
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="prepare failed"):
        await ToolFactory.create_all_tools(_FailingPrepareConfig(calls))

    assert calls == ["prepare", "release"]


@pytest.mark.asyncio
async def test_prepare_error_wins_when_release_also_fails(caplog):
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="prepare failed"):
        await ToolFactory.create_all_tools(_FailingPrepareAndReleaseConfig(calls))

    assert calls == ["prepare", "release"]
    assert "Failed to release prepared tool-factory runtime" in caplog.text


@pytest.mark.asyncio
async def test_release_error_propagates_without_primary_error(monkeypatch):
    calls: list[str] = []

    async def build_tools(config, apply_user_override_filter=True):
        calls.append("build")
        return []

    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", build_tools)

    with pytest.raises(ValueError, match="release failed"):
        await ToolFactory.create_all_tools(_FailingReleaseConfig(calls))

    assert calls == ["build", "release"]


@pytest.mark.asyncio
async def test_primary_build_error_wins_when_verified_handoff_fails(
    monkeypatch, caplog
):
    calls: list[str] = []

    async def build_tools(config, apply_user_override_filter=True):
        calls.append("build")
        raise RuntimeError("build sentinel")

    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", build_tools)

    with pytest.raises(RuntimeError, match="build sentinel"):
        await ToolFactory.create_all_tools(_FailingVerifiedHandoffConfig(calls))

    assert calls == ["build", "handoff"]
    assert "Failed to hand off tool-factory runtime" in caplog.text


@pytest.mark.asyncio
async def test_real_web_config_primary_error_identity_wins_over_handoff_fault(
    monkeypatch, tmp_path, caplog
):
    from xagent.web.models.tool_config import ToolConfig
    from xagent.web.tools.config import WebToolConfig

    engine = create_engine(
        f"sqlite:///{tmp_path / 'handoff-primary.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        connect_args={"check_same_thread": False},
    )
    ToolConfig.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    live_db = factory()
    config = WebToolConfig(db=live_db, db_factory=factory, request=None, user_id=1)
    sentinel = RuntimeError("build sentinel")
    handoff_fault = ValueError("independent handoff fault")
    real_handoff = WebToolConfig.handoff_factory_runtime

    async def prepare(_config):
        return None

    async def build_tools(config, apply_user_override_filter=True):
        config.db.query(ToolConfig).all()
        raise sentinel

    def failing_handoff(config):
        real_handoff(config)
        raise handoff_fault

    monkeypatch.setattr(WebToolConfig, "prepare_factory_runtime", prepare)
    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", build_tools)
    monkeypatch.setattr(WebToolConfig, "handoff_factory_runtime", failing_handoff)
    try:
        with pytest.raises(RuntimeError) as caught:
            await ToolFactory.create_all_tools(config)

        assert caught.value is sentinel
        assert engine.pool.checkedout() == 0
        assert "Failed to hand off tool-factory runtime" in caplog.text
    finally:
        live_db.close()
        config.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_primary_error_identity_wins_over_real_session_boundary_failure(
    monkeypatch, tmp_path, caplog
):
    from xagent.web.models.tool_config import ToolConfig
    from xagent.web.models.user import User
    from xagent.web.tools.config import WebToolConfig

    engine = create_engine(
        f"sqlite:///{tmp_path / 'handoff-boundary-primary.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        connect_args={"check_same_thread": False},
    )
    ToolConfig.__table__.create(bind=engine)
    User.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    live_db = factory()
    config = WebToolConfig(db=live_db, db_factory=factory, request=object(), user_id=1)
    sentinel = RuntimeError("primary sentinel")

    async def prepare(_config):
        return None

    async def build_tools(_config, apply_user_override_filter=True):
        raise sentinel

    monkeypatch.setattr(WebToolConfig, "prepare_factory_runtime", prepare)
    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", build_tools)
    try:
        live_db.query(ToolConfig).all()
        live_db.add(
            User(username="boundary-pending", password_hash="hash", is_admin=False)
        )
        assert engine.pool.checkedout() == 1

        with pytest.raises(RuntimeError) as caught:
            await ToolFactory.create_all_tools(config)

        assert caught.value is sentinel
        assert config._live_db is live_db
        assert engine.pool.checkedout() == 1
        assert "Failed to hand off tool-factory runtime" in caplog.text
    finally:
        live_db.rollback()
        live_db.close()
        config.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_release_db_before_sandbox_workspace_setup(monkeypatch):
    calls: list[str] = []

    async def fake_create_registered_tools(config):
        return []

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        staticmethod(fake_create_registered_tools),
    )

    from xagent.core.tools.adapters.vibe.sandboxed_tool import (
        sandboxed_tool_wrapper,
    )

    async def fake_create_workspace_in_sandbox(sandbox, workspace):
        calls.append("sandbox_exec")

    monkeypatch.setattr(
        sandboxed_tool_wrapper,
        "create_workspace_in_sandbox",
        fake_create_workspace_in_sandbox,
    )

    await ToolFactory.create_all_tools(_FakeConfig(calls))

    assert "sandbox_exec" in calls
    assert "release_db" in calls
    # The DB release happens after the last config DB reads (overrides /
    # allowlist) and before the sandbox workspace exec.
    assert calls.index("release_db") > calls.index("load_overrides")
    assert calls.index("release_db") > calls.index("load_allowlist")
    assert calls.index("release_db") < calls.index("sandbox_exec")
