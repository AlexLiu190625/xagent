import asyncio
import os
import tempfile
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.core.tools.adapters.vibe.agent_tool as mod
from xagent.core.agent.result import tool_result_succeeded
from xagent.core.tools.adapters.vibe.agent_tool import AgentTool
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import User
from xagent.web.services.llm_utils import UserAwareModelStorage


class _Stop(Exception):
    """Halt the run before the sub-agent executes."""


class _DelegatedQuery:
    def __init__(self, agent):
        self._agent = agent

    def filter(self, *_args):
        return self

    def first(self):
        return self._agent


class _DelegatedSession:
    def __init__(self, agent):
        self._agent = agent

    def query(self, *_args):
        return _DelegatedQuery(self._agent)

    def commit(self):
        return None

    def close(self):
        return None


class _FailingCloseConfig:
    def close(self):
        raise ValueError("cleanup sentinel")


class _SucceedingCloseConfig:
    def close(self):
        return None


def _delegated_agent_tool() -> AgentTool:
    return AgentTool(
        agent_id=1,
        agent_name="Delegated",
        agent_description="d",
        session_factory=lambda: _DelegatedSession(
            SimpleNamespace(
                id=1,
                name="Delegated",
                instructions=None,
                knowledge_bases=None,
                skills=None,
                tool_categories=[],
                models={"general": 1},
                execution_mode=None,
            )
        ),
        user_id=1,
        tool_name="delegated",
        tool_description="d",
    )


def _patch_delegated_runtime(
    monkeypatch, execute_task, *, close_config=_FailingCloseConfig
):
    import xagent.core.agent.service as service_module
    import xagent.core.tools.adapters.vibe.agent_model_resolution as resolution

    class FakeAgentService:
        workspace = None

        def __init__(self, **_kwargs):
            return None

        async def execute_task(self, **_kwargs):
            return await execute_task()

    monkeypatch.setattr(mod, "WebToolConfig", lambda **_kwargs: close_config())
    monkeypatch.setattr(service_module, "AgentService", FakeAgentService)
    monkeypatch.setattr(
        resolution,
        "resolve_agent_model_llms",
        lambda *_args: (object(), None, None, None),
    )


@pytest.mark.asyncio
async def test_agent_tool_maps_successful_body_cleanup_failure_to_boundary_error(
    monkeypatch,
):
    async def execute_task():
        return {"output": "completed"}

    _patch_delegated_runtime(monkeypatch, execute_task)
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["response"].endswith("Tool runtime cleanup could not be completed.")
    assert result["success"] is False


@pytest.mark.asyncio
async def test_agent_tool_preserves_body_failure_when_cleanup_also_fails(
    monkeypatch, caplog
):
    primary = RuntimeError("body sentinel")

    async def execute_task():
        raise primary

    _patch_delegated_runtime(monkeypatch, execute_task)
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["response"].endswith("body sentinel")
    assert result["success"] is False
    assert "Failed to close delegated agent tool runtime after execution" in caplog.text


@pytest.mark.asyncio
async def test_agent_tool_preserves_cancelled_error_identity_when_cleanup_fails(
    monkeypatch,
):
    primary = asyncio.CancelledError("cancelled sentinel")

    async def execute_task():
        raise primary

    _patch_delegated_runtime(monkeypatch, execute_task)
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    with pytest.raises(asyncio.CancelledError) as caught:
        await tool.run_json_async({"task": "run"})

    assert caught.value is primary


@pytest.mark.asyncio
async def test_agent_tool_child_waiting_returns_classified_nested_failure(
    monkeypatch,
):
    """A child that paused for user input must not surface as a success.

    Even when the child left a non-empty partial ``output`` behind, the
    delegated call cannot forward the interactive prompt one level up, so the
    parent must see a classified failure rather than a half-finished answer.
    """

    async def execute_task():
        return {"status": "waiting_for_user", "output": "Here is what I have so far"}

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["success"] is False
    assert result["is_error"] is True
    assert result["status"] == "error"
    assert result["failure_code"] == "unsupported_nested_interaction"
    assert isinstance(result["error"], str) and result["error"]
    assert isinstance(result["output"], str) and result["output"]
    assert isinstance(result["response"], str) and result["response"]
    assert list(result.keys()) == [
        "success",
        "is_error",
        "status",
        "failure_code",
        "error",
        "output",
        "response",
    ]
    assert tool_result_succeeded(result) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "child_result",
    [
        {"success": True, "status": "failed"},
        {"success": False, "status": "completed"},
    ],
)
async def test_agent_tool_or_precedence_disagreement_classifies_as_generic_failure(
    monkeypatch, child_result
):
    """``success`` and ``status`` disagreeing must still fail closed.

    Whichever field says "not done" wins; neither can veto the other back to
    a happy-path result.
    """

    async def execute_task():
        return dict(child_result)

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["success"] is False
    assert result["is_error"] is True
    assert result["status"] == "error"
    assert "failure_code" not in result
    assert tool_result_succeeded(result) is False


@pytest.mark.asyncio
async def test_agent_tool_interrupt_uses_error_text_over_placeholder_output(
    monkeypatch,
):
    """The interrupted child's real diagnostic must win over its placeholder text.

    The execution adapter replaces ``output`` with a user-facing placeholder
    message for interrupted runs, but keeps the real diagnostic in ``error``.
    The classified failure must surface the diagnostic, not the placeholder.
    """

    async def execute_task():
        return {
            "status": "interrupted",
            "success": False,
            "error": "child tool call was cancelled mid-flight",
            "output": "The assistant was interrupted before finishing.",
        }

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["success"] is False
    assert "failure_code" not in result
    assert result["error"] == "child tool call was cancelled mid-flight"
    assert tool_result_succeeded(result) is False


@pytest.mark.asyncio
async def test_agent_tool_normal_child_result_unchanged(monkeypatch):
    """A plain completed child result must keep its exact legacy shape.

    This also pins the "absent status/success is inert" rule: fakes that omit
    both fields (as this one does) must stay on the happy path.
    """

    async def execute_task():
        return {"output": "all good"}

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result == {"response": "all good"}


@pytest.mark.asyncio
async def test_agent_tool_catchall_does_not_rewrap_classified_failure(monkeypatch):
    """The classified failure must return before the catch-all can touch it.

    Using a close config that does not raise proves this end to end: the
    classified dict built ahead of the ``except Exception`` block must reach
    the caller unrewrapped, not folded into the generic
    ``Error executing agent ...`` message.
    """

    async def execute_task():
        return {"status": "waiting_for_user", "output": "partial"}

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["failure_code"] == "unsupported_nested_interaction"
    assert not str(result["response"]).startswith("Error executing agent")
    assert tool_result_succeeded(result) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "child_result",
    [
        {"status": "waiting_for_user", "output": "partial"},
        {"success": False, "status": "failed", "error": "child blew up"},
    ],
)
async def test_agent_tool_classified_failure_still_traces_delegation_error(
    monkeypatch, child_result
):
    """Both classified paths must emit the delegation terminal event.

    A delegated run's public outcome is derived from the
    ``workforce_delegation_start``/``_end``/``_error`` trace events, so a
    classified failure that returned without tracing one would leave the
    child showing as still running. Emitting the terminal event is part of
    the classified-failure contract, not an optional extra.
    """

    traced = []

    async def execute_task():
        return dict(child_result)

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    async def _record(status, **kwargs):
        traced.append((status, kwargs))

    monkeypatch.setattr(tool, "_trace_delegation", _record)

    result = await tool.run_json_async({"task": "run"})

    assert tool_result_succeeded(result) is False
    assert [status for status, _ in traced] == ["start", "error"]
    assert traced[-1][1]["error"] == result["error"]
    assert traced[-1][1]["execution_task_id"] is not None


def _create_factory() -> tuple[sessionmaker, str]:
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    db_url = f"sqlite:///{temp_db.name}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal, temp_db.name


def test_agent_tool_does_not_share_a_live_session_with_child_config(monkeypatch):
    """The child WebToolConfig must be built with a factory, never a live session."""
    SessionLocal, db_path = _create_factory()
    try:
        seed = SessionLocal()
        try:
            user = User(username="iso_owner", password_hash="x", is_admin=False)
            seed.add(user)
            seed.commit()
            seed.refresh(user)

            model = Model(
                model_id="general-model",
                model_provider="openai",
                model_name="General Model",
                api_key="x",
            )
            seed.add(model)
            seed.commit()
            seed.refresh(model)

            agent = Agent(
                user_id=user.id,
                name="Iso Worker",
                status=AgentStatus.PUBLISHED,
                models={"general": model.id},
            )
            seed.add(agent)
            seed.commit()
            seed.refresh(agent)

            agent_id = agent.id
            user_id = user.id
        finally:
            seed.close()

        # Make model resolution succeed so we reach the WebToolConfig build.
        monkeypatch.setattr(
            UserAwareModelStorage,
            "get_llm_by_name_with_access",
            lambda self, model_id, uid: object(),
        )

        captured: dict = {}

        def spy(*args, **kwargs):
            captured["db"] = kwargs.get("db")
            captured["db_factory"] = kwargs.get("db_factory")
            raise _Stop()

        monkeypatch.setattr(mod, "WebToolConfig", spy)

        tool = AgentTool(
            agent_id=agent_id,
            agent_name="Iso Worker",
            agent_description="d",
            session_factory=SessionLocal,
            user_id=user_id,
            tool_name="t",
            tool_description="d",
        )

        try:
            asyncio.run(tool.run_json_async({"task": "hi"}))
        except _Stop:
            pass

        assert captured["db"] is None
        assert captured["db_factory"] is SessionLocal
    finally:
        try:
            os.remove(db_path)
        except OSError:
            pass
