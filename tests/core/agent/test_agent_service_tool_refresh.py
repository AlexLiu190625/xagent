from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent.service import AgentService
from xagent.core.tools.adapters.vibe.config import (
    MCPToolLoadSummary,
    MCPUnavailableSummary,
    RequiredMCPUnavailableError,
)
from xagent.core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ConnectorRuntimeError,
)


class NamedTool:
    def __init__(self, name: str) -> None:
        self.name = name


class RefreshingToolConfig:
    _workspace_config = None

    def __init__(self, allowed_tools: list[str] | None = None) -> None:
        self.refresh_count = 0
        self.allowed_tools = allowed_tools

    def refresh_user_tool_overrides(self) -> None:
        self.refresh_count += 1

    def get_user_tool_overrides(self) -> dict[str, dict[str, bool]]:
        return {"disabled": {"enabled": self.refresh_count < 2}}

    def get_allowed_tools(self) -> list[str] | None:
        return self.allowed_tools


class AllowedToolsConfig:
    _workspace_config = None

    def __init__(self, allowed_tools: list[str] | None = None) -> None:
        self.allowed_tools = allowed_tools

    def get_user_tool_overrides(self) -> dict[str, dict[str, bool]]:
        return {}

    def get_allowed_tools(self) -> list[str] | None:
        return self.allowed_tools


class DelegationRuntimeConfig:
    _workspace_config = None

    def __init__(self) -> None:
        self.parent_task_id = "parent-task-1"
        self.parent_tracer = object()

    def get_user_tool_overrides(self) -> dict[str, dict[str, bool]]:
        return {}

    def get_allowed_tools(self) -> list[str] | None:
        return None

    def get_allowed_agent_ids(self) -> list[int] | None:
        return None

    def get_agent_tool_overrides(self) -> dict[int, dict[str, Any]]:
        return {}

    def get_enable_global_agent_tools(self) -> bool:
        return True

    def get_allow_cross_user_agent_ids(self) -> bool:
        return False

    def get_parent_task_id(self) -> str:
        return self.parent_task_id

    def get_parent_tracer(self) -> object:
        return self.parent_tracer

    def get_agent_call_stack(self) -> list[int]:
        return []


@pytest.mark.asyncio
async def test_agent_service_refreshes_initialized_tools_when_policy_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_config = RefreshingToolConfig()
    tool_sets: list[list[Any]] = [
        [NamedTool("allowed"), NamedTool("disabled")],
        [NamedTool("allowed")],
    ]

    async def create_all_tools(config: Any) -> list[Any]:
        assert config is tool_config
        return tool_sets.pop(0)

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )

    service = AgentService(
        name="tool-refresh-test",
        id="tool-refresh-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    await service._ensure_tools_initialized()
    assert {tool.name for tool in service.tools} == {"allowed", "disabled"}

    await service._ensure_tools_initialized()
    assert {tool.name for tool in service.tools} == {"allowed"}
    assert tool_config.refresh_count == 2


@pytest.mark.asyncio
async def test_agent_service_rebuild_reuses_mcp_summary_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SummaryObservingConfig(RefreshingToolConfig):
        def __init__(self) -> None:
            super().__init__()
            self.summaries: list[MCPToolLoadSummary] = []

        async def emit_mcp_load_summary(self, summary: MCPToolLoadSummary) -> None:
            self.summaries.append(summary)

    tool_config = SummaryObservingConfig()
    observed_config_ids: list[int] = []

    async def create_all_tools(config: Any) -> list[Any]:
        observed_config_ids.append(id(config))
        await config.emit_mcp_load_summary(MCPToolLoadSummary())
        return [NamedTool("allowed")]

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )
    service = AgentService(
        name="mcp-summary-observer-refresh-test",
        id="mcp-summary-observer-refresh-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    await service._ensure_tools_initialized()
    await service._ensure_tools_initialized()

    assert observed_config_ids == [id(tool_config), id(tool_config)]
    assert tool_config.summaries == [MCPToolLoadSummary(), MCPToolLoadSummary()]


@pytest.mark.asyncio
async def test_agent_service_refreshes_when_allowed_tools_changes_from_all_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_config = AllowedToolsConfig(allowed_tools=None)

    async def create_all_tools(config: Any) -> list[Any]:
        assert config is tool_config
        return [NamedTool("allowed"), NamedTool("disabled")]

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )

    service = AgentService(
        name="allowed-tools-refresh-test",
        id="allowed-tools-refresh-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    await service._ensure_tools_initialized()
    assert {tool.name for tool in service.tools} == {"allowed", "disabled"}

    tool_config.allowed_tools = []
    await service._ensure_tools_initialized()
    assert service.tools == []


@pytest.mark.asyncio
async def test_agent_service_refreshes_when_delegation_parent_context_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_config = DelegationRuntimeConfig()
    tool_sets: list[list[Any]] = [
        [NamedTool("initial")],
        [NamedTool("new-task")],
        [NamedTool("new-tracer")],
    ]

    async def create_all_tools(config: Any) -> list[Any]:
        assert config is tool_config
        return tool_sets.pop(0)

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )

    service = AgentService(
        name="delegation-runtime-refresh-test",
        id="delegation-runtime-refresh-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    await service._ensure_tools_initialized()
    assert [tool.name for tool in service.tools] == ["initial"]

    tool_config.parent_task_id = "parent-task-2"
    await service._ensure_tools_initialized()
    assert [tool.name for tool in service.tools] == ["new-task"]

    tool_config.parent_tracer = object()
    await service._ensure_tools_initialized()
    assert [tool.name for tool in service.tools] == ["new-tracer"]


@pytest.mark.asyncio
async def test_agent_service_preserves_connector_runtime_initialization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_config = AllowedToolsConfig()
    runtime_error = ConnectorRuntimeError(
        ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
        "Connector runtime context is unavailable.",
        details={"reason": "runtime_view_resolution_failed"},
        status_code=503,
    )

    async def create_all_tools(config: Any) -> list[Any]:
        assert config is tool_config
        raise runtime_error

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )

    service = AgentService(
        name="runtime-error-test",
        id="runtime-error-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        await service._ensure_tools_initialized()

    assert exc_info.value is runtime_error
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_agent_service_preserves_required_mcp_initialization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_config = AllowedToolsConfig()
    required_error = RequiredMCPUnavailableError(
        [MCPUnavailableSummary.from_values("Gmail", "oauth_token_required")]
    )

    async def create_all_tools(config: Any) -> list[Any]:
        assert config is tool_config
        raise required_error

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )
    service = AgentService(
        name="required-mcp-error-test",
        id="required-mcp-error-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    with pytest.raises(RequiredMCPUnavailableError) as exc_info:
        await service._ensure_tools_initialized()

    assert exc_info.value is required_error
