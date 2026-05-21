from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent.service import AgentService


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
