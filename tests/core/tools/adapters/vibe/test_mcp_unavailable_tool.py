from __future__ import annotations

import logging
import re

import pytest
from mcp.types import Tool as MCPTool

from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPToolAdapter,
    UnavailableMCPTool,
)
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec


def _unavailable_tool(
    *,
    server_name: str = "Google Drive",
    server_id: int | None = 42,
    allow_users: list[str] | None = None,
) -> UnavailableMCPTool:
    return UnavailableMCPTool(
        server_name=server_name,
        server_id=server_id,
        allow_users=allow_users,
    )


def _unavailable_config(
    *,
    name: str | None = "Google Drive",
    server_id: int | None = 42,
    allow_users: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "transport": "unavailable",
        "description": f"{name} server",
        "config": {
            "unavailable": True,
            "reason": "oauth_token_resolver_failed",
            "message": "MCP server credentials are unavailable.",
            "server_id": server_id,
        },
        "allow_users": allow_users,
    }


def test_unavailable_tool_name_is_llm_safe_and_uses_server_id():
    tool = _unavailable_tool(server_name="Google Drive", server_id=42)

    assert tool.name == "mcp_google_drive_42_unavailable"
    assert re.fullmatch(r"[A-Za-z0-9_]+", tool.name)


def test_unavailable_tool_names_avoid_flat_collisions_with_server_id():
    first = _unavailable_tool(server_name="Google Drive", server_id=1)
    second = _unavailable_tool(server_name="google-drive", server_id=2)

    assert first.name == "mcp_google_drive_1_unavailable"
    assert second.name == "mcp_google_drive_2_unavailable"
    assert first.name != second.name


def test_unavailable_tool_empty_server_name_has_stable_fallback():
    tool = _unavailable_tool(server_name="!!!", server_id=None)

    assert tool.name == "mcp_server_unavailable"
    assert re.fullmatch(r"[A-Za-z0-9_]+", tool.name)


def test_unavailable_tool_metadata_is_mcp_and_server_scoped():
    tool = _unavailable_tool(allow_users=["7"])

    assert tool.metadata.category == ToolCategory.MCP
    assert tool.metadata.source_server == "google_drive"
    assert tool.metadata.allow_users == ["7"]
    assert tool.metadata.read_only is True
    assert tool.metadata.concurrency_safe is True


@pytest.mark.asyncio
async def test_unavailable_tool_async_authorized_returns_clean_error(monkeypatch):
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    tool = _unavailable_tool(allow_users=["7"])

    result = await tool.run_json_async({})

    assert result["is_error"] is True
    text = result["content"][0]["text"]
    assert "MCP server credentials are unavailable" in text
    assert "resolver" not in text
    assert "provider" not in text
    assert "RuntimeError" not in text


def test_unavailable_tool_sync_authorized_returns_clean_error(monkeypatch):
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    tool = _unavailable_tool(allow_users=["7"])

    result = tool.run_json_sync({})

    assert result["is_error"] is True
    assert "MCP server credentials are unavailable" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_unavailable_tool_async_unauthorized_uses_mcp_access_denied(
    monkeypatch,
):
    monkeypatch.setenv("XAGENT_USER_ID", "8")
    tool = _unavailable_tool(allow_users=["7"])

    result = await tool.run_json_async({})

    assert result == {
        "content": [
            {
                "text": (
                    "Access denied: User 8 is not authorized to use tool "
                    "mcp_google_drive_42_unavailable"
                )
            }
        ],
        "is_error": True,
    }


def test_unavailable_tool_sync_unauthorized_uses_mcp_access_denied(monkeypatch):
    monkeypatch.setenv("XAGENT_USER_ID", "8")
    tool = _unavailable_tool(allow_users=["7"])

    result = tool.run_json_sync({})

    assert result["is_error"] is True
    assert "Access denied" in result["content"][0]["text"]


def test_unavailable_tool_missing_user_permission_regression(monkeypatch):
    monkeypatch.delenv("XAGENT_USER_ID", raising=False)

    assert _unavailable_tool(allow_users=None).run_json_sync({})["is_error"] is True
    assert (
        "MCP server credentials"
        in _unavailable_tool(allow_users=None).run_json_sync({})["content"][0]["text"]
    )
    assert (
        "Access denied"
        in _unavailable_tool(allow_users=["7"]).run_json_sync({})["content"][0]["text"]
    )
    assert (
        "MCP server credentials"
        in _unavailable_tool(allow_users=["system"]).run_json_sync({})["content"][0][
            "text"
        ]
    )


def test_mcp_tool_adapter_permission_helper_regression(monkeypatch):
    adapter = MCPToolAdapter(
        MCPTool(name="remote_tool", description="Remote tool", inputSchema={}),
        {"transport": "stdio", "command": "echo"},
        allow_users=["7"],
    )

    monkeypatch.delenv("XAGENT_USER_ID", raising=False)
    assert adapter._get_current_user_id() is None
    assert adapter._is_user_allowed(None) is False

    system_adapter = MCPToolAdapter(
        MCPTool(name="remote_tool", description="Remote tool", inputSchema={}),
        {"transport": "stdio", "command": "echo"},
        allow_users=["system"],
    )
    assert system_adapter._is_user_allowed(None) is True

    unrestricted_adapter = MCPToolAdapter(
        MCPTool(name="remote_tool", description="Remote tool", inputSchema={}),
        {"transport": "stdio", "command": "echo"},
        allow_users=None,
    )
    assert unrestricted_adapter._is_user_allowed(None) is True

    monkeypatch.setenv("XAGENT_USER_ID", "8")
    assert adapter._get_current_user_id() == "8"
    assert adapter._is_user_allowed("8") is False
    assert adapter._is_user_allowed("7") is True


def test_unavailable_tool_return_value_as_string_extracts_text():
    tool = _unavailable_tool()

    assert (
        tool.return_value_as_string(
            {"content": [{"text": "first"}, {"text": "second"}], "is_error": True}
        )
        == "first\nsecond"
    )


@pytest.mark.asyncio
async def test_factory_builds_unavailable_tools_without_normal_loader(monkeypatch):
    async def fail_loader(*args, **kwargs):
        raise AssertionError("normal loader should not be called")

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        fail_loader,
    )

    tools = await ToolFactory._create_mcp_tools_from_configs([_unavailable_config()])

    assert [tool.name for tool in tools] == ["mcp_google_drive_42_unavailable"]


@pytest.mark.asyncio
async def test_factory_normal_loader_failure_keeps_unavailable_tool(monkeypatch):
    async def fail_loader(*args, **kwargs):
        raise RuntimeError("loader failed")

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        fail_loader,
    )

    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            _unavailable_config(),
            {"name": "normal", "transport": "stdio", "config": {"command": "echo"}},
        ]
    )

    assert [tool.name for tool in tools] == ["mcp_google_drive_42_unavailable"]


@pytest.mark.asyncio
async def test_factory_malformed_normal_config_keeps_unavailable_tool(caplog):
    caplog.set_level(logging.WARNING)

    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            _unavailable_config(),
            {"name": "normal", "transport": "stdio", "config": None},
        ]
    )

    assert [tool.name for tool in tools] == ["mcp_google_drive_42_unavailable"]
    assert "MCP server config 'config' field for server 'normal'" in caplog.text
    assert "dictionary, got NoneType" in caplog.text


@pytest.mark.asyncio
async def test_factory_does_not_pass_unavailable_config_to_normal_loader(monkeypatch):
    seen_connections = None

    async def loader(connections, **kwargs):
        nonlocal seen_connections
        seen_connections = connections
        return []

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        loader,
    )

    await ToolFactory._create_mcp_tools_from_configs(
        [
            _unavailable_config(),
            {
                "name": "normal",
                "transport": "stdio",
                "config": {"command": "echo", "args": "--flag value"},
            },
        ]
    )

    assert seen_connections == {
        "normal": {"transport": "stdio", "command": "echo", "args": ["--flag", "value"]}
    }


@pytest.mark.asyncio
async def test_factory_preserves_runtime_keys_for_normal_configs(monkeypatch):
    seen_connections = None

    async def loader(connections, **kwargs):
        nonlocal seen_connections
        seen_connections = connections
        return []

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        loader,
    )

    await ToolFactory._create_mcp_tools_from_configs(
        [
            {
                "name": "normal",
                "transport": "streamable_http",
                "config": {"url": "https://mcp.example.test"},
                "runtime_bindings": [{"binding": "value"}],
                "runtime_input_schema": {"context": {"account": {"type": "string"}}},
                "connector_runtime": {"context": {"account": "a"}},
                "allow_delegated_authorization": True,
            },
        ]
    )

    assert seen_connections == {
        "normal": {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "runtime_bindings": [{"binding": "value"}],
            "runtime_input_schema": {"context": {"account": {"type": "string"}}},
            "connector_runtime": {"context": {"account": "a"}},
            "allow_delegated_authorization": True,
        }
    }


@pytest.mark.asyncio
async def test_factory_propagates_connector_runtime_error(monkeypatch):
    async def fail_loader(*args, **kwargs):
        raise ConnectorRuntimeError(
            "connector_runtime_unavailable",
            "Connector runtime context is unavailable.",
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        fail_loader,
    )

    with pytest.raises(ConnectorRuntimeError):
        await ToolFactory._create_mcp_tools_from_configs(
            [
                _unavailable_config(),
                {"name": "normal", "transport": "stdio", "config": {"command": "echo"}},
            ]
        )


@pytest.mark.asyncio
async def test_factory_returns_unavailable_tools_before_normal_tools(monkeypatch):
    normal_tool = _unavailable_tool(server_name="Normal", server_id=99)

    async def loader(connections, **kwargs):
        return [normal_tool]

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        loader,
    )

    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            {"name": "normal", "transport": "stdio", "config": {"command": "echo"}},
            _unavailable_config(),
        ]
    )

    assert [tool.name for tool in tools] == [
        "mcp_google_drive_42_unavailable",
        "mcp_normal_99_unavailable",
    ]


@pytest.mark.asyncio
async def test_factory_malformed_unavailable_config_does_not_drop_valid_one():
    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            _unavailable_config(name=None, server_id=None),
            _unavailable_config(name="Google Drive", server_id=42),
        ]
    )

    assert [tool.name for tool in tools] == [
        "mcp_server_unavailable",
        "mcp_google_drive_42_unavailable",
    ]


def test_selection_plain_mcp_admits_unavailable_tool():
    tool = _unavailable_tool()
    spec = ToolSelectionSpec.from_raw(tool_categories=["mcp"])

    assert spec.compute_allowed_names([tool]) == frozenset({tool.name})


def test_selection_scoped_mcp_admits_unavailable_tool_by_source_server():
    tool = _unavailable_tool()
    spec = ToolSelectionSpec.from_raw(tool_categories=["mcp:Google Drive"])

    assert spec.compute_allowed_names([tool]) == frozenset({tool.name})


def test_selection_unrelated_scoped_mcp_rejects_unavailable_tool():
    tool = _unavailable_tool()
    spec = ToolSelectionSpec.from_raw(tool_categories=["mcp:Slack"])

    assert spec.compute_allowed_names([tool]) == frozenset()
