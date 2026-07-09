from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPToolAdapter,
    _build_mcp_tool_adapter,
    _exception_indicates_http_401,
    _mcp_return_value_as_string,
)


def test_build_mcp_tool_adapter_stamps_normalized_source_server():
    """``_build_mcp_tool_adapter`` carries the originating server identity
    onto ``metadata.source_server``, normalized once via the shared SSOT,
    while the LLM-visible name keeps its original casing. Server-scoped
    selection matches on the structured field, not the tool name."""
    mcp_tool = SimpleNamespace(
        name="send_message",
        description="Send a message",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = _build_mcp_tool_adapter(
        "Google Drive",
        {"transport": "stdio", "command": "python", "args": []},
        mcp_tool,
    )

    assert adapter.source_server == "google_drive"
    assert adapter.metadata.source_server == "google_drive"
    # LLM-visible name keeps original casing / spacing folded to underscores.
    assert adapter.name == "mcp_Google Drive_send_message".replace(" ", "_")


def test_mcp_tool_adapter_source_server_defaults_none():
    """A directly constructed adapter with no server origin reports
    ``source_server`` as ``None`` (no scoped-selection match)."""
    mcp_tool = SimpleNamespace(
        name="ping",
        description="ping",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )
    assert adapter.source_server is None
    assert adapter.metadata.source_server is None


def test_mcp_tool_adapter_defaults_to_not_concurrency_safe():
    mcp_tool = SimpleNamespace(
        name="list_messages",
        description="List messages",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    assert adapter.metadata.concurrency_safe is False


def test_build_mcp_tool_adapter_marks_all_tools_safe_when_server_opts_in():
    mcp_tool = SimpleNamespace(
        name="list_messages",
        description="List messages",
        inputSchema={"type": "object", "properties": {}},
    )

    adapter = _build_mcp_tool_adapter(
        "mail",
        {"transport": "stdio", "command": "python", "args": []},
        mcp_tool,
        concurrency_safe=True,
    )

    assert adapter.metadata.concurrency_safe is True


def test_exception_indicates_http_401_uses_bounded_status_signals():
    class StatusError(RuntimeError):
        status_code = 401

    assert _exception_indicates_http_401(StatusError("request failed"))
    assert _exception_indicates_http_401(RuntimeError("HTTP status 401"))
    assert _exception_indicates_http_401(RuntimeError("401 Unauthorized"))
    assert not _exception_indicates_http_401(RuntimeError("Unauthorized"))
    assert not _exception_indicates_http_401(
        RuntimeError("connection reset on port 401")
    )
    assert not _exception_indicates_http_401(RuntimeError("tool returned id 40123"))


def test_mcp_return_value_as_string_keeps_malformed_scalar_content_together():
    assert _mcp_return_value_as_string({"content": "error"}) == "error"


def test_build_mcp_tool_adapter_honors_concurrent_tool_allowlist():
    safe_tool = SimpleNamespace(
        name="list_messages",
        description="List messages",
        inputSchema={"type": "object", "properties": {}},
    )
    unsafe_tool = SimpleNamespace(
        name="delete_message",
        description="Delete a message",
        inputSchema={"type": "object", "properties": {}},
    )

    safe_adapter = _build_mcp_tool_adapter(
        "mail",
        {"transport": "stdio", "command": "python", "args": []},
        safe_tool,
        concurrency_safe=True,
        concurrent_tools=["list_messages"],
    )
    unsafe_adapter = _build_mcp_tool_adapter(
        "mail",
        {"transport": "stdio", "command": "python", "args": []},
        unsafe_tool,
        concurrency_safe=True,
        concurrent_tools=["list_messages"],
    )

    assert safe_adapter.metadata.concurrency_safe is True
    assert unsafe_adapter.metadata.concurrency_safe is False


def test_build_args_model_handles_optional_array_schema():
    mcp_tool = SimpleNamespace(
        name="gmail_manage_labels",
        description="Manage Gmail labels",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "add_label_ids": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": ["action"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    args_model = adapter.args_type()
    parsed = args_model(action="modify_message", add_label_ids=["TRASH"])

    assert parsed.add_label_ids == ["TRASH"]


def test_normalize_args_by_schema_wraps_scalar_for_array_only_field():
    mcp_tool = SimpleNamespace(
        name="gmail_manage_labels",
        description="Manage Gmail labels",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "add_label_ids": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": ["action"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    normalized = adapter._normalize_args_by_schema(
        {"action": "modify_message", "add_label_ids": "TRASH"}
    )

    assert normalized["add_label_ids"] == ["TRASH"]


def test_normalize_args_by_schema_keeps_scalar_for_union_scalar_or_array_field():
    mcp_tool = SimpleNamespace(
        name="multi_shape_tool",
        description="Accept string or string array input",
        inputSchema={
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                }
            },
            "required": ["value"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    normalized = adapter._normalize_args_by_schema({"value": "abc"})

    assert normalized["value"] == "abc"


def test_build_args_model_handles_anyof_multi_type_schema():
    mcp_tool = SimpleNamespace(
        name="multi_type_tool",
        description="Accept string or integer input",
        inputSchema={
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ]
                }
            },
            "required": ["value"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    args_model = adapter.args_type()

    assert args_model(value="abc").value == "abc"
    assert args_model(value=123).value == 123


def test_build_args_model_handles_multi_value_type_list():
    mcp_tool = SimpleNamespace(
        name="multi_value_type_tool",
        description="Accept string or integer input",
        inputSchema={
            "type": "object",
            "properties": {"value": {"type": ["string", "integer", "null"]}},
            "required": ["value"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    args_model = adapter.args_type()

    assert args_model(value="abc").value == "abc"
    assert args_model(value=123).value == 123


@pytest.mark.asyncio
async def test_runtime_bindings_hide_and_inject_mcp_meta_and_tool_arguments(
    monkeypatch,
):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "account_id": {"type": "string"},
            },
            "required": ["query", "account_id"],
        },
    )
    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "runtime_bindings": [
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "tool_arguments", "key": "account_id"},
            },
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "mcp_meta", "key": "account_id"},
            },
        ],
        "connector_runtime": {
            "context": {"account_id": "6185"},
            "secrets": {},
            "auth_selector": {},
        },
    }
    adapter = MCPToolAdapter(mcp_tool=mcp_tool, connection=connection)
    captured = {}

    class _FakeSession:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            captured["name"] = name
            captured["arguments"] = arguments
            captured["kwargs"] = kwargs
            return SimpleNamespace(content=[], isError=False)

    @asynccontextmanager
    async def _fake_create_session(_connection):
        yield _FakeSession()

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _fake_create_session,
    )

    args_model = adapter.args_type()
    assert "account_id" not in args_model.model_fields

    result = await adapter.run_json_async(
        {"query": "active", "account_id": "llm-supplied"}
    )

    assert result["is_error"] is False
    assert captured["name"] == "list_clients"
    assert captured["arguments"] == {"query": "active", "account_id": "6185"}
    assert captured["kwargs"]["meta"] == {"account_id": "6185"}


def test_mcp_runtime_tool_argument_missing_source_warns(caplog):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={
            "transport": "stdio",
            "command": "python",
            "args": [],
            "runtime_bindings": [
                {
                    "source": {"input_type": "context", "key": "account_id"},
                    "target": {"target_type": "tool_arguments", "key": "account_id"},
                },
            ],
            "connector_runtime": {"context": {}, "secrets": {}, "auth_selector": {}},
        },
    )

    caplog.set_level("WARNING")
    assert adapter._runtime_tool_arguments() == {}
    assert (
        "Skipping runtime MCP tool argument binding for missing context source"
        in caplog.text
    )
    assert "account_id" in caplog.text
    assert "list_clients" in caplog.text


@pytest.mark.asyncio
async def test_delegated_authorization_401_refreshes_connection_once(monkeypatch):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={"type": "object", "properties": {}},
    )
    connections = []
    refresh_calls = 0

    def _refresh_connection():
        nonlocal refresh_calls
        refresh_calls += 1
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "headers": {"Authorization": "Bearer fresh-token"},
        }

    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "headers": {"Authorization": "Bearer expired-token"},
        "_connector_runtime_refresh": _refresh_connection,
    }
    adapter = MCPToolAdapter(mcp_tool=mcp_tool, connection=connection)

    class _FakeSession:
        def __init__(self, connection):
            self._connection = connection

        async def initialize(self):
            if self._connection["headers"]["Authorization"] == "Bearer expired-token":
                raise RuntimeError("HTTP 401 Unauthorized")

        async def call_tool(self, name, arguments, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(model_dump=lambda: {"text": "ok"})],
                isError=False,
            )

    @asynccontextmanager
    async def _fake_create_session(connection):
        connections.append(connection)
        yield _FakeSession(connection)

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _fake_create_session,
    )

    result = await adapter.run_json_async({})

    assert result == {"content": [{"text": "ok"}], "is_error": False}
    assert refresh_calls == 1
    assert [item["headers"]["Authorization"] for item in connections] == [
        "Bearer expired-token",
        "Bearer fresh-token",
    ]


@pytest.mark.asyncio
async def test_delegated_authorization_401_with_non_mapping_connection_does_not_crash(
    monkeypatch,
):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection=SimpleNamespace(transport="streamable_http"),
    )

    class _FakeSession:
        async def initialize(self):
            raise RuntimeError("HTTP 401 Unauthorized")

    @asynccontextmanager
    async def _fake_create_session(connection):
        yield _FakeSession()

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _fake_create_session,
    )

    result = await adapter.run_json_async({})

    assert result["is_error"] is True
    assert result["content"][0]["text"] == "Error executing MCP tool."
    assert "AttributeError" not in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_mcp_tool_execution_error_does_not_echo_raw_exception(monkeypatch):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "streamable_http", "url": "https://mcp.example.test"},
    )

    class _FakeSession:
        async def initialize(self):
            raise RuntimeError("transport failed with Bearer runtime-token")

    @asynccontextmanager
    async def _fake_create_session(connection):
        yield _FakeSession()

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _fake_create_session,
    )

    result = await adapter.run_json_async({})

    assert result["is_error"] is True
    assert result["content"][0]["text"] == "Error executing MCP tool."
    assert "runtime-token" not in repr(result)


@pytest.mark.asyncio
async def test_delegated_authorization_401_after_refresh_returns_safe_error(
    monkeypatch,
):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={"type": "object", "properties": {}},
    )
    refresh_calls = 0

    def _refresh_connection():
        nonlocal refresh_calls
        refresh_calls += 1
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "headers": {"Authorization": "Bearer still-expired-token"},
        }

    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "headers": {"Authorization": "Bearer expired-token"},
        "_connector_runtime_refresh": _refresh_connection,
    }
    adapter = MCPToolAdapter(mcp_tool=mcp_tool, connection=connection)

    class _FakeSession:
        async def initialize(self):
            raise RuntimeError("HTTP 401 Unauthorized")

    @asynccontextmanager
    async def _fake_create_session(connection):
        yield _FakeSession()

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _fake_create_session,
    )

    result = await adapter.run_json_async({})

    assert refresh_calls == 1
    assert result["is_error"] is True
    assert "delegated_authorization_failed" in result["content"][0]["text"]
    assert "expired-token" not in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_delegated_authorization_retry_failure_does_not_leak_token(
    monkeypatch,
    caplog,
):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={"type": "object", "properties": {}},
    )

    def _refresh_connection():
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "headers": {"Authorization": "Bearer fresh-runtime-token"},
        }

    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "headers": {"Authorization": "Bearer expired-token"},
        "_connector_runtime_refresh": _refresh_connection,
    }
    adapter = MCPToolAdapter(mcp_tool=mcp_tool, connection=connection)
    calls = 0

    class _FakeSession:
        def __init__(self, connection):
            self._connection = connection

        async def initialize(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("HTTP 401 Unauthorized")
            raise RuntimeError(
                f"transport failed with {self._connection['headers']['Authorization']}"
            )

    @asynccontextmanager
    async def _fake_create_session(connection):
        yield _FakeSession(connection)

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _fake_create_session,
    )
    caplog.set_level("ERROR")

    result = await adapter.run_json_async({})

    assert calls == 2
    assert result["is_error"] is True
    assert result["content"][0]["text"] == (
        "Error executing MCP tool after delegated authorization retry."
    )
    public_output = repr(result) + caplog.text
    assert "fresh-runtime-token" not in public_output
    assert "expired-token" not in public_output
