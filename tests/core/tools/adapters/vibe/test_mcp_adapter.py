import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPToolAdapter,
    _build_mcp_tool_adapter,
    _exception_indicates_http_401,
    _mcp_return_value_as_string,
)


def _http_status_error(
    *, status_code: int = 401, authenticate: list[str] | None = None
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://mcp.example.test/tools")
    headers = [(b"www-authenticate", value.encode()) for value in (authenticate or [])]
    response = httpx.Response(status_code, headers=headers, request=request)
    return httpx.HTTPStatusError(
        "planted-secret-must-not-be-evidence",
        request=request,
        response=response,
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


def test_resolver_challenge_extracts_from_nested_group_cause_and_context():
    cause = RuntimeError("outer")
    cause.__cause__ = _http_status_error(
        authenticate=['Bearer error="invalid_token", scope="records.read"']
    )
    context = RuntimeError("context")
    context.__context__ = cause
    nested = ExceptionGroup("nested", [RuntimeError("noise"), context])

    challenge = mcp_adapter_module._resolver_invalid_token_challenge(
        ExceptionGroup("root", [nested])
    )

    assert challenge is not None
    assert challenge.params["error"] == "invalid_token"
    assert challenge.scope == "records.read"


def test_resolver_challenge_traversal_handles_cycles():
    outer = RuntimeError("outer")
    inner = RuntimeError("inner")
    outer.__cause__ = inner
    inner.__context__ = outer

    assert mcp_adapter_module._resolver_invalid_token_challenge(outer) is None


def test_resolver_challenge_traversal_stops_at_named_node_budget():
    current: BaseException = _http_status_error(
        authenticate=['Bearer error="invalid_token"']
    )
    for index in range(mcp_adapter_module._RESOLVER_HTTP_401_NODE_LIMIT):
        wrapper = RuntimeError(f"wrapper-{index}")
        wrapper.__cause__ = current
        current = wrapper

    assert mcp_adapter_module._resolver_invalid_token_challenge(current) is None


def test_resolver_challenge_uses_all_www_authenticate_header_values():
    exc = _http_status_error(
        authenticate=[
            'Basic realm="legacy"',
            (
                'Bearer error="invalid_token", '
                'resource_metadata="https://mcp.example.test/.well-known/resource"'
            ),
        ]
    )

    challenge = mcp_adapter_module._resolver_invalid_token_challenge(exc)

    assert challenge is not None
    assert challenge.resource_metadata_url == (
        "https://mcp.example.test/.well-known/resource"
    )


@pytest.mark.parametrize(
    "exc",
    [
        _http_status_error(authenticate=[]),
        _http_status_error(authenticate=['Bearer error="invalid_token']),
        _http_status_error(authenticate=['Basic error="invalid_token"']),
        _http_status_error(authenticate=['Bearer error="insufficient_scope"']),
        _http_status_error(authenticate=['Bearer scope="records.read"']),
        _http_status_error(
            status_code=403, authenticate=['Bearer error="invalid_token"']
        ),
        RuntimeError("HTTP 401 Unauthorized; Bearer error=invalid_token"),
    ],
)
def test_resolver_challenge_rejects_non_refreshable_evidence(exc):
    assert mcp_adapter_module._resolver_invalid_token_challenge(exc) is None


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


def _resolver_retry_adapter(connection):
    return MCPToolAdapter(
        mcp_tool=SimpleNamespace(
            name="list_clients",
            description="List clients",
            inputSchema={"type": "object", "properties": {}},
        ),
        connection=connection,
    )


@pytest.mark.asyncio
async def test_resolver_retry_analyzes_initial_401_chain_once(monkeypatch):
    from xagent.core.agent.result import ClassifiedToolFailure

    strict_401_calls = 0
    strict_401_responses = mcp_adapter_module._strict_http_401_responses

    def counting_strict_401_responses(exc, **kwargs):
        nonlocal strict_401_calls
        strict_401_calls += 1
        yield from strict_401_responses(exc, **kwargs)

    async def _resolver_refresh(challenge):
        return ClassifiedToolFailure(failure_code="oauth_token_required")

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )
    monkeypatch.setattr(
        mcp_adapter_module,
        "_strict_http_401_responses",
        counting_strict_401_responses,
    )

    result = await adapter._retry_resolver_401(
        _http_status_error(authenticate=['Bearer error="invalid_token"']),
        {},
        {},
    )

    assert strict_401_calls == 1
    assert result is not None
    assert result["failure_code"] == "oauth_token_required"


@pytest.mark.asyncio
async def test_resolver_401_has_priority_and_retries_rebuilt_connection_once(
    monkeypatch,
):
    resolver_calls = []
    connector_calls = 0
    fresh_connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "headers": {"Authorization": "Bearer fresh-token"},
    }

    async def _resolver_refresh(challenge):
        resolver_calls.append(challenge)
        return fresh_connection

    def _connector_refresh():
        nonlocal connector_calls
        connector_calls += 1
        return fresh_connection

    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "_oauth_token_resolver_refresh": _resolver_refresh,
        "_connector_runtime_refresh": _connector_refresh,
    }
    adapter = _resolver_retry_adapter(connection)
    attempted_connections = []

    async def _execute(attempted, tool_args, tool_meta):
        attempted_connections.append(attempted)
        if attempted is connection:
            raise _http_status_error(
                authenticate=['Bearer error="invalid_token", scope="records.read"']
            )
        return {"content": [{"text": "ok"}], "is_error": False}

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert result == {"content": [{"text": "ok"}], "is_error": False}
    assert len(resolver_calls) == 1
    assert resolver_calls[0].params["error"] == "invalid_token"
    assert connector_calls == 0
    assert attempted_connections == [connection, fresh_connection]


@pytest.mark.asyncio
async def test_real_mcp_session_retries_nested_resolver_401_once(monkeypatch, caplog):
    initial_token = "real-initial-access-token-secret"
    refreshed_token = "real-refreshed-access-token-secret"
    generation = "real-resolver-generation-secret"
    raw_exception_secret = "real-http-status-error-secret"
    tool_args = {"query": "active", "account_id": "6185"}
    tool_meta = {"account_id": "6185"}
    initial_requests = []
    refreshed_requests = []
    initial_client_builds = 0
    refreshed_client_builds = 0
    refresh_calls = []

    async def _initial_handler(request):
        payload = json.loads(request.content) if request.content else {}
        initial_requests.append((request, payload))
        assert payload["method"] == "initialize"
        return httpx.Response(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer error="invalid_token", scope="records.read"'
                )
            },
            extensions={"reason_phrase": raw_exception_secret.encode()},
            request=request,
        )

    async def _refreshed_handler(request):
        payload = json.loads(request.content) if request.content else {}
        refreshed_requests.append((request, payload))
        method = payload.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-mcp", "version": "1"},
            }
        elif method == "tools/call":
            result = {
                "content": [{"type": "text", "text": "clients-ok"}],
                "isError": False,
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "list_clients",
                        "description": "List clients",
                        "inputSchema": {"type": "object"},
                    }
                ]
            }
        else:
            return httpx.Response(405 if request.method == "GET" else 202)
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "mcp-session-id": "refreshed-session",
            },
            json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
            request=request,
        )

    def _initial_client_factory(headers=None, timeout=None, auth=None):
        nonlocal initial_client_builds
        initial_client_builds += 1
        return httpx.AsyncClient(
            transport=httpx.MockTransport(_initial_handler),
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    def _refreshed_client_factory(headers=None, timeout=None, auth=None):
        nonlocal refreshed_client_builds
        refreshed_client_builds += 1
        return httpx.AsyncClient(
            transport=httpx.MockTransport(_refreshed_handler),
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    async def _resolver_refresh(challenge):
        refresh_calls.append((challenge, generation))
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "headers": {"Authorization": f"Bearer {refreshed_token}"},
            "httpx_client_factory": _refreshed_client_factory,
            "terminate_on_close": False,
        }

    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "headers": {"Authorization": f"Bearer {initial_token}"},
        "httpx_client_factory": _initial_client_factory,
        "terminate_on_close": False,
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
        "_oauth_token_resolver_refresh": _resolver_refresh,
    }
    adapter = MCPToolAdapter(
        mcp_tool=SimpleNamespace(
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
        ),
        connection=connection,
    )
    nested_failures = []
    retry = adapter._retry_after_authorization_failure

    async def _observe_nested_failure(exc, attempted_args, attempted_meta):
        nested_failures.append(exc)
        responses = list(mcp_adapter_module._strict_http_401_responses(exc))
        assert len(responses) == 1
        assert responses[0].headers.get_list("WWW-Authenticate") == [
            'Bearer error="invalid_token", scope="records.read"'
        ]
        return await retry(exc, attempted_args, attempted_meta)

    monkeypatch.setattr(
        adapter, "_retry_after_authorization_failure", _observe_nested_failure
    )
    caplog.set_level("DEBUG")

    result = await adapter.run_json_async(
        {"query": "active", "account_id": "llm-supplied"}
    )

    assert result == {
        "content": [
            {
                "type": "text",
                "text": "clients-ok",
                "annotations": None,
                "meta": None,
            }
        ],
        "is_error": False,
    }
    assert initial_client_builds == 1
    assert refreshed_client_builds == 1
    assert len(refresh_calls) == 1
    assert refresh_calls[0][0].params["error"] == "invalid_token"
    assert refresh_calls[0][0].scope == "records.read"
    assert len(nested_failures) == 1
    assert raw_exception_secret in repr(nested_failures[0])
    assert len(initial_requests) == 1
    assert initial_requests[0][0].headers["Authorization"] == (
        f"Bearer {initial_token}"
    )
    tool_calls = [
        (request, payload)
        for request, payload in refreshed_requests
        if payload.get("method") == "tools/call"
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0][0].headers["Authorization"] == f"Bearer {refreshed_token}"
    assert tool_calls[0][1]["params"] == {
        "_meta": tool_meta,
        "name": "list_clients",
        "arguments": tool_args,
    }
    public_output = repr(result) + caplog.text
    assert initial_token not in public_output
    assert refreshed_token not in public_output
    assert generation not in public_output
    assert raw_exception_secret not in public_output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authenticate",
    [
        [],
        ['Bearer error="invalid_token'],
        ['Basic error="invalid_token"'],
        ['Bearer error="insufficient_scope"'],
        ['Bearer scope="records.read"'],
    ],
)
async def test_resolver_owned_invalid_401_challenge_fails_without_connector_fallback(
    monkeypatch, authenticate
):
    resolver_calls = 0
    connector_calls = 0

    def _resolver_refresh(challenge):
        nonlocal resolver_calls
        resolver_calls += 1
        return {}

    def _connector_refresh():
        nonlocal connector_calls
        connector_calls += 1
        return {}

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
            "_connector_runtime_refresh": _connector_refresh,
        }
    )

    async def _execute(connection, tool_args, tool_meta):
        raise _http_status_error(authenticate=authenticate)

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert result["is_error"] is True
    assert "delegated_authorization_failed" in result["content"][0]["text"]
    assert resolver_calls == 0
    assert connector_calls == 0


@pytest.mark.asyncio
async def test_resolver_owned_text_only_401_does_not_use_either_refresh_callback(
    monkeypatch,
):
    resolver_calls = 0
    connector_calls = 0

    def _resolver_refresh(challenge):
        nonlocal resolver_calls
        resolver_calls += 1
        return {}

    def _connector_refresh():
        nonlocal connector_calls
        connector_calls += 1
        return {}

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
            "_connector_runtime_refresh": _connector_refresh,
        }
    )

    async def _execute(connection, tool_args, tool_meta):
        raise RuntimeError("HTTP 401 Unauthorized; Bearer error=invalid_token")

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert result == {
        "content": [{"text": "Error executing MCP tool."}],
        "is_error": True,
    }
    assert resolver_calls == 0
    assert connector_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_behavior", ["none", "raise", "malformed"])
async def test_resolver_refresh_failure_is_fixed_and_sanitized(
    monkeypatch, caplog, refresh_behavior
):
    secret = f"resolver-{refresh_behavior}-secret"

    async def _resolver_refresh(challenge):
        if refresh_behavior == "raise":
            raise RuntimeError(secret)
        if refresh_behavior == "malformed":
            return SimpleNamespace(secret=secret)
        return None

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )

    async def _execute(connection, tool_args, tool_meta):
        raise _http_status_error(authenticate=['Bearer error="invalid_token"'])

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)
    caplog.set_level("ERROR")

    result = await adapter.run_json_async({})

    assert result["is_error"] is True
    assert "delegated_authorization_failed" in result["content"][0]["text"]
    assert secret not in repr(result) + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refreshed_connection",
    [
        {},
        {"transport": "unsupported", "url": "https://mcp.example.test"},
        {"transport": "stdio"},
        {"transport": "stdio", "command": ""},
        {"transport": "stdio", "command": ["python"]},
        {"transport": "stdio", "command": "python"},
        {"transport": "sse"},
        {"transport": "streamable_http"},
        {"transport": "streamable_http", "url": ""},
        {"transport": "streamable_http", "url": 42},
        {"transport": "websocket"},
    ],
)
async def test_resolver_refresh_rejects_non_executable_connection_before_retry(
    monkeypatch, refreshed_connection
):
    refresh_calls = 0

    async def _resolver_refresh(challenge):
        nonlocal refresh_calls
        refresh_calls += 1
        return refreshed_connection

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )
    execution_calls = 0

    async def _execute(connection, tool_args, tool_meta):
        nonlocal execution_calls
        execution_calls += 1
        raise _http_status_error(authenticate=['Bearer error="invalid_token"'])

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert refresh_calls == 1
    assert execution_calls == 1
    assert result["is_error"] is True
    assert "delegated_authorization_failed" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_resolver_retry_second_401_does_not_refresh_twice(monkeypatch):
    refresh_calls = 0

    async def _resolver_refresh(challenge):
        nonlocal refresh_calls
        refresh_calls += 1
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )
    execution_calls = 0

    async def _execute(connection, tool_args, tool_meta):
        nonlocal execution_calls
        execution_calls += 1
        raise _http_status_error(authenticate=['Bearer error="invalid_token"'])

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert execution_calls == 2
    assert refresh_calls == 1
    assert result["is_error"] is True
    assert "delegated_authorization_failed" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_resolver_retry_classifies_same_401_instance_without_second_refresh(
    monkeypatch,
):
    refresh_calls = 0

    async def _resolver_refresh(challenge):
        nonlocal refresh_calls
        refresh_calls += 1
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )
    execution_calls = 0
    repeated_error = _http_status_error(authenticate=['Bearer error="invalid_token"'])

    async def _execute(connection, tool_args, tool_meta):
        nonlocal execution_calls
        execution_calls += 1
        raise repeated_error

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert execution_calls == 2
    assert refresh_calls == 1
    assert result["is_error"] is True
    assert "delegated_authorization_failed" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_resolver_retry_ignores_rewrapped_initial_401_response(monkeypatch):
    async def _resolver_refresh(challenge):
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )
    initial_error = _http_status_error(authenticate=['Bearer error="invalid_token"'])
    rewrapped_initial_response = httpx.HTTPStatusError(
        "rewrapped initial response",
        request=initial_error.request,
        response=initial_error.response,
    )
    execution_calls = 0

    async def _execute(connection, tool_args, tool_meta):
        nonlocal execution_calls
        execution_calls += 1
        if execution_calls == 1:
            raise initial_error
        raise rewrapped_initial_response

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert execution_calls == 2
    assert result == {
        "content": [
            {"text": "Error executing MCP tool after delegated authorization retry."}
        ],
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_resolver_retry_prunes_over_budget_initial_exception_subtree(
    monkeypatch,
):
    async def _resolver_refresh(challenge):
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )
    deep_original: BaseException = _http_status_error(
        authenticate=['Bearer error="invalid_token"']
    )
    for index in range(mcp_adapter_module._RESOLVER_HTTP_401_NODE_LIMIT + 1):
        wrapper = RuntimeError(f"initial-wrapper-{index}")
        wrapper.__cause__ = deep_original
        deep_original = wrapper
    initial_error = ExceptionGroup(
        "initial",
        [
            _http_status_error(authenticate=['Bearer error="invalid_token"']),
            deep_original,
        ],
    )
    execution_calls = 0

    async def _execute(connection, tool_args, tool_meta):
        nonlocal execution_calls
        execution_calls += 1
        if execution_calls == 1:
            raise initial_error
        raise RuntimeError("retry transport failed")

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert execution_calls == 2
    assert result == {
        "content": [
            {"text": "Error executing MCP tool after delegated authorization retry."}
        ],
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_resolver_retry_non_401_failure_does_not_leak_secrets(
    monkeypatch, caplog
):
    initial_secret = "initial-resolver-secret"
    refreshed_secret = "refreshed-resolver-secret"

    async def _resolver_refresh(challenge):
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "headers": {"Authorization": f"Bearer {refreshed_secret}"},
        }

    initial_connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "headers": {"Authorization": f"Bearer {initial_secret}"},
        "_oauth_token_resolver_refresh": _resolver_refresh,
    }
    adapter = _resolver_retry_adapter(initial_connection)

    async def _execute(connection, tool_args, tool_meta):
        if connection is initial_connection:
            raise _http_status_error(authenticate=['Bearer error="invalid_token"'])
        raise RuntimeError(f"transport failed with {refreshed_secret}")

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)
    caplog.set_level("ERROR")

    result = await adapter.run_json_async({})

    assert result == {
        "content": [
            {"text": "Error executing MCP tool after delegated authorization retry."}
        ],
        "is_error": True,
    }
    public_output = repr(result) + caplog.text
    assert initial_secret not in public_output
    assert refreshed_secret not in public_output


@pytest.mark.asyncio
async def test_connector_refresh_empty_dict_preserves_legacy_retry_failure(
    monkeypatch,
):
    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "_connector_runtime_refresh": lambda: {},
    }
    adapter = _resolver_retry_adapter(connection)
    attempted_connections = []

    async def _execute(attempted, tool_args, tool_meta):
        attempted_connections.append(attempted)
        if attempted is connection:
            raise RuntimeError("HTTP 401 Unauthorized")
        raise RuntimeError("malformed connector retry connection")

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert attempted_connections == [connection, {}]
    assert result == {
        "content": [
            {"text": "Error executing MCP tool after delegated authorization retry."}
        ],
        "is_error": True,
    }


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
