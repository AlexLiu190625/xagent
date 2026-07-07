from unittest.mock import patch

import pytest

from xagent.core.tools.adapters.vibe.api_tool_adapter import (
    CustomApiTool,
    create_custom_api_tools,
)


def test_custom_api_tool_init():
    tool = CustomApiTool(
        name="my-test api",
        description="A test API",
        env={"API_KEY": "secret123", "API_KEY_BACKUP": "secret456"},
        url="https://api.example.com/hello",
        method="POST",
        headers={"Authorization": "Bearer $API_KEY"},
    )

    assert tool.name == "api_my_test_api_call"
    # Structured originating-server identity, normalized once via the SSOT,
    # so a scoped mcp:<server> selector matches this wrapper by equality.
    assert tool.source_server == "my_test_api"
    assert tool.metadata.source_server == "my_test_api"
    assert "A test API" in tool.description
    assert "Configured endpoint: https://api.example.com/hello" in tool.description
    assert "Configured method: POST" in tool.description
    assert "- API_KEY" in tool.description
    assert "- API_KEY_BACKUP" in tool.description


def test_custom_api_tool_replace_secrets():
    # Use unencrypted secrets for simplicity since decrypt_value handles unencrypted fallback or we can mock it
    # We will mock decrypt_value to just return the value for testing replace
    with patch(
        "xagent.core.tools.adapters.vibe.api_tool_adapter.decrypt_value",
        side_effect=lambda x: x,
    ):
        tool = CustomApiTool(
            name="test",
            description="test",
            env={"API_KEY": "secret123", "API_KEY_BACKUP": "secret456"},
        )

        # Test word boundaries
        result = tool._replace_secrets("Bearer $API_KEY")
        assert result == "Bearer secret123"

        # Test word boundaries avoiding partial replacement
        result2 = tool._replace_secrets("Bearer $API_KEY_BACKUP")
        assert result2 == "Bearer secret456"

        # Test bracket notation
        result3 = tool._replace_secrets("Bearer ${API_KEY}")
        assert result3 == "Bearer secret123"

        # Test recursive
        dict_val = {
            "url": "http://example.com?key=$API_KEY",
            "headers": {"Authorization": "Bearer ${API_KEY_BACKUP}"},
            "list": ["$API_KEY", "normal"],
        }
        res_dict = tool._replace_secrets(dict_val)
        assert res_dict["url"] == "http://example.com?key=secret123"
        assert res_dict["headers"]["Authorization"] == "Bearer secret456"
        assert res_dict["list"] == ["secret123", "normal"]


@pytest.mark.asyncio
async def test_run_json_async():
    with (
        patch(
            "xagent.core.tools.adapters.vibe.api_tool_adapter.decrypt_value",
            side_effect=lambda x: x,
        ),
        patch(
            "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api"
        ) as mock_call_api,
    ):
        mock_call_api.return_value = {
            "success": True,
            "status_code": 200,
            "headers": {},
            "body": {"data": "test"},
            "error": None,
        }

        tool = CustomApiTool(name="test", description="test", env={"KEY": "val"})

        args = {"url": "http://test.com/$KEY", "method": "GET"}

        res = await tool.run_json_async(args)
        assert res["success"] is True
        assert res["status_code"] == 200
        assert res["body"] == {"data": "test"}

        mock_call_api.assert_called_once_with(
            url="http://test.com/val", method="GET", headers={}, params={}, body=None
        )


@pytest.mark.asyncio
async def test_run_json_async_uses_configured_endpoint_defaults():
    with (
        patch(
            "xagent.core.tools.adapters.vibe.api_tool_adapter.decrypt_value",
            side_effect=lambda x: x,
        ),
        patch(
            "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api"
        ) as mock_call_api,
    ):
        mock_call_api.return_value = {
            "success": True,
            "status_code": 200,
            "headers": {},
            "body": {"data": "test"},
            "error": None,
        }

        tool = CustomApiTool(
            name="HelloAPI",
            description="test",
            env={"TOKEN": "secret"},
            url="https://api.example.com/hello",
            method="POST",
            headers={"Authorization": "Bearer $TOKEN"},
        )

        res = await tool.run_json_async({"body": {"name": "Ada"}})
        assert res["success"] is True

        mock_call_api.assert_called_once_with(
            url="https://api.example.com/hello",
            method="POST",
            headers={"Authorization": "Bearer secret"},
            params={},
            body={"name": "Ada"},
        )


@pytest.mark.asyncio
async def test_run_json_async_error_does_not_echo_raw_exception():
    with patch(
        "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api"
    ) as mock_call_api:
        mock_call_api.side_effect = RuntimeError(
            "transport failed with Bearer runtime-token"
        )
        tool = CustomApiTool(
            name="ShiftCare",
            description="test",
            env={},
            url="https://api.example.com/clients",
        )

        res = await tool.run_json_async({})

        assert res["success"] is False
        assert res["error"] == "Error executing Custom API."
        assert "runtime-token" not in repr(res)


@pytest.mark.asyncio
async def test_run_json_async_merges_configured_and_call_headers():
    with (
        patch(
            "xagent.core.tools.adapters.vibe.api_tool_adapter.decrypt_value",
            side_effect=lambda x: x,
        ),
        patch(
            "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api"
        ) as mock_call_api,
    ):
        mock_call_api.return_value = {
            "success": True,
            "status_code": 200,
            "headers": {},
            "body": {"data": "test"},
            "error": None,
        }

        tool = CustomApiTool(
            name="HelloAPI",
            description="test",
            env={"TOKEN": "secret"},
            url="https://api.example.com/hello",
            headers={
                "Authorization": "Bearer $TOKEN",
                "X-Default": "1",
                "X-Override": "default",
            },
        )

        res = await tool.run_json_async(
            {"headers": {"X-Custom": "2", "X-Override": "caller"}}
        )
        assert res["success"] is True

        mock_call_api.assert_called_once_with(
            url="https://api.example.com/hello",
            method="GET",
            headers={
                "Authorization": "Bearer secret",
                "X-Default": "1",
                "X-Custom": "2",
                "X-Override": "caller",
            },
            params={},
            body=None,
        )


@pytest.mark.asyncio
async def test_run_json_async_applies_runtime_headers_and_body_fields():
    with patch(
        "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api"
    ) as mock_call_api:
        mock_call_api.return_value = {
            "success": True,
            "status_code": 200,
            "headers": {},
            "body": {"data": "test"},
            "error": None,
        }
        tool = CustomApiTool(
            name="ShiftCare",
            description="test",
            env={},
            url="https://api.example.com/clients",
            method="POST",
            headers={"X-Account": "static"},
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "account_id"},
                    "target": {"target_type": "headers", "key": "X-Account"},
                },
                {
                    "source": {"input_type": "context", "key": "account_id"},
                    "target": {
                        "target_type": "body_field",
                        "path": "scope.account_id",
                    },
                },
            ],
            connector_runtime={
                "context": {"account_id": "6185"},
                "secrets": {},
                "auth_selector": {},
            },
        )

        res = await tool.run_json_async({"body": {"scope": {"account_id": "llm"}}})
        assert res["success"] is True

        mock_call_api.assert_called_once_with(
            url="https://api.example.com/clients",
            method="POST",
            headers={"X-Account": "6185"},
            params={},
            body={"scope": {"account_id": "6185"}},
        )


def test_runtime_bindings_hide_custom_api_headers_and_body_from_llm_schema():
    tool = CustomApiTool(
        name="ShiftCare",
        description="test",
        env={},
        url="https://api.example.com/clients",
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account"},
            },
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {
                    "target_type": "body_field",
                    "path": "scope.account_id",
                },
            },
        ],
    )

    fields = tool.args_type().model_fields

    assert "headers" not in fields
    assert "body" not in fields
    assert "url" in fields
    assert "method" in fields
    assert "params" in fields


def test_runtime_bindings_sanitize_custom_api_trace_args_before_execution():
    tool = CustomApiTool(
        name="ShiftCare",
        description="test",
        env={},
        url="https://api.example.com/clients",
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account"},
            },
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {
                    "target_type": "body_field",
                    "path": "scope.account_id",
                },
            },
        ],
    )

    sanitized = tool.sanitize_tool_args_for_trace(
        {
            "headers": {"X-Account": "llm"},
            "body": {"scope": {"account_id": "llm"}},
            "params": {"q": "client"},
        }
    )

    assert sanitized == {"params": {"q": "client"}}


@pytest.mark.asyncio
async def test_run_json_async_requires_flag_for_runtime_authorization_header():
    with patch(
        "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api"
    ) as mock_call_api:
        mock_call_api.return_value = {
            "success": True,
            "status_code": 200,
            "headers": {},
            "body": {"data": "test"},
            "error": None,
        }
        tool = CustomApiTool(
            name="ShiftCare",
            description="test",
            env={},
            url="https://api.example.com/clients",
            runtime_bindings=[
                {
                    "source": {"input_type": "secrets", "key": "authorization"},
                    "target": {"target_type": "headers", "key": "Authorization"},
                }
            ],
            connector_runtime={
                "context": {},
                "secrets": {"authorization": "Bearer tenant-token"},
                "auth_selector": {},
            },
            allow_delegated_authorization=False,
        )

        res = await tool.run_json_async({})
        assert res["success"] is True

        mock_call_api.assert_called_once_with(
            url="https://api.example.com/clients",
            method="GET",
            headers={},
            params={},
            body=None,
        )


@pytest.mark.asyncio
async def test_run_json_async_ignores_non_scalar_runtime_header_values():
    with patch(
        "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api"
    ) as mock_call_api:
        mock_call_api.return_value = {
            "success": True,
            "status_code": 200,
            "headers": {},
            "body": {"data": "test"},
            "error": None,
        }
        tool = CustomApiTool(
            name="ShiftCare",
            description="test",
            env={},
            url="https://api.example.com/clients",
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "tuple_value"},
                    "target": {"target_type": "headers", "key": "X-Tuple"},
                },
                {
                    "source": {"input_type": "context", "key": "set_value"},
                    "target": {"target_type": "headers", "key": "X-Set"},
                },
            ],
            connector_runtime={
                "context": {
                    "tuple_value": ("account", "6185"),
                    "set_value": {"account", "6185"},
                },
                "secrets": {},
                "auth_selector": {},
            },
        )

        res = await tool.run_json_async({})
        assert res["success"] is True

        mock_call_api.assert_called_once_with(
            url="https://api.example.com/clients",
            method="GET",
            headers={},
            params={},
            body=None,
        )


@pytest.mark.asyncio
async def test_run_json_async_warns_when_runtime_body_binding_discards_non_object_body(
    caplog,
):
    with patch(
        "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api"
    ) as mock_call_api:
        mock_call_api.return_value = {
            "success": True,
            "status_code": 200,
            "headers": {},
            "body": {"data": "test"},
            "error": None,
        }
        tool = CustomApiTool(
            name="ShiftCare",
            description="test",
            env={},
            url="https://api.example.com/clients",
            body='"static text body"',
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "account_id"},
                    "target": {"target_type": "body_field", "path": "account.id"},
                },
            ],
            connector_runtime={
                "context": {"account_id": "6185"},
                "secrets": {},
                "auth_selector": {},
            },
        )

        caplog.set_level("WARNING")
        res = await tool.run_json_async({})

        assert res["success"] is True
        assert "discard non-object body" in caplog.text
        mock_call_api.assert_called_once_with(
            url="https://api.example.com/clients",
            method="GET",
            headers={},
            params={},
            body={"account": {"id": "6185"}},
        )


@pytest.mark.asyncio
async def test_run_json_async_warns_when_runtime_body_binding_replaces_scalar_parent(
    caplog,
):
    with patch(
        "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api"
    ) as mock_call_api:
        mock_call_api.return_value = {
            "success": True,
            "status_code": 200,
            "headers": {},
            "body": {"data": "test"},
            "error": None,
        }
        tool = CustomApiTool(
            name="ShiftCare",
            description="test",
            env={},
            url="https://api.example.com/clients",
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "name"},
                    "target": {"target_type": "body_field", "path": "user.name"},
                },
            ],
            connector_runtime={
                "context": {"name": "Alice"},
                "secrets": {},
                "auth_selector": {},
            },
        )

        caplog.set_level("WARNING")
        res = await tool.run_json_async({"body": {"user": "legacy"}})

        assert res["success"] is True
        assert "overrides non-object intermediate field user" in caplog.text
        mock_call_api.assert_called_once_with(
            url="https://api.example.com/clients",
            method="GET",
            headers={},
            params={},
            body={"user": {"name": "Alice"}},
        )


@pytest.mark.asyncio
async def test_run_json_async_returns_error_without_any_url():
    tool = CustomApiTool(name="test", description="test", env={})

    res = await tool.run_json_async({})

    assert res["success"] is False
    assert "URL is required" in res["error"]


def test_run_json_sync_raises_runtime_error():
    tool = CustomApiTool(name="test", description="test", env={})

    # Since pytest-asyncio runs tests in an event loop if marked with @pytest.mark.asyncio
    # We can test that calling the sync version raises an error when a loop is running
    async def inner():
        with pytest.raises(RuntimeError, match="Event loop is already running"):
            tool.run_json_sync({"url": "http://test", "method": "GET"})

    import asyncio

    asyncio.run(inner())


def test_create_custom_api_tools():
    configs = [
        {
            "name": "api1",
            "description": "desc1",
            "env": {"k1": "v1"},
            "url": "https://api.example.com/api1",
            "method": "POST",
            "headers": {"X-Key": "$k1"},
        },
        {"name": "api2", "description": "desc2", "env": {"k2": "v2"}},
    ]
    tools = create_custom_api_tools(configs)
    assert len(tools) == 2
    assert tools[0].name == "api_api1_call"
    assert tools[1].name == "api_api2_call"
    assert "Configured endpoint: https://api.example.com/api1" in tools[0].description
