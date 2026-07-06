import pytest

from xagent.core.tools.adapters.vibe.connector_runtime import (
    CONNECTOR_TYPE_CUSTOM_API,
    CONNECTOR_TYPE_MCP,
    ERROR_DELEGATED_AUTHORIZATION_FAILED,
    ERROR_MCP_OAUTH_AUTHORIZATION_FAILED,
    ERROR_RUNTIME_SECRET_UNAVAILABLE,
    REDACTED_RUNTIME_SECRET,
    ConnectorRef,
    ConnectorRuntimeError,
    redact_runtime_value,
    validate_runtime_source_key,
)


def test_connector_ref_round_trips_wire_shape() -> None:
    ref = ConnectorRef.from_wire(
        {"connector_type": CONNECTOR_TYPE_MCP, "connector_id": 123}
    )

    assert ref.connector_type == CONNECTOR_TYPE_MCP
    assert ref.connector_id == 123
    assert ref.storage_key == "mcp:123"
    assert ref.to_wire() == {"connector_type": "mcp", "connector_id": 123}


@pytest.mark.parametrize(
    "value",
    [
        None,
        "mcp:123",
        {"connector_type": "mcp", "connector_id": "123"},
        {"connector_type": "unknown", "connector_id": 123},
        {"connector_type": CONNECTOR_TYPE_CUSTOM_API, "connector_id": 0},
        {
            "connector_type": CONNECTOR_TYPE_CUSTOM_API,
            "connector_id": 1,
            "name": "ambiguous",
        },
    ],
)
def test_connector_ref_rejects_ambiguous_or_invalid_wire_shape(value) -> None:
    with pytest.raises(ValueError):
        ConnectorRef.from_wire(value)


@pytest.mark.parametrize("key", ["account_id", "tenant-123", "user_42", "A1"])
def test_validate_runtime_source_key_accepts_simple_keys(key: str) -> None:
    assert validate_runtime_source_key(key) == key


@pytest.mark.parametrize("key", ["", "shiftcare.com/account_id", "a.b", "space key"])
def test_validate_runtime_source_key_rejects_keys_with_parse_ambiguity(
    key: str,
) -> None:
    with pytest.raises(ValueError):
        validate_runtime_source_key(key)


def test_connector_runtime_error_is_public_safe_and_contains_connector_ref() -> None:
    ref = ConnectorRef(connector_type=CONNECTOR_TYPE_MCP, connector_id=7)

    error = ConnectorRuntimeError(
        ERROR_MCP_OAUTH_AUTHORIZATION_FAILED,
        "MCP OAuth authorization is unavailable",
        connector_ref=ref,
        details={"reason": "missing_grant"},
        status_code=401,
    )

    assert str(error) == (
        "mcp_oauth_authorization_failed: MCP OAuth authorization is unavailable"
    )
    assert error.status_code == 401
    assert error.to_public_error() == {
        "code": ERROR_MCP_OAUTH_AUTHORIZATION_FAILED,
        "message": "MCP OAuth authorization is unavailable",
        "details": {
            "reason": "missing_grant",
            "connector_ref": {"connector_type": "mcp", "connector_id": 7},
        },
    }


def test_delegated_and_managed_oauth_errors_are_distinct() -> None:
    assert ERROR_DELEGATED_AUTHORIZATION_FAILED == "delegated_authorization_failed"
    assert ERROR_MCP_OAUTH_AUTHORIZATION_FAILED == "mcp_oauth_authorization_failed"
    assert ERROR_DELEGATED_AUTHORIZATION_FAILED != ERROR_MCP_OAUTH_AUTHORIZATION_FAILED


def test_redact_runtime_value_does_not_preserve_secret_material() -> None:
    value = {
        "authorization": "Bearer secret-token",
        "nested": {"resource_owner_key": "person-1"},
    }

    redacted = redact_runtime_value(value)

    assert redacted == {
        "authorization": REDACTED_RUNTIME_SECRET,
        "nested": REDACTED_RUNTIME_SECRET,
    }
    assert "secret-token" not in repr(redacted)
    assert "person-1" not in repr(redacted)


@pytest.mark.asyncio
async def test_tool_registry_does_not_swallow_connector_runtime_error() -> None:
    from xagent.core.tools.adapters.vibe.factory import ToolRegistry

    saved_creators = list(ToolRegistry._tool_creators)
    saved_imported = ToolRegistry._modules_imported
    ToolRegistry._tool_creators = []
    ToolRegistry._modules_imported = True

    async def _creator(_config):
        raise ConnectorRuntimeError(
            ERROR_RUNTIME_SECRET_UNAVAILABLE,
            "Required runtime secret is unavailable.",
            details={"reason": "store_lost"},
        )

    try:
        ToolRegistry.register(_creator, categories={"mcp"})
        with pytest.raises(ConnectorRuntimeError) as exc_info:
            await ToolRegistry.create_registered_tools(object())
    finally:
        ToolRegistry._tool_creators = saved_creators
        ToolRegistry._modules_imported = saved_imported

    assert exc_info.value.code == ERROR_RUNTIME_SECRET_UNAVAILABLE
