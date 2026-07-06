"""Shared connector runtime context contracts.

This module is intentionally free of web/ORM imports so both tool adapters and
web runtime services can use the same connector identity and error shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Mapping

ConnectorType = Literal["mcp", "custom_api"]

CONNECTOR_TYPE_MCP = "mcp"
CONNECTOR_TYPE_CUSTOM_API = "custom_api"
ALLOWED_CONNECTOR_TYPES = frozenset({CONNECTOR_TYPE_MCP, CONNECTOR_TYPE_CUSTOM_API})

RUNTIME_INPUT_CONTEXT = "context"
RUNTIME_INPUT_SECRETS = "secrets"
RUNTIME_INPUT_AUTH_SELECTOR = "auth_selector"

TARGET_MCP_META = "mcp_meta"
TARGET_TRANSPORT_HEADERS = "transport_headers"
TARGET_TOOL_ARGUMENTS = "tool_arguments"
TARGET_HEADERS = "headers"
TARGET_BODY_FIELD = "body_field"

RUNTIME_SOURCE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
REDACTED_RUNTIME_SECRET = "[REDACTED_RUNTIME_SECRET]"
MISSING_RUNTIME_VALUE = object()

ERROR_CONNECTOR_NOT_FOUND = "connector_not_found"
ERROR_INVALID_RUNTIME_CONTEXT = "invalid_runtime_context"
ERROR_MISSING_RUNTIME_CONTEXT = "missing_runtime_context"
ERROR_RUNTIME_CONTEXT_IMMUTABLE = "runtime_context_immutable"
ERROR_RUNTIME_SECRET_NOT_ALLOWED = "runtime_secret_not_allowed"
ERROR_RUNTIME_SECRET_UNAVAILABLE = "runtime_secret_unavailable"
ERROR_SCHEDULED_SECRET_UNAVAILABLE = "scheduled_secret_unavailable"
ERROR_RUNTIME_BINDING_NOT_ALLOWED = "runtime_binding_not_allowed"
ERROR_RUNTIME_BINDING_CONFLICT = "runtime_binding_conflict"
ERROR_MCP_OAUTH_AUTHORIZATION_FAILED = "mcp_oauth_authorization_failed"
ERROR_DELEGATED_AUTHORIZATION_FAILED = "delegated_authorization_failed"

RUNTIME_SECRET_REASON_NOT_PROVIDED = "not_provided"
RUNTIME_SECRET_REASON_STORE_LOST = "store_lost"
RUNTIME_SECRET_REASON_CLEANED = "cleaned"
RUNTIME_SECRET_REASON_EXPIRED = "expired"


@dataclass(frozen=True, order=True)
class ConnectorRef:
    """Stable runtime identity for a connector selected by a task."""

    connector_type: ConnectorType
    connector_id: int

    _WIRE_KEYS: ClassVar[frozenset[str]] = frozenset({"connector_type", "connector_id"})

    def __post_init__(self) -> None:
        if self.connector_type not in ALLOWED_CONNECTOR_TYPES:
            raise ValueError(f"unsupported connector_type: {self.connector_type!r}")
        if not isinstance(self.connector_id, int) or self.connector_id <= 0:
            raise ValueError("connector_id must be a positive integer")

    @classmethod
    def from_wire(cls, value: Any) -> "ConnectorRef":
        if not isinstance(value, dict):
            raise ValueError("connector ref must be an object")
        extra = set(value) - cls._WIRE_KEYS
        if extra:
            raise ValueError(f"connector ref has unknown field(s): {sorted(extra)!r}")
        connector_type = value.get("connector_type")
        connector_id = value.get("connector_id")
        if connector_type not in ALLOWED_CONNECTOR_TYPES:
            raise ValueError("connector_type must be 'mcp' or 'custom_api'")
        if not isinstance(connector_id, int):
            raise ValueError("connector_id must be an integer")
        return cls(connector_type=connector_type, connector_id=connector_id)

    def to_wire(self) -> dict[str, Any]:
        return {
            "connector_type": self.connector_type,
            "connector_id": self.connector_id,
        }

    @property
    def storage_key(self) -> str:
        return f"{self.connector_type}:{self.connector_id}"


class ConnectorRuntimeError(RuntimeError):
    """Public-safe runtime context error.

    ``message`` and ``details`` must be safe to return to API callers and logs
    after normal structured redaction. Raw secret/auth selector values should
    never be attached to this exception.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        connector_ref: ConnectorRef | None = None,
        details: dict[str, Any] | None = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.connector_ref = connector_ref
        self.details = dict(details or {})
        self.status_code = status_code
        if connector_ref is not None:
            self.details.setdefault("connector_ref", connector_ref.to_wire())

    def __str__(self) -> str:
        return f"{self.code}: {self.safe_message}"

    def to_public_error(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.safe_message,
            "details": dict(self.details),
        }


def validate_runtime_source_key(key: str) -> str:
    if not isinstance(key, str) or not RUNTIME_SOURCE_KEY_RE.fullmatch(key):
        raise ValueError("runtime input key must match [A-Za-z0-9_-]+")
    return key


def redact_runtime_value(value: Any) -> Any:
    """Redact runtime secrets/auth selectors without preserving raw structure."""

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): REDACTED_RUNTIME_SECRET for key in value}
    if isinstance(value, list):
        return [REDACTED_RUNTIME_SECRET for _ in value]
    return REDACTED_RUNTIME_SECRET


def runtime_bindings_from_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    bindings = config.get("runtime_bindings")
    if not isinstance(bindings, list):
        return []
    return [binding for binding in bindings if isinstance(binding, dict)]


def connector_runtime_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = config.get("connector_runtime")
    return runtime if isinstance(runtime, dict) else {}


def binding_source(binding: Mapping[str, Any]) -> dict[str, Any]:
    source = binding.get("source")
    if isinstance(source, dict):
        return source
    if isinstance(source, str) and "." in source:
        input_type, key = source.split(".", 1)
        return {"input_type": input_type, "key": key}
    return {}


def binding_target(binding: Mapping[str, Any]) -> dict[str, Any]:
    target = binding.get("target")
    return target if isinstance(target, dict) else {}


def binding_source_value(
    binding: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    allowed_input_types: set[str],
) -> Any:
    source = binding_source(binding)
    input_type = source.get("input_type") or source.get("type") or source.get("section")
    key = source.get("key")
    if input_type not in allowed_input_types or not isinstance(key, str):
        return MISSING_RUNTIME_VALUE
    section = runtime.get(input_type)
    if not isinstance(section, dict) or key not in section:
        return MISSING_RUNTIME_VALUE
    return section[key]
