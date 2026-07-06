"""Connector runtime context validation and task binding.

The web layer owns invocation trust, connector visibility, selected-ref
snapshots, and task-bound non-secret context persistence. Tool adapters only
consume the resolved runtime view later in the execution path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterable, cast

from sqlalchemy.orm import Session

from ...core.tools.adapters.vibe.connector_runtime import (
    CONNECTOR_TYPE_CUSTOM_API,
    CONNECTOR_TYPE_MCP,
    ERROR_CONNECTOR_NOT_FOUND,
    ERROR_INVALID_RUNTIME_CONTEXT,
    ERROR_MISSING_RUNTIME_CONTEXT,
    ERROR_RUNTIME_CONTEXT_IMMUTABLE,
    ERROR_RUNTIME_SECRET_NOT_ALLOWED,
    ERROR_RUNTIME_SECRET_UNAVAILABLE,
    RUNTIME_INPUT_AUTH_SELECTOR,
    RUNTIME_INPUT_CONTEXT,
    RUNTIME_INPUT_SECRETS,
    RUNTIME_SECRET_REASON_NOT_PROVIDED,
    RUNTIME_SECRET_REASON_STORE_LOST,
    ConnectorRef,
    ConnectorRuntimeError,
    ConnectorType,
    validate_runtime_source_key,
)
from ...core.tools.adapters.vibe.selection_spec import (
    ToolSelectionSpec,
    normalize_mcp_server_name,
)
from ..models.agent import Agent
from ..models.custom_api import CustomApi, UserCustomApi
from ..models.mcp import MCPServer, UserMCPServer
from ..models.task import Task, TaskConnectorRuntimeContext


@dataclass(frozen=True)
class ConnectorRuntimePayload:
    ref: ConnectorRef
    context: dict[str, Any]
    secrets: dict[str, Any]
    auth_selector: dict[str, Any]


@dataclass(frozen=True)
class ConnectorRuntimeCreatePlan:
    selected_refs: tuple[ConnectorRef, ...]
    context_by_ref: dict[ConnectorRef, dict[str, Any]]
    ephemeral_by_ref: dict[ConnectorRef, dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class ConnectorRuntimeAppendPlan:
    ephemeral_by_ref: dict[ConnectorRef, dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class ConnectorRuntimeValues:
    context: dict[str, Any]
    secrets: dict[str, Any]
    auth_selector: dict[str, Any]

    def to_runtime_config(self) -> dict[str, Any]:
        return {
            RUNTIME_INPUT_CONTEXT: dict(self.context),
            RUNTIME_INPUT_SECRETS: dict(self.secrets),
            RUNTIME_INPUT_AUTH_SELECTOR: dict(self.auth_selector),
        }


@dataclass(frozen=True)
class ConnectorRuntimeRequest:
    task_id: int
    turn_id: str | None
    user_id: int | None
    connector_ref: ConnectorRef
    values: ConnectorRuntimeValues


ConnectorRuntimeResolver = Callable[
    [ConnectorRuntimeRequest], ConnectorRuntimeValues | None
]


_EPHEMERAL_RUNTIME_VALUES: dict[str, dict[str, Any]] = {}
_EPHEMERAL_RUNTIME_VALUES_LOCK = RLock()
_RUNTIME_RESOLVER: ConnectorRuntimeResolver | None = None


def set_connector_runtime_resolver_for_testing(
    resolver: ConnectorRuntimeResolver | None,
) -> None:
    global _RUNTIME_RESOLVER
    _RUNTIME_RESOLVER = resolver


def store_ephemeral_runtime_values(
    turn_id: str, values_by_ref: dict[ConnectorRef, dict[str, dict[str, Any]]]
) -> None:
    """Store per-turn secrets/auth selectors by turn id."""

    if not values_by_ref:
        return
    encoded = {
        ref.storage_key: {
            section: dict(values) for section, values in sections.items() if values
        }
        for ref, sections in values_by_ref.items()
    }
    with _EPHEMERAL_RUNTIME_VALUES_LOCK:
        _EPHEMERAL_RUNTIME_VALUES[turn_id] = encoded


def pop_ephemeral_runtime_values(turn_id: str) -> dict[str, Any] | None:
    with _EPHEMERAL_RUNTIME_VALUES_LOCK:
        return _EPHEMERAL_RUNTIME_VALUES.pop(turn_id, None)


def get_ephemeral_runtime_values(turn_id: str) -> dict[str, Any] | None:
    with _EPHEMERAL_RUNTIME_VALUES_LOCK:
        values = _EPHEMERAL_RUNTIME_VALUES.get(turn_id)
        return dict(values) if isinstance(values, dict) else None


def load_connector_runtime_view(
    *,
    db: Session,
    task_id: int,
    turn_id: str | None,
    user_id: int | None,
) -> dict[str, dict[str, Any]]:
    """Resolve task-bound and per-turn runtime values for tool creation."""

    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        return {}

    selected_refs = _load_task_selected_refs(task)
    if not selected_refs:
        return {}

    persisted_context = _load_task_context_rows(db, task_id=task_id)
    ephemeral_by_ref = (
        get_ephemeral_runtime_values(turn_id) if isinstance(turn_id, str) else None
    )
    visible = (
        _load_visible_runtime_connectors(db, user_id=user_id)
        if user_id is not None
        else {}
    )

    runtime_view: dict[str, dict[str, Any]] = {}
    for ref in selected_refs:
        connector = visible.get(ref)
        if connector is None:
            continue
        raw_ephemeral = (
            ephemeral_by_ref.get(ref.storage_key, {})
            if isinstance(ephemeral_by_ref, dict)
            else {}
        )
        values = ConnectorRuntimeValues(
            context=dict(persisted_context.get(ref, {})),
            secrets=dict(
                raw_ephemeral.get(RUNTIME_INPUT_SECRETS, {})
                if isinstance(raw_ephemeral, dict)
                else {}
            ),
            auth_selector=dict(
                raw_ephemeral.get(RUNTIME_INPUT_AUTH_SELECTOR, {})
                if isinstance(raw_ephemeral, dict)
                else {}
            ),
        )
        values = _resolve_runtime_values(
            task_id=task_id,
            turn_id=turn_id,
            user_id=user_id,
            ref=ref,
            values=values,
        )
        _require_context_values(ref, connector, values.context)
        _require_ephemeral_values_at_binding(ref, connector, values)
        runtime_view[ref.storage_key] = values.to_runtime_config()

    return runtime_view


def prepare_create_connector_runtime(
    *,
    db: Session,
    agent: Agent,
    payload_items: Iterable[Any] | None,
) -> ConnectorRuntimeCreatePlan:
    visible = _load_visible_runtime_connectors(db, user_id=int(agent.user_id))
    selected_refs = _plan_selected_refs(agent, visible)
    payload_by_ref = _parse_payload_items(payload_items)
    _validate_payload_refs(payload_by_ref, visible=visible, selected_refs=selected_refs)

    context_by_ref: dict[ConnectorRef, dict[str, Any]] = {}
    ephemeral_by_ref: dict[ConnectorRef, dict[str, dict[str, Any]]] = {}

    for ref in selected_refs:
        connector = visible[ref]
        payload = payload_by_ref.get(ref)
        context = dict(payload.context) if payload is not None else {}
        secrets = dict(payload.secrets) if payload is not None else {}
        auth_selector = dict(payload.auth_selector) if payload is not None else {}
        _validate_values_against_schema(ref, connector, context, secrets, auth_selector)
        _require_context_values(ref, connector, context)
        _require_ephemeral_values(ref, connector, secrets, auth_selector)
        if context:
            context_by_ref[ref] = context
        if secrets or auth_selector:
            ephemeral_by_ref[ref] = {
                RUNTIME_INPUT_SECRETS: secrets,
                RUNTIME_INPUT_AUTH_SELECTOR: auth_selector,
            }

    return ConnectorRuntimeCreatePlan(
        selected_refs=tuple(sorted(selected_refs)),
        context_by_ref=context_by_ref,
        ephemeral_by_ref=ephemeral_by_ref,
    )


def persist_create_connector_runtime_context(
    *, db: Session, task_id: int, plan: ConnectorRuntimeCreatePlan
) -> None:
    for ref, context in plan.context_by_ref.items():
        db.add(
            TaskConnectorRuntimeContext(
                task_id=task_id,
                connector_type=ref.connector_type,
                connector_id=ref.connector_id,
                context=_canonical_json_value(context),
            )
        )


def prepare_append_connector_runtime(
    *,
    db: Session,
    agent: Agent,
    task: Task,
    payload_items: Iterable[Any] | None,
) -> ConnectorRuntimeAppendPlan:
    selected_refs = _load_task_selected_refs(task)
    payload_by_ref = _parse_payload_items(payload_items)
    visible = _load_visible_runtime_connectors(db, user_id=int(agent.user_id))
    _validate_payload_refs(payload_by_ref, visible=visible, selected_refs=selected_refs)

    persisted_context = _load_task_context_rows(db, task_id=int(task.id))
    ephemeral_by_ref: dict[ConnectorRef, dict[str, dict[str, Any]]] = {}

    for ref in selected_refs:
        connector = visible.get(ref)
        if connector is None:
            _raise_runtime_error(ERROR_CONNECTOR_NOT_FOUND, ref)
        payload = payload_by_ref.get(ref)
        context = dict(payload.context) if payload is not None else {}
        secrets = dict(payload.secrets) if payload is not None else {}
        auth_selector = dict(payload.auth_selector) if payload is not None else {}
        _validate_values_against_schema(ref, connector, context, secrets, auth_selector)
        _require_ephemeral_values(ref, connector, secrets, auth_selector)
        stored = persisted_context.get(ref, {})
        _require_context_values(ref, connector, stored)
        if context and _canonical_json_value(context) != _canonical_json_value(stored):
            _raise_runtime_error(ERROR_RUNTIME_CONTEXT_IMMUTABLE, ref)
        if secrets or auth_selector:
            ephemeral_by_ref[ref] = {
                RUNTIME_INPUT_SECRETS: secrets,
                RUNTIME_INPUT_AUTH_SELECTOR: auth_selector,
            }

    return ConnectorRuntimeAppendPlan(ephemeral_by_ref=ephemeral_by_ref)


def _parse_payload_items(
    payload_items: Iterable[Any] | None,
) -> dict[ConnectorRef, ConnectorRuntimePayload]:
    result: dict[ConnectorRef, ConnectorRuntimePayload] = {}
    for item in payload_items or ():
        raw_ref = _read_field(item, "connector_ref")
        if hasattr(raw_ref, "model_dump"):
            raw_ref = raw_ref.model_dump()
        try:
            ref = ConnectorRef.from_wire(raw_ref)
        except ValueError as exc:
            raise ConnectorRuntimeError(
                ERROR_INVALID_RUNTIME_CONTEXT,
                "Invalid connector runtime context.",
                details={"reason": str(exc)},
            ) from exc
        if ref in result:
            _raise_runtime_error(
                ERROR_INVALID_RUNTIME_CONTEXT, ref, reason="duplicate_ref"
            )
        context = _optional_mapping(_read_field(item, RUNTIME_INPUT_CONTEXT))
        secrets = _optional_mapping(_read_field(item, RUNTIME_INPUT_SECRETS))
        auth_selector = _optional_mapping(
            _read_field(item, RUNTIME_INPUT_AUTH_SELECTOR)
        )
        if ref.connector_type == CONNECTOR_TYPE_CUSTOM_API and auth_selector:
            _raise_runtime_error(
                ERROR_INVALID_RUNTIME_CONTEXT, ref, reason="auth_selector_not_supported"
            )
        result[ref] = ConnectorRuntimePayload(
            ref=ref,
            context=context,
            secrets=secrets,
            auth_selector=auth_selector,
        )
    return result


def _read_field(item: Any, field: str) -> Any:
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)


def _optional_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConnectorRuntimeError(
            ERROR_INVALID_RUNTIME_CONTEXT,
            "Invalid connector runtime context.",
            details={"reason": "runtime section must be an object"},
        )
    return dict(value)


def _load_visible_runtime_connectors(
    db: Session, *, user_id: int
) -> dict[ConnectorRef, Any]:
    visible: dict[ConnectorRef, Any] = {}
    mcp_rows = (
        db.query(MCPServer)
        .join(UserMCPServer, MCPServer.id == UserMCPServer.mcpserver_id)
        .filter(UserMCPServer.user_id == user_id, UserMCPServer.is_active)
        .all()
    )
    for server in mcp_rows:
        visible[
            ConnectorRef(cast(ConnectorType, CONNECTOR_TYPE_MCP), int(server.id))
        ] = server

    custom_api_rows = (
        db.query(CustomApi)
        .join(UserCustomApi, CustomApi.id == UserCustomApi.custom_api_id)
        .filter(UserCustomApi.user_id == user_id, UserCustomApi.is_active)
        .all()
    )
    for api in custom_api_rows:
        visible[
            ConnectorRef(cast(ConnectorType, CONNECTOR_TYPE_CUSTOM_API), int(api.id))
        ] = api
    return visible


def _plan_selected_refs(
    agent: Agent, visible: dict[ConnectorRef, Any]
) -> tuple[ConnectorRef, ...]:
    tool_categories = (
        list(agent.tool_categories) if isinstance(agent.tool_categories, list) else None
    )
    spec = ToolSelectionSpec.from_raw(tool_categories=tool_categories)
    selected: list[ConnectorRef] = []
    scoped_mcp_servers = spec.scoped_mcp_servers() if spec is not None else None
    for ref, connector in visible.items():
        if not _has_runtime_declaration(connector):
            continue
        name_key = normalize_mcp_server_name(str(connector.name or ""))
        if ref.connector_type == CONNECTOR_TYPE_MCP:
            if not spec.includes_mcp():
                continue
            if scoped_mcp_servers is not None and name_key not in scoped_mcp_servers:
                continue
        elif ref.connector_type == CONNECTOR_TYPE_CUSTOM_API:
            if not spec.includes_custom_api():
                continue
            if scoped_mcp_servers is not None and name_key not in scoped_mcp_servers:
                continue
        selected.append(ref)
    return tuple(sorted(selected))


def _has_runtime_declaration(connector: Any) -> bool:
    return bool(getattr(connector, "runtime_input_schema", None)) or bool(
        getattr(connector, "runtime_bindings", None)
    )


def _validate_payload_refs(
    payload_by_ref: dict[ConnectorRef, ConnectorRuntimePayload],
    *,
    visible: dict[ConnectorRef, Any],
    selected_refs: tuple[ConnectorRef, ...],
) -> None:
    selected = set(selected_refs)
    for ref in payload_by_ref:
        if ref not in visible:
            _raise_runtime_error(ERROR_CONNECTOR_NOT_FOUND, ref)
        if ref not in selected:
            _raise_runtime_error(
                ERROR_INVALID_RUNTIME_CONTEXT, ref, reason="connector_not_selected"
            )


def _load_task_selected_refs(task: Task) -> tuple[ConnectorRef, ...]:
    raw_refs = task.connector_runtime_selected_refs
    if raw_refs is None:
        return ()
    if not isinstance(raw_refs, list):
        raise ConnectorRuntimeError(
            ERROR_INVALID_RUNTIME_CONTEXT,
            "Invalid connector runtime context.",
            details={"reason": "stored selected refs must be a list"},
        )
    try:
        return tuple(sorted(ConnectorRef.from_wire(raw_ref) for raw_ref in raw_refs))
    except ValueError as exc:
        raise ConnectorRuntimeError(
            ERROR_INVALID_RUNTIME_CONTEXT,
            "Invalid connector runtime context.",
            details={"reason": str(exc)},
        ) from exc


def _load_task_context_rows(
    db: Session, *, task_id: int
) -> dict[ConnectorRef, dict[str, Any]]:
    rows = (
        db.query(TaskConnectorRuntimeContext)
        .filter(TaskConnectorRuntimeContext.task_id == task_id)
        .all()
    )
    result: dict[ConnectorRef, dict[str, Any]] = {}
    for row in rows:
        connector_type = cast(ConnectorType, str(row.connector_type))
        ref = ConnectorRef(connector_type, int(row.connector_id))
        context: dict[str, Any] = row.context if isinstance(row.context, dict) else {}
        result[ref] = dict(context)
    return result


def _validate_values_against_schema(
    ref: ConnectorRef,
    connector: Any,
    context: dict[str, Any],
    secrets: dict[str, Any],
    auth_selector: dict[str, Any],
) -> None:
    schema = _runtime_input_schema(connector)
    for section_name, values in (
        (RUNTIME_INPUT_CONTEXT, context),
        (RUNTIME_INPUT_SECRETS, secrets),
        (RUNTIME_INPUT_AUTH_SELECTOR, auth_selector),
    ):
        declarations = _schema_section(schema, section_name)
        if (
            section_name == RUNTIME_INPUT_AUTH_SELECTOR
            and ref.connector_type != CONNECTOR_TYPE_MCP
        ):
            if values:
                _raise_runtime_error(
                    ERROR_INVALID_RUNTIME_CONTEXT,
                    ref,
                    reason="auth_selector_not_supported",
                )
            continue
        for key in values:
            try:
                validate_runtime_source_key(key)
            except ValueError as exc:
                _raise_runtime_error(
                    ERROR_INVALID_RUNTIME_CONTEXT, ref, reason=str(exc)
                )
            if key not in declarations:
                _raise_runtime_error(
                    ERROR_INVALID_RUNTIME_CONTEXT,
                    ref,
                    reason=f"undeclared_{section_name}_key",
                )


def _runtime_input_schema(connector: Any) -> dict[str, Any]:
    schema = getattr(connector, "runtime_input_schema", None)
    return schema if isinstance(schema, dict) else {}


def _schema_section(schema: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = schema.get(section_name)
    return section if isinstance(section, dict) else {}


def _require_context_values(
    ref: ConnectorRef, connector: Any, context: dict[str, Any]
) -> None:
    declarations = _schema_section(
        _runtime_input_schema(connector), RUNTIME_INPUT_CONTEXT
    )
    for key, declaration in declarations.items():
        try:
            validate_runtime_source_key(key)
        except ValueError as exc:
            _raise_runtime_error(ERROR_INVALID_RUNTIME_CONTEXT, ref, reason=str(exc))
        if _is_required(declaration) and key not in context:
            _raise_runtime_error(
                ERROR_MISSING_RUNTIME_CONTEXT, ref, reason=f"missing_context.{key}"
            )


def _require_ephemeral_values(
    ref: ConnectorRef,
    connector: Any,
    secrets: dict[str, Any],
    auth_selector: dict[str, Any],
) -> None:
    schema = _runtime_input_schema(connector)
    for section_name, values in (
        (RUNTIME_INPUT_SECRETS, secrets),
        (RUNTIME_INPUT_AUTH_SELECTOR, auth_selector),
    ):
        declarations = _schema_section(schema, section_name)
        for key, declaration in declarations.items():
            try:
                validate_runtime_source_key(key)
            except ValueError as exc:
                _raise_runtime_error(
                    ERROR_INVALID_RUNTIME_CONTEXT, ref, reason=str(exc)
                )
            if _is_required(declaration) and key not in values:
                if section_name == RUNTIME_INPUT_SECRETS:
                    _raise_runtime_error(
                        ERROR_RUNTIME_SECRET_UNAVAILABLE,
                        ref,
                        reason=RUNTIME_SECRET_REASON_NOT_PROVIDED,
                    )
                _raise_runtime_error(
                    ERROR_RUNTIME_SECRET_UNAVAILABLE,
                    ref,
                    reason=RUNTIME_SECRET_REASON_NOT_PROVIDED,
                )


def _require_ephemeral_values_at_binding(
    ref: ConnectorRef,
    connector: Any,
    values: ConnectorRuntimeValues,
) -> None:
    schema = _runtime_input_schema(connector)
    for section_name, section_values in (
        (RUNTIME_INPUT_SECRETS, values.secrets),
        (RUNTIME_INPUT_AUTH_SELECTOR, values.auth_selector),
    ):
        declarations = _schema_section(schema, section_name)
        for key, declaration in declarations.items():
            if _is_required(declaration) and key not in section_values:
                _raise_runtime_error(
                    ERROR_RUNTIME_SECRET_UNAVAILABLE,
                    ref,
                    reason=RUNTIME_SECRET_REASON_STORE_LOST,
                )


def _resolve_runtime_values(
    *,
    task_id: int,
    turn_id: str | None,
    user_id: int | None,
    ref: ConnectorRef,
    values: ConnectorRuntimeValues,
) -> ConnectorRuntimeValues:
    if _RUNTIME_RESOLVER is None:
        return values
    resolved = _RUNTIME_RESOLVER(
        ConnectorRuntimeRequest(
            task_id=task_id,
            turn_id=turn_id,
            user_id=user_id,
            connector_ref=ref,
            values=values,
        )
    )
    return resolved if resolved is not None else values


def _is_required(declaration: Any) -> bool:
    return isinstance(declaration, dict) and bool(declaration.get("required"))


def _canonical_json_value(value: dict[str, Any]) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"))),
    )


def _raise_runtime_error(
    code: str, ref: ConnectorRef, *, reason: str | None = None
) -> None:
    details = {}
    if reason is not None:
        details["reason"] = reason
    raise ConnectorRuntimeError(
        code,
        _message_for_code(code),
        connector_ref=ref,
        details=details,
        status_code=_status_for_code(code),
    )


def _message_for_code(code: str) -> str:
    return {
        ERROR_CONNECTOR_NOT_FOUND: "Connector not found or not accessible.",
        ERROR_INVALID_RUNTIME_CONTEXT: "Invalid connector runtime context.",
        ERROR_MISSING_RUNTIME_CONTEXT: "Required connector runtime context is missing.",
        ERROR_RUNTIME_CONTEXT_IMMUTABLE: "Connector runtime context cannot change after task creation.",
        ERROR_RUNTIME_SECRET_UNAVAILABLE: "Required runtime secret is unavailable.",
        ERROR_RUNTIME_SECRET_NOT_ALLOWED: "Runtime secret is not allowed for this entrypoint.",
    }.get(code, "Invalid connector runtime context.")


def _status_for_code(code: str) -> int:
    if code == ERROR_CONNECTOR_NOT_FOUND:
        return 404
    if code == ERROR_RUNTIME_CONTEXT_IMMUTABLE:
        return 409
    return 400
