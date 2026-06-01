import json
import logging
import re
from dataclasses import dataclass
from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy.orm import Session, selectinload

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.user import User
from xagent.web.services.llm_utils import UserAwareModelStorage

from ..models.workforce import Workforce, WorkforceAgent, WorkforceBuilderMessage
from .agent_access import list_accessible_published_agents
from .workforce_access import (
    ensure_agent_access,
    ensure_workforce_access,
)
from .workforce_names import workforce_name_exists
from .workforce_snapshot import normalize_text
from .workforce_workers import create_workforce_worker

logger = logging.getLogger(__name__)

SUPPORTED_BUILDER_OPS = {
    "update_workforce",
    "add_existing_worker",
    "update_worker",
    "remove_worker",
}

PLACEHOLDER_PATCH_WARNING = (
    "Could not confidently infer a structured change. "
    "Please refine the instruction or edit manually."
)


@dataclass(frozen=True)
class WorkforceBuilderProposalResult:
    user_message: WorkforceBuilderMessage
    assistant_message: WorkforceBuilderMessage
    proposed_patch: dict[str, Any]
    requires_confirmation: bool = True


@dataclass(frozen=True)
class WorkforceBuilderApplyResult:
    workforce: Workforce
    message: WorkforceBuilderMessage


def serialize_builder_message(message: WorkforceBuilderMessage) -> dict[str, Any]:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "status": message.status,
        "proposed_patch": message.proposed_patch,
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def list_builder_messages(
    db: Session,
    user: User,
    workforce: Workforce,
) -> list[WorkforceBuilderMessage]:
    workforce = ensure_workforce_access(db, user, workforce, action="view")
    return (
        db.query(WorkforceBuilderMessage)
        .filter(WorkforceBuilderMessage.workforce_id == workforce.id)
        .order_by(WorkforceBuilderMessage.id.asc())
        .all()
    )


def _load_builder_workforce(db: Session, workforce: Workforce) -> Workforce:
    if workforce.id is None:
        return workforce
    loaded = (
        db.query(Workforce)
        .options(
            selectinload(Workforce.manager_agent),
            selectinload(Workforce.workers).selectinload(WorkforceAgent.agent),
        )
        .filter(Workforce.id == workforce.id)
        .first()
    )
    if loaded is None:
        raise HTTPException(status_code=404, detail="Workforce not found")
    return loaded


def _ensure_builder_workforce_access(
    db: Session,
    user: User,
    workforce: Workforce,
    *,
    action: str,
) -> Workforce:
    workforce = ensure_workforce_access(db, user, workforce, action=action)
    return _load_builder_workforce(db, workforce)


def _serialize_workers_for_prompt(workforce: Workforce) -> list[dict[str, Any]]:
    workers = sorted(
        workforce.workers,
        key=lambda item: (item.sort_order or 0, item.id or 0),
    )
    return [
        {
            "member_id": worker.id,
            "agent_id": worker.agent_id,
            "agent_name": worker.agent.name,
            "alias": worker.alias,
            "assignment_instructions": worker.assignment_instructions,
            "enabled": worker.enabled,
            "sort_order": worker.sort_order,
        }
        for worker in workers
    ]


def _make_builder_context(workforce: Workforce) -> dict[str, Any]:
    return {
        "workforce": {
            "id": workforce.id,
            "name": workforce.name,
            "description": workforce.description,
            "status": workforce.status,
            "manager_agent_id": workforce.manager_agent_id,
            "manager_agent_name": workforce.manager_agent.name
            if workforce.manager_agent
            else None,
            "manager_instructions": workforce.manager_instructions,
        },
        "workers": _serialize_workers_for_prompt(workforce),
    }


def _available_published_worker_agents(
    db: Session,
    user: User,
    workforce: Workforce,
) -> list[Agent]:
    excluded_agent_ids = {int(workforce.manager_agent_id)}
    excluded_agent_ids.update(int(worker.agent_id) for worker in workforce.workers)
    return list_accessible_published_agents(
        db,
        user,
        exclude_agent_ids=excluded_agent_ids,
    )


def _serialize_available_agents_for_prompt(
    db: Session,
    user: User,
    workforce: Workforce,
) -> list[dict[str, Any]]:
    return [
        {
            "agent_id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "status": getattr(agent.status, "value", str(agent.status)),
        }
        for agent in _available_published_worker_agents(db, user, workforce)
    ]


def _clean_patch(candidate: dict[str, Any]) -> dict[str, Any]:
    summary = (
        str(candidate.get("summary") or "").strip() or "Update workforce configuration."
    )
    warnings = candidate.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    clean_warnings = [str(item).strip() for item in warnings if str(item).strip()]
    clarification = str(candidate.get("clarification") or "").strip() or None

    operations = candidate.get("operations")
    if not isinstance(operations, list):
        operations = []

    clean_operations: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        op_name = str(operation.get("op") or "").strip()
        if op_name not in SUPPORTED_BUILDER_OPS:
            continue
        clean_operations.append(operation)

    return {
        "summary": summary,
        "operations": clean_operations,
        "warnings": clean_warnings,
        "clarification": clarification,
    }


def _normalize_optional_bool(
    operation: dict[str, Any],
    field_name: str,
) -> bool | None:
    if field_name not in operation:
        return None
    value = operation.get(field_name)
    if not isinstance(value, bool):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a boolean",
        )
    return value


def _has_meaningful_operations(patch: dict[str, Any]) -> bool:
    operations = patch.get("operations")
    if not isinstance(operations, list) or not operations:
        return False
    if len(operations) != 1:
        return True

    only_operation = operations[0]
    if not isinstance(only_operation, dict):
        return False
    if only_operation.get("op") != "update_workforce":
        return True

    fields = only_operation.get("fields")
    return isinstance(fields, dict) and bool(fields)


def _find_worker_by_name(workforce: Workforce, raw_name: str) -> WorkforceAgent | None:
    target = raw_name.strip().lower()
    if not target:
        return None
    for raw_worker in workforce.workers:
        worker = cast(WorkforceAgent, raw_worker)
        candidates = [
            worker.alias or "",
            worker.agent.name if worker.agent else "",
        ]
        for candidate in candidates:
            if candidate and candidate.lower() == target:
                return worker
    for raw_worker in workforce.workers:
        worker = cast(WorkforceAgent, raw_worker)
        candidates = [
            worker.alias or "",
            worker.agent.name if worker.agent else "",
        ]
        for candidate in candidates:
            if candidate and target in candidate.lower():
                return worker
    return None


def _find_agent_candidate_for_message(
    db: Session,
    user: User,
    workforce: Workforce,
    message: str,
) -> Agent | None:
    lower = message.lower()
    for agent in _available_published_worker_agents(db, user, workforce):
        if agent.name.lower() in lower:
            return agent
    return None


def _extract_assignment_instructions(message: str) -> str | None:
    quoted = [str(item).strip() for item in re.findall(r'"([^"]+)"', message)]
    if quoted:
        return quoted[-1] or None

    match = re.search(
        r"(?:to|for)\s+(?:handle|focus on|work on|cover)\s+(.+)$",
        message,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().rstrip(".")
    return None


def _fallback_patch_from_message(
    db: Session,
    user: User,
    workforce: Workforce,
    message: str,
) -> dict[str, Any]:
    text = message.strip()
    lower = text.lower()
    operations: list[dict[str, Any]] = []
    warnings: list[str] = []
    summary_parts: list[str] = []

    quoted_texts = re.findall(r'"([^"]+)"', text)
    workforce_name_target = quoted_texts[0].strip() if quoted_texts else None

    rename_match = re.search(r"\brename\b|\brenamed\b|\bname it\b|\bcall it\b", lower)
    if rename_match and workforce_name_target:
        operations.append(
            {
                "op": "update_workforce",
                "fields": {"name": workforce_name_target},
            }
        )
        summary_parts.append(f'Rename workforce to "{workforce_name_target}"')

    description_match = re.search(
        r"(?:description|desc)\s*(?:to|as|=)?\s*\"([^\"]+)\"",
        text,
        flags=re.IGNORECASE,
    )
    if description_match:
        operations.append(
            {
                "op": "update_workforce",
                "fields": {"description": description_match.group(1).strip()},
            }
        )
        summary_parts.append("Update workforce description")

    manager_instructions_match = re.search(
        r"(manager instructions?|manager prompt)\s*(?:to|as|=)?\s*\"([^\"]+)\"",
        text,
        flags=re.IGNORECASE,
    )
    if manager_instructions_match:
        operations.append(
            {
                "op": "update_workforce",
                "fields": {
                    "manager_instructions": manager_instructions_match.group(2).strip()
                },
            }
        )
        summary_parts.append("Update manager instructions")

    remove_match = re.search(
        r"\bremove\b\s+([a-zA-Z0-9 _-]+)",
        text,
        flags=re.IGNORECASE,
    )
    if remove_match:
        raw_target = remove_match.group(1).strip().rstrip(".")
        matched_worker = _find_worker_by_name(workforce, raw_target)
        if matched_worker is not None:
            operations.append({"op": "remove_worker", "member_id": matched_worker.id})
            warnings.append(
                f'Removing worker "{matched_worker.alias or matched_worker.agent.name}".'
            )
            summary_parts.append(
                f'Remove worker "{matched_worker.alias or matched_worker.agent.name}"'
            )

    update_match = re.search(
        (
            r"(?:make|update|change)\s+([a-zA-Z0-9 _-]+?)\s+"
            r"(?:focus on|to handle|to work on|to)\s+(.+)"
        ),
        text,
        flags=re.IGNORECASE,
    )
    if update_match:
        target_name = update_match.group(1).strip()
        instructions = update_match.group(2).strip().rstrip(".")
        matched_worker = _find_worker_by_name(workforce, target_name)
        if matched_worker is not None and instructions:
            operations.append(
                {
                    "op": "update_worker",
                    "member_id": matched_worker.id,
                    "assignment_instructions": instructions,
                }
            )
            summary_parts.append(
                f'Update worker "{matched_worker.alias or matched_worker.agent.name}"'
            )

    if "add" in lower and "worker" in lower:
        agent = _find_agent_candidate_for_message(db, user, workforce, text)
        instructions = _extract_assignment_instructions(text)
        if agent is not None and instructions:
            operations.append(
                {
                    "op": "add_existing_worker",
                    "agent_id": int(agent.id),
                    "alias": agent.name,
                    "assignment_instructions": instructions,
                }
            )
            summary_parts.append(f'Add worker "{agent.name}"')

    if not operations:
        operations.append(
            {
                "op": "update_workforce",
                "fields": {},
            }
        )
        warnings.append(PLACEHOLDER_PATCH_WARNING)
        summary_parts.append("No safe structured changes inferred")

    return {
        "summary": ". ".join(summary_parts) + ".",
        "operations": operations,
        "warnings": warnings,
        "clarification": None,
    }


async def _llm_generate_patch(
    db: Session,
    user: User,
    workforce: Workforce,
    message: str,
) -> dict[str, Any] | None:
    try:
        storage = UserAwareModelStorage(db)
        default_llm, _, _, _ = storage.get_configured_defaults(int(user.id))
        llm = default_llm
        if not llm:
            default_llm, _, _, _ = storage.get_configured_defaults(None)
            llm = default_llm
        if not llm:
            return None

        system_prompt = (
            "You are a Workforce Builder assistant. "
            "Team means human organization and must never be modified. "
            "Workforce means AI orchestration and may be modified only at the relationship layer. "
            "You may only output JSON for a proposed patch. "
            "You can modify workforce name, description, manager instructions, "
            "worker membership, worker alias, worker assignment instructions, "
            "and worker order. "
            "Do not modify underlying agent instructions, models, tools, skills, "
            "or knowledge bases. "
            "Supported operations are: update_workforce, add_existing_worker, "
            "update_worker, remove_worker. "
            "Workers must be existing published agents from available_published_agents. "
            "Do not create new agents or add workers from templates. "
            "For destructive operations like remove_worker, include a warning. "
            "If the user's intent is unclear, return no operations and include a "
            "clarification question. "
            "Return a JSON object with keys summary, operations, warnings, "
            "clarification. No markdown fences."
        )
        user_prompt = json.dumps(
            {
                "request": message,
                "current_state": _make_builder_context(workforce),
                "available_published_agents": _serialize_available_agents_for_prompt(
                    db,
                    user,
                    workforce,
                ),
            },
            ensure_ascii=False,
        )
        response = await llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = (
            response["content"]
            if isinstance(response, dict) and "content" in response
            else response
        )
        if not isinstance(content, str):
            content = str(content)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return None
        return _clean_patch(parsed)
    except Exception as exc:
        logger.warning("Failed to generate builder patch with LLM: %s", exc)
        return None


async def generate_builder_patch(
    db: Session,
    user: User,
    workforce: Workforce,
    message: str,
) -> tuple[str, dict[str, Any]]:
    workforce = _ensure_builder_workforce_access(db, user, workforce, action="edit")
    normalized_message = normalize_text(message, "message", required=True)

    llm_patch = await _llm_generate_patch(db, user, workforce, normalized_message)
    if llm_patch is not None:
        if _has_meaningful_operations(llm_patch):
            return (
                f"I prepared {len(llm_patch['operations'])} change(s) for review.",
                llm_patch,
            )
        clarification = llm_patch.get("clarification")
        if clarification:
            return (str(clarification), llm_patch)
        return (
            "I could not translate the request into an executable Workforce change. "
            "Review the proposal details and refine the prompt if needed.",
            llm_patch,
        )

    fallback_patch = _clean_patch(
        _fallback_patch_from_message(db, user, workforce, normalized_message)
    )
    if _has_meaningful_operations(fallback_patch):
        return (
            f"I prepared {len(fallback_patch['operations'])} change(s) using rule-based parsing because no LLM was available.",
            fallback_patch,
        )

    return (
        "I could not confidently translate the request into a safe structured change. "
        "Review the warning and refine the prompt if needed.",
        fallback_patch,
    )


def _apply_update_workforce(
    workforce: Workforce,
    operation: dict[str, Any],
    db: Session,
) -> None:
    workforce_row = cast(Any, workforce)
    fields = operation.get("fields")
    if not isinstance(fields, dict):
        return
    unsupported_fields = set(fields) - {"name", "description", "manager_instructions"}
    if unsupported_fields:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported workforce update fields: "
                + ", ".join(sorted(unsupported_fields))
            ),
        )
    if "name" in fields:
        name = normalize_text(cast(str | None, fields.get("name")), "name", True)
        if workforce_name_exists(
            db,
            scope_type=str(workforce.scope_type),
            scope_id=str(workforce.scope_id),
            name=name,
            exclude_workforce_id=int(workforce.id),
        ):
            raise HTTPException(status_code=409, detail="Workforce name already exists")
        workforce_row.name = name
    if "description" in fields:
        workforce_row.description = normalize_text(
            cast(str | None, fields.get("description")),
            "description",
        )
    if "manager_instructions" in fields:
        workforce_row.manager_instructions = normalize_text(
            cast(str | None, fields.get("manager_instructions")),
            "manager_instructions",
        )


def _apply_add_existing_worker(
    workforce: Workforce,
    operation: dict[str, Any],
    db: Session,
    user: User,
) -> None:
    agent_id = operation.get("agent_id")
    if not isinstance(agent_id, int):
        raise HTTPException(
            status_code=400,
            detail="agent_id is required for add_existing_worker",
        )

    agent = ensure_agent_access(
        db.query(Agent).filter(Agent.id == agent_id).first(),
        user,
        db,
        require_published=True,
    )
    agent_id_value = int(agent.id)
    if agent_id_value == int(workforce.manager_agent_id):
        raise HTTPException(
            status_code=400,
            detail="Manager agent cannot also be a worker",
        )
    existing = (
        db.query(WorkforceAgent)
        .filter(
            WorkforceAgent.workforce_id == workforce.id,
            WorkforceAgent.agent_id == agent.id,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="Agent already added to workforce")

    assignment_instructions = normalize_text(
        cast(str | None, operation.get("assignment_instructions")),
        "assignment_instructions",
        required=True,
    )
    enabled = _normalize_optional_bool(operation, "enabled")
    create_workforce_worker(
        db,
        workforce,
        user,
        source_type="existing",
        assignment_instructions=assignment_instructions,
        alias=cast(str | None, operation.get("alias")),
        agent_id=agent_id_value,
        enabled=True if enabled is None else enabled,
        sort_order=operation.get("sort_order")
        if isinstance(operation.get("sort_order"), int)
        else None,
    )


def _apply_update_worker(
    workforce: Workforce,
    operation: dict[str, Any],
    db: Session,
) -> None:
    member_id = operation.get("member_id")
    if not isinstance(member_id, int):
        raise HTTPException(
            status_code=400,
            detail="member_id is required for update_worker",
        )
    worker = (
        db.query(WorkforceAgent)
        .filter(
            WorkforceAgent.id == member_id,
            WorkforceAgent.workforce_id == workforce.id,
        )
        .first()
    )
    if worker is None:
        raise HTTPException(status_code=404, detail="Workforce worker not found")
    worker_row = cast(Any, worker)

    alias = None
    if "alias" in operation:
        alias = normalize_text(
            cast(str | None, operation.get("alias")),
            "alias",
        )
    assignment_instructions = None
    if "assignment_instructions" in operation:
        assignment_instructions = normalize_text(
            cast(str | None, operation.get("assignment_instructions")),
            "assignment_instructions",
            required=True,
        )
    enabled = _normalize_optional_bool(operation, "enabled")

    if "alias" in operation:
        worker_row.alias = alias
    if "assignment_instructions" in operation:
        worker_row.assignment_instructions = assignment_instructions
    if enabled is not None:
        worker_row.enabled = enabled
    if "sort_order" in operation and isinstance(operation.get("sort_order"), int):
        worker_row.sort_order = int(operation["sort_order"])


def _apply_remove_worker(
    workforce: Workforce,
    operation: dict[str, Any],
    db: Session,
) -> None:
    member_id = operation.get("member_id")
    if not isinstance(member_id, int):
        raise HTTPException(
            status_code=400,
            detail="member_id is required for remove_worker",
        )
    worker = (
        db.query(WorkforceAgent)
        .filter(
            WorkforceAgent.id == member_id,
            WorkforceAgent.workforce_id == workforce.id,
        )
        .first()
    )
    if worker is None:
        raise HTTPException(status_code=404, detail="Workforce worker not found")
    db.delete(worker)
    db.flush()


def _ensure_published_agent(agent: Agent | None) -> Agent:
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.status != AgentStatus.PUBLISHED:
        raise HTTPException(
            status_code=400,
            detail="Workforce agents must be published",
        )
    return agent


def _validate_active_workforce_configuration(
    db: Session,
    workforce: Workforce,
) -> None:
    if workforce.status != "active":
        return

    _ensure_published_agent(db.get(Agent, int(workforce.manager_agent_id)))
    workers = (
        db.query(WorkforceAgent)
        .filter(WorkforceAgent.workforce_id == workforce.id)
        .order_by(WorkforceAgent.sort_order.asc(), WorkforceAgent.id.asc())
        .all()
    )
    enabled_workers = [worker for worker in workers if worker.enabled]
    if not enabled_workers:
        raise HTTPException(
            status_code=400,
            detail="Workforce requires at least one enabled worker",
        )

    for worker in enabled_workers:
        _ensure_published_agent(worker.agent)
        normalize_text(
            cast(str | None, worker.assignment_instructions),
            "assignment_instructions",
            required=True,
        )
        if int(worker.agent_id) == int(workforce.manager_agent_id):
            raise HTTPException(
                status_code=400,
                detail="Manager agent cannot also be a worker",
            )


def apply_builder_patch(
    db: Session,
    user: User,
    workforce: Workforce,
    patch: dict[str, Any],
) -> Workforce:
    workforce = ensure_workforce_access(db, user, workforce, action="edit")
    clean_patch = _clean_patch(patch)
    operations = clean_patch["operations"]

    for operation in operations:
        op_name = operation["op"]
        if op_name == "update_workforce":
            _apply_update_workforce(workforce, operation, db)
        elif op_name == "add_existing_worker":
            _apply_add_existing_worker(workforce, operation, db, user)
        elif op_name == "update_worker":
            _apply_update_worker(workforce, operation, db)
        elif op_name == "remove_worker":
            _apply_remove_worker(workforce, operation, db)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported builder operation: {op_name}",
            )

    db.flush()
    _validate_active_workforce_configuration(db, workforce)
    return workforce


async def propose_workforce_builder_changes(
    db: Session,
    user: User,
    workforce: Workforce,
    *,
    message: str,
) -> WorkforceBuilderProposalResult:
    workforce = ensure_workforce_access(db, user, workforce, action="edit")
    user_message_text = normalize_text(message, "message", required=True)

    assistant_message_text, patch = await generate_builder_patch(
        db,
        user,
        workforce,
        user_message_text,
    )

    try:
        user_message = WorkforceBuilderMessage(
            workforce_id=int(workforce.id),
            user_id=int(user.id),
            role="user",
            content=user_message_text,
            status="message",
        )
        db.add(user_message)
        assistant_message = WorkforceBuilderMessage(
            workforce_id=int(workforce.id),
            user_id=int(user.id),
            role="assistant",
            content=assistant_message_text,
            proposed_patch=patch,
            status="proposed",
        )
        db.add(assistant_message)
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(user_message)
    db.refresh(assistant_message)
    return WorkforceBuilderProposalResult(
        user_message=user_message,
        assistant_message=assistant_message,
        proposed_patch=cast(dict[str, Any], assistant_message.proposed_patch),
    )


def apply_workforce_builder_changes(
    db: Session,
    user: User,
    workforce: Workforce,
    *,
    message_id: int,
    proposed_patch: dict[str, Any],
) -> WorkforceBuilderApplyResult:
    workforce = ensure_workforce_access(db, user, workforce, action="edit")
    message = (
        db.query(WorkforceBuilderMessage)
        .filter(
            WorkforceBuilderMessage.id == message_id,
            WorkforceBuilderMessage.workforce_id == workforce.id,
        )
        .first()
    )
    if message is None:
        raise HTTPException(status_code=404, detail="Builder message not found")
    if message.role != "assistant":
        raise HTTPException(status_code=400, detail="Builder message is not applicable")
    if message.status != "proposed" or not isinstance(message.proposed_patch, dict):
        raise HTTPException(
            status_code=400,
            detail="Builder message has no pending patch",
        )
    if proposed_patch != message.proposed_patch:
        raise HTTPException(
            status_code=400,
            detail="Proposed patch does not match message",
        )

    try:
        workforce = apply_builder_patch(db, user, workforce, proposed_patch)
        message.status = "applied"
        message.proposed_patch = proposed_patch
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(workforce)
    db.refresh(message)
    return WorkforceBuilderApplyResult(workforce=workforce, message=message)
