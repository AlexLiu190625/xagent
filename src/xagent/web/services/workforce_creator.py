import json
import logging
import re
from dataclasses import dataclass
from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
from xagent.web.models.user import User
from xagent.web.services.llm_utils import UserAwareModelStorage

from ..models.workforce import Workforce, WorkforceBuilderMessage
from .agent_access import list_accessible_published_agents
from .agent_store import AgentStore
from .hot_path_cache import invalidate_agent_cache
from .workforce_access import (
    can_create_workforce,
    resolve_create_scope,
)
from .workforce_names import resolve_unique_agent_name, resolve_unique_workforce_name
from .workforce_snapshot import normalize_text
from .workforce_workers import create_workforce_worker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkforcePromptCreationResult:
    workforce: Workforce
    plan: dict[str, Any]
    messages: list[WorkforceBuilderMessage]


def _serialize_available_agents(agents: list[Agent]) -> list[dict[str, Any]]:
    return [
        {
            "agent_id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "status": getattr(agent.status, "value", str(agent.status)),
        }
        for agent in agents
    ]


def _short_name_from_prompt(prompt: str) -> str:
    words = re.findall(r"[\w-]+", prompt, flags=re.UNICODE)
    if not words:
        return "New Workforce"
    name = " ".join(words[:6]).strip()
    if len(name) > 80:
        name = name[:80].rstrip()
    return f"{name} Workforce" if "workforce" not in name.lower() else name


def _clean_creation_plan(
    candidate: dict[str, Any],
    available_agent_ids: set[int],
    prompt: str,
) -> dict[str, Any]:
    name = str(candidate.get("name") or "").strip() or _short_name_from_prompt(prompt)
    description = str(candidate.get("description") or "").strip() or prompt.strip()

    manager = candidate.get("manager")
    if not isinstance(manager, dict):
        manager = {}
    manager_name = str(manager.get("name") or "").strip() or f"{name} Manager"
    manager_description = (
        str(manager.get("description") or "").strip()
        or "Coordinates this Workforce and synthesizes worker outputs."
    )
    manager_instructions = str(manager.get("instructions") or "").strip() or (
        "Understand the user's goal, delegate focused work to the available workers, "
        "compare their outputs, and return one coherent final answer."
    )

    workers = candidate.get("workers")
    if not isinstance(workers, list):
        workers = []
    clean_workers: list[dict[str, Any]] = []
    seen_agent_ids: set[int] = set()
    for item in workers:
        if not isinstance(item, dict):
            continue
        agent_id = item.get("agent_id")
        if not isinstance(agent_id, int):
            continue
        if agent_id not in available_agent_ids or agent_id in seen_agent_ids:
            continue
        instructions = str(item.get("assignment_instructions") or "").strip()
        if not instructions:
            continue
        enabled = item.get("enabled", True)
        clean_workers.append(
            {
                "agent_id": agent_id,
                "alias": str(item.get("alias") or "").strip() or None,
                "assignment_instructions": instructions,
                "enabled": enabled if isinstance(enabled, bool) else True,
            }
        )
        seen_agent_ids.add(agent_id)

    warnings = candidate.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    clean_warnings = [str(item).strip() for item in warnings if str(item).strip()]
    if not clean_workers:
        clean_warnings.append(
            "No published worker agents were selected. Add workers before running."
        )

    return {
        "name": name[:200],
        "description": description,
        "manager": {
            "name": manager_name[:200],
            "description": manager_description,
            "instructions": manager_instructions,
        },
        "workers": clean_workers,
        "warnings": clean_warnings,
    }


def _fallback_creation_plan(prompt: str, agents: list[Agent]) -> dict[str, Any]:
    lower_prompt = prompt.lower()
    selected: list[Agent] = []
    for agent in agents:
        haystack = f"{agent.name} {agent.description or ''}".lower()
        if any(
            token and token in haystack
            for token in re.findall(r"[a-z0-9]+", lower_prompt)
        ):
            selected.append(agent)
        if len(selected) >= 3:
            break
    if not selected:
        selected = agents[: min(3, len(agents))]

    name = _short_name_from_prompt(prompt)
    return _clean_creation_plan(
        {
            "name": name,
            "description": prompt,
            "manager": {
                "name": f"{name} Manager",
                "description": "Coordinates this Workforce and synthesizes worker outputs.",
                "instructions": (
                    "Break the user request into worker assignments, call the right "
                    "workers, reconcile their outputs, and provide a concise final answer."
                ),
            },
            "workers": [
                {
                    "agent_id": int(agent.id),
                    "alias": agent.name,
                    "assignment_instructions": (
                        f"Contribute to the Workforce goal using the strengths of {agent.name}."
                    ),
                    "enabled": True,
                }
                for agent in selected
            ],
            "warnings": [
                "Created from a rule-based fallback because no LLM plan was available."
            ],
        },
        {int(agent.id) for agent in agents},
        prompt,
    )


async def generate_workforce_creation_plan(
    db: Session,
    user: User,
    prompt: str,
) -> dict[str, Any]:
    normalized_prompt = normalize_text(prompt, "prompt", required=True)
    agents = list_accessible_published_agents(db, user)
    available_agent_ids = {int(agent.id) for agent in agents}

    try:
        storage = UserAwareModelStorage(db)
        default_llm, _, _, _ = storage.get_configured_defaults(int(user.id))
        llm = default_llm
        if not llm:
            default_llm, _, _, _ = storage.get_configured_defaults(None)
            llm = default_llm
        if not llm:
            return _fallback_creation_plan(normalized_prompt, agents)

        system_prompt = (
            "You create initial AI Workforce drafts from a user's goal. "
            "A Workforce is AI orchestration, not a human team. "
            "Create a manager spec that can coordinate workers and synthesize results. "
            "Select workers only from available_published_agents. "
            "Do not create worker agents. If no worker fits, return no workers and add a warning. "
            "Return JSON only with keys: name, description, manager, workers, warnings. "
            "manager has name, description, instructions. "
            "Each worker has agent_id, alias, assignment_instructions, enabled. "
            "Keep instructions concrete and task-oriented."
        )
        user_prompt = json.dumps(
            {
                "request": normalized_prompt,
                "available_published_agents": _serialize_available_agents(agents),
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
            return _fallback_creation_plan(normalized_prompt, agents)
        return _clean_creation_plan(parsed, available_agent_ids, normalized_prompt)
    except Exception as exc:
        logger.warning("Failed to generate Workforce creation plan with LLM: %s", exc)
        return _fallback_creation_plan(normalized_prompt, agents)


def _create_manager_agent_from_plan(
    db: Session,
    user: User,
    name: str,
    manager_plan: dict[str, Any],
) -> Agent:
    return AgentStore(db).add_agent(
        user_id=int(user.id),
        name=resolve_unique_agent_name(
            db,
            user_id=int(user.id),
            name=str(manager_plan.get("name") or f"{name} Manager"),
        ),
        description=normalize_text(
            cast(str | None, manager_plan.get("description")),
            "description",
        ),
        instructions=normalize_text(
            cast(str | None, manager_plan.get("instructions")),
            "instructions",
        ),
        execution_mode="think",
        models=None,
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        status=AgentStatus.PUBLISHED,
        widget_enabled=False,
        allowed_domains=[],
    )


def invalidate_workforce_creation_cache(
    *,
    owner_user_id: int,
    manager_agent_id: int,
    workforce_id: int,
) -> None:
    del workforce_id
    invalidate_agent_cache(owner_user_id, manager_agent_id)


async def create_workforce_from_prompt(
    db: Session,
    user: User,
    *,
    prompt: str,
) -> WorkforcePromptCreationResult:
    normalized_prompt = normalize_text(prompt, "prompt", required=True)
    scope_type, scope_id = resolve_create_scope(db, user)
    if not can_create_workforce(db, user, scope_type, scope_id):
        raise HTTPException(status_code=403, detail="Access denied")

    owner_user_id = int(user.id)
    try:
        plan = await generate_workforce_creation_plan(db, user, normalized_prompt)
        name = resolve_unique_workforce_name(
            db,
            scope_type=scope_type,
            scope_id=scope_id,
            name=str(plan["name"]),
        )
        manager_plan = cast(dict[str, Any], plan["manager"])
        manager_agent = _create_manager_agent_from_plan(db, user, name, manager_plan)

        workforce = Workforce(
            owner_user_id=int(user.id),
            scope_type=scope_type,
            scope_id=scope_id,
            name=name,
            description=normalize_text(
                cast(str | None, plan.get("description")),
                "description",
            ),
            manager_agent_id=int(manager_agent.id),
            status="draft",
        )
        db.add(workforce)
        db.flush()

        for index, worker in enumerate(
            cast(list[dict[str, Any]], plan.get("workers") or [])
        ):
            create_workforce_worker(
                db,
                workforce,
                user,
                source_type="existing",
                agent_id=cast(int, worker["agent_id"]),
                alias=cast(str | None, worker.get("alias")),
                assignment_instructions=str(worker["assignment_instructions"]),
                enabled=bool(worker.get("enabled", True)),
                sort_order=index + 1,
            )

        user_message = WorkforceBuilderMessage(
            workforce_id=int(workforce.id),
            user_id=int(user.id),
            role="user",
            content=normalized_prompt,
            status="message",
        )
        db.add(user_message)
        warnings = cast(list[str], plan.get("warnings") or [])
        assistant_content = "Created an initial Workforce draft from your prompt."
        if warnings:
            assistant_content = f"{assistant_content}\n\n" + "\n".join(
                f"- {warning}" for warning in warnings
            )
        assistant_message = WorkforceBuilderMessage(
            workforce_id=int(workforce.id),
            user_id=int(user.id),
            role="assistant",
            content=assistant_content,
            status="message",
        )
        db.add(assistant_message)
        manager_agent_id = int(manager_agent.id)
        workforce_id = int(workforce.id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    else:
        invalidate_workforce_creation_cache(
            owner_user_id=owner_user_id,
            manager_agent_id=manager_agent_id,
            workforce_id=workforce_id,
        )
    db.refresh(workforce)
    db.refresh(user_message)
    db.refresh(assistant_message)
    return WorkforcePromptCreationResult(
        workforce=workforce,
        plan=plan,
        messages=[user_message, assistant_message],
    )
