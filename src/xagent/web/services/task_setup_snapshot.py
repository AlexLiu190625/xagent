"""Read-only snapshot of the synchronous DB state required to bootstrap
a task-bound ``AgentService``.

Background:
    ``AgentServiceManager.get_agent_for_task`` runs a contiguous block
    of synchronous DB queries (Task row + per-task LLM resolution +
    optional Agent Builder lookup with up to 4 ``DBModel`` queries and
    4 user-aware LLM access checks) on the main asyncio event loop. On
    a fully-configured Agent Builder task that adds up to 8-12 DB
    round-trips. Under load the block measures 20+ seconds of asyncio
    slow-callback time and blocks every other request on the same
    worker (issue #427 — ``_schedule_bg._runner took 23.371s``
    observed locally on 2026-05-20).

    This module batches those reads into a single function intended to
    be invoked through ``asyncio.to_thread``. The function opens its
    own ``SessionLocal``, eagerly reads everything, closes the session,
    and returns a frozen primitive snapshot. ORM rows MUST NOT escape
    the loader -- a downstream caller that mistakenly held an ORM
    reference past the close would hit ``DetachedInstanceError`` on
    its next attribute access.

Out of scope (first cut, by design):
    * ``UploadedFile`` selected-files loop -- contains writes
      (``UploadedFile.task_id`` assignment + ``db.flush()``), so it
      stays on the main loop with the request session.
    * ``_load_persisted_conversation_history`` /
      ``_load_persisted_execution_context`` -- already separate async
      helpers; can be migrated in a follow-up.
    * ToolFactory inner DB I/O -- tool subclasses hold ``self._db``;
      that refactor is Step 4 in the PR3 sequence.
    * MCP server configs -- async + OAuth refresh path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from sqlalchemy.orm import Session

from ...config import (
    get_agent_pattern_for_execution_mode,
    get_default_task_execution_mode,
)
from ...core.model.chat.basic.base import BaseLLM
from ..models.agent import Agent, AgentStatus
from ..models.database import get_session_local
from ..models.model import Model as DBModel
from ..models.task import Task

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TaskFields:
    """Primitive subset of the ``Task`` row needed past the snapshot."""

    id: int
    user_id: int
    status: Any  # ``TaskStatus`` enum value (frozen, not ORM).
    agent_id: Optional[int]
    agent_config: Any  # JSON column -- ``dict | None`` in practice.
    model_name: Optional[str]
    compact_model_name: Optional[str]
    execution_mode: Optional[str]
    agent_type: Optional[str]


@dataclass(frozen=True)
class _AgentFields:
    """Primitive subset of the ``Agent`` row when ``task.agent_id`` is set."""

    id: int
    name: str
    status: AgentStatus  # enum, not ORM
    instructions: Optional[str]


@dataclass(frozen=True)
class TaskSetupSnapshot:
    """All synchronous DB state that ``get_agent_for_task`` needs to
    bootstrap a task-bound ``AgentService``.

    Strict invariant: every field is a primitive, an enum, a frozen
    dataclass, or a fully-constructed application-layer object
    (``BaseLLM``) that is safe to read off the loop thread. ORM rows
    must not leak.
    """

    task: _TaskFields
    task_pattern: str
    # Final resolved LLMs after the agent-builder override (if any).
    task_llm: Optional[BaseLLM]
    task_fast_llm: Optional[BaseLLM]
    task_vision_llm: Optional[BaseLLM]
    task_compact_llm: Optional[BaseLLM]
    # Agent Builder configuration -- only populated when
    # ``task.agent_id`` resolves to an existing ``Agent`` row.
    agent: Optional[_AgentFields]
    agent_config: Optional[dict]
    excluded_agent_id: Optional[int]


def _resolve_task_llm_ids_sync(task_row: Task, session: Session) -> List[Optional[str]]:
    """Normalize the four task LLM identifiers using the snapshot's
    own session. Mirrors ``AgentServiceManager._get_task_llm_ids``.
    """
    from .llm_utils import CoreStorage, make_normalize_model_id

    core_storage = CoreStorage(session, DBModel)
    normalize = make_normalize_model_id(core_storage)
    return [
        normalize(
            getattr(task_row, "model_id", None),
            getattr(task_row, "model_name", None),
        ),
        normalize(
            getattr(task_row, "small_fast_model_id", None),
            getattr(task_row, "small_fast_model_name", None),
        ),
        normalize(
            getattr(task_row, "visual_model_id", None),
            getattr(task_row, "visual_model_name", None),
        ),
        normalize(
            getattr(task_row, "compact_model_id", None),
            getattr(task_row, "compact_model_name", None),
        ),
    ]


def _load_agent_builder_config_sync(
    agent_row: Agent, session: Session, user_id: int
) -> dict:
    """Eagerly load Agent Builder configuration into a primitive dict
    inside the snapshot's session. Mirrors
    ``AgentServiceManager._load_agent_builder_config`` but is kept here
    so the snapshot loader does not depend on the manager class.
    """
    from .llm_utils import UserAwareModelStorage

    storage = UserAwareModelStorage(session)

    default_llm: Optional[BaseLLM] = None
    fast_llm: Optional[BaseLLM] = None
    vision_llm: Optional[BaseLLM] = None
    compact_llm: Optional[BaseLLM] = None

    raw_models: Any = agent_row.models or {}
    models: dict[str, Any] = dict(raw_models) if isinstance(raw_models, dict) else {}

    def _resolve(slot: str) -> Optional[BaseLLM]:
        db_row_id = models.get(slot)
        if not db_row_id:
            return None
        db_model = session.query(DBModel).filter(DBModel.id == db_row_id).first()
        if not db_model:
            return None
        return storage.get_llm_by_name_with_access(str(db_model.model_id), user_id)

    default_llm = _resolve("general")
    fast_llm = _resolve("small_fast")
    vision_llm = _resolve("visual")
    compact_llm = _resolve("compact")

    return {
        "llms": (default_llm, fast_llm, vision_llm, compact_llm),
        "execution_mode": agent_row.execution_mode,
        "instructions": agent_row.instructions,
        "skills": list(agent_row.skills or []),
        "knowledge_bases": list(agent_row.knowledge_bases or []),
        "tool_categories": list(agent_row.tool_categories or []),
    }


def load_task_setup_snapshot_sync(
    task_id: int,
    user_id: Optional[int],
) -> Optional[TaskSetupSnapshot]:
    """Open a dedicated ``SessionLocal``, read every synchronous field
    ``get_agent_for_task`` needs for normal (non-reconstruct) creation,
    close the session, and return a primitive snapshot.

    Designed to be called from the event loop via
    ``await asyncio.to_thread(load_task_setup_snapshot_sync, ...)`` so
    the main loop stays responsive during the read (issue #427).

    Returns ``None`` when the task row is missing -- callers fall back
    to whatever behaviour the legacy in-line code already implements
    for that case (default LLM, no agent-builder override).
    """
    from .llm_utils import resolve_llms_from_names

    session_factory = get_session_local()
    session: Session = session_factory()
    try:
        task_row = session.query(Task).filter(Task.id == task_id).first()
        if task_row is None:
            return None

        task_fields = _TaskFields(
            id=int(task_row.id),
            user_id=int(task_row.user_id),
            status=task_row.status,
            agent_id=int(task_row.agent_id) if task_row.agent_id is not None else None,
            agent_config=(
                dict(task_row.agent_config)
                if isinstance(task_row.agent_config, dict)
                else task_row.agent_config
            ),
            model_name=(
                str(task_row.model_name) if task_row.model_name is not None else None
            ),
            compact_model_name=(
                str(task_row.compact_model_name)
                if task_row.compact_model_name is not None
                else None
            ),
            execution_mode=getattr(task_row, "execution_mode", None),
            agent_type=(
                str(task_row.agent_type) if task_row.agent_type is not None else None
            ),
        )

        task_execution_mode = task_fields.execution_mode
        if not task_execution_mode:
            task_execution_mode = get_default_task_execution_mode(
                agent_id=task_fields.agent_id,
            )
        task_pattern = get_agent_pattern_for_execution_mode(task_execution_mode)

        llm_ids = _resolve_task_llm_ids_sync(task_row, session)
        (
            task_llm,
            task_fast_llm,
            task_vision_llm,
            task_compact_llm,
        ) = resolve_llms_from_names(llm_ids, session, user_id)

        agent_fields: Optional[_AgentFields] = None
        agent_config: Optional[dict] = None
        excluded_agent_id: Optional[int] = None

        if task_fields.agent_id is not None:
            agent_row = (
                session.query(Agent)
                .filter(
                    Agent.id == task_fields.agent_id,
                    Agent.user_id == task_fields.user_id,
                )
                .first()
            )
            if agent_row is not None:
                agent_fields = _AgentFields(
                    id=int(agent_row.id),
                    name=str(agent_row.name),
                    status=agent_row.status,
                    instructions=(
                        str(agent_row.instructions)
                        if agent_row.instructions is not None
                        else None
                    ),
                )
                if agent_row.status == AgentStatus.PUBLISHED:
                    excluded_agent_id = agent_fields.id

                agent_config = _load_agent_builder_config_sync(
                    agent_row, session, int(task_fields.user_id)
                )
                (
                    task_llm,
                    task_fast_llm,
                    task_vision_llm,
                    task_compact_llm,
                ) = agent_config["llms"]

        return TaskSetupSnapshot(
            task=task_fields,
            task_pattern=task_pattern,
            task_llm=task_llm,
            task_fast_llm=task_fast_llm,
            task_vision_llm=task_vision_llm,
            task_compact_llm=task_compact_llm,
            agent=agent_fields,
            agent_config=agent_config,
            excluded_agent_id=excluded_agent_id,
        )
    finally:
        session.close()
