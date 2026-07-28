"""Request-to-worker database session handoff for Skill runtime reads."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import Depends
from sqlalchemy.orm import Session

from xagent.skills.library import SkillScopeContext, SkillScopeMetadataValue
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.database import get_db, release_db_connection_if_clean
from xagent.web.models.user import User


class SkillRuntimeSessionBoundaryError(RuntimeError):
    """The caller Session cannot yield its connection to a Skill worker."""


def handoff_skill_runtime_session(caller_db: Session) -> None:
    """Release a clean caller transaction before worker-owned database I/O."""
    if not release_db_connection_if_clean(caller_db):
        raise SkillRuntimeSessionBoundaryError(
            "Cannot start Skill runtime database work while the caller "
            "database session has pending writes"
        )


def build_runtime_skill_scope(
    *,
    user_id: int | None,
    metadata: Mapping[str, SkillScopeMetadataValue] | None = None,
    caller_db: Session,
) -> SkillScopeContext:
    """Detach request identity, then hand the caller's pool slot to the worker."""
    context = SkillScopeContext(user_id=user_id, metadata=metadata or {})
    handoff_skill_runtime_session(caller_db)
    return context


def get_skill_runtime_scope(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SkillScopeContext:
    """Resolve detached Skill identity before the route body starts."""
    metadata: dict[str, SkillScopeMetadataValue] = {}
    team_id = getattr(current_user, "_saas_team_id", None)
    if isinstance(team_id, int):
        metadata["team_id"] = team_id
    return build_runtime_skill_scope(
        user_id=int(current_user.id) if current_user.id is not None else None,
        metadata=metadata,
        caller_db=db,
    )
