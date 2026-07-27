"""Optional application hooks for team-owned knowledge bases.

Standalone xagent keeps collections scoped to the requesting user's id. A
multi-tenant application can install these hooks to expose a logical team
collection while its files and vector rows remain in the original storage
user's namespace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import Session

KnowledgeBaseAction = Literal["read", "edit", "delete"]


@dataclass(frozen=True)
class KnowledgeBaseAccess:
    name: str
    storage_user_id: int
    team_owned: bool = False
    can_edit: bool = True
    can_delete: bool = True


KnowledgeBaseVisibilityHook = Callable[[Session | None, int], list[KnowledgeBaseAccess]]
KnowledgeBaseAccessHook = Callable[
    [Session | None, int, str, KnowledgeBaseAction], KnowledgeBaseAccess | None
]
KnowledgeBaseLifecycleHook = Callable[[Session | None, int, str, str | None], None]

_visibility_hook: KnowledgeBaseVisibilityHook | None = None
_access_hook: KnowledgeBaseAccessHook | None = None
_renamed_hook: KnowledgeBaseLifecycleHook | None = None
_deleted_hook: KnowledgeBaseLifecycleHook | None = None


def set_knowledge_base_team_hooks(
    *,
    visibility: KnowledgeBaseVisibilityHook | None = None,
    access: KnowledgeBaseAccessHook | None = None,
    renamed: KnowledgeBaseLifecycleHook | None = None,
    deleted: KnowledgeBaseLifecycleHook | None = None,
) -> None:
    """Install or clear application-owned knowledge-base hooks."""

    global _visibility_hook, _access_hook, _renamed_hook, _deleted_hook
    _visibility_hook = visibility
    _access_hook = access
    _renamed_hook = renamed
    _deleted_hook = deleted


def has_knowledge_base_visibility_hook() -> bool:
    """Return whether the application installed a team visibility hook."""

    return _visibility_hook is not None


def visible_team_knowledge_bases(
    db: Session | None, user_id: int
) -> list[KnowledgeBaseAccess]:
    """Return visible team KBs through the application-owned hook.

    Callers may invoke the hook from a worker thread. When ``db`` is ``None``,
    the hook must create, use, and close its own thread-confined session and
    must not depend on event-loop thread-local state.
    """
    if _visibility_hook is None:
        return []
    return list(_visibility_hook(db, int(user_id)))


def resolve_knowledge_base_access(
    db: Session | None,
    user_id: int,
    name: str,
    action: KnowledgeBaseAction = "read",
) -> KnowledgeBaseAccess:
    """Resolve ownership; hooks must open a session when ``db`` is None."""

    if _access_hook is not None:
        resolved = _access_hook(db, int(user_id), name, action)
        if resolved is not None:
            return resolved
    return KnowledgeBaseAccess(name=name, storage_user_id=int(user_id))


def notify_knowledge_base_renamed(
    db: Session | None, user_id: int, old_name: str, new_name: str
) -> None:
    """Notify the app; hooks must open a session when ``db`` is None."""
    if _renamed_hook is not None and old_name != new_name:
        _renamed_hook(db, int(user_id), old_name, new_name)


def notify_knowledge_base_deleted(db: Session | None, user_id: int, name: str) -> None:
    """Notify the app; hooks must open a session when ``db`` is None."""
    if _deleted_hook is not None:
        _deleted_hook(db, int(user_id), name, None)
