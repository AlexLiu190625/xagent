"""Application-owned access seam for personal management API keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class PersonalKeyAccessScope:
    """The personal-key owners an authenticated actor may manage."""

    owner_user_ids: tuple[int, ...]
    can_manage_others: bool


_personal_key_scope_hook: Any = None


def set_personal_key_scope_hook(hook: Any) -> None:
    """Install or clear the application-owned personal-key scope resolver."""
    global _personal_key_scope_hook
    _personal_key_scope_hook = hook


def get_personal_key_access_scope(db: Any, actor: Any) -> PersonalKeyAccessScope:
    """Resolve a fail-closed scope, always retaining the actor's own keys."""
    actor_id = int(actor.id)
    if _personal_key_scope_hook is None:
        return PersonalKeyAccessScope((actor_id,), False)

    try:
        proposed = cast(PersonalKeyAccessScope, _personal_key_scope_hook(db, actor))
        if not isinstance(proposed, PersonalKeyAccessScope):
            raise TypeError("Personal key scope hook returned an invalid value")
        if not proposed.can_manage_others:
            return PersonalKeyAccessScope((actor_id,), False)
        owner_ids = tuple(
            dict.fromkeys(
                (actor_id, *(int(owner_id) for owner_id in proposed.owner_user_ids))
            )
        )
        return PersonalKeyAccessScope(owner_ids, True)
    except (AttributeError, TypeError, ValueError):
        return PersonalKeyAccessScope((actor_id,), False)
