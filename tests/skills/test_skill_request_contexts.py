from types import SimpleNamespace
from typing import Any

import pytest

from xagent.skills.library import SkillScopeContext, SkillWriteContext
from xagent.web.api.skill_hub import _get_scoped_manager, _write_context
from xagent.web.api.skills import _request_skill_manager
from xagent.web.services.skill_runtime import SkillRuntimeSessionBoundaryError


def test_skill_hub_write_context_reuses_detached_scope_identity() -> None:
    user = SimpleNamespace(id=7, _saas_team_id=11)
    request = object()
    db = object()
    scope = SkillScopeContext(user_id=7, metadata={"team_id": 11})

    context = _write_context(request, user, db, scope)

    assert context == SkillWriteContext(
        user=user,
        user_id=7,
        db=db,
        request=request,
        metadata={"team_id": 11},
    )


@pytest.mark.asyncio
async def test_skills_api_manager_hands_off_caller_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    db = object()
    scope = SkillScopeContext(user_id=7)

    class _Manager:
        async def ensure_initialized(self) -> None:
            captured["initialized"] = True

    def _create_skill_manager(*, context):
        captured["context"] = context
        return _Manager()

    def _unexpected_session_factory():
        raise AssertionError("skills API must not own a read session")

    def _handoff(caller_db):
        captured["caller_db"] = caller_db

    monkeypatch.setattr(
        "xagent.skills.utils.create_skill_manager",
        _create_skill_manager,
    )
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        _unexpected_session_factory,
    )
    monkeypatch.setattr(
        "xagent.web.api.skills.handoff_skill_runtime_session",
        _handoff,
    )

    manager = await _request_skill_manager(
        SimpleNamespace(),
        scope,
        db,
    )

    assert isinstance(manager, _Manager)
    assert captured == {
        "caller_db": db,
        "context": scope,
        "initialized": True,
    }


@pytest.mark.asyncio
async def test_skill_hub_manager_fails_before_provider_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = SkillScopeContext(user_id=7)
    db = object()

    def _reject_handoff(caller_db):
        assert caller_db is db
        raise SkillRuntimeSessionBoundaryError("pending writes")

    monkeypatch.setattr(
        "xagent.web.api.skill_hub.handoff_skill_runtime_session",
        _reject_handoff,
    )

    with pytest.raises(SkillRuntimeSessionBoundaryError, match="pending writes"):
        await _get_scoped_manager(SimpleNamespace(), scope, db)
