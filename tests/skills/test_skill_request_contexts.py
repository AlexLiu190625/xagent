from types import SimpleNamespace
from typing import Any

import pytest

from xagent.skills.library import SkillScopeContext, SkillWriteContext
from xagent.web.api.skill_hub import _scope_context, _write_context
from xagent.web.api.skills import _request_skill_manager


def test_skill_hub_read_context_contains_only_detached_scope_identity() -> None:
    user = SimpleNamespace(id=7, _saas_team_id=11)

    context = _scope_context(user)

    assert context == SkillScopeContext(user_id=7, metadata={"team_id": 11})


def test_skill_hub_write_context_keeps_request_resources_out_of_reads() -> None:
    user = SimpleNamespace(id=7, _saas_team_id=11)
    request = object()
    db = object()

    context = _write_context(request, user, db)

    assert context == SkillWriteContext(
        user=user,
        user_id=7,
        db=db,
        request=request,
        metadata={"team_id": 11},
    )


@pytest.mark.asyncio
async def test_skills_api_manager_does_not_open_a_route_owned_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Manager:
        async def ensure_initialized(self) -> None:
            captured["initialized"] = True

    def _create_skill_manager(*, context):
        captured["context"] = context
        return _Manager()

    def _unexpected_session_factory():
        raise AssertionError("skills API must not own a read session")

    monkeypatch.setattr(
        "xagent.skills.utils.create_skill_manager",
        _create_skill_manager,
    )
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        _unexpected_session_factory,
    )

    manager = await _request_skill_manager(
        SimpleNamespace(),
        SimpleNamespace(id=7),
    )

    assert isinstance(manager, _Manager)
    assert captured == {
        "context": SkillScopeContext(user_id=7),
        "initialized": True,
    }
