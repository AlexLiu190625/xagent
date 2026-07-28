from dataclasses import fields

import pytest

from xagent.skills.library import (
    CompositeSkillLibraryProvider,
    SkillRecord,
    SkillScopeContext,
    SkillWriteContext,
)
from xagent.skills.manager import SkillManager


def test_skill_scope_context_contains_only_detached_runtime_identity() -> None:
    context = SkillScopeContext(user_id=7, metadata={"team_id": 11})

    assert {field.name for field in fields(context)} == {"user_id", "metadata"}
    assert context.user_id == 7
    assert context.metadata == {"team_id": 11}


def test_skill_scope_context_rejects_request_owned_metadata() -> None:
    with pytest.raises(TypeError, match="detached scalar"):
        SkillScopeContext(user_id=7, metadata={"db": object()})


def test_skill_scope_context_copies_and_freezes_metadata() -> None:
    metadata = {"team_id": 11}
    context = SkillScopeContext(user_id=7, metadata=metadata)

    metadata["team_id"] = 12

    assert context.metadata == {"team_id": 11}
    with pytest.raises(TypeError):
        context.metadata["team_id"] = 12  # type: ignore[index]


def test_skill_write_context_keeps_request_owned_resources_explicit() -> None:
    user = object()
    db = object()
    request = object()

    context = SkillWriteContext(
        user=user,
        user_id=7,
        db=db,
        request=request,
        metadata={"team_id": 11},
    )

    assert context.user is user
    assert context.user_id == 7
    assert context.db is db
    assert context.request is request
    assert context.metadata == {"team_id": 11}


class StaticProvider:
    def __init__(self, source: str, records: list[SkillRecord]):
        self.source = source
        self.records = records

    async def list_records(self, context: SkillScopeContext) -> list[SkillRecord]:
        return self.records

    async def read_file(
        self, context: SkillScopeContext, record: SkillRecord, path: str
    ) -> bytes:
        return record.files[path]


def _record(name: str, source: str, description: str) -> SkillRecord:
    return SkillRecord(
        name=name,
        source=source,
        scope=source,
        files={
            "SKILL.md": (
                f"---\ndescription: {description!r}\n"
                f"when_to_use: Use {source}.\n---\n# {name}\n"
            ).encode()
        },
    )


@pytest.mark.asyncio
async def test_composite_provider_later_records_override_earlier_by_name():
    provider = CompositeSkillLibraryProvider(
        [
            StaticProvider("builtin", [_record("writer", "builtin", "builtin")]),
            StaticProvider("team", [_record("writer", "team", "team")]),
            StaticProvider("personal", [_record("writer", "personal", "personal")]),
        ]
    )

    manager = SkillManager(provider=provider, context=SkillScopeContext(user_id=7))
    await manager.initialize()

    skill = await manager.get_skill("writer")

    assert skill is not None
    assert skill["description"] == "personal"
    assert skill["source"] == "personal"
    assert skill["scope"] == "personal"


@pytest.mark.asyncio
async def test_composite_provider_can_return_visible_records_with_shadowed_state():
    provider = CompositeSkillLibraryProvider(
        [
            StaticProvider("team", [_record("writer", "team", "team")]),
            StaticProvider("personal", [_record("writer", "personal", "personal")]),
        ]
    )

    records = await provider.list_visible_records(SkillScopeContext(user_id=7))

    assert [(r.scope, r.name, r.effective, r.shadowed_by) for r in records] == [
        ("team", "writer", False, "personal"),
        ("personal", "writer", True, None),
    ]
