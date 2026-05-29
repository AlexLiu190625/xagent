import json
from typing import Any

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.web.models import (
    Agent,
    Base,
    User,
    Workforce,
    WorkforceAgent,
    WorkforceBuilderMessage,
)
from xagent.web.models.agent import AgentOrigin, AgentStatus
from xagent.web.services import workforce_builder as builder_module
from xagent.web.services import workforce_creator as creator_module
from xagent.web.services.agent_store import AgentStore
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    set_cache_backend_for_testing,
)
from xagent.web.services.workforce_access import WorkforcePolicy, set_workforce_policy
from xagent.web.services.workforce_builder import (
    apply_workforce_builder_changes,
    generate_builder_patch,
    list_builder_messages,
    propose_workforce_builder_changes,
    serialize_builder_message,
)
from xagent.web.services.workforce_creator import (
    create_workforce_from_prompt,
    generate_workforce_creation_plan,
)
from xagent.web.services.workforce_workers import create_workforce_worker


@pytest.fixture()
def db_session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture(autouse=True)
def reset_workforce_policy() -> None:
    set_workforce_policy(WorkforcePolicy())
    yield
    set_workforce_policy(WorkforcePolicy())


def _create_user(db: Session, username: str, *, is_admin: bool = False) -> User:
    user = User(
        username=username,
        password_hash="hash",
        is_admin=is_admin,
    )
    db.add(user)
    db.flush()
    return user


def _create_agent(
    db: Session,
    user: User,
    name: str,
    *,
    execution_mode: str = "balanced",
    status: AgentStatus = AgentStatus.PUBLISHED,
) -> Agent:
    agent = Agent(
        user_id=int(user.id),
        name=name,
        description=f"{name} description",
        instructions=f"{name} instructions",
        execution_mode=execution_mode,
        models={"general": "test-model"},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        status=status,
    )
    db.add(agent)
    db.flush()
    return agent


def _create_workforce(
    db: Session,
    user: User,
    manager: Agent,
    *,
    status: str = "draft",
    name: str = "Research Team",
) -> Workforce:
    workforce = Workforce(
        owner_user_id=int(user.id),
        scope_type="user",
        scope_id=str(user.id),
        name=name,
        description="Coordinates research tasks",
        manager_agent_id=int(manager.id),
        manager_instructions="Prefer concise synthesis.",
        status=status,
    )
    db.add(workforce)
    db.flush()
    return workforce


def _add_worker(
    db: Session,
    user: User,
    workforce: Workforce,
    worker_agent: Agent,
    *,
    alias: str = "Research Analyst",
    enabled: bool = True,
) -> WorkforceAgent:
    return create_workforce_worker(
        db,
        workforce,
        user,
        source_type="existing",
        agent_id=int(worker_agent.id),
        alias=alias,
        assignment_instructions="Collect evidence and cite sources.",
        enabled=enabled,
    )


class _FakeLLM:
    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"content": json.dumps(self.response)}


class _FakeModelStorage:
    def __init__(self, db: Session, llm: _FakeLLM | None = None):
        del db
        self.llm = llm

    def get_configured_defaults(
        self, user_id: int | None
    ) -> tuple[_FakeLLM | None, None, None, None]:
        del user_id
        return (self.llm, None, None, None)


@pytest.mark.asyncio
async def test_generate_workforce_creation_plan_cleans_llm_output(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _create_user(db_session, "owner")
    other = _create_user(db_session, "other")
    analyst = _create_agent(db_session, owner, "Analyst")
    draft_agent = _create_agent(
        db_session,
        owner,
        "Draft Worker",
        status=AgentStatus.DRAFT,
    )
    other_agent = _create_agent(db_session, other, "Other Worker")
    fake_llm = _FakeLLM(
        {
            "name": " Launch Research ",
            "description": "Coordinate launch analysis",
            "manager": {
                "name": "Launch Manager",
                "description": "Runs the workflow",
                "instructions": "Synthesize worker findings.",
            },
            "workers": [
                {
                    "agent_id": int(analyst.id),
                    "alias": "Analyst",
                    "assignment_instructions": "Gather research.",
                    "enabled": True,
                },
                {
                    "agent_id": int(draft_agent.id),
                    "assignment_instructions": "Should be ignored.",
                },
                {
                    "agent_id": int(other_agent.id),
                    "assignment_instructions": "Should be ignored.",
                },
                {
                    "agent_id": int(analyst.id),
                    "assignment_instructions": "Duplicate should be ignored.",
                },
                {"agent_id": "bad", "assignment_instructions": "Invalid."},
            ],
            "warnings": [" keep this "],
        }
    )
    monkeypatch.setattr(
        creator_module,
        "UserAwareModelStorage",
        lambda db: _FakeModelStorage(db, fake_llm),
    )

    plan = await generate_workforce_creation_plan(
        db_session,
        owner,
        "Launch research",
    )

    sent_prompt = json.loads(fake_llm.calls[0]["messages"][1]["content"])
    sent_agent_ids = {
        item["agent_id"] for item in sent_prompt["available_published_agents"]
    }
    assert sent_agent_ids == {analyst.id}
    assert plan["name"] == "Launch Research"
    assert plan["manager"]["name"] == "Launch Manager"
    assert plan["workers"] == [
        {
            "agent_id": analyst.id,
            "alias": "Analyst",
            "assignment_instructions": "Gather research.",
            "enabled": True,
        }
    ]
    assert plan["warnings"] == ["keep this"]


@pytest.mark.asyncio
async def test_create_workforce_from_prompt_creates_draft_and_builder_messages(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _create_user(db_session, "owner")
    worker_agent = _create_agent(db_session, owner, "Analyst")
    existing_manager = _create_agent(db_session, owner, "Launch Manager")
    existing_workforce = _create_workforce(
        db_session,
        owner,
        existing_manager,
        name="Launch Workforce",
    )
    db_session.commit()

    async def fake_plan(db: Session, user: User, prompt: str) -> dict[str, Any]:
        del db, user
        return {
            "name": "launch workforce",
            "description": prompt,
            "manager": {
                "name": "launch manager",
                "description": "Coordinates launch work.",
                "instructions": "Delegate and summarize.",
            },
            "manager_instructions": "Delegate and summarize.",
            "workers": [
                {
                    "agent_id": int(worker_agent.id),
                    "alias": "Analyst",
                    "assignment_instructions": "Collect launch research.",
                    "enabled": True,
                }
            ],
            "warnings": ["Review before publishing."],
        }

    monkeypatch.setattr(
        creator_module,
        "generate_workforce_creation_plan",
        fake_plan,
    )

    result = await create_workforce_from_prompt(
        db_session,
        owner,
        prompt="Plan a product launch",
    )

    workforce = result.workforce
    assert workforce.id != existing_workforce.id
    assert workforce.name == "launch workforce 2"
    assert workforce.status == "draft"
    assert workforce.manager_agent.name == "launch manager 2"
    assert workforce.manager_agent.status == AgentStatus.PUBLISHED
    assert (
        workforce.manager_agent.origin == AgentOrigin.WORKFORCE_GENERATED_MANAGER.value
    )
    assert workforce.manager_agent.published_at is not None
    assert workforce.manager_agent.widget_enabled is False
    assert workforce.manager_agent.execution_mode == "think"
    assert len(workforce.workers) == 1
    assert workforce.workers[0].agent_id == worker_agent.id
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert "Review before publishing." in result.messages[1].content
    assert db_session.query(WorkforceBuilderMessage).count() == 2


@pytest.mark.asyncio
async def test_create_workforce_from_prompt_hides_generated_manager_from_agent_list(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_cache_backend_for_testing(InMemoryTTLCache())
    try:
        owner = _create_user(db_session, "owner")
        worker_agent = _create_agent(db_session, owner, "Analyst")
        db_session.commit()
        store = AgentStore(db_session)
        cached_agent_names = {
            item["name"] for item in store.list_agent_items(int(owner.id))
        }
        assert cached_agent_names == {"Analyst"}

        async def fake_plan(db: Session, user: User, prompt: str) -> dict[str, Any]:
            del db, user, prompt
            return {
                "name": "Launch Workforce",
                "description": "Plan a product launch",
                "manager": {
                    "name": "Launch Manager",
                    "description": "Coordinates launch work.",
                    "instructions": "Delegate and summarize.",
                },
                "manager_instructions": "Delegate and summarize.",
                "workers": [
                    {
                        "agent_id": int(worker_agent.id),
                        "alias": "Analyst",
                        "assignment_instructions": "Collect launch research.",
                        "enabled": True,
                    }
                ],
                "warnings": [],
            }

        monkeypatch.setattr(
            creator_module,
            "generate_workforce_creation_plan",
            fake_plan,
        )

        result = await create_workforce_from_prompt(
            db_session,
            owner,
            prompt="Plan a product launch",
        )

        refreshed_agent_names = {
            item["name"] for item in store.list_agent_items(int(owner.id))
        }
        assert result.workforce.manager_agent.name == "Launch Manager"
        assert refreshed_agent_names == {"Analyst"}
    finally:
        set_cache_backend_for_testing(None)


@pytest.mark.asyncio
async def test_generate_builder_patch_uses_llm_and_filters_available_agents(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _create_user(db_session, "owner")
    manager = _create_agent(db_session, owner, "Manager")
    existing_worker_agent = _create_agent(db_session, owner, "Analyst")
    available_agent = _create_agent(db_session, owner, "Editor")
    draft_agent = _create_agent(
        db_session,
        owner,
        "Draft Worker",
        status=AgentStatus.DRAFT,
    )
    workforce = _create_workforce(db_session, owner, manager)
    _add_worker(db_session, owner, workforce, existing_worker_agent)
    fake_llm = _FakeLLM(
        {
            "summary": "Update the workforce.",
            "operations": [
                {"op": "unknown", "fields": {"name": "ignored"}},
                {
                    "op": "add_existing_worker",
                    "agent_id": int(available_agent.id),
                    "assignment_instructions": "Edit the final brief.",
                },
            ],
            "warnings": [" check "],
        }
    )
    monkeypatch.setattr(
        builder_module,
        "UserAwareModelStorage",
        lambda db: _FakeModelStorage(db, fake_llm),
    )

    assistant_text, patch = await generate_builder_patch(
        db_session,
        owner,
        workforce,
        "Add an editor",
    )

    sent_prompt = json.loads(fake_llm.calls[0]["messages"][1]["content"])
    sent_agent_ids = {
        item["agent_id"] for item in sent_prompt["available_published_agents"]
    }
    assert sent_agent_ids == {available_agent.id}
    assert draft_agent.id not in sent_agent_ids
    assert "1 change" in assistant_text
    assert patch == {
        "summary": "Update the workforce.",
        "operations": [
            {
                "op": "add_existing_worker",
                "agent_id": available_agent.id,
                "assignment_instructions": "Edit the final brief.",
            }
        ],
        "warnings": ["check"],
        "clarification": None,
    }


def test_load_builder_workforce_preloads_prompt_relationships(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "owner")
    manager = _create_agent(db_session, owner, "Manager")
    analyst = _create_agent(db_session, owner, "Analyst")
    editor = _create_agent(db_session, owner, "Editor")
    workforce = _create_workforce(db_session, owner, manager)
    _add_worker(db_session, owner, workforce, analyst)
    _add_worker(db_session, owner, workforce, editor, alias="Editor")
    workforce_id = int(workforce.id)
    db_session.commit()
    db_session.expire_all()

    unloaded_workforce = db_session.get(Workforce, workforce_id)
    assert unloaded_workforce is not None

    loaded = builder_module._load_builder_workforce(db_session, unloaded_workforce)

    assert "manager_agent" not in inspect(loaded).unloaded
    assert "workers" not in inspect(loaded).unloaded
    assert {worker.agent.name for worker in loaded.workers} == {"Analyst", "Editor"}
    assert all("agent" not in inspect(worker).unloaded for worker in loaded.workers)


@pytest.mark.asyncio
async def test_builder_fallback_can_update_description_without_rename(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _create_user(db_session, "owner")
    manager = _create_agent(db_session, owner, "Manager")
    worker_agent = _create_agent(db_session, owner, "Analyst")
    workforce = _create_workforce(db_session, owner, manager)
    _add_worker(db_session, owner, workforce, worker_agent)
    monkeypatch.setattr(
        builder_module,
        "UserAwareModelStorage",
        lambda db: _FakeModelStorage(db, None),
    )

    assistant_text, patch = await generate_builder_patch(
        db_session,
        owner,
        workforce,
        'Set description to "Focus on launch planning"',
    )

    assert "rule-based parsing" in assistant_text
    assert patch["operations"] == [
        {
            "op": "update_workforce",
            "fields": {"description": "Focus on launch planning"},
        }
    ]


@pytest.mark.asyncio
async def test_builder_fallback_can_add_policy_visible_agent(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class VisibleAgentPolicy(WorkforcePolicy):
        def __init__(self, visible_agent_ids: set[int]):
            self.visible_agent_ids = visible_agent_ids

        def get_visible_agent_ids(
            self,
            db: Session,
            user: User,
            purpose: str,
        ) -> set[int]:
            del db, user, purpose
            return self.visible_agent_ids

    owner = _create_user(db_session, "owner")
    other = _create_user(db_session, "other")
    manager = _create_agent(db_session, owner, "Manager")
    existing_worker_agent = _create_agent(db_session, owner, "Analyst")
    visible_agent = _create_agent(db_session, other, "Data Miner")
    workforce = _create_workforce(db_session, owner, manager)
    _add_worker(db_session, owner, workforce, existing_worker_agent)
    set_workforce_policy(VisibleAgentPolicy({int(visible_agent.id)}))
    monkeypatch.setattr(
        builder_module,
        "UserAwareModelStorage",
        lambda db: _FakeModelStorage(db, None),
    )

    assistant_text, patch = await generate_builder_patch(
        db_session,
        owner,
        workforce,
        'Add worker Data Miner to handle "dataset analysis"',
    )
    apply_builder_result = builder_module.apply_builder_patch(
        db_session,
        owner,
        workforce,
        patch,
    )

    assert "rule-based parsing" in assistant_text
    assert patch["operations"] == [
        {
            "op": "add_existing_worker",
            "agent_id": visible_agent.id,
            "alias": "Data Miner",
            "assignment_instructions": "dataset analysis",
        }
    ]
    worker_agent_ids = {
        worker.agent_id
        for worker in db_session.query(WorkforceAgent)
        .filter(WorkforceAgent.workforce_id == apply_builder_result.id)
        .all()
    }
    assert worker_agent_ids == {
        existing_worker_agent.id,
        visible_agent.id,
    }


@pytest.mark.asyncio
async def test_propose_builder_changes_persists_user_and_assistant_messages(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _create_user(db_session, "owner")
    manager = _create_agent(db_session, owner, "Manager")
    worker_agent = _create_agent(db_session, owner, "Analyst")
    workforce = _create_workforce(db_session, owner, manager)
    _add_worker(db_session, owner, workforce, worker_agent)

    async def fake_patch(
        db: Session,
        user: User,
        target_workforce: Workforce,
        message: str,
    ) -> tuple[str, dict[str, Any]]:
        del db, user, target_workforce
        assert not db_session.new
        assert not db_session.dirty
        assert not db_session.deleted
        return (
            f"Prepared patch for {message}",
            {
                "summary": "Rename workforce.",
                "operations": [
                    {
                        "op": "update_workforce",
                        "fields": {"name": "Launch Team"},
                    }
                ],
                "warnings": [],
                "clarification": None,
            },
        )

    monkeypatch.setattr(builder_module, "generate_builder_patch", fake_patch)

    result = await propose_workforce_builder_changes(
        db_session,
        owner,
        workforce,
        message="rename it",
    )

    messages = list_builder_messages(db_session, owner, workforce)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert result.user_message.id == messages[0].id
    assert result.assistant_message.status == "proposed"
    assert result.requires_confirmation is True
    assert result.proposed_patch["operations"][0]["fields"]["name"] == "Launch Team"
    assert (
        serialize_builder_message(result.assistant_message)["proposed_patch"]["summary"]
        == "Rename workforce."
    )


def test_list_builder_messages_requires_workforce_view_access(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "owner")
    other = _create_user(db_session, "other")
    manager = _create_agent(db_session, owner, "Manager")
    workforce = _create_workforce(db_session, owner, manager)
    message = WorkforceBuilderMessage(
        workforce_id=int(workforce.id),
        user_id=int(owner.id),
        role="assistant",
        content="Prepared patch.",
        status="message",
    )
    db_session.add(message)
    db_session.commit()

    assert list_builder_messages(db_session, owner, workforce) == [message]
    with pytest.raises(HTTPException) as denied:
        list_builder_messages(db_session, other, workforce)

    assert denied.value.status_code == 403
    assert denied.value.detail == "Access denied"


def test_apply_builder_changes_validates_patch_and_updates_message(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "owner")
    manager = _create_agent(db_session, owner, "Manager")
    worker_agent = _create_agent(db_session, owner, "Analyst")
    workforce = _create_workforce(db_session, owner, manager, status="active")
    _add_worker(db_session, owner, workforce, worker_agent)
    patch = {
        "summary": "Rename workforce.",
        "operations": [
            {
                "op": "update_workforce",
                "fields": {"name": "Launch Workforce"},
            }
        ],
        "warnings": [],
        "clarification": None,
    }
    message = WorkforceBuilderMessage(
        workforce_id=int(workforce.id),
        user_id=int(owner.id),
        role="assistant",
        content="Prepared patch.",
        proposed_patch=patch,
        status="proposed",
    )
    db_session.add(message)
    db_session.commit()

    with pytest.raises(HTTPException) as mismatch:
        apply_workforce_builder_changes(
            db_session,
            owner,
            workforce,
            message_id=int(message.id),
            proposed_patch={**patch, "summary": "Different"},
        )
    assert mismatch.value.status_code == 400
    assert mismatch.value.detail == "Proposed patch does not match message"

    result = apply_workforce_builder_changes(
        db_session,
        owner,
        workforce,
        message_id=int(message.id),
        proposed_patch=patch,
    )

    assert result.workforce.name == "Launch Workforce"
    assert result.message.status == "applied"
    assert result.message.proposed_patch == patch


def test_apply_builder_patch_rejects_workforce_status_update(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "owner")
    manager = _create_agent(db_session, owner, "Manager")
    worker_agent = _create_agent(db_session, owner, "Analyst")
    workforce = _create_workforce(db_session, owner, manager, status="draft")
    _add_worker(db_session, owner, workforce, worker_agent)
    patch = {
        "summary": "Publish workforce.",
        "operations": [
            {
                "op": "update_workforce",
                "fields": {"status": "active"},
            }
        ],
        "warnings": [],
        "clarification": None,
    }
    message = WorkforceBuilderMessage(
        workforce_id=int(workforce.id),
        user_id=int(owner.id),
        role="assistant",
        content="Prepared patch.",
        proposed_patch=patch,
        status="proposed",
    )
    db_session.add(message)
    db_session.commit()

    with pytest.raises(HTTPException) as invalid:
        apply_workforce_builder_changes(
            db_session,
            owner,
            workforce,
            message_id=int(message.id),
            proposed_patch=patch,
        )

    assert invalid.value.status_code == 400
    assert invalid.value.detail == "Unsupported workforce update fields: status"
    assert db_session.get(Workforce, workforce.id).status == "draft"


def test_apply_builder_patch_rejects_case_insensitive_workforce_name_duplicate(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "owner")
    manager = _create_agent(db_session, owner, "Manager")
    first_workforce = _create_workforce(
        db_session,
        owner,
        manager,
        name="Launch Workforce",
    )
    second_workforce = _create_workforce(
        db_session,
        owner,
        manager,
        name="Research Team",
    )
    patch = {
        "summary": "Rename workforce.",
        "operations": [
            {
                "op": "update_workforce",
                "fields": {"name": "launch workforce"},
            }
        ],
        "warnings": [],
        "clarification": None,
    }

    with pytest.raises(HTTPException) as conflict:
        builder_module.apply_builder_patch(db_session, owner, second_workforce, patch)

    assert first_workforce.name == "Launch Workforce"
    assert second_workforce.name == "Research Team"
    assert conflict.value.status_code == 409
    assert conflict.value.detail == "Workforce name already exists"


def test_admin_can_apply_active_workforce_metadata_patch_without_run_scope(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "owner")
    admin = _create_user(db_session, "admin", is_admin=True)
    manager = _create_agent(db_session, owner, "Manager")
    worker_agent = _create_agent(db_session, owner, "Analyst")
    workforce = _create_workforce(db_session, owner, manager, status="active")
    _add_worker(db_session, owner, workforce, worker_agent)
    patch = {
        "summary": "Rename workforce.",
        "operations": [
            {
                "op": "update_workforce",
                "fields": {"name": "Admin Renamed Workforce"},
            }
        ],
        "warnings": [],
        "clarification": None,
    }
    message = WorkforceBuilderMessage(
        workforce_id=int(workforce.id),
        user_id=int(admin.id),
        role="assistant",
        content="Prepared patch.",
        proposed_patch=patch,
        status="proposed",
    )
    db_session.add(message)
    db_session.commit()

    result = apply_workforce_builder_changes(
        db_session,
        admin,
        workforce,
        message_id=int(message.id),
        proposed_patch=patch,
    )

    assert result.workforce.name == "Admin Renamed Workforce"
    assert result.message.status == "applied"


def test_apply_builder_patch_revalidates_active_workforce_configuration(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "owner")
    manager = _create_agent(db_session, owner, "Manager")
    worker_agent = _create_agent(db_session, owner, "Analyst")
    workforce = _create_workforce(db_session, owner, manager, status="active")
    worker = _add_worker(db_session, owner, workforce, worker_agent)
    patch = {
        "summary": "Disable the only worker.",
        "operations": [
            {
                "op": "update_worker",
                "member_id": int(worker.id),
                "enabled": False,
            }
        ],
        "warnings": [],
        "clarification": None,
    }
    message = WorkforceBuilderMessage(
        workforce_id=int(workforce.id),
        user_id=int(owner.id),
        role="assistant",
        content="Prepared patch.",
        proposed_patch=patch,
        status="proposed",
    )
    db_session.add(message)
    db_session.commit()

    with pytest.raises(HTTPException) as invalid:
        apply_workforce_builder_changes(
            db_session,
            owner,
            workforce,
            message_id=int(message.id),
            proposed_patch=patch,
        )

    assert invalid.value.status_code == 400
    assert invalid.value.detail == "Workforce requires at least one enabled worker"
    assert db_session.get(WorkforceAgent, worker.id).enabled is True


def test_apply_builder_patch_rejects_non_boolean_worker_enabled(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "owner")
    manager = _create_agent(db_session, owner, "Manager")
    worker_agent = _create_agent(db_session, owner, "Analyst")
    workforce = _create_workforce(db_session, owner, manager)
    worker = _add_worker(db_session, owner, workforce, worker_agent)
    patch = {
        "summary": "Disable worker.",
        "operations": [
            {
                "op": "update_worker",
                "member_id": int(worker.id),
                "enabled": "false",
            }
        ],
        "warnings": [],
        "clarification": None,
    }
    message = WorkforceBuilderMessage(
        workforce_id=int(workforce.id),
        user_id=int(owner.id),
        role="assistant",
        content="Prepared patch.",
        proposed_patch=patch,
        status="proposed",
    )
    db_session.add(message)
    db_session.commit()

    with pytest.raises(HTTPException) as invalid:
        apply_workforce_builder_changes(
            db_session,
            owner,
            workforce,
            message_id=int(message.id),
            proposed_patch=patch,
        )

    assert invalid.value.status_code == 400
    assert invalid.value.detail == "enabled must be a boolean"
    assert db_session.get(WorkforceAgent, worker.id).enabled is True


def test_apply_builder_patch_rejects_non_boolean_added_worker_enabled(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "owner")
    manager = _create_agent(db_session, owner, "Manager")
    worker_agent = _create_agent(db_session, owner, "Analyst")
    editor_agent = _create_agent(db_session, owner, "Editor")
    workforce = _create_workforce(db_session, owner, manager)
    _add_worker(db_session, owner, workforce, worker_agent)
    patch = {
        "summary": "Add editor.",
        "operations": [
            {
                "op": "add_existing_worker",
                "agent_id": int(editor_agent.id),
                "assignment_instructions": "Edit the final brief.",
                "enabled": "false",
            }
        ],
        "warnings": [],
        "clarification": None,
    }
    message = WorkforceBuilderMessage(
        workforce_id=int(workforce.id),
        user_id=int(owner.id),
        role="assistant",
        content="Prepared patch.",
        proposed_patch=patch,
        status="proposed",
    )
    db_session.add(message)
    db_session.commit()

    with pytest.raises(HTTPException) as invalid:
        apply_workforce_builder_changes(
            db_session,
            owner,
            workforce,
            message_id=int(message.id),
            proposed_patch=patch,
        )

    assert invalid.value.status_code == 400
    assert invalid.value.detail == "enabled must be a boolean"
    assert (
        db_session.query(WorkforceAgent)
        .filter(
            WorkforceAgent.workforce_id == workforce.id,
            WorkforceAgent.agent_id == editor_agent.id,
        )
        .first()
        is None
    )


@pytest.mark.asyncio
async def test_builder_services_require_workforce_edit_access(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "owner")
    other = _create_user(db_session, "other")
    manager = _create_agent(db_session, owner, "Manager")
    worker_agent = _create_agent(db_session, owner, "Analyst")
    workforce = _create_workforce(db_session, owner, manager)
    _add_worker(db_session, owner, workforce, worker_agent)

    with pytest.raises(HTTPException) as denied:
        await propose_workforce_builder_changes(
            db_session,
            other,
            workforce,
            message="rename it",
        )

    assert denied.value.status_code == 403
    assert denied.value.detail == "Access denied"
    assert db_session.query(WorkforceBuilderMessage).count() == 0
