"""Integration tests for agent management endpoints."""

from typing import Any

import pytest

from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce
from xagent.web.services.workforce_access import WorkforcePolicy, set_workforce_policy

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _reset_workforce_policy() -> None:
    set_workforce_policy(WorkforcePolicy())
    yield
    set_workforce_policy(WorkforcePolicy())


def _create_agent(headers: dict[str, str], name: str = "Test Agent") -> int:
    resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "test",
            "instructions": "You are a test agent.",
            "execution_mode": "balanced",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def _create_agent_row(
    *,
    user_id: int,
    name: str,
    status: AgentStatus = AgentStatus.DRAFT,
    origin: str = AgentOrigin.USER.value,
    widget_enabled: bool = True,
    allowed_domains: list[str] | None = None,
) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=user_id,
            name=name,
            description=f"{name} description",
            instructions=f"{name} instructions",
            execution_mode="balanced",
            origin=origin,
            status=status,
            widget_enabled=widget_enabled,
            allowed_domains=allowed_domains or [],
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None
        return int(user.id)
    finally:
        db.close()


class _VisibleAgentPolicy(WorkforcePolicy):
    def __init__(self, visible_agent_ids: set[int]) -> None:
        self.visible_agent_ids = visible_agent_ids

    def get_visible_agent_ids(
        self,
        db: Any,
        user: User,
        purpose: str,
    ) -> set[int]:
        del db, user, purpose
        return self.visible_agent_ids


def test_list_agents_includes_owned_agents_and_policy_visible_agents() -> None:
    _admin_headers()
    bob_headers = _register_second_user()
    admin_id = _user_id("admin")
    bob_id = _user_id("bob")

    bob_draft_id = _create_agent_row(user_id=bob_id, name="Bob Draft")
    bob_published_id = _create_agent_row(
        user_id=bob_id,
        name="Bob Published",
        status=AgentStatus.PUBLISHED,
    )
    shared_published_id = _create_agent_row(
        user_id=admin_id,
        name="Shared Published",
        status=AgentStatus.PUBLISHED,
    )
    shared_draft_id = _create_agent_row(
        user_id=admin_id,
        name="Shared Draft",
        status=AgentStatus.DRAFT,
    )
    set_workforce_policy(_VisibleAgentPolicy({shared_published_id, shared_draft_id}))

    response = client.get("/api/agents", headers=bob_headers)
    assert response.status_code == 200, response.text
    items_by_id = {item["id"]: item for item in response.json()}

    assert {
        bob_draft_id,
        bob_published_id,
        shared_published_id,
        shared_draft_id,
    }.issubset(items_by_id)

    assert items_by_id[bob_draft_id]["access"] == "owner"
    assert items_by_id[bob_draft_id]["readonly"] is False
    assert items_by_id[bob_draft_id]["can_edit"] is True
    assert items_by_id[bob_draft_id]["can_publish"] is True
    assert items_by_id[bob_draft_id]["can_delete"] is True

    assert items_by_id[shared_published_id]["access"] == "policy"
    assert items_by_id[shared_published_id]["readonly"] is True
    assert items_by_id[shared_published_id]["can_edit"] is False
    assert items_by_id[shared_published_id]["can_publish"] is False
    assert items_by_id[shared_published_id]["can_delete"] is False
    assert items_by_id[shared_draft_id]["access"] == "policy"
    assert items_by_id[shared_draft_id]["status"] == "draft"
    assert items_by_id[shared_draft_id]["readonly"] is True


def test_agent_lists_keep_reusable_managers_and_hide_generated_managers() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    reusable_manager_id = _create_agent_row(
        user_id=owner_id,
        name="Reusable Manager",
        status=AgentStatus.PUBLISHED,
    )
    generated_manager_id = _create_agent_row(
        user_id=owner_id,
        name="Generated Manager",
        status=AgentStatus.PUBLISHED,
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )
    worker_id = _create_agent_row(
        user_id=owner_id,
        name="Reusable Worker",
        status=AgentStatus.PUBLISHED,
    )

    db = _direct_db_session()
    try:
        workforce = Workforce(
            owner_user_id=owner_id,
            scope_type="user",
            scope_id=str(owner_id),
            name="Reusable Manager Workforce",
            manager_agent_id=reusable_manager_id,
            status="draft",
        )
        db.add(workforce)
        db.commit()
    finally:
        db.close()

    response = client.get("/api/agents", headers=headers)
    assert response.status_code == 200, response.text
    agent_ids = {item["id"] for item in response.json()}
    assert reusable_manager_id in agent_ids
    assert generated_manager_id not in agent_ids
    assert worker_id in agent_ids

    options_response = client.get("/api/workforces/agent-options", headers=headers)
    assert options_response.status_code == 200, options_response.text
    option_ids = {item["id"] for item in options_response.json()}
    assert reusable_manager_id in option_ids
    assert generated_manager_id not in option_ids
    assert worker_id in option_ids


def test_agent_name_conflicts_ignore_generated_workforce_managers() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    generated_name = "Generated Manager Name"
    generated_manager_id = _create_agent_row(
        user_id=owner_id,
        name=generated_name,
        status=AgentStatus.PUBLISHED,
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )

    create_response = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": generated_name,
            "description": "Reusable agent",
            "instructions": "You are reusable.",
            "execution_mode": "balanced",
        },
    )

    assert create_response.status_code == 200, create_response.text
    created_agent_id = create_response.json()["id"]
    assert created_agent_id != generated_manager_id

    update_target_id = _create_agent_row(user_id=owner_id, name="Update Target")
    hidden_update_name = "Hidden Update Manager Name"
    _create_agent_row(
        user_id=owner_id,
        name=hidden_update_name,
        status=AgentStatus.PUBLISHED,
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )

    update_response = client.put(
        f"/api/agents/{update_target_id}",
        headers=headers,
        json={"name": hidden_update_name},
    )

    assert update_response.status_code == 200, update_response.text
    assert update_response.json()["name"] == hidden_update_name


def test_generated_workforce_manager_agents_cannot_authenticate_widget() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    generated_manager_id = _create_agent_row(
        user_id=owner_id,
        name="Generated Widget Manager",
        status=AgentStatus.PUBLISHED,
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        widget_enabled=True,
        allowed_domains=["*"],
    )

    response = client.post(
        "/api/widget/auth",
        json={"agent_id": generated_manager_id, "guest_id": "guest-1"},
        headers={"origin": "https://example.com"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Widget owner not found or invalid agent_id"


class TestDeleteAgent:
    """DELETE /api/agents/{agent_id} - remove an agent."""

    def test_with_tasks_keeps_tasks_and_nulls_agent_id(self):
        headers = _admin_headers()
        agent_id = _create_agent(headers)
        client.post(f"/api/agents/{agent_id}/api-key", headers=headers)

        db = _direct_db_session()
        try:
            admin_user = db.query(User).filter(User.username == "admin").first()
            assert admin_user is not None
            task = Task(
                user_id=admin_user.id,
                title="task tied to agent",
                description="task tied to agent",
                status=TaskStatus.PENDING,
                agent_id=agent_id,
            )
            db.add(task)
            db.commit()
            db.refresh(task)
            task_id = task.id
        finally:
            db.close()

        delete_resp = client.delete(f"/api/agents/{agent_id}", headers=headers)
        assert delete_resp.status_code == 200, delete_resp.text

        db = _direct_db_session()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            assert task is not None
            assert task.agent_id is None
            assert (
                db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).all()
                == []
            )
        finally:
            db.close()
