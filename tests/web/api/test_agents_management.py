"""Integration tests for agent management endpoints."""

import pytest

from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


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
