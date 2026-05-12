"""Integration tests for /v1/chat/tasks/* endpoints.

Phase 1 surface tested here:
  - D: POST /v1/chat/tasks  (this commit)
  - E: POST /v1/chat/tasks/{id}/messages, GET /v1/chat/tasks/{id}  (next)
  - F: GET /v1/chat/tasks/{id}/steps  (after E)

D tests mock the background-execution kickoff so the suite doesn't
need to spin up an actual AgentService / LLM. The behaviors under
test are HTTP shape + DB rows + which background helper was called
with which arguments -- not the LLM call itself.
"""

from typing import Tuple
from unittest.mock import AsyncMock, patch

import pytest

from xagent.web.models.task import Task, TaskStatus

from ..conftest import _admin_headers, _direct_db_session, client

# Opt this file into the shared conftest ``_test_db`` fixture; see the
# note in test_agent_api_keys.py for why we use ``usefixtures`` with a
# string name rather than importing the fixture.
pytestmark = pytest.mark.usefixtures("_test_db")


# ===== helpers =====


def _create_agent_with_key() -> Tuple[int, str]:
    """Create one agent under the admin user + generate its API key.

    Returns: (agent_id, full_key)
    """
    headers = _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "v1 tasks test agent",
            "description": "test",
            "instructions": "you are a test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]

    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    return agent_id, key_resp.json()["full_key"]


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


# ``start_task_in_background`` does real work (spawns an asyncio.Task that
# calls the AgentService coroutine). The D-level tests don't want to
# exercise the agent execution itself -- that's covered separately --
# so every test patches the helper to a no-op AsyncMock and asserts on
# the call args.
@pytest.fixture(autouse=True)
def mock_start_task():
    with patch(
        "xagent.web.api.v1.tasks.start_task_in_background",
        new=AsyncMock(),
    ) as mocked:
        yield mocked


# ===== POST /v1/chat/tasks =====


def test_create_task_happy_path(mock_start_task):
    """Returns 202 + task_id, writes Task with source='sdk' + input,
    persists first user message, kicks off background.
    """
    agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["agent_id"] == agent_id
    assert body["status"] == "pending"
    assert "task_id" in body
    assert "created_at" in body
    task_id = body["task_id"]

    # DB: Task row exists, owned by admin user, source='sdk', input set,
    # status PENDING
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        assert task is not None
        assert task.agent_id == agent_id
        assert task.source == "sdk"
        assert task.input == "first user message"
        assert task.status == TaskStatus.PENDING

        # task_chat_messages: one user-role message written
        from xagent.web.models.chat_message import TaskChatMessage

        msgs = (
            db.query(TaskChatMessage).filter(TaskChatMessage.task_id == task_id).all()
        )
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "first user message"
    finally:
        db.close()

    # Background kickoff was called exactly once for this task
    assert mock_start_task.await_count == 1
    kwargs = mock_start_task.await_args.kwargs
    assert kwargs["task"].id == task_id
    assert kwargs["user_message"] == "first user message"


def test_create_task_missing_authorization_returns_401(mock_start_task):
    """No Authorization header -> 401 invalid_api_key envelope."""
    agent_id, _key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "hi"},
        },
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "invalid_api_key"
    # No DB side effects
    assert mock_start_task.await_count == 0


def test_create_task_agent_id_mismatch_returns_404(mock_start_task):
    """body.agent_id != authed agent.id -> 404 agent_not_found."""
    _agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": 999999,  # not the bound agent
            "message": {"role": "user", "content": "hi"},
        },
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "agent_not_found"
    assert mock_start_task.await_count == 0


def test_create_task_empty_message_returns_422(mock_start_task):
    """Empty message.content fails Pydantic min_length=1 -> 422."""
    agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": ""},
        },
    )
    assert resp.status_code == 422
    assert mock_start_task.await_count == 0


def test_create_task_wrong_role_returns_422(mock_start_task):
    """role != 'user' fails Pydantic Literal check -> 422."""
    agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "assistant", "content": "hi"},
        },
    )
    assert resp.status_code == 422
    assert mock_start_task.await_count == 0


def test_create_task_revoked_key_returns_401(mock_start_task):
    """Revoked key can't create tasks -> 401 invalid_api_key."""
    agent_id, full_key = _create_agent_with_key()
    # Revoke the key via the admin endpoint
    admin = _admin_headers()
    revoke = client.delete(f"/api/agents/{agent_id}/api-key", headers=admin)
    assert revoke.status_code == 200
    assert revoke.json()["revoked"] is True

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "hi"},
        },
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"
    assert mock_start_task.await_count == 0


def test_create_task_cross_user_agent_returns_404(mock_start_task):
    """Bob's key cannot target Alice's agent_id -> 404 agent_not_found.

    Defense: the key is bound to agent X. Putting agent_id=Y in the
    body where Y != X always returns 404 regardless of whether Y
    exists, owned by a different user, etc.
    """
    # Admin (alice) creates agent A and a key for it.
    alice_agent_id, _alice_key = _create_agent_with_key()

    # Register bob and create agent B + key, then have bob attempt to
    # POST against alice's agent_id using bob's own key.
    from ..conftest import _register_second_user

    bob_headers = _register_second_user()
    bob_agent = client.post(
        "/api/agents",
        headers=bob_headers,
        json={
            "name": "bob agent",
            "description": "test",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    ).json()
    bob_agent_id = bob_agent["id"]
    bob_key = client.post(
        f"/api/agents/{bob_agent_id}/api-key", headers=bob_headers
    ).json()["full_key"]

    # Bob's key + Alice's agent_id in body -> 404
    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(bob_key),
        json={
            "agent_id": alice_agent_id,
            "message": {"role": "user", "content": "hi"},
        },
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent_not_found"
    assert mock_start_task.await_count == 0
