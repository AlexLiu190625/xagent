"""Integration tests for /v1/chat/tasks/* endpoints.

Phase 1 surface tested here:
  - D: POST /v1/chat/tasks
  - E: POST /v1/chat/tasks/{id}/messages, GET /v1/chat/tasks/{id}
  - F: GET /v1/chat/tasks/{id}/steps  (this commit)

D / E / F tests mock the background-execution kickoff so the suite
doesn't need to spin up an actual AgentService / LLM. The behaviors
under test are HTTP shape + DB rows + which background helper was
called with which arguments -- not the LLM call itself. F's steps
endpoint exercises real :class:`TraceEvent` rows inserted directly
into the test DB to drive the mapping.
"""

from datetime import datetime, timezone
from typing import Tuple
from unittest.mock import AsyncMock, patch

import pytest

from xagent.web.models.task import Task, TaskStatus, TraceEvent

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


# ===== Shared helper for E tests: create a task via POST then return its id =====


def _create_task(full_key: str, agent_id: int, content: str = "hello") -> int:
    """Drive POST /v1/chat/tasks and return the resulting task_id."""
    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": content}},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


# ===== POST /v1/chat/tasks/{task_id}/messages =====


def test_append_message_happy_path(mock_start_task):
    """Returns 202 + accepted_at, persists new user message, kicks off bg."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="first turn")
    mock_start_task.reset_mock()  # discard the create-task call so we count just the append

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "second turn"},
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    assert body["status"] == "pending"
    assert "accepted_at" in body

    # Two user messages now exist for this task; task.input is the latest
    from xagent.web.models.chat_message import TaskChatMessage

    db = _direct_db_session()
    try:
        msgs = (
            db.query(TaskChatMessage)
            .filter(TaskChatMessage.task_id == task_id)
            .order_by(TaskChatMessage.id)
            .all()
        )
        assert len(msgs) == 2
        assert [m.content for m in msgs] == ["first turn", "second turn"]
        task = db.query(Task).filter(Task.id == task_id).first()
        assert task is not None
        assert task.input == "second turn"
    finally:
        db.close()

    assert mock_start_task.await_count == 1


def test_append_message_to_running_task_returns_409(mock_start_task):
    """Appending to a RUNNING task is rejected as task_busy."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    mock_start_task.reset_mock()

    # Flip status to RUNNING directly so we don't have to actually run
    # the agent service in tests.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.RUNNING
        db.commit()
    finally:
        db.close()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "hello"}},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "task_busy"
    # No new background kickoff happened
    assert mock_start_task.await_count == 0


def test_append_message_to_missing_task_returns_404(mock_start_task):
    """Appending to a task that doesn't exist -> 404 task_not_found."""
    agent_id, full_key = _create_agent_with_key()
    resp = client.post(
        "/v1/chat/tasks/9999999/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "hi"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"
    assert mock_start_task.await_count == 0


def test_append_message_to_other_agents_task_returns_404(mock_start_task):
    """Bob can't append to Alice's task even if he knows the id."""
    alice_agent_id, alice_key = _create_agent_with_key()
    alice_task_id = _create_task(alice_key, alice_agent_id)
    mock_start_task.reset_mock()

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

    resp = client.post(
        f"/v1/chat/tasks/{alice_task_id}/messages",
        headers=_bearer(bob_key),
        json={"agent_id": bob_agent_id, "message": {"role": "user", "content": "hi"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"
    assert mock_start_task.await_count == 0


def test_append_message_body_agent_id_mismatch_returns_404(mock_start_task):
    """body.agent_id != authed agent.id -> 404 agent_not_found.

    Distinct from cross-agent task ownership (which is task_not_found) --
    here the task IS the caller's, but the body claims a different
    agent_id; consistent with POST /v1/chat/tasks behavior.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    mock_start_task.reset_mock()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": 999999, "message": {"role": "user", "content": "hi"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent_not_found"
    assert mock_start_task.await_count == 0


# ===== GET /v1/chat/tasks/{task_id} =====


def test_get_task_pending(mock_start_task):
    """Fresh task: status='pending', input set, output/error null, completed_at null."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="seed input")

    resp = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    assert body["status"] == "pending"
    assert body["input"] == "seed input"
    assert body["output"] is None
    assert body["error"] is None
    assert "created_at" in body
    assert body["completed_at"] is None


def test_get_task_completed_returns_output(mock_start_task):
    """Completed task: output populated, completed_at set."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    # Flip status + write output to simulate completed background turn
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.COMPLETED
        task.output = "final answer"
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["output"] == "final answer"
    assert body["completed_at"] is not None


def test_get_task_failed_returns_error(mock_start_task):
    """Failed task: error populated, completed_at set."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.FAILED
        task.error_message = "agent crashed"
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error"] == "agent crashed"
    assert body["output"] is None
    assert body["completed_at"] is not None


def test_get_missing_task_returns_404(mock_start_task):
    _agent_id, full_key = _create_agent_with_key()
    resp = client.get("/v1/chat/tasks/9999999", headers=_bearer(full_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


def test_get_other_agents_task_returns_404(mock_start_task):
    """Cross-agent task access -> 404 (not leaking existence)."""
    alice_agent_id, alice_key = _create_agent_with_key()
    alice_task_id = _create_task(alice_key, alice_agent_id)

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

    resp = client.get(f"/v1/chat/tasks/{alice_task_id}", headers=_bearer(bob_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


# ===== F: GET /v1/chat/tasks/{task_id}/steps =====


def _insert_trace_event(
    *,
    task_id: int,
    event_type: str,
    event_id: str,
    timestamp: datetime,
    data: dict,
    step_id: str | None = None,
) -> None:
    """Insert one TraceEvent row directly via the test DB.

    Bypasses the production trace handler (which runs through asyncio
    + thread pool) so tests can assert on the GET /steps surface
    without spinning up the agent runtime.
    """
    db = _direct_db_session()
    try:
        ev = TraceEvent(
            task_id=task_id,
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            step_id=step_id,
            data=data,
        )
        db.add(ev)
        db.commit()
    finally:
        db.close()


def test_get_steps_returns_mapped_steps_in_order(mock_start_task):
    """Insert react_action + tool_execution + ai_message + filtered
    llm_call events, GET /steps, assert 3 public steps in started_at
    order with correct types and statuses.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Public step 1: thinking phase=action (react_action_start/end)
    _insert_trace_event(
        task_id=task_id,
        event_type="react_action_start",
        event_id="evt-1",
        timestamp=base.replace(second=1),
        step_id="step-A",
        data={},
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="react_action_end",
        event_id="evt-2",
        timestamp=base.replace(second=2),
        step_id="step-A",
        data={},
    )

    # Filtered: llm_call_start / end -- must not appear
    _insert_trace_event(
        task_id=task_id,
        event_type="llm_call_start",
        event_id="evt-3",
        timestamp=base.replace(second=3),
        step_id="step-A",
        data={},
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="llm_call_end",
        event_id="evt-4",
        timestamp=base.replace(second=4),
        step_id="step-A",
        data={},
    )

    # Public step 2: tool_call (execute_python)
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="evt-5",
        timestamp=base.replace(second=5),
        step_id="step-A",
        data={
            "tool_name": "execute_python",
            "tool_args": {"code": "print(1)"},
            "tool_execution_id": "tx-1",
        },
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_end",
        event_id="evt-6",
        timestamp=base.replace(second=6),
        step_id="step-A",
        data={
            "tool_name": "execute_python",
            "tool_args": {"code": "print(1)"},
            "tool_execution_id": "tx-1",
            "result": {"output": "1\n"},
            "success": True,
        },
    )

    # Public step 3: message role=assistant (ai_message)
    _insert_trace_event(
        task_id=task_id,
        event_type="ai_message",
        event_id="evt-7",
        timestamp=base.replace(second=7),
        data={"content": "Here's the result"},
    )

    resp = client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    steps = body["steps"]
    assert len(steps) == 3

    # Assert ordering and types
    assert steps[0]["type"] == "thinking"
    assert steps[0]["data"]["phase"] == "action"
    assert steps[0]["status"] == "completed"

    assert steps[1]["type"] == "tool_call"
    assert steps[1]["data"]["name"] == "execute_python"
    assert steps[1]["data"]["args"] == {"code": "print(1)"}
    assert steps[1]["data"]["result"] == {"output": "1\n"}

    assert steps[2]["type"] == "message"
    assert steps[2]["data"] == {"role": "assistant", "content": "Here's the result"}


def test_get_steps_task_not_found_returns_404(mock_start_task):
    """Non-existent task_id -> 404 task_not_found."""
    _agent_id, full_key = _create_agent_with_key()
    resp = client.get("/v1/chat/tasks/9999999/steps", headers=_bearer(full_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


def test_get_steps_other_agents_task_returns_404(mock_start_task):
    """Cross-agent steps access -> 404 (not leaking existence)."""
    alice_agent_id, alice_key = _create_agent_with_key()
    alice_task_id = _create_task(alice_key, alice_agent_id)

    from ..conftest import _register_second_user

    bob_headers = _register_second_user()
    bob_agent = client.post(
        "/api/agents",
        headers=bob_headers,
        json={
            "name": "bob agent steps",
            "description": "test",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    ).json()
    bob_agent_id = bob_agent["id"]
    bob_key = client.post(
        f"/api/agents/{bob_agent_id}/api-key", headers=bob_headers
    ).json()["full_key"]

    resp = client.get(f"/v1/chat/tasks/{alice_task_id}/steps", headers=_bearer(bob_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


def test_get_steps_empty_task_returns_empty_array(mock_start_task):
    """Task with no trace events yet -> 200 + empty steps array."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    resp = client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    assert body["steps"] == []
