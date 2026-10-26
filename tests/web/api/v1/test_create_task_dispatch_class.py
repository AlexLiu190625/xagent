"""The ``dispatch_class`` field of ``POST /v1/chat/tasks`` and its effective value."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_event_bridge, task_execution_admission

from ..conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")

ACCEPT = "XAGENT_DISPATCH_ACCEPT_REQUESTED_CLASS"
FAIR = "XAGENT_DISPATCHER_FAIR_SHARE"


@pytest.fixture(autouse=True)
def _local_runner_stub():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg", new=MagicMock()
    ) as mocked:
        yield mocked


@pytest.fixture(autouse=True)
def _upstream_classifier():
    task_execution_admission.reset_command_classifier()
    yield
    task_execution_admission.reset_command_classifier()


@pytest.fixture
def shared_mode(monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())


def _agent_with_key() -> tuple[int, dict[str, str]]:
    headers = _admin_headers()
    agent = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "declared class agent",
            "description": "test",
            "instructions": "you are a test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent.status_code == 200, agent.text
    agent_id = agent.json()["id"]
    key = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key.status_code == 200, key.text
    return agent_id, {"Authorization": f"Bearer {key.json()['full_key']}"}


def _create(agent_id, headers, **extra):
    return client.post(
        "/v1/chat/tasks",
        headers=headers,
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "go"},
            **extra,
        },
    )


def _stored_declaration(task_id):
    db = _direct_db_session()
    try:
        return (
            db.query(Task.requested_dispatch_class).filter(Task.id == task_id).scalar()
        )
    finally:
        db.close()


def _command_classes(task_id):
    db = _direct_db_session()
    try:
        rows = (
            db.query(TaskExecutionCommand.kind, TaskExecutionCommand.dispatch_class)
            .filter(TaskExecutionCommand.task_id == task_id)
            .order_by(TaskExecutionCommand.id)
            .all()
        )
        return [(str(kind), klass) for kind, klass in rows]
    finally:
        db.close()


@pytest.mark.parametrize("value", ["Interactive", "", "urgent", 1, True])
def test_an_invalid_dispatch_class_is_rejected(value):
    agent_id, headers = _agent_with_key()
    response = _create(agent_id, headers, dispatch_class=value)
    assert response.status_code == 422, response.text
    db = _direct_db_session()
    try:
        assert db.query(Task).count() == 0
    finally:
        db.close()


@pytest.mark.parametrize("declared", [None, "interactive", "batch"])
def test_the_declaration_is_stored_as_sent_with_the_gates_closed(shared_mode, declared):
    agent_id, headers = _agent_with_key()
    extra = {} if declared is None else {"dispatch_class": declared}
    response = _create(agent_id, headers, **extra)
    assert response.status_code == 202, response.text
    assert _stored_declaration(response.json()["task_id"]) == declared


def test_an_explicit_null_is_the_same_as_omitting_the_field(shared_mode):
    agent_id, headers = _agent_with_key()
    response = _create(agent_id, headers, dispatch_class=None)
    assert response.status_code == 202, response.text
    assert _stored_declaration(response.json()["task_id"]) is None


@pytest.mark.parametrize(
    "declared,accept,fair,expected",
    [
        ("interactive", False, False, "batch"),
        ("interactive", True, False, "batch"),
        ("interactive", False, True, "batch"),
        ("interactive", True, True, "interactive"),
        ("batch", True, True, "batch"),
        (None, True, True, "batch"),
        (None, False, False, "batch"),
    ],
)
def test_the_response_reports_the_class_the_start_command_got(
    shared_mode, monkeypatch, declared, accept, fair, expected
):
    monkeypatch.setenv(ACCEPT, str(accept))
    monkeypatch.setenv(FAIR, str(fair))
    agent_id, headers = _agent_with_key()
    extra = {} if declared is None else {"dispatch_class": declared}
    response = _create(agent_id, headers, **extra)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["dispatch_class"] == expected
    assert _command_classes(body["task_id"]) == [("start", expected)]
    assert _stored_declaration(body["task_id"]) == declared


def test_without_a_dispatch_queue_the_response_class_is_null_and_the_declaration_kept():
    agent_id, headers = _agent_with_key()
    response = _create(agent_id, headers, dispatch_class="interactive")
    assert response.status_code == 202, response.text
    body = response.json()
    assert "dispatch_class" in body
    assert body["dispatch_class"] is None
    assert _stored_declaration(body["task_id"]) == "interactive"
    assert _command_classes(body["task_id"]) == []


def test_an_append_ignores_the_field_and_follows_the_tasks_declaration(
    shared_mode, monkeypatch
):
    monkeypatch.setenv(ACCEPT, "true")
    monkeypatch.setenv(FAIR, "true")
    agent_id, headers = _agent_with_key()
    created = _create(agent_id, headers, dispatch_class="interactive").json()
    task_id = created["task_id"]
    db = _direct_db_session()
    try:
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.task_id == task_id
        ).update({"status": "completed"})
        db.query(Task).filter(Task.id == task_id).update(
            {
                "status": TaskStatus.COMPLETED,
                "control_state": "completed",
                "run_id": created["run_id"],
            }
        )
        db.commit()
    finally:
        db.close()

    appended = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=headers,
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "more"},
            "dispatch_class": "batch",
        },
    )
    assert appended.status_code == 202, appended.text
    assert "dispatch_class" not in appended.json()
    assert _command_classes(task_id) == [
        ("start", "interactive"),
        ("start", "interactive"),
    ]
    assert _stored_declaration(task_id) == "interactive"
