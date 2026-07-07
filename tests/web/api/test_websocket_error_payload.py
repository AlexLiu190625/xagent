from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import _terminal_task_error_payload
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.task_lease_service import get_runner_id

from .conftest import _direct_db_session


def test_terminal_task_error_payload_marks_task_failed(_test_db):
    db = _direct_db_session()
    try:
        user = User(username="owner", password_hash="hash")
        db.add(user)
        db.commit()

        task = Task(
            user_id=user.id,
            title="Failing task",
            description="Failing task",
            status=TaskStatus.RUNNING,
            runner_id=get_runner_id(),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        db.add(task)
        db.commit()
        task_id = task.id

        payload = _terminal_task_error_payload(task_id, "Runtime error")

        assert payload["type"] == "agent_error"
        assert payload["message"] == "Runtime error"
        assert payload["task"]["id"] == task_id
        assert payload["task"]["status"] == "failed"

        db.expire_all()
        persisted_task = db.query(Task).filter(Task.id == task_id).one()
        assert persisted_task.status == TaskStatus.FAILED
        assert persisted_task.runner_id is None
        assert persisted_task.lease_expires_at is None
        assert persisted_task.error_message == "Runtime error"
    finally:
        db.close()


def test_terminal_task_error_payload_persists_error_chat_message(_test_db):
    """Failures before agent execution (no trace events, e.g. sandbox
    capacity rejection) must persist the error as an assistant message so
    a history reload shows the real error instead of a generic bubble."""
    from xagent.web.models.chat_message import TaskChatMessage

    db = _direct_db_session()
    try:
        user = User(username="owner", password_hash="hash")
        db.add(user)
        db.commit()

        task = Task(
            user_id=user.id,
            title="Rejected task",
            description="Rejected task",
            status=TaskStatus.RUNNING,
            runner_id=get_runner_id(),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        db.add(task)
        db.commit()
        task_id = task.id

        error_text = "Sandbox capacity limit reached (2 containers, cap 2)"
        _terminal_task_error_payload(task_id, error_text, event_type="task_error")

        db.expire_all()
        messages = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
            )
            .all()
        )
        assert len(messages) == 1
        assert error_text in messages[0].content
    finally:
        db.close()


@pytest.mark.asyncio
async def test_handle_chat_message_access_denied_does_not_fail_task(
    _test_db, monkeypatch
):
    db = _direct_db_session()
    try:
        owner = User(username="owner", password_hash="hash")
        intruder = User(username="intruder", password_hash="hash")
        db.add_all([owner, intruder])
        db.commit()

        task = Task(
            user_id=owner.id,
            title="Private task",
            description="Private task",
            status=TaskStatus.RUNNING,
            runner_id=get_runner_id(),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        db.add(task)
        db.commit()
        task_id = task.id
        intruder_id = intruder.id
    finally:
        db.close()

    sent_messages = []
    broadcasts = []

    async def fake_send_personal_message(message, websocket):
        sent_messages.append(message)

    async def fake_broadcast_to_task(message, broadcast_task_id):
        broadcasts.append((broadcast_task_id, message))

    monkeypatch.setattr(
        websocket_api.manager,
        "send_personal_message",
        fake_send_personal_message,
    )
    monkeypatch.setattr(
        websocket_api.manager,
        "broadcast_to_task",
        fake_broadcast_to_task,
    )

    await websocket_api.handle_chat_message(
        object(),
        task_id,
        {
            "message": "use this task",
            "user": SimpleNamespace(id=intruder_id, is_admin=False),
        },
    )

    assert broadcasts == []
    assert sent_messages
    assert sent_messages[0]["type"] == "error"
    assert "Access denied" in sent_messages[0]["message"]

    db = _direct_db_session()
    try:
        persisted_task = db.query(Task).filter(Task.id == task_id).one()
        assert persisted_task.status == TaskStatus.RUNNING
        assert persisted_task.runner_id == get_runner_id()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_handle_execute_task_unauthenticated_does_not_fail_task(
    _test_db, monkeypatch
):
    db = _direct_db_session()
    try:
        user = User(username="owner", password_hash="hash")
        db.add(user)
        db.commit()

        task = Task(
            user_id=user.id,
            title="Private task",
            description="Private task",
            status=TaskStatus.RUNNING,
            runner_id=get_runner_id(),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        db.add(task)
        db.commit()
        task_id = task.id
    finally:
        db.close()

    sent_messages = []
    broadcasts = []

    async def fake_send_personal_message(message, websocket):
        sent_messages.append(message)

    async def fake_broadcast_to_task(message, broadcast_task_id):
        broadcasts.append((broadcast_task_id, message))

    monkeypatch.setattr(
        websocket_api.manager,
        "send_personal_message",
        fake_send_personal_message,
    )
    monkeypatch.setattr(
        websocket_api.manager,
        "broadcast_to_task",
        fake_broadcast_to_task,
    )

    await websocket_api.handle_execute_task(object(), task_id, {})

    assert broadcasts == []
    assert sent_messages
    assert sent_messages[0]["type"] == "error"
    assert "authentication required" in sent_messages[0]["message"].lower()

    db = _direct_db_session()
    try:
        persisted_task = db.query(Task).filter(Task.id == task_id).one()
        assert persisted_task.status == TaskStatus.RUNNING
        assert persisted_task.runner_id == get_runner_id()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_execute_task_background_error_marks_task_failed(_test_db, monkeypatch):
    db = _direct_db_session()
    try:
        user = User(username="owner", password_hash="hash")
        db.add(user)
        db.commit()

        task = Task(
            user_id=user.id,
            title="Failing background task",
            description="Failing background task",
            status=TaskStatus.RUNNING,
            runner_id=get_runner_id(),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        db.add(task)
        db.commit()
        task_id = task.id
        user_id = user.id
    finally:
        db.close()

    broadcasts = []

    async def fake_broadcast_to_task(message, broadcast_task_id):
        broadcasts.append((broadcast_task_id, message))

    class FailingAgentManager:
        async def get_agent_for_task(self, *args, **kwargs):
            raise RuntimeError("setup failed")

    monkeypatch.setattr(
        websocket_api.manager,
        "broadcast_to_task",
        fake_broadcast_to_task,
    )

    await websocket_api.execute_task_background(
        task_id=task_id,
        user_message="run",
        context={},
        agent_manager=FailingAgentManager(),
        task_owner_user_id=user_id,
    )

    assert broadcasts
    broadcast_task_id, payload = broadcasts[-1]
    assert broadcast_task_id == task_id
    assert payload["type"] == "task_error"
    assert payload["task"]["status"] == "failed"
    assert payload["error"] == "setup failed"

    db = _direct_db_session()
    try:
        persisted_task = db.query(Task).filter(Task.id == task_id).one()
        assert persisted_task.status == TaskStatus.FAILED
        assert persisted_task.runner_id is None
        assert persisted_task.lease_expires_at is None
        assert persisted_task.error_message == "setup failed"
    finally:
        db.close()
