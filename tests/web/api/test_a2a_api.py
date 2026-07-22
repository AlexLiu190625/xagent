import asyncio
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from sqlalchemy import create_engine
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.web.api import a2a as a2a_api
from xagent.web.models import database as database_module
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services.a2a_protocol import A2A_MAX_MESSAGE_TEXT_LENGTH
from xagent.web.services.api_keys import AgentApiIdentity
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    MAX_COMMAND_DEFERS,
    MAX_COMMAND_FAILURES,
)
from xagent.web.services.task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
)
from xagent.web.services.task_orchestrator import TaskTurnError

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


def _bearer(full_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {full_key}",
        "A2A-Version": "1.0",
    }


def _request_with_json(path: str, payload: dict[str, object]) -> Request:
    body = json.dumps(payload).encode()
    delivered = False

    async def receive() -> dict[str, object]:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("testserver", 80),
        },
        receive,
    )


def _detached_identity(agent: Agent, key: AgentApiKey) -> AgentApiIdentity:
    raw_categories = agent.tool_categories
    return AgentApiIdentity(
        agent_id=int(agent.id),
        user_id=int(agent.user_id),
        execution_mode=(
            str(agent.execution_mode) if agent.execution_mode is not None else None
        ),
        tool_categories=(
            tuple(raw_categories) if isinstance(raw_categories, list) else None
        ),
        status=str(getattr(agent.status, "value", agent.status)),
        origin=str(agent.origin),
        key_prefix=str(key.key_prefix),
    )


@pytest.fixture()
def single_connection_a2a_db(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Real one-slot pool shared by an A2A request and runtime workers."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'a2a-turn-boundary.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.15,
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(database_module, "_SessionLocal", session_local)

    db = session_local()
    user = User(
        username="a2a-turn-boundary-owner",
        password_hash="hash",
        is_admin=False,
    )
    db.add(user)
    db.flush()
    agent = Agent(
        user_id=int(user.id),
        name="A2A turn boundary agent",
        description="test",
        instructions="test",
        execution_mode="balanced",
        models={},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        status=AgentStatus.PUBLISHED,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    key = SimpleNamespace(key_prefix="a2atest")
    try:
        yield engine, db, agent, key
    finally:
        db.close()
        engine.dispose()


def _create_agent(headers: dict[str, str], name: str = "A2A Test Agent") -> int:
    response = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "A2A test agent",
            "instructions": "You are an A2A test agent.",
            "execution_mode": "balanced",
            "suggested_prompts": ["Summarize this"],
        },
    )
    assert response.status_code == 200, response.text
    return int(response.json()["id"])


def _publish_agent(headers: dict[str, str], agent_id: int) -> None:
    response = client.post(f"/api/agents/{agent_id}/publish", headers=headers)
    assert response.status_code == 200, response.text


def _create_published_agent_with_key() -> tuple[int, str]:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    _publish_agent(headers, agent_id)
    key_response = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_response.status_code == 200, key_response.text
    return agent_id, key_response.json()["full_key"]


def test_agent_card_exposes_published_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAGENT_PUBLIC_API_BASE_URL", raising=False)
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    _publish_agent(headers, agent_id)

    response = client.get(f"/api/a2a/agents/{agent_id}/.well-known/agent-card.json")

    assert response.status_code == 200, response.text
    assert response.headers["a2a-version"] == "1.0"
    assert response.headers["content-type"].startswith("application/a2a+json")
    body = response.json()
    assert body["name"] == "A2A Test Agent"
    assert body["defaultInputModes"] == ["text/plain", "application/json"]
    assert body["defaultOutputModes"] == ["text/plain"]
    assert body["supportedInterfaces"] == [
        {
            "url": f"http://testserver/api/a2a/agents/{agent_id}",
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }
    ]
    assert body["securitySchemes"]["xagentAgentApiKey"] == {
        "httpAuthSecurityScheme": {
            "scheme": "Bearer",
            "description": "Xagent agent API key",
        }
    }
    assert body["securityRequirements"] == [{"schemes": {"xagentAgentApiKey": {}}}]
    assert body["skills"][0]["examples"] == ["Summarize this"]


def test_agent_card_uses_public_api_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", " https://sg.cloud.xagent.co/ ")
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    _publish_agent(headers, agent_id)

    response = client.get(f"/api/a2a/agents/{agent_id}/.well-known/agent-card.json")

    assert response.status_code == 200, response.text
    assert response.json()["supportedInterfaces"][0]["url"] == (
        f"https://sg.cloud.xagent.co/api/a2a/agents/{agent_id}"
    )


def test_agent_card_hides_draft_agent() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    response = client.get(f"/api/a2a/agents/{agent_id}/.well-known/agent-card.json")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == 404
    assert error["status"] == "NOT_FOUND"
    assert error["details"][0]["reason"] == "AGENT_NOT_FOUND"


def test_agent_card_does_not_expose_private_instructions() -> None:
    headers = _admin_headers()
    response = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "Private Prompt Agent",
            "instructions": "secret system prompt",
            "execution_mode": "balanced",
        },
    )
    assert response.status_code == 200, response.text
    agent_id = int(response.json()["id"])
    _publish_agent(headers, agent_id)

    card = client.get(f"/api/a2a/agents/{agent_id}/.well-known/agent-card.json").json()

    assert card["description"] == "Private Prompt Agent"
    assert "secret system prompt" not in json.dumps(card)


def test_message_send_creates_hidden_a2a_task() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as schedule_bg:
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-1",
                    "contextId": "ctx-1",
                    "role": "ROLE_USER",
                    "parts": [{"text": "hello from a2a"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    task = body["task"]
    assert task["contextId"] == "ctx-1"
    assert task["status"]["state"] == "TASK_STATE_WORKING"

    db = _direct_db_session()
    try:
        row = db.query(Task).filter(Task.id == int(task["id"])).one()
        assert row.agent_id == agent_id
        assert row.source == "a2a"
        assert row.is_visible is False
        assert row.input == "hello from a2a"
        assert row.agent_config == {"a2a_context_id": "ctx-1"}
        assert row.status == TaskStatus.RUNNING

        key = db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).one()
        assert key.usage_month == datetime.now(UTC).strftime("%Y-%m")
        assert key.usage_month_calls == 1
    finally:
        db.close()

    assert schedule_bg.call_count == 1
    assert schedule_bg.call_args.kwargs["task_id"] == int(task["id"])


@pytest.mark.asyncio
async def test_message_send_releases_one_slot_request_pool_before_turn_start(
    single_connection_a2a_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A2A startup must not pin the request checkout needed by the worker."""

    engine, db, agent, key = single_connection_a2a_db
    from xagent.web.services import task_orchestrator as orchestrator_module

    observed_checked_out: list[int] = []
    original_begin = a2a_api.TaskTurnOrchestrator.create_and_begin_turn
    original_atomic = orchestrator_module._create_turn_atomic_sync

    async def observed_begin(**kwargs: object) -> object:
        observed_checked_out.append(engine.pool.checkedout())
        return await original_begin(**kwargs)  # type: ignore[arg-type]

    def slow_atomic(*args: object, **kwargs: object) -> object:
        time.sleep(0.05)
        return original_atomic(*args, **kwargs)  # type: ignore[arg-type]

    ticks = 0
    stop_ticker = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop_ticker.is_set():
            ticks += 1
            await asyncio.sleep(0.005)

    monkeypatch.setattr(
        a2a_api.TaskTurnOrchestrator,
        "create_and_begin_turn",
        observed_begin,
    )
    monkeypatch.setattr(orchestrator_module, "_create_turn_atomic_sync", slow_atomic)
    monkeypatch.setattr(a2a_api, "record_key_usage", AsyncMock())
    identity = _detached_identity(agent, key)
    db.close()

    path = f"/api/a2a/agents/{int(agent.id)}/message:send"
    request = _request_with_json(
        path,
        {
            "message": {
                "messageId": "msg-one-slot",
                "role": "ROLE_USER",
                "parts": [{"text": "keep the request pool free"}],
            },
            "configuration": {"returnImmediately": True},
        },
    )

    ticker_task = asyncio.create_task(ticker())
    try:
        with patch(
            "xagent.web.services.task_orchestrator._schedule_bg",
            new=MagicMock(),
        ):
            response = await a2a_api.send_message(
                int(agent.id),
                request,
                identity,
            )
    finally:
        stop_ticker.set()
        await ticker_task

    assert response.status_code == 200
    assert observed_checked_out == [0]
    assert ticks >= 3, "A2A turn startup blocked the asyncio event loop"


@pytest.mark.asyncio
async def test_create_cancellation_after_commit_drains_turn_start(
    single_connection_a2a_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation drains scheduling after the atomic create commit."""

    _engine, db, agent, _key = single_connection_a2a_db
    from xagent.web.services import task_orchestrator as orchestrator_module

    agent_id = int(agent.id)
    owner_id = int(agent.user_id)
    db.close()

    committed = threading.Event()
    allow_worker_return = threading.Event()
    original_create = orchestrator_module._create_turn_atomic_sync

    def create_then_block(*args: object, **kwargs: object):
        created = original_create(*args, **kwargs)  # type: ignore[arg-type]
        committed.set()
        assert allow_worker_return.wait(timeout=1.0)
        return created

    monkeypatch.setattr(
        orchestrator_module,
        "_create_turn_atomic_sync",
        create_then_block,
    )

    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as schedule_bg:
        request_task = asyncio.create_task(
            a2a_api._start_a2a_turn(
                agent_id=agent_id,
                task_owner_user_id=owner_id,
                execution_mode="balanced",
                text="cancel after commit",
                message_id="msg-cancel-after-commit",
                context_id="ctx-cancel-after-commit",
                task_id=None,
            )
        )
        try:
            assert await asyncio.to_thread(committed.wait, 1.0)
            with database_module.get_session_local()() as verify_db:
                started = verify_db.query(Task).filter(Task.source == "a2a").one()
                assert started.status == TaskStatus.RUNNING

            request_task.cancel()
            await asyncio.sleep(0)
            assert not request_task.done()
        finally:
            allow_worker_return.set()

        with pytest.raises(asyncio.CancelledError):
            await request_task

    with database_module.get_session_local()() as verify_db:
        pending = (
            verify_db.query(Task)
            .filter(Task.source == "a2a", Task.status == TaskStatus.PENDING)
            .count()
        )
        assert pending == 0
        started = verify_db.query(Task).filter(Task.source == "a2a").one()
        assert started.status == TaskStatus.RUNNING

    schedule_bg.assert_called_once()


@pytest.mark.asyncio
async def test_wait_poll_checkout_does_not_block_the_event_loop(
    single_connection_a2a_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contended A2A status poll belongs on a worker, not the main loop."""

    engine, db, agent, _key = single_connection_a2a_db
    agent_id = int(agent.id)
    owner_id = int(agent.user_id)
    db.close()
    held_connection = engine.connect()
    task = Task(
        id=999,
        user_id=owner_id,
        title="poll boundary",
        status=TaskStatus.RUNNING,
        agent_id=agent_id,
        source="a2a",
        agent_config={"a2a_context_id": "ctx-poll-boundary"},
    )
    monkeypatch.setattr(a2a_api, "A2A_BLOCKING_WAIT_TIMEOUT_SECONDS", 1.0)

    tick_times: list[float] = []
    stop_ticker = asyncio.Event()

    async def ticker() -> None:
        while not stop_ticker.is_set():
            tick_times.append(time.monotonic())
            await asyncio.sleep(0.005)

    ticker_task = asyncio.create_task(ticker())
    try:
        with pytest.raises(Exception, match="QueuePool limit"):
            await a2a_api._wait_for_task(agent_id, a2a_api._task_snapshot(task))
        # Let the ticker record the gap across the failed checkout.
        await asyncio.sleep(0.02)
    finally:
        stop_ticker.set()
        await ticker_task
        held_connection.close()

    gaps = [right - left for left, right in zip(tick_times, tick_times[1:])]
    assert len(tick_times) >= 10
    assert max(gaps, default=0.0) < 0.08, (
        "A2A polling performed the QueuePool wait on the asyncio event loop"
    )


def test_message_send_rejects_key_bound_to_different_agent() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    headers = _admin_headers()
    other_agent_id = _create_agent(headers, name="Other A2A Agent")
    _publish_agent(headers, other_agent_id)

    response = client.post(
        f"/api/a2a/agents/{other_agent_id}/message:send",
        headers=_bearer(full_key),
        json={
            "message": {
                "messageId": "msg-1",
                "role": "ROLE_USER",
                "parts": [{"text": "wrong target"}],
            },
            "configuration": {"returnImmediately": True},
        },
    )

    assert other_agent_id != agent_id
    assert response.status_code == 404
    assert response.json()["error"]["details"][0]["reason"] == "AGENT_NOT_FOUND"


def test_message_send_rejects_draft_agent_key() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    key_response = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_response.status_code == 200, key_response.text

    response = client.post(
        f"/api/a2a/agents/{agent_id}/message:send",
        headers=_bearer(key_response.json()["full_key"]),
        json={
            "message": {
                "messageId": "msg-1",
                "role": "ROLE_USER",
                "parts": [{"text": "draft should not run"}],
            },
            "configuration": {"returnImmediately": True},
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["details"][0]["reason"] == "AGENT_NOT_FOUND"


def test_message_send_requires_supported_a2a_version() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    payload = {
        "message": {
            "messageId": "msg-version",
            "role": "ROLE_USER",
            "parts": [{"text": "hello"}],
        },
        "configuration": {"returnImmediately": True},
    }

    response = client.post(
        f"/api/a2a/agents/{agent_id}/message:send",
        headers={"Authorization": f"Bearer {full_key}"},
        json=payload,
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["message"] == "A2A-Version header or query parameter is required."
    assert error["details"][0]["reason"] == "VERSION_NOT_SUPPORTED"
    assert error["details"][0]["metadata"]["supportedVersions"] == "1.0"

    incompatible = client.post(
        f"/api/a2a/agents/{agent_id}/message:send",
        headers={
            "Authorization": f"Bearer {full_key}",
            "A2A-Version": "2.0",
        },
        json=payload,
    )

    assert incompatible.status_code == 400
    incompatible_error = incompatible.json()["error"]
    assert incompatible_error["details"][0]["reason"] == "VERSION_NOT_SUPPORTED"
    assert incompatible_error["details"][0]["metadata"]["supportedVersions"] == ("1.0")


def test_message_send_rejects_oversized_content() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    response = client.post(
        f"/api/a2a/agents/{agent_id}/message:send",
        headers=_bearer(full_key),
        json={
            "message": {
                "messageId": "msg-too-large",
                "role": "ROLE_USER",
                "parts": [{"text": "x" * (A2A_MAX_MESSAGE_TEXT_LENGTH + 1)}],
            },
            "configuration": {"returnImmediately": True},
        },
    )

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["status"] == "RESOURCE_EXHAUSTED"
    assert error["details"][0]["metadata"]["maxLength"] == str(
        A2A_MAX_MESSAGE_TEXT_LENGTH
    )


def test_message_send_requires_message_id() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    response = client.post(
        f"/api/a2a/agents/{agent_id}/message:send",
        headers=_bearer(full_key),
        json={
            "message": {
                "role": "ROLE_USER",
                "parts": [{"text": "missing id"}],
            },
            "configuration": {"returnImmediately": True},
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["details"][0]["reason"] == "INVALID_ARGUMENT"
    assert error["details"][0]["metadata"]["field"] == "message.messageId"


def test_a2a_fastapi_validation_uses_protocol_error_shape() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    response = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"pageSize": 0},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["status"] == "INVALID_ARGUMENT"
    assert error["details"][0]["reason"] == "INVALID_ARGUMENT"


def test_a2a_auth_error_includes_bearer_challenge() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    _publish_agent(headers, agent_id)

    response = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers={"A2A-Version": "1.0"},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["status"] == "UNAUTHENTICATED"


def test_message_send_blocks_by_default_until_task_finishes() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    original_create = a2a_api.TaskTurnOrchestrator.create_and_begin_turn

    async def _complete_turn(**kwargs: object) -> object:
        started = await original_create(**kwargs)  # type: ignore[arg-type]
        db = _direct_db_session()
        try:
            row = db.query(Task).filter(Task.id == started.task_id).one()
            row.status = TaskStatus.COMPLETED
            row.output = "blocking response"
            db.commit()
        finally:
            db.close()
        return started

    with (
        patch(
            "xagent.web.api.a2a.TaskTurnOrchestrator.create_and_begin_turn",
            new=_complete_turn,
        ),
        patch(
            "xagent.web.services.task_orchestrator._schedule_bg",
            new=MagicMock(),
        ),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-blocking",
                    "role": "ROLE_USER",
                    "parts": [{"text": "wait for me"}],
                }
            },
        )

    assert response.status_code == 200, response.text
    task = response.json()["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"] == [{"text": "blocking response"}]


def test_message_send_returns_working_task_when_wait_deadline_expires(
    monkeypatch,
) -> None:
    agent_id, full_key = _create_published_agent_with_key()
    monkeypatch.setattr(a2a_api, "A2A_BLOCKING_WAIT_TIMEOUT_SECONDS", 0.0)

    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-wait-timeout",
                    "role": "ROLE_USER",
                    "parts": [{"text": "keep working"}],
                }
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["task"]["status"]["state"] == "TASK_STATE_WORKING"


def test_failed_create_does_not_leave_pending_a2a_task() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    with patch(
        "xagent.web.api.a2a.TaskTurnOrchestrator.create_and_begin_turn",
        side_effect=TaskTurnError("busy"),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-failed-create",
                    "role": "ROLE_USER",
                    "parts": [{"text": "fail before scheduling"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 400, response.text
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.source == "a2a").count() == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_pool_timeout_during_atomic_turn_create_leaves_no_a2a_task(
    single_connection_a2a_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed atomic create leaves no task and keeps the event loop live."""

    engine, db, agent, key = single_connection_a2a_db
    recovery = MagicMock()
    monkeypatch.setattr(a2a_api, "_recover_failed_turn_start", recovery)
    monkeypatch.setattr(a2a_api, "record_key_usage", AsyncMock())
    identity = _detached_identity(agent, key)
    db.close()
    held_connection = engine.connect()

    path = f"/api/a2a/agents/{int(agent.id)}/message:send"
    request = _request_with_json(
        path,
        {
            "message": {
                "messageId": "msg-pool-timeout",
                "role": "ROLE_USER",
                "parts": [{"text": "do not retry the exhausted pool"}],
            },
            "configuration": {"returnImmediately": True},
        },
    )

    ticks = 0
    stop_ticker = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop_ticker.is_set():
            ticks += 1
            await asyncio.sleep(0.005)

    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as schedule_bg:
        ticker_task = asyncio.create_task(ticker())
        try:
            with pytest.raises(SQLAlchemyTimeoutError):
                await a2a_api.send_message(
                    int(agent.id),
                    request,
                    identity,
                )
        finally:
            held_connection.close()
            stop_ticker.set()
            await ticker_task

    with database_module.get_session_local()() as verify_db:
        pending = (
            verify_db.query(Task)
            .filter(Task.source == "a2a", Task.status == TaskStatus.PENDING)
            .count()
        )
        assert pending == 0

    assert ticks >= 5, "pool checkout blocked the asyncio event loop"
    recovery.assert_not_called()
    schedule_bg.assert_not_called()


def test_unexpected_a2a_error_uses_internal_protocol_envelope() -> None:
    agent_id, full_key = _create_published_agent_with_key()

    with patch(
        "xagent.web.api.a2a.TaskTurnOrchestrator.create_and_begin_turn",
        side_effect=RuntimeError("sensitive implementation detail"),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-error",
                    "role": "ROLE_USER",
                    "parts": [{"text": "fail"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["status"] == "INTERNAL"
    assert error["details"][0]["reason"] == "INTERNAL"
    assert "sensitive implementation detail" not in response.text
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.source == "a2a").count() == 0
    finally:
        db.close()


def test_get_and_list_tasks_use_rest_binding_shapes() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        created = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-list",
                    "contextId": "ctx-list",
                    "role": "ROLE_USER",
                    "parts": [{"text": "list me"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
    task_id = created.json()["task"]["id"]

    db = _direct_db_session()
    try:
        row = db.query(Task).filter(Task.id == int(task_id)).one()
        row.status = TaskStatus.COMPLETED
        row.output = "listed output"
        db.commit()
    finally:
        db.close()

    fetched = client.get(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}",
        headers=_bearer(full_key),
    )
    listed = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"contextId": "ctx-list"},
    )
    listed_with_artifacts = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"includeArtifacts": "true"},
    )

    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["id"] == task_id
    assert "task" not in fetched.json()
    assert fetched.json()["artifacts"][0]["parts"] == [{"text": "listed output"}]
    assert listed.json()["totalSize"] == 1
    assert "artifacts" not in listed.json()["tasks"][0]
    assert listed_with_artifacts.json()["tasks"][0]["artifacts"][0]["parts"] == [
        {"text": "listed output"}
    ]


def test_follow_up_infers_context_for_input_required_task() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        created = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-initial",
                    "contextId": "ctx-follow-up",
                    "role": "ROLE_USER",
                    "parts": [{"text": "initial"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
    task_id = created.json()["task"]["id"]
    db = _direct_db_session()
    try:
        row = db.query(Task).filter(Task.id == int(task_id)).one()
        row.status = TaskStatus.WAITING_FOR_USER
        db.commit()
    finally:
        db.close()

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=True)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    begin_turn = AsyncMock()
    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=agent_manager,
        ),
        patch("xagent.web.api.a2a._schedule_waiting_a2a_resume") as schedule_resume,
        patch(
            "xagent.web.api.a2a.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-follow-up",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "follow up"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["task"]["contextId"] == "ctx-follow-up"
    assert response.json()["task"]["status"]["state"] == "TASK_STATE_WORKING"
    agent_service.post_user_message.assert_awaited_once_with(
        str(task_id),
        execution_message="follow up",
        display_message="follow up",
        turn_id=f"a2a:{task_id}:msg-follow-up",
        request_interrupt=False,
        reason="A2A input-required response",
    )
    begin_turn.assert_not_awaited()
    schedule_resume.assert_called_once()
    db = _direct_db_session()
    try:
        resumed = db.query(Task).filter(Task.id == int(task_id)).one()
        assert resumed.status == TaskStatus.RUNNING
        assert resumed.input == "follow up"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_waiting_resume_uses_one_offloop_scope_and_no_request_session(
    single_connection_a2a_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint resume passes one frozen setup/scope into the live runtime."""

    _engine, db, agent, key = single_connection_a2a_db
    agent_id = int(agent.id)
    owner_id = int(agent.user_id)
    waiting = Task(
        user_id=owner_id,
        title="waiting boundary",
        status=TaskStatus.WAITING_FOR_USER,
        agent_id=agent_id,
        source="a2a",
        is_visible=False,
        agent_config={"a2a_context_id": "ctx-waiting-boundary"},
    )
    db.add(waiting)
    db.commit()
    db.refresh(waiting)
    task_id = int(waiting.id)
    identity = _detached_identity(agent, key)
    db.close()

    main_thread = threading.get_ident()
    scope = object()
    setup_snapshot = SimpleNamespace(runtime_user=SimpleNamespace(id=owner_id))
    scope_threads: list[int] = []
    snapshot_threads: list[int] = []

    def resolve_scope(resolved_task_id: int) -> object:
        assert resolved_task_id == task_id
        scope_threads.append(threading.get_ident())
        return scope

    def load_snapshot(resolved_task_id: int, user_id: int):
        assert (resolved_task_id, user_id) == (task_id, owner_id)
        snapshot_threads.append(threading.get_ident())
        return setup_snapshot

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=True)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    schedule_resume = MagicMock()

    monkeypatch.setattr(
        a2a_api,
        "resolve_execution_scope",
        resolve_scope,
        raising=False,
    )
    monkeypatch.setattr(
        a2a_api,
        "load_task_setup_snapshot_sync",
        load_snapshot,
        raising=False,
    )
    monkeypatch.setattr(a2a_api, "_schedule_waiting_a2a_resume", schedule_resume)
    monkeypatch.setattr(a2a_api, "record_key_usage", AsyncMock())

    path = f"/api/a2a/agents/{agent_id}/message:send"
    request = _request_with_json(
        path,
        {
            "message": {
                "messageId": "msg-waiting-boundary",
                "taskId": task_id,
                "role": "ROLE_USER",
                "parts": [{"text": "resume safely"}],
            },
            "configuration": {"returnImmediately": True},
        },
    )

    with patch(
        "xagent.web.api.chat.get_agent_manager",
        return_value=agent_manager,
    ):
        response = await a2a_api.send_message(
            agent_id,
            request,
            identity,
        )

    assert response.status_code == 200
    assert scope_threads and all(thread != main_thread for thread in scope_threads)
    assert snapshot_threads and all(
        thread != main_thread for thread in snapshot_threads
    )
    assert len(scope_threads) == 1
    assert len(snapshot_threads) == 1
    manager_args = agent_manager.get_agent_for_task.await_args.args
    manager_kwargs = agent_manager.get_agent_for_task.await_args.kwargs
    assert manager_args[:2] == (task_id, None)
    assert manager_kwargs["task_setup_snapshot"] is setup_snapshot
    assert manager_kwargs["resolved_execution_scope"] is scope
    assert schedule_resume.call_args.kwargs["resolved_execution_scope"] is scope


@pytest.mark.asyncio
async def test_waiting_resume_cancellation_drains_finalize_before_schedule(
    single_connection_a2a_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled request cannot strand a committed resume claim."""

    _engine, db, agent, _key = single_connection_a2a_db
    agent_id = int(agent.id)
    owner_id = int(agent.user_id)
    waiting = Task(
        user_id=owner_id,
        title="waiting cancellation",
        status=TaskStatus.WAITING_FOR_USER,
        agent_id=agent_id,
        source="a2a",
        is_visible=False,
        agent_config={"a2a_context_id": "ctx-waiting-cancel"},
    )
    db.add(waiting)
    db.commit()
    db.refresh(waiting)
    task_id = int(waiting.id)
    db.close()

    setup_snapshot = SimpleNamespace(runtime_user=SimpleNamespace(id=owner_id))
    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=True)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    finalize_started = threading.Event()
    allow_finalize = threading.Event()
    schedule_resume = MagicMock()
    original_finalize = a2a_api._finalize_waiting_a2a_resume

    def delayed_finalize(**kwargs):
        finalize_started.set()
        assert allow_finalize.wait(timeout=1.0)
        return original_finalize(**kwargs)

    monkeypatch.setattr(a2a_api, "resolve_execution_scope", lambda _task_id: None)
    monkeypatch.setattr(
        a2a_api,
        "load_task_setup_snapshot_sync",
        lambda _task_id, _owner_id: setup_snapshot,
    )
    monkeypatch.setattr(
        a2a_api,
        "_finalize_waiting_a2a_resume",
        delayed_finalize,
    )
    monkeypatch.setattr(a2a_api, "_schedule_waiting_a2a_resume", schedule_resume)

    with patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager):
        resume = asyncio.create_task(
            a2a_api._resume_waiting_a2a_task(
                task_id=task_id,
                agent_id=agent_id,
                task_owner_user_id=owner_id,
                text="resume after cancellation",
                message_id="msg-waiting-cancel",
            )
        )
        assert await asyncio.to_thread(finalize_started.wait, 1.0)
        resume.cancel()
        allow_finalize.set()
        with pytest.raises(asyncio.CancelledError):
            await resume

    schedule_resume.assert_called_once()
    with database_module.get_session_local()() as verify_db:
        claimed = verify_db.query(Task).filter(Task.id == task_id).one()
        assert claimed.status == TaskStatus.RUNNING
        assert claimed.runner_id is not None
        assert claimed.run_id is not None
        assert claimed.lease_expires_at is not None
        assert schedule_resume.call_args.kwargs["run_id"] == claimed.run_id


@pytest.mark.asyncio
async def test_waiting_resume_finalize_pool_timeout_retains_fenced_lease(
    single_connection_a2a_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pool timeout retains the exact claim for TTL without another checkout."""

    _engine, db, agent, _key = single_connection_a2a_db
    agent_id = int(agent.id)
    owner_id = int(agent.user_id)
    waiting = Task(
        user_id=owner_id,
        title="waiting pool timeout",
        status=TaskStatus.WAITING_FOR_USER,
        agent_id=agent_id,
        source="a2a",
        is_visible=False,
        agent_config={"a2a_context_id": "ctx-waiting-pool-timeout"},
    )
    db.add(waiting)
    db.commit()
    db.refresh(waiting)
    task_id = int(waiting.id)
    db.close()

    setup_snapshot = SimpleNamespace(runtime_user=SimpleNamespace(id=owner_id))
    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=True)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    restore_claim = MagicMock()
    schedule_resume = MagicMock()

    monkeypatch.setattr(a2a_api, "resolve_execution_scope", lambda _task_id: None)
    monkeypatch.setattr(
        a2a_api,
        "load_task_setup_snapshot_sync",
        lambda _task_id, _owner_id: setup_snapshot,
    )
    monkeypatch.setattr(
        a2a_api,
        "_finalize_waiting_a2a_resume",
        lambda **_kwargs: (_ for _ in ()).throw(
            SQLAlchemyTimeoutError("QueuePool limit reached")
        ),
    )
    monkeypatch.setattr(a2a_api, "_restore_waiting_resume_claim", restore_claim)
    monkeypatch.setattr(a2a_api, "_schedule_waiting_a2a_resume", schedule_resume)

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
        pytest.raises(SQLAlchemyTimeoutError, match="QueuePool limit"),
    ):
        await a2a_api._resume_waiting_a2a_task(
            task_id=task_id,
            agent_id=agent_id,
            task_owner_user_id=owner_id,
            text="resume during pool exhaustion",
            message_id="msg-waiting-pool-timeout",
        )

    restore_claim.assert_not_called()
    schedule_resume.assert_not_called()
    with database_module.get_session_local()() as verify_db:
        claimed = verify_db.query(Task).filter(Task.id == task_id).one()
        assert claimed.status == TaskStatus.RUNNING
        assert claimed.runner_id is not None
        assert claimed.run_id is not None
        assert claimed.lease_expires_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("heartbeat_outcome", "expected_exception", "expected_match"),
    [
        (
            TaskLeaseHeartbeatOutcome(lease_lost=True),
            RuntimeError,
            "lease heartbeat lost",
        ),
        (
            TaskLeaseHeartbeatOutcome(
                pool_timeout=SQLAlchemyTimeoutError("heartbeat pool timeout")
            ),
            SQLAlchemyTimeoutError,
            "heartbeat pool timeout",
        ),
    ],
)
async def test_waiting_resume_heartbeat_starts_before_post_and_unhealthy_lease_stops_work(
    monkeypatch: pytest.MonkeyPatch,
    heartbeat_outcome: TaskLeaseHeartbeatOutcome,
    expected_exception: type[BaseException],
    expected_match: str,
) -> None:
    """A slow checkpoint post must stay fenced until finalize and schedule."""

    task_id = 701
    agent_id = 702
    owner_id = 703
    lease = TaskLease(
        task_id=task_id,
        runner_id="a2a-resume-runner",
        run_id="a2a-resume-run",
    )
    heartbeat_started = asyncio.Event()

    async def lost_heartbeat(
        heartbeat_lease: TaskLease,
        stop_event: asyncio.Event,
    ) -> TaskLeaseHeartbeatOutcome:
        assert heartbeat_lease == lease
        heartbeat_started.set()
        await stop_event.wait()
        return heartbeat_outcome

    async def post_user_message(*_args, **_kwargs) -> bool:
        assert heartbeat_started.is_set(), (
            "lease heartbeat started after external await"
        )
        return True

    setup_snapshot = SimpleNamespace(runtime_user=SimpleNamespace(id=owner_id))
    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(side_effect=post_user_message)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    finalize = MagicMock()
    restore = MagicMock()
    schedule = MagicMock()

    monkeypatch.setattr(a2a_api, "resolve_execution_scope", lambda _task_id: None)
    monkeypatch.setattr(
        a2a_api,
        "load_task_setup_snapshot_sync",
        lambda _task_id, _owner_id: setup_snapshot,
    )
    monkeypatch.setattr(
        a2a_api,
        "_claim_waiting_a2a_resume",
        lambda _task_id, _agent_id: lease,
    )
    monkeypatch.setattr(
        a2a_api, "run_task_lease_heartbeat", lost_heartbeat, raising=False
    )
    monkeypatch.setattr(a2a_api, "_finalize_waiting_a2a_resume", finalize)
    monkeypatch.setattr(a2a_api, "_restore_waiting_resume_claim", restore)
    monkeypatch.setattr(a2a_api, "_schedule_waiting_a2a_resume", schedule)

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
        pytest.raises(expected_exception, match=expected_match),
    ):
        await a2a_api._resume_waiting_a2a_task(
            task_id=task_id,
            agent_id=agent_id,
            task_owner_user_id=owner_id,
            text="slow checkpoint response",
            message_id="message-with-live-lease",
        )

    finalize.assert_not_called()
    restore.assert_not_called()
    schedule.assert_not_called()


@pytest.mark.asyncio
async def test_waiting_resume_schedule_failure_restores_exact_claim(
    single_connection_a2a_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local scheduling failure releases only this resume's exact lease."""

    _engine, db, agent, _key = single_connection_a2a_db
    agent_id = int(agent.id)
    owner_id = int(agent.user_id)
    waiting = Task(
        user_id=owner_id,
        title="waiting schedule failure",
        status=TaskStatus.WAITING_FOR_USER,
        agent_id=agent_id,
        source="a2a",
        is_visible=False,
        agent_config={"a2a_context_id": "ctx-waiting-schedule-failure"},
    )
    db.add(waiting)
    db.commit()
    db.refresh(waiting)
    task_id = int(waiting.id)
    db.close()

    setup_snapshot = SimpleNamespace(runtime_user=SimpleNamespace(id=owner_id))
    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=True)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)

    monkeypatch.setattr(a2a_api, "resolve_execution_scope", lambda _task_id: None)
    monkeypatch.setattr(
        a2a_api,
        "load_task_setup_snapshot_sync",
        lambda _task_id, _owner_id: setup_snapshot,
    )
    monkeypatch.setattr(
        a2a_api,
        "_schedule_waiting_a2a_resume",
        MagicMock(side_effect=RuntimeError("scheduler unavailable")),
    )

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
        pytest.raises(RuntimeError, match="scheduler unavailable"),
    ):
        await a2a_api._resume_waiting_a2a_task(
            task_id=task_id,
            agent_id=agent_id,
            task_owner_user_id=owner_id,
            text="resume before scheduling",
            message_id="msg-waiting-schedule-failure",
        )

    with database_module.get_session_local()() as verify_db:
        restored = verify_db.query(Task).filter(Task.id == task_id).one()
        assert restored.status == TaskStatus.WAITING_FOR_USER
        assert restored.runner_id is None
        assert restored.lease_expires_at is None


@pytest.mark.asyncio
async def test_waiting_resume_recovery_does_not_overwrite_replacement_owner(
    single_connection_a2a_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Late recovery is fenced by both runner_id and run_id."""

    _engine, db, agent, _key = single_connection_a2a_db
    agent_id = int(agent.id)
    owner_id = int(agent.user_id)
    waiting = Task(
        user_id=owner_id,
        title="waiting replacement owner",
        status=TaskStatus.WAITING_FOR_USER,
        agent_id=agent_id,
        source="a2a",
        is_visible=False,
        agent_config={"a2a_context_id": "ctx-waiting-replacement-owner"},
    )
    db.add(waiting)
    db.commit()
    db.refresh(waiting)
    task_id = int(waiting.id)
    db.close()

    setup_snapshot = SimpleNamespace(runtime_user=SimpleNamespace(id=owner_id))
    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=True)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)

    def replace_owner_then_fail(**_kwargs) -> None:
        with database_module.get_session_local()() as replacement_db:
            replacement = replacement_db.query(Task).filter(Task.id == task_id).one()
            replacement.runner_id = "replacement-runner"
            replacement.run_id = "replacement-run"
            replacement_db.commit()
        raise RuntimeError("scheduler unavailable after replacement")

    monkeypatch.setattr(a2a_api, "resolve_execution_scope", lambda _task_id: None)
    monkeypatch.setattr(
        a2a_api,
        "load_task_setup_snapshot_sync",
        lambda _task_id, _owner_id: setup_snapshot,
    )
    monkeypatch.setattr(
        a2a_api,
        "_schedule_waiting_a2a_resume",
        replace_owner_then_fail,
    )

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
        pytest.raises(RuntimeError, match="scheduler unavailable after replacement"),
    ):
        await a2a_api._resume_waiting_a2a_task(
            task_id=task_id,
            agent_id=agent_id,
            task_owner_user_id=owner_id,
            text="resume before ownership change",
            message_id="msg-waiting-replacement-owner",
        )

    with database_module.get_session_local()() as verify_db:
        replacement = verify_db.query(Task).filter(Task.id == task_id).one()
        assert replacement.status == TaskStatus.RUNNING
        assert replacement.runner_id == "replacement-runner"
        assert replacement.run_id == "replacement-run"
        assert replacement.lease_expires_at is not None


@pytest.mark.asyncio
async def test_waiting_resume_without_checkpoint_releases_lease_for_fallback(
    single_connection_a2a_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """posted=False leaves a clean PAUSED row for the APPEND fallback."""

    _engine, db, agent, _key = single_connection_a2a_db
    agent_id = int(agent.id)
    owner_id = int(agent.user_id)
    waiting = Task(
        user_id=owner_id,
        title="waiting missing checkpoint",
        status=TaskStatus.WAITING_FOR_USER,
        agent_id=agent_id,
        source="a2a",
        is_visible=False,
        agent_config={"a2a_context_id": "ctx-waiting-missing-checkpoint"},
    )
    db.add(waiting)
    db.commit()
    db.refresh(waiting)
    task_id = int(waiting.id)
    db.close()

    setup_snapshot = SimpleNamespace(runtime_user=SimpleNamespace(id=owner_id))
    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=False)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    schedule_resume = MagicMock()

    monkeypatch.setattr(a2a_api, "resolve_execution_scope", lambda _task_id: None)
    monkeypatch.setattr(
        a2a_api,
        "load_task_setup_snapshot_sync",
        lambda _task_id, _owner_id: setup_snapshot,
    )
    monkeypatch.setattr(a2a_api, "_schedule_waiting_a2a_resume", schedule_resume)

    with patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager):
        posted, finalized = await a2a_api._resume_waiting_a2a_task(
            task_id=task_id,
            agent_id=agent_id,
            task_owner_user_id=owner_id,
            text="fallback to a fresh turn",
            message_id="msg-waiting-missing-checkpoint",
        )

    assert posted is False
    assert finalized.status == TaskStatus.PAUSED
    schedule_resume.assert_not_called()
    with database_module.get_session_local()() as verify_db:
        fallback = verify_db.query(Task).filter(Task.id == task_id).one()
        assert fallback.status == TaskStatus.PAUSED
        assert fallback.runner_id is None
        assert fallback.lease_expires_at is None


def test_failed_follow_up_restores_input_required_status() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="waiting",
            status=TaskStatus.WAITING_FOR_USER,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-recover"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(return_value=False)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=agent_manager,
        ),
        patch(
            "xagent.web.api.a2a.TaskTurnOrchestrator.begin_turn",
            side_effect=TaskTurnError("busy"),
        ),
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-recover",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "retry"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 400, response.text
    db = _direct_db_session()
    try:
        recovered = db.query(Task).filter(Task.id == task_id).one()
        assert recovered.status == TaskStatus.WAITING_FOR_USER
    finally:
        db.close()


def test_checkpoint_resume_exception_restores_input_required_status() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="waiting",
            status=TaskStatus.WAITING_FOR_USER,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-resume-error"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    agent_service = MagicMock()
    agent_service.post_user_message = AsyncMock(
        side_effect=RuntimeError("checkpoint callback failed")
    )
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    with patch(
        "xagent.web.api.chat.get_agent_manager",
        return_value=agent_manager,
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-resume-error",
                    "taskId": task_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "retry safely"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )

    assert response.status_code == 500
    agent_service.post_user_message.assert_awaited_once_with(
        str(task_id),
        execution_message="retry safely",
        display_message="retry safely",
        turn_id=f"a2a:{task_id}:msg-resume-error",
        request_interrupt=False,
        reason="A2A input-required response",
    )
    db = _direct_db_session()
    try:
        recovered = db.query(Task).filter(Task.id == task_id).one()
        assert recovered.status == TaskStatus.WAITING_FOR_USER
    finally:
        db.close()


def test_list_tasks_uses_database_filters_and_pagination() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    base_time = datetime.now(UTC) - timedelta(minutes=10)
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task_specs = [
            (TaskStatus.PENDING, "ctx-a", {}, 0),
            (TaskStatus.RUNNING, "ctx-b", {}, 1),
            (TaskStatus.FAILED, "ctx-a", {"a2a_state": "TASK_STATE_CANCELED"}, 2),
            (TaskStatus.FAILED, "ctx-b", {}, 3),
            (TaskStatus.COMPLETED, "ctx-a", {}, 4),
        ]
        tasks: list[Task] = []
        for status, context_id, extra_config, minute in task_specs:
            task = Task(
                user_id=owner_id,
                title=f"task-{minute}",
                status=status,
                updated_at=base_time + timedelta(minutes=minute),
                agent_id=agent_id,
                source="a2a",
                is_visible=False,
                agent_config={"a2a_context_id": context_id, **extra_config},
            )
            db.add(task)
            tasks.append(task)
        db.commit()
        task_ids = [int(task.id) for task in tasks]
    finally:
        db.close()

    first = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"pageSize": 2},
    )
    second = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"pageSize": 2, "pageToken": first.json()["nextPageToken"]},
    )
    canceled = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"status": "TASK_STATE_CANCELED"},
    )
    failed = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"status": "TASK_STATE_FAILED"},
    )
    context = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"contextId": "ctx-a"},
    )
    recent = client.get(
        f"/api/a2a/agents/{agent_id}/tasks",
        headers=_bearer(full_key),
        params={"statusTimestampAfter": (base_time + timedelta(minutes=2)).isoformat()},
    )

    assert first.status_code == 200, first.text
    assert first.json()["totalSize"] == 5
    assert [item["id"] for item in first.json()["tasks"]] == [
        str(task_ids[4]),
        str(task_ids[3]),
    ]
    assert first.json()["nextPageToken"] == "2"
    assert [item["id"] for item in second.json()["tasks"]] == [
        str(task_ids[2]),
        str(task_ids[1]),
    ]
    assert canceled.json()["totalSize"] == 1
    assert canceled.json()["tasks"][0]["id"] == str(task_ids[2])
    assert failed.json()["totalSize"] == 1
    assert failed.json()["tasks"][0]["id"] == str(task_ids[3])
    assert context.json()["totalSize"] == 3
    assert recent.json()["totalSize"] == 2


@pytest.mark.asyncio
async def test_stream_artifact_updates_are_incremental_and_finalize(
    monkeypatch,
) -> None:
    task = Task(
        id=101,
        user_id=1,
        title="stream",
        status=TaskStatus.RUNNING,
        output="part",
        agent_id=7,
        source="a2a",
        agent_config={"a2a_context_id": "ctx-artifact"},
    )
    running = Task(
        id=101,
        user_id=1,
        title="stream",
        status=TaskStatus.RUNNING,
        output="partial",
        agent_id=7,
        source="a2a",
        agent_config={"a2a_context_id": "ctx-artifact"},
    )
    completed = Task(
        id=101,
        user_id=1,
        title="stream",
        status=TaskStatus.COMPLETED,
        output="partial",
        agent_id=7,
        source="a2a",
        agent_config={"a2a_context_id": "ctx-artifact"},
    )
    fresh_tasks = iter([running, completed])

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(a2a_api.asyncio, "sleep", no_sleep)

    async def fetch_fresh(_agent_id: int, _task_id: int):
        return a2a_api._task_snapshot(next(fresh_tasks))

    monkeypatch.setattr(a2a_api, "_fetch_fresh_a2a_task", fetch_fresh)

    response = a2a_api._task_stream_response(7, a2a_api._task_snapshot(task))
    events = [
        json.loads(chunk.removeprefix("data: "))
        async for chunk in response.body_iterator
    ]
    artifact_updates = [
        event["artifactUpdate"] for event in events if "artifactUpdate" in event
    ]

    assert artifact_updates[0]["artifact"]["parts"] == [{"text": "ial"}]
    assert artifact_updates[0]["append"] is True
    assert artifact_updates[0]["lastChunk"] is False
    assert artifact_updates[1]["artifact"]["parts"] == [{"text": "partial"}]
    assert artifact_updates[1]["append"] is False
    assert artifact_updates[1]["lastChunk"] is True


def test_cancel_is_idempotent_and_subscribe_rejects_terminal_task() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        created = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-cancel",
                    "role": "ROLE_USER",
                    "parts": [{"text": "cancel me"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
    task_id = created.json()["task"]["id"]

    first = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:cancel",
        headers=_bearer(full_key),
    )
    second = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:cancel",
        headers=_bearer(full_key),
    )
    subscribed = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:subscribe",
        headers=_bearer(full_key),
    )

    assert first.status_code == 200, first.text
    assert first.json()["status"]["state"] == "TASK_STATE_CANCELED"
    assert second.status_code == 200, second.text
    assert second.json()["status"]["state"] == "TASK_STATE_CANCELED"
    assert subscribed.status_code == 400
    assert subscribed.json()["error"]["details"][0]["reason"] == "UNSUPPORTED_OPERATION"


@pytest.mark.asyncio
async def test_cancel_existing_command_releases_request_session_before_poll(
    single_connection_a2a_db,
) -> None:
    """An idempotent cancel must not pin the only connection while polling."""

    engine, db, agent, key = single_connection_a2a_db
    agent_id = int(agent.id)
    owner_id = int(agent.user_id)
    task = Task(
        user_id=owner_id,
        title="already canceled",
        status=TaskStatus.FAILED,
        agent_id=agent_id,
        source="a2a",
        is_visible=False,
        state_version=7,
        agent_config={
            "a2a_context_id": "ctx-existing-cancel",
            "a2a_state": "TASK_STATE_CANCELED",
        },
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    task_id = int(task.id)
    command_identity = f"cancel:{task_id}:7"
    db.add(
        TaskExecutionCommand(
            task_id=task_id,
            actor_user_id=owner_id,
            command_id=command_identity,
            kind="cancel",
            payload={"agent_id": agent_id},
            status=COMMAND_COMPLETED,
        )
    )
    db.commit()
    identity = _detached_identity(agent, key)
    db.close()

    checked_out_at_dispatch: list[int] = []

    async def observe_dispatch(_executor, *, command_db_id=None):
        checked_out_at_dispatch.append(engine.pool.checkedout())
        return False

    with patch.object(
        a2a_api,
        "dispatch_one_task_command",
        new=observe_dispatch,
    ):
        response = await a2a_api.cancel_task(
            agent_id,
            task_id,
            identity,
        )

    assert response.status_code == 200
    assert checked_out_at_dispatch == [0]
    assert response.body is not None
    assert json.loads(response.body)["status"]["state"] == "TASK_STATE_CANCELED"


def test_cancel_retries_a_previous_terminal_transport_failure() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        created = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-retry-cancel",
                    "role": "ROLE_USER",
                    "parts": [{"text": "cancel me after a retry"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
    assert created.status_code == 200, created.text
    task_id = int(created.json()["task"]["id"])

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = db.query(Task).filter(Task.id == task_id).one()
        db.add(
            TaskExecutionCommand(
                task_id=task_id,
                actor_user_id=int(agent.user_id),
                command_id=f"cancel:{task_id}:{task.run_id or task.state_version}",
                kind="cancel",
                payload={"agent_id": agent_id},
                target_run_id=str(task.run_id) if task.run_id is not None else None,
                target_runner_id=None,
                status=COMMAND_FAILED,
                attempt_count=1,
                failure_count=MAX_COMMAND_FAILURES,
                defer_count=MAX_COMMAND_DEFERS,
                error="temporary transport failure",
                completed_at=datetime.now(UTC),
            )
        )
        db.commit()
    finally:
        db.close()

    response = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:cancel",
        headers=_bearer(full_key),
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"]["state"] == "TASK_STATE_CANCELED"
    db = _direct_db_session()
    try:
        command = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.task_id == task_id)
            .one()
        )
        assert command.status == "completed"
        assert command.failure_count == 0
        assert command.defer_count == 0
    finally:
        db.close()


def test_cancel_maps_stale_run_rejection_to_conflict() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        created = client.post(
            f"/api/a2a/agents/{agent_id}/message:send",
            headers=_bearer(full_key),
            json={
                "message": {
                    "messageId": "msg-stale-cancel",
                    "role": "ROLE_USER",
                    "parts": [{"text": "rotate before cancel"}],
                },
                "configuration": {"returnImmediately": True},
            },
        )
    assert created.status_code == 200, created.text
    task_id = int(created.json()["task"]["id"])
    real_dispatch = a2a_api.dispatch_one_task_command

    async def rotate_then_dispatch(executor, *, command_db_id=None):
        db = _direct_db_session()
        try:
            task = db.query(Task).filter(Task.id == task_id).one()
            task.run_id = "rotated-before-cancel"
            task.runner_id = None
            task.lease_expires_at = None
            db.commit()
        finally:
            db.close()
        return await real_dispatch(executor, command_db_id=command_db_id)

    with patch.object(
        a2a_api,
        "dispatch_one_task_command",
        new=rotate_then_dispatch,
    ):
        response = client.post(
            f"/api/a2a/agents/{agent_id}/tasks/{task_id}:cancel",
            headers=_bearer(full_key),
        )

    assert response.status_code == 409, response.text
    assert response.json()["error"]["details"][0]["reason"] == "INVALID_REQUEST"
    assert "run changed" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_cancel_does_not_overwrite_a_concurrent_completion() -> None:
    agent_id, _full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title="cancel completion race",
            status=TaskStatus.RUNNING,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-cancel-race"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)

        async def complete_during_cancel(_task_id: int) -> None:
            concurrent_db = _direct_db_session()
            try:
                concurrent_task = (
                    concurrent_db.query(Task).filter(Task.id == task_id).one()
                )
                concurrent_task.status = TaskStatus.COMPLETED
                concurrent_task.output = "completed concurrently"
                concurrent_db.commit()
            finally:
                concurrent_db.close()

        with patch(
            "xagent.web.api.websocket.background_task_manager.cancel_task",
            new=AsyncMock(side_effect=complete_during_cancel),
        ):
            await a2a_api._cancel_task_unserialized(
                task_id=task_id,
                agent_id=agent_id,
            )

        db.expire_all()
        completed = db.query(Task).filter(Task.id == task_id).one()
        assert completed.status == TaskStatus.COMPLETED
        assert completed.output == "completed concurrently"
        assert completed.error_message is None
    finally:
        db.close()


def test_subscribe_stream_starts_with_wrapped_task_snapshot() -> None:
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="paused",
            status=TaskStatus.PAUSED,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-stream"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    response = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:subscribe",
        headers=_bearer(full_key),
    )

    assert response.status_code == 200, response.text
    data_lines = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(data_lines) == 1
    event = json.loads(data_lines[0])
    assert event["task"]["id"] == str(task_id)
    assert event["task"]["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"


def test_subscribe_stream_ends_at_server_lifetime_limit(monkeypatch) -> None:
    agent_id, full_key = _create_published_agent_with_key()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
        task = Task(
            user_id=owner_id,
            title="running",
            status=TaskStatus.RUNNING,
            agent_id=agent_id,
            source="a2a",
            is_visible=False,
            agent_config={"a2a_context_id": "ctx-stream-limit"},
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()
    monkeypatch.setattr(a2a_api, "A2A_STREAM_MAX_DURATION_SECONDS", 0.0)

    response = client.post(
        f"/api/a2a/agents/{agent_id}/tasks/{task_id}:subscribe",
        headers=_bearer(full_key),
    )

    assert response.status_code == 200, response.text
    data_lines = [
        line for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert len(data_lines) == 1
    event = json.loads(data_lines[0].removeprefix("data: "))
    assert event["task"]["status"]["state"] == "TASK_STATE_WORKING"
