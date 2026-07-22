from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine, event, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.web.api.v1 import tasks as v1_tasks
from xagent.web.models import database as database_module
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.schemas.v1 import AppendMessageRequest, CreateTaskRequest
from xagent.web.services import sdk_task_service
from xagent.web.services.api_keys import AgentApiIdentity
from xagent.web.services.sdk_task_service import (
    SdkAgentMismatchError,
    SdkTaskNotFoundError,
    load_sdk_task_snapshot_sync,
    load_sdk_task_steps_snapshot_sync,
    load_sdk_task_steps_version_sync,
    prepare_append_sdk_task_sync,
    prepare_create_sdk_task_sync,
    resolve_sdk_upload_owner_sync,
)
from xagent.web.services.task_orchestrator import TaskTurnError


@pytest.fixture()
def sdk_task_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'sdk-task-worker.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=1,
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    monkeypatch.setattr(database_module, "_SessionLocal", session_local)

    with session_local() as db:
        user = User(
            username="sdk-task-worker-owner",
            password_hash="hash",
            is_admin=False,
        )
        db.add(user)
        db.flush()
        agent = Agent(
            user_id=int(user.id),
            name="SDK task worker agent",
            description="test",
            instructions="test",
            execution_mode="balanced",
            models={},
            knowledge_bases=[],
            skills=[],
            tool_categories=[],
            suggested_prompts=[],
        )
        db.add(agent)
        db.commit()
        db.refresh(user)
        db.refresh(agent)
        user_id = int(user.id)
        agent_id = int(agent.id)

    identity = AgentApiIdentity(
        agent_id=agent_id,
        user_id=user_id,
        execution_mode="balanced",
        tool_categories=(),
        status="draft",
        origin="user",
        key_prefix="sdk-worker-key",
    )
    try:
        yield engine, session_local, identity
    finally:
        engine.dispose()


def _insert_sdk_task(session_local, identity: AgentApiIdentity) -> int:
    with session_local() as db:
        task = Task(
            user_id=identity.user_id,
            title="existing sdk task",
            description="first",
            status=TaskStatus.COMPLETED,
            agent_id=identity.agent_id,
            input="first",
            source="sdk",
            is_visible=False,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return int(task.id)


def test_five_task_routes_do_not_accept_request_sessions() -> None:
    for endpoint in (
        v1_tasks.upload_task_files,
        v1_tasks.create_chat_task,
        v1_tasks.append_message_to_task,
        v1_tasks.get_chat_task,
        v1_tasks.get_chat_task_steps,
    ):
        assert "db" not in inspect.signature(endpoint).parameters


def test_upload_owner_defaults_to_authenticated_agents_user(sdk_task_db) -> None:
    _engine, _session_local, identity = sdk_task_db

    owner_user_id = resolve_sdk_upload_owner_sync(
        task_id=None,
        authenticated_agent_id=identity.agent_id,
        default_user_id=identity.user_id,
    )

    assert owner_user_id == identity.user_id


def test_task_aware_upload_uses_persisted_owner_after_agent_owner_drift(
    sdk_task_db,
) -> None:
    _engine, session_local, identity = sdk_task_db
    task_id = _insert_sdk_task(session_local, identity)
    with session_local() as db:
        replacement_owner = User(
            username="sdk-task-current-agent-owner",
            password_hash="hash",
            is_admin=False,
        )
        db.add(replacement_owner)
        db.flush()
        replacement_owner_id = int(replacement_owner.id)
        db.query(Agent).filter(Agent.id == identity.agent_id).update(
            {"user_id": replacement_owner_id}
        )
        db.commit()

    owner_user_id = resolve_sdk_upload_owner_sync(
        task_id=task_id,
        authenticated_agent_id=identity.agent_id,
        default_user_id=replacement_owner_id,
    )

    assert owner_user_id == identity.user_id


def test_task_aware_upload_hides_other_agent_and_non_sdk_tasks(sdk_task_db) -> None:
    _engine, session_local, identity = sdk_task_db
    task_id = _insert_sdk_task(session_local, identity)

    with pytest.raises(SdkTaskNotFoundError):
        resolve_sdk_upload_owner_sync(
            task_id=task_id,
            authenticated_agent_id=identity.agent_id + 1,
            default_user_id=identity.user_id,
        )

    with session_local() as db:
        db.query(Task).filter(Task.id == task_id).update({"source": "web"})
        db.commit()

    with pytest.raises(SdkTaskNotFoundError):
        resolve_sdk_upload_owner_sync(
            task_id=task_id,
            authenticated_agent_id=identity.agent_id,
            default_user_id=identity.user_id,
        )


def test_create_preparation_materializes_after_session_checkin(
    sdk_task_db,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, session_local, identity = sdk_task_db
    local_path = tmp_path / "attachment.txt"
    local_path.write_text("hello", encoding="utf-8")
    with session_local() as db:
        uploaded = UploadedFile(
            file_id="attachment-1",
            user_id=identity.user_id,
            task_id=None,
            filename=local_path.name,
            storage_path=str(local_path),
            storage_status="legacy",
            mime_type="text/plain",
            file_size=local_path.stat().st_size,
        )
        db.add(uploaded)
        db.commit()

    original_materialize = sdk_task_service.materialize_turn_file_lookups
    checked_out: list[int] = []

    def observed_materialize(lookups):
        checked_out.append(engine.pool.checkedout())
        return original_materialize(lookups)

    monkeypatch.setattr(
        sdk_task_service,
        "materialize_turn_file_lookups",
        observed_materialize,
    )

    prepared = prepare_create_sdk_task_sync(
        agent_id=identity.agent_id,
        task_owner_user_id=identity.user_id,
        actor_user_id=identity.user_id,
        tool_categories=identity.tool_categories,
        content="first message",
        file_ids=("attachment-1",),
        connector_runtime_context=None,
    )

    assert prepared.agent_id == identity.agent_id
    assert prepared.file_infos[0]["file_id"] == "attachment-1"
    assert checked_out == [0]
    assert engine.pool.checkedout() == 0


def test_append_preserves_not_found_before_body_agent_mismatch(sdk_task_db) -> None:
    _engine, session_local, identity = sdk_task_db

    with pytest.raises(SdkTaskNotFoundError):
        prepare_append_sdk_task_sync(
            task_id=999999,
            authenticated_agent_id=identity.agent_id,
            actor_user_id=identity.user_id,
            requested_agent_id=identity.agent_id + 1,
            file_ids=(),
            connector_runtime_context=None,
        )

    task_id = _insert_sdk_task(session_local, identity)
    with pytest.raises(SdkAgentMismatchError):
        prepare_append_sdk_task_sync(
            task_id=task_id,
            authenticated_agent_id=identity.agent_id,
            actor_user_id=identity.user_id,
            requested_agent_id=identity.agent_id + 1,
            file_ids=(),
            connector_runtime_context=None,
        )


def test_read_snapshots_are_detached_primitives(sdk_task_db) -> None:
    _engine, session_local, identity = sdk_task_db
    task_id = _insert_sdk_task(session_local, identity)
    timestamp = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
    with session_local() as db:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id="evt-1",
                event_type="ai_message",
                timestamp=timestamp,
                data={"content": "done"},
            )
        )
        db.commit()

    task = load_sdk_task_snapshot_sync(task_id, identity.agent_id)
    version = load_sdk_task_steps_version_sync(task_id, identity.agent_id)
    steps = load_sdk_task_steps_snapshot_sync(task_id, identity.agent_id)

    assert task is not None
    assert task.task_id == task_id
    assert task.status is TaskStatus.COMPLETED
    assert version is not None and version.max_event_id > 0
    assert steps is not None and steps.max_event_id == version.max_event_id
    assert steps.events[0].event_type == "ai_message"
    assert not hasattr(task, "_sa_instance_state")
    assert not hasattr(steps.events[0], "_sa_instance_state")


def test_steps_snapshot_copies_trace_payloads_after_session_checkin(
    sdk_task_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, session_local, identity = sdk_task_db
    task_id = _insert_sdk_task(session_local, identity)
    with session_local() as db:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id="evt-copy-boundary",
                event_type="ai_message",
                timestamp=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
                data={"content": "done", "nested": {"value": [1, 2, 3]}},
            )
        )
        db.commit()

    original_deepcopy = sdk_task_service.deepcopy
    checked_out: list[int] = []

    def observed_deepcopy(value):
        checked_out.append(engine.pool.checkedout())
        return original_deepcopy(value)

    monkeypatch.setattr(sdk_task_service, "deepcopy", observed_deepcopy)

    snapshot = load_sdk_task_steps_snapshot_sync(task_id, identity.agent_id)

    assert snapshot is not None
    assert snapshot.events[0].data["nested"] == {"value": [1, 2, 3]}
    assert checked_out == [0]


def test_pending_failure_compensation_does_not_overwrite_concurrent_claim(
    sdk_task_db,
) -> None:
    engine, session_local, identity = sdk_task_db
    with session_local() as db:
        task = Task(
            user_id=identity.user_id,
            title="pending compensation race",
            status=TaskStatus.PENDING,
            agent_id=identity.agent_id,
            input="race",
            source="sdk",
            is_visible=False,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)

    injected = False

    def claim_immediately_before_compensation(
        connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        nonlocal injected
        normalized = statement.lower()
        if (
            injected
            or not normalized.lstrip().startswith("update tasks set")
            or "control_state" not in normalized
        ):
            return
        injected = True
        connection.execute(
            update(Task)
            .where(Task.id == task_id, Task.status == TaskStatus.PENDING)
            .values(
                status=TaskStatus.RUNNING,
                control_state="running",
                run_id="concurrent-run",
                state_version=Task.state_version + 1,
            )
        )

    event.listen(engine, "before_cursor_execute", claim_immediately_before_compensation)
    try:
        sdk_task_service.mark_pending_sdk_task_failed_sync(
            task_id,
            "compensation must not win",
            expected_agent_id=identity.agent_id,
            expected_owner_user_id=identity.user_id,
        )
    finally:
        event.remove(
            engine,
            "before_cursor_execute",
            claim_immediately_before_compensation,
        )

    assert injected
    with session_local() as db:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status is TaskStatus.RUNNING
        assert task.control_state == "running"
        assert task.run_id == "concurrent-run"
        assert task.error_message is None


async def _wait_for_async_event(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=2)


@pytest.mark.asyncio
async def test_create_workflow_finishes_turn_claim_after_caller_cancellation(
    sdk_task_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine, session_local, identity = sdk_task_db
    entered = asyncio.Event()
    release = asyncio.Event()
    original_begin_turn = v1_tasks.TaskTurnOrchestrator.begin_turn

    async def gated_begin_turn(**kwargs):
        entered.set()
        await release.wait()
        return await original_begin_turn(**kwargs)

    monkeypatch.setattr(
        v1_tasks.TaskTurnOrchestrator,
        "begin_turn",
        gated_begin_turn,
    )
    monkeypatch.setattr(
        "xagent.web.services.task_orchestrator._schedule_bg",
        MagicMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(v1_tasks, "record_key_usage", AsyncMock())

    caller = asyncio.create_task(
        v1_tasks.create_chat_task(
            CreateTaskRequest(
                agent_id=identity.agent_id,
                message={"role": "user", "content": "cancel after commit"},
            ),
            identity,
        )
    )
    await _wait_for_async_event(entered)
    caller.cancel()
    await asyncio.sleep(0)
    assert not caller.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await caller

    with session_local() as db:
        task = db.query(Task).filter(Task.input == "cancel after commit").one()
        assert task.status is TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_create_workflow_compensates_begin_failure_without_pending_task(
    sdk_task_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine, session_local, identity = sdk_task_db
    monkeypatch.setattr(
        v1_tasks.TaskTurnOrchestrator,
        "begin_turn",
        AsyncMock(side_effect=TaskTurnError("bg_inflight")),
    )
    monkeypatch.setattr(v1_tasks, "record_key_usage", AsyncMock())

    with pytest.raises(v1_tasks.V1ApiError):
        await v1_tasks.create_chat_task(
            CreateTaskRequest(
                agent_id=identity.agent_id,
                message={"role": "user", "content": "begin failure"},
            ),
            identity,
        )

    with session_local() as db:
        task = db.query(Task).filter(Task.input == "begin failure").one()
        assert task.status is TaskStatus.FAILED
        assert task.error_message == "Task turn start failed."


@pytest.mark.asyncio
async def test_append_workflow_finishes_turn_claim_after_caller_cancellation(
    sdk_task_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine, session_local, identity = sdk_task_db
    task_id = _insert_sdk_task(session_local, identity)
    entered = asyncio.Event()
    release = asyncio.Event()
    original_begin_turn = v1_tasks.TaskTurnOrchestrator.begin_turn

    async def gated_begin_turn(**kwargs):
        entered.set()
        await release.wait()
        return await original_begin_turn(**kwargs)

    monkeypatch.setattr(
        v1_tasks.TaskTurnOrchestrator,
        "begin_turn",
        gated_begin_turn,
    )
    monkeypatch.setattr(
        "xagent.web.services.task_orchestrator._schedule_bg",
        MagicMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(v1_tasks, "record_key_usage", AsyncMock())

    caller = asyncio.create_task(
        v1_tasks.append_message_to_task(
            task_id,
            AppendMessageRequest(
                agent_id=identity.agent_id,
                message={"role": "user", "content": "cancel append"},
            ),
            identity,
        )
    )
    await _wait_for_async_event(entered)
    caller.cancel()
    await asyncio.sleep(0)
    assert not caller.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await caller

    with session_local() as db:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status is TaskStatus.RUNNING
        assert task.input == "cancel append"
