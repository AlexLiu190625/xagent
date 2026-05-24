from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE, CHECKPOINT_TYPE
from xagent.core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    TraceScope,
)
from xagent.web.api.trace_handlers import DatabaseTraceHandler
from xagent.web.api.websocket import (
    _is_agent_checkpoint_data,
    _is_duplicate_user_message_turn,
    _persist_agent_outbound_event,
    create_final_answer_stream_event,
    create_stream_event,
    send_historical_data_as_stream,
)
from xagent.web.api.ws_trace_handlers import (
    WebSocketTraceHandler,
    get_event_type_mapping,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task import TraceEvent as DatabaseTraceEvent
from xagent.web.models.user import User


def test_agent_checkpoint_is_not_converted_to_websocket_stream_event() -> None:
    event = TraceEvent(
        CHECKPOINT_EVENT_TYPE,
        task_id="365",
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": "365",
            "snapshot": {"label": "dag_before_llm"},
        },
    )

    stream_event = WebSocketTraceHandler(365)._convert_trace_event_to_stream_event(
        event
    )

    assert stream_event is None


def test_action_tool_error_maps_to_tool_execution_failed() -> None:
    event = TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.ERROR, TraceCategory.TOOL),
        task_id="365",
        step_id="default",
        data={"tool_name": "execute_python_code", "error_message": "failed"},
    )

    assert get_event_type_mapping(event) == "tool_execution_failed"


def test_action_llm_error_maps_to_llm_call_failed() -> None:
    event = TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.ERROR, TraceCategory.LLM),
        task_id="365",
        step_id="365",
        data={"error_message": "read timed out"},
    )

    assert get_event_type_mapping(event) == "llm_call_failed"


def test_historical_stream_identifies_agent_checkpoint_payload() -> None:
    assert _is_agent_checkpoint_data(
        {
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": "365",
            "snapshot": {"label": "dag_before_llm"},
        }
    )
    assert _is_agent_checkpoint_data(
        {
            "type": "checkpoint",
            "execution_id": "365",
            "pattern_state": {"status": "running"},
            "context": {"messages": []},
        }
    )
    assert not _is_agent_checkpoint_data({"event": "ai_message"})


def test_final_answer_stream_event_is_not_trace_event() -> None:
    event = create_final_answer_stream_event(
        "final_answer_delta",
        365,
        {
            "type": "final_answer_delta",
            "message_id": "final_answer_1",
            "delta": "hello",
        },
    )

    assert event["type"] == "final_answer_delta"
    assert event["task_id"] == 365
    assert event["message_id"] == "final_answer_1"
    assert event["delta"] == "hello"
    assert "event_type" not in event
    assert "data" not in event


def test_persist_agent_outbound_event_uses_payload_ids(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="tester", password_hash="hashed_password", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        task = Task(
            user_id=int(user.id),
            title="Chat task",
            description="Task chat",
            status=TaskStatus.PENDING,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.api.websocket.get_db", get_test_db)

    event = create_stream_event(
        "agent_message",
        int(task.id),
        {
            "event_id": "agent-event-1",
            "step_id": "react-step-1",
            "message": "Need input",
            "expect_response": False,
        },
    )

    _persist_agent_outbound_event(int(task.id), event)

    db = SessionLocal()
    try:
        trace_event = db.query(DatabaseTraceEvent).filter_by(task_id=int(task.id)).one()
        assert trace_event.event_id == "agent-event-1"
        assert trace_event.event_type == "agent_message"
        assert trace_event.step_id == "react-step-1"
    finally:
        db.close()


def test_database_trace_handler_dedupes_user_message_turn_id() -> None:
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="tester", password_hash="hashed_password", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        task = Task(
            user_id=int(user.id),
            title="Chat task",
            description="Task chat",
            status=TaskStatus.PENDING,
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        handler = DatabaseTraceHandler(int(task.id))
        event_type = TraceEventType(
            TraceScope.TASK,
            TraceAction.START,
            TraceCategory.MESSAGE,
        )
        first = TraceEvent(
            event_type,
            task_id=str(task.id),
            data={"message": "Repeat", "turn_id": "turn-1"},
        )
        duplicate = TraceEvent(
            event_type,
            task_id=str(task.id),
            data={"message": "Repeat", "turn_id": "turn-1"},
        )
        different_turn = TraceEvent(
            event_type,
            task_id=str(task.id),
            data={"message": "Repeat", "turn_id": "turn-2"},
        )

        handler._save_trace_event(db, first)
        handler._save_trace_event(db, duplicate)
        handler._save_trace_event(db, different_turn)

        rows = (
            db.query(DatabaseTraceEvent)
            .filter_by(task_id=int(task.id), event_type="user_message")
            .order_by(DatabaseTraceEvent.id)
            .all()
        )
        assert [row.data["turn_id"] for row in rows] == ["turn-1", "turn-2"]
    finally:
        db.close()


def test_websocket_trace_handler_dedupes_prior_user_message_turn_id(
    monkeypatch,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="tester", password_hash="hashed_password", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        task = Task(
            user_id=int(user.id),
            title="Chat task",
            description="Task chat",
            status=TaskStatus.PENDING,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="first-event",
                event_type="user_message",
                timestamp=task.created_at,
                data={"message": "Repeat", "turn_id": "turn-1"},
            )
        )
        db.commit()
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)

    handler = WebSocketTraceHandler(task_id)
    assert not handler._has_prior_user_message_turn(
        "user_message", {"turn_id": "turn-1"}, "first-event"
    )
    assert handler._has_prior_user_message_turn(
        "user_message", {"turn_id": "turn-1"}, "second-event"
    )
    assert not handler._has_prior_user_message_turn(
        "user_message", {"turn_id": "turn-2"}, "second-event"
    )


def test_historical_replay_duplicate_turn_helper_allows_distinct_turns() -> None:
    seen: set[str] = set()

    assert not _is_duplicate_user_message_turn(
        "user_message", {"message": "Repeat", "turn_id": "turn-1"}, seen
    )
    assert _is_duplicate_user_message_turn(
        "user_message", {"message": "Repeat", "turn_id": "turn-1"}, seen
    )
    assert not _is_duplicate_user_message_turn(
        "user_message", {"message": "Repeat", "turn_id": "turn-2"}, seen
    )


@pytest.mark.asyncio
async def test_historical_replay_uses_turn_id_before_legacy_content_dedupe(
    monkeypatch,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="tester", password_hash="hashed_password", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        task = Task(
            user_id=int(user.id),
            title="Chat task",
            description="Chat task",
            status=TaskStatus.COMPLETED,
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        task_id = int(task.id)
        user_id = int(user.id)
        base_time = datetime(2026, 5, 22, tzinfo=timezone.utc)
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="trace-turn-a",
                event_type="user_message",
                timestamp=base_time + timedelta(seconds=1),
                data={"message": "Repeat", "turn_id": "turn-A"},
            )
        )
        db.add_all(
            [
                TaskChatMessage(
                    task_id=task_id,
                    user_id=user_id,
                    role="user",
                    content="Repeat",
                    message_type="user_message",
                    turn_id="turn-A",
                    created_at=base_time + timedelta(seconds=2),
                ),
                TaskChatMessage(
                    task_id=task_id,
                    user_id=user_id,
                    role="user",
                    content="Repeat",
                    message_type="user_message",
                    turn_id="turn-B",
                    created_at=base_time + timedelta(seconds=3),
                ),
            ]
        )
        db.commit()
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    sent_events: list[dict] = []

    async def send_personal_message(event: dict, websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)
    monkeypatch.setattr("xagent.web.api.websocket.cache_get", lambda *args: None)
    monkeypatch.setattr(
        "xagent.web.api.websocket.cache_set", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.send_personal_message",
        send_personal_message,
    )

    await send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )

    user_message_events = [
        event
        for event in sent_events
        if event.get("type") == "trace_event"
        and event.get("event_type") == "user_message"
    ]

    assert [
        (event["data"].get("message"), event["data"].get("turn_id"))
        for event in user_message_events
    ] == [("Repeat", "turn-A"), ("Repeat", "turn-B")]


@pytest.mark.asyncio
async def test_historical_replay_dedupes_file_only_turns_by_turn_id(
    monkeypatch,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(username="tester", password_hash="hashed_password", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        task = Task(
            user_id=int(user.id),
            title="Chat task",
            description="Chat task",
            status=TaskStatus.COMPLETED,
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        task_id = int(task.id)
        user_id = int(user.id)
        base_time = datetime(2026, 5, 22, tzinfo=timezone.utc)
        attachments = [{"file_id": "fid-only", "name": "only.pdf"}]
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="trace-file-only",
                event_type="user_message",
                timestamp=base_time + timedelta(seconds=1),
                data={"message": "", "turn_id": "turn-file", "files": attachments},
            )
        )
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=user_id,
                role="user",
                content="",
                message_type="user_message",
                turn_id="turn-file",
                attachments=attachments,
                created_at=base_time + timedelta(seconds=2),
            )
        )
        db.commit()
    finally:
        db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    sent_events: list[dict] = []

    async def send_personal_message(event: dict, websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)
    monkeypatch.setattr("xagent.web.api.websocket.cache_get", lambda *args: None)
    monkeypatch.setattr(
        "xagent.web.api.websocket.cache_set", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.send_personal_message",
        send_personal_message,
    )

    await send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )

    user_message_events = [
        event
        for event in sent_events
        if event.get("type") == "trace_event"
        and event.get("event_type") == "user_message"
    ]

    assert [
        (event["data"].get("turn_id"), event["data"].get("files"))
        for event in user_message_events
    ] == [("turn-file", attachments)]
