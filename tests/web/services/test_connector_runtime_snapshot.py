from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.web.models import Agent, Base, MCPServer, Task, User, UserMCPServer
from xagent.web.models.agent import AgentStatus
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.task import TaskStatus
from xagent.web.services.connector_runtime import (
    bind_connector_runtime_selection_snapshot,
    prepare_connector_runtime_selection_snapshot,
)


@pytest.fixture()
def db_session() -> Iterator[Session]:
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


def _create_user(db: Session, username: str) -> User:
    user = User(username=username, password_hash="hash")
    db.add(user)
    db.flush()
    return user


def _create_runtime_mcp(db: Session, user: User, name: str) -> MCPServer:
    server = MCPServer(
        name=name,
        description=f"{name} description",
        managed="external",
        transport="streamable_http",
        url="https://example.com/mcp",
        runtime_input_schema={
            "context": {"account_id": {"type": "string", "required": False}}
        },
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "mcp_meta", "key": "account_id"},
            }
        ],
    )
    db.add(server)
    db.flush()
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=True,
        )
    )
    db.flush()
    return server


def _create_runtime_custom_api(db: Session, user: User, name: str) -> CustomApi:
    api = CustomApi(
        name=name,
        description=f"{name} description",
        url="https://example.com/api",
        method="GET",
        runtime_input_schema={
            "context": {"account_id": {"type": "string", "required": False}}
        },
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
    )
    db.add(api)
    db.flush()
    db.add(
        UserCustomApi(
            user_id=user.id,
            custom_api_id=api.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=True,
        )
    )
    db.flush()
    return api


def test_task_model_defaults_connector_runtime_selected_refs_to_empty_list(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    task = Task(user_id=user.id, title="default refs", status=TaskStatus.PENDING)
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    assert task.connector_runtime_selected_refs == []

    legacy_task = Task(
        user_id=user.id,
        title="legacy null refs",
        status=TaskStatus.PENDING,
        connector_runtime_selected_refs=None,
    )
    db_session.add(legacy_task)
    db_session.commit()
    db_session.refresh(legacy_task)

    assert legacy_task.connector_runtime_selected_refs is None


def test_selection_snapshot_uses_runtime_owner_connector_visibility(
    db_session: Session,
) -> None:
    agent_owner = _create_user(db_session, "agent-owner")
    task_owner = _create_user(db_session, "task-owner")
    agent = Agent(
        user_id=agent_owner.id,
        name="Published Agent",
        description="shared",
        instructions="Use tools.",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()

    owner_server = _create_runtime_mcp(db_session, agent_owner, "owner-server")
    task_owner_server = _create_runtime_mcp(db_session, task_owner, "task-owner-server")

    selected_refs = prepare_connector_runtime_selection_snapshot(
        db=db_session,
        agent=agent,
        connector_user_id=int(task_owner.id),
    )
    task = Task(
        user_id=task_owner.id,
        agent_id=agent.id,
        title="published agent task",
        status=TaskStatus.PENDING,
    )
    bind_connector_runtime_selection_snapshot(task=task, selected_refs=selected_refs)

    assert task.connector_runtime_selected_refs == [
        {"connector_type": "mcp", "connector_id": int(task_owner_server.id)}
    ]
    assert {"connector_type": "mcp", "connector_id": int(owner_server.id)} not in (
        task.connector_runtime_selected_refs or []
    )


def test_selection_snapshot_scopes_custom_api_by_source_server_name(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    agent = Agent(
        user_id=user.id,
        name="Scoped Connector Agent",
        description="shared",
        instructions="Use scoped tools.",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=["mcp:Records"],
    )
    db_session.add(agent)
    db_session.flush()

    selected_mcp = _create_runtime_mcp(db_session, user, "Records")
    selected_api = _create_runtime_custom_api(db_session, user, "Records")
    unselected_api = _create_runtime_custom_api(db_session, user, "Billing")

    selected_refs = prepare_connector_runtime_selection_snapshot(
        db=db_session,
        agent=agent,
        connector_user_id=int(user.id),
    )
    task = Task(
        user_id=user.id,
        agent_id=agent.id,
        title="scoped connector task",
        status=TaskStatus.PENDING,
    )
    bind_connector_runtime_selection_snapshot(task=task, selected_refs=selected_refs)

    assert task.connector_runtime_selected_refs == [
        {"connector_type": "custom_api", "connector_id": int(selected_api.id)},
        {"connector_type": "mcp", "connector_id": int(selected_mcp.id)},
    ]
    assert {
        "connector_type": "custom_api",
        "connector_id": int(unselected_api.id),
    } not in (task.connector_runtime_selected_refs or [])
