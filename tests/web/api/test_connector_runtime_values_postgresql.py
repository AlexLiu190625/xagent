"""Real-PostgreSQL half of the connector-runtime values endpoint's
compare-and-swap proof.

The SQLite half of both constructs lives in
``tests/web/test_connector_runtime_entrypoints_e2e.py`` and runs on every
PR (see the module docstring there for why the SQLite half is the one
that must never be skipped -- it is the lane that actually exercises the
defect these two constructs guard against; this file exists to additionally
prove the same construct on the backend the production ``json`` column
type differs for).

Obtains its database through ``tests/shared/postgres_disposable.py``
(``disposable_database_factory``), the same disposable-CREATE-DATABASE
helper the other ``*_postgresql.py`` suites in this repo use. That helper
reads ``XAGENT_TEST_POSTGRES_URL`` and skips the whole module when it is
unset.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.task import Task, TaskConnectorRuntimeContext, TaskStatus
from xagent.web.models.user import User
from xagent.web.schemas.connector_runtime import ConnectorRuntimeValueItem
from xagent.web.services.connector_runtime import (
    apply_task_connector_runtime_context_values,
    bind_connector_runtime_selection_snapshot,
    prepare_connector_runtime_selection_snapshot,
)

pytestmark = pytest.mark.postgresql


@pytest.fixture()
def session_factory():
    with disposable_database_factory(
        "xagent_connector_runtime_values"
    ) as make_database:
        engine = make_database("values")
        Base.metadata.create_all(bind=engine)
        yield sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _seed_task_with_connector(
    session_factory, *, key_names: list[str]
) -> tuple[int, int]:
    """One MCP connector declaring the given (all optional) context keys,
    one task that selected it. Returns (task_id, server_id)."""
    with session_factory() as db:
        user = User(username="pg-values-user", password_hash="x", is_admin=False)
        db.add(user)
        db.flush()
        server = MCPServer(
            name="pg-values-server",
            description="pg-values-server description",
            managed="external",
            transport="streamable_http",
            url="https://example.com/mcp",
            runtime_input_schema={
                "context": {
                    key: {"type": "string", "required": False} for key in key_names
                }
            },
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": key},
                    "target": {"target_type": "mcp_meta", "key": key},
                }
                for key in key_names
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
        agent = Agent(
            user_id=user.id,
            name="pg-values-agent",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            tool_categories=["mcp"],
        )
        db.add(agent)
        db.commit()

        selected_refs = prepare_connector_runtime_selection_snapshot(
            db=db, agent=agent, connector_user_id=int(user.id)
        )
        task = Task(
            user_id=user.id,
            agent_id=agent.id,
            title="pg-values-task",
            source="sdk",
            status=TaskStatus.PENDING,
        )
        db.add(task)
        db.flush()
        bind_connector_runtime_selection_snapshot(
            task=task, selected_refs=selected_refs
        )
        db.commit()
        return int(task.id), int(server.id)


def _direct_apply(
    session: Session, task_id: int, server_id: int, context: dict[str, Any]
) -> Any:
    task = session.query(Task).filter(Task.id == task_id).one()
    agent = (
        session.query(Agent).filter(Agent.id == task.agent_id).one()
        if task.agent_id is not None
        else None
    )
    item = ConnectorRuntimeValueItem(
        connector_ref={"connector_type": "mcp", "connector_id": server_id},
        context=context,
    )
    return apply_task_connector_runtime_context_values(
        db=session, task=task, agent=agent, payload_items=[item]
    )


def _stored_context(
    session_factory, task_id: int, server_id: int
) -> dict[str, Any] | None:
    with session_factory() as db:
        row = (
            db.query(TaskConnectorRuntimeContext)
            .filter(
                TaskConnectorRuntimeContext.task_id == task_id,
                TaskConnectorRuntimeContext.connector_type == "mcp",
                TaskConnectorRuntimeContext.connector_id == server_id,
            )
            .one_or_none()
        )
        return dict(row.context) if row is not None else None


@pytest.mark.parametrize(
    ("a_context", "expect_conflict"),
    [
        ({"b": "2"}, False),
        ({"x": "0"}, False),
        ({"x": "9"}, True),
    ],
)
def test_concurrent_overwrite_of_an_existing_row_postgresql(
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
    a_context: dict[str, str],
    expect_conflict: bool,
) -> None:
    """Same construct as the SQLite half in
    ``test_connector_runtime_entrypoints_e2e.py`` -- a row already has
    ``{"x": "0"}``. Session A read it before session B committed a change
    to the same row; B commits (adding "a"); A's conditional UPDATE, built
    on the text it read before B's write, cannot match the row's current
    text on PostgreSQL's ``json`` column either -- rowcount 0, not an
    error -- and must retry: reread, remerge, and either succeed (A's own
    key doesn't collide with B's) or conflict (A resubmits "x" with a
    different value than what's on the row now).
    """
    task_id, server_id = _seed_task_with_connector(
        session_factory, key_names=["x", "a", "b"]
    )
    with session_factory() as seed_session:
        _direct_apply(seed_session, task_id, server_id, {"x": "0"})
        seed_session.commit()

    from xagent.web.services import connector_runtime as connector_runtime_service

    real_read = connector_runtime_service._load_task_context_row_snapshots

    session_a = session_factory()
    try:
        stale_snapshot = real_read(session_a, task_id=task_id)

        with session_factory() as session_b:
            _direct_apply(session_b, task_id, server_id, {"a": "1"})
            session_b.commit()

        state = {"calls": 0}

        def _stale_first_read(db: Session, *, task_id: int) -> dict[str, Any]:
            state["calls"] += 1
            if state["calls"] == 1:
                return stale_snapshot
            return real_read(db, task_id=task_id)

        monkeypatch.setattr(
            connector_runtime_service,
            "_load_task_context_row_snapshots",
            _stale_first_read,
        )

        if expect_conflict:
            with pytest.raises(ConnectorRuntimeError) as exc_info:
                _direct_apply(session_a, task_id, server_id, a_context)
            assert exc_info.value.details["reason"] == "conflict.context.x"
            session_a.rollback()
        else:
            _direct_apply(session_a, task_id, server_id, a_context)
            session_a.commit()
    finally:
        session_a.close()

    if expect_conflict:
        assert _stored_context(session_factory, task_id, server_id) == {
            "x": "0",
            "a": "1",
        }
    else:
        stored = _stored_context(session_factory, task_id, server_id)
        assert stored["x"] == "0"
        for key, value in a_context.items():
            assert stored[key] == value


def _overwrite_context_row_text(
    session_factory, task_id: int, server_id: int, raw_text: str
) -> None:
    with session_factory() as db:
        row = (
            db.query(TaskConnectorRuntimeContext)
            .filter(
                TaskConnectorRuntimeContext.task_id == task_id,
                TaskConnectorRuntimeContext.connector_type == "mcp",
                TaskConnectorRuntimeContext.connector_id == server_id,
            )
            .one()
        )
        db.execute(
            sa_text(
                "UPDATE task_connector_runtime_contexts SET context = :t WHERE id = :i"
            ),
            {"t": raw_text, "i": row.id},
        )
        db.commit()


@pytest.mark.parametrize(
    "raw_text",
    [
        # Non-canonical rendering: different key order plus extra
        # whitespace than SQLAlchemy's default `json.dumps` would produce.
        '{"b": 1,   "a": 2}',
        # Non-ASCII, unescaped -- `ensure_ascii=True` (json.dumps's
        # default) would have turned this into \\uXXXX escapes.
        '{"name": "会议室预订"}',
    ],
)
def test_values_endpoint_expected_old_value_is_the_database_rendering_postgresql(
    session_factory, raw_text: str
) -> None:
    """PostgreSQL's ``json`` column type stores the exact text it was
    given -- this is the backend the "the database's own
    rendering" half of the design claim is actually about. A row whose
    stored text was never produced by this codebase's own ``json.dumps``
    call must still accept a new key; a Python-side re-serialization of the
    read value as the CAS predicate's expected-old-value would never match
    a row like this, on either backend.
    """
    task_id, server_id = _seed_task_with_connector(
        session_factory, key_names=["auth_token"]
    )
    with session_factory() as seed_session:
        seed_session.add(
            TaskConnectorRuntimeContext(
                task_id=task_id,
                connector_type="mcp",
                connector_id=server_id,
                context={"placeholder": "will be overwritten"},
            )
        )
        seed_session.commit()
    _overwrite_context_row_text(session_factory, task_id, server_id, raw_text)

    with session_factory() as db:
        _direct_apply(db, task_id, server_id, {"auth_token": "new-value"})
        db.commit()

    assert (
        _stored_context(session_factory, task_id, server_id)["auth_token"]
        == "new-value"
    )
