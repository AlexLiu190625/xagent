"""Fair-share selection against PostgreSQL: the listing, and which index each query uses."""

import json
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from tests.alembic.lock_retry_support import driver_engines
from tests.shared.postgres_disposable import disposable_database_factory
from tests.web.services.fair_selection_support import World
from xagent.web.models.database import Base
from xagent.web.services import task_command_transport as transport
from xagent.web.services.task_dispatch_cap import DispatchCapSnapshot, DispatchPolicy

INDEX = "ix_task_commands_claimable_fair"


@pytest.fixture
def pg_factory():
    with disposable_database_factory("xagent_fair_selection") as make:
        yield make


@pytest.fixture(params=["psycopg2", "psycopg"])
def pg(request, pg_factory):
    engines = driver_engines(pg_factory, request.param)
    Base.metadata.create_all(engines["psycopg2"])
    try:
        yield engines[request.param]
    finally:
        if request.param != "psycopg2":
            engines[request.param].dispose()


QUEUED_ROWS = [
    # Sorts before every other key, so a listing that starts from it by mistake shows.
    ("0", "batch", "completed"),
    ("a", "batch", "pending"),
    ("a", "batch", "processing"),
    ("a", "interactive", "pending"),
    ("b", "batch", "completed"),
    ("b", "batch", "failed"),
    ("b", "interactive", "pending"),
    ("c", "batch", "pending"),
    ("user:10", "batch", "pending"),
    ("user:2", "interactive", "processing"),
    ("user:2", "batch", "pending"),
]


@pytest.mark.postgresql
def test_the_listing_gives_the_same_tenants_as_on_sqlite(pg, tmp_path) -> None:
    lite = sa.create_engine(f"sqlite:///{tmp_path / 'lite.db'}")
    Base.metadata.create_all(lite)
    found = {}
    for name, engine in (("sqlite", lite), ("postgresql", pg)):
        world = World(engine)
        for tenant, dispatch_class, status in QUEUED_ROWS:
            world.add(tenant, dispatch_class, status=status)
        with world.sessions() as db:
            found[name] = transport._list_queued_tenants(db)
    lite.dispose()

    assert found["postgresql"] == found["sqlite"]
    assert set(found["postgresql"]) == {"a", "b", "c", "user:10", "user:2"}
    assert set(found["postgresql"]["b"]) == {"interactive"}


@pytest.fixture
def busy_table(pg):
    """Fifty thousand queued commands, nine in ten of one tenant, among as many settled."""
    with pg.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO users (id, username, password_hash, is_admin) "
                "VALUES (1, 'u', 'h', false)"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO tasks (id, user_id, title, status) "
                "SELECT g, 1, 't', 'PENDING' FROM generate_series(1, 100) g"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO task_execution_commands (task_id, command_id, kind, "
                "payload, status, attempt_count, dispatch_class, fairness_key, "
                "created_at, updated_at) "
                "SELECT (g % 100) + 1, 'h' || g, 'start', '{}', 'completed', 1, "
                "'batch', 'user:' || (g % 7 + 1), now(), now() "
                "FROM generate_series(1, 50000) g"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO task_execution_commands (task_id, command_id, kind, "
                "payload, status, attempt_count, dispatch_class, fairness_key, "
                "created_at, updated_at) "
                "SELECT (g % 100) + 1, 'q' || g, 'start', '{}', 'pending', 0, "
                "'batch', CASE WHEN g % 10 = 0 THEN 'user:' || (g % 7 + 2) "
                "ELSE 'user:1' END, now(), now() "
                "FROM generate_series(1, 50000) g"
            )
        )
    with pg.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.execute(sa.text("VACUUM (ANALYZE) task_execution_commands"))
        connection.execute(sa.text("ANALYZE tasks"))
    return pg


def index_names(plan) -> set[str]:
    """Every index a plan reads, anywhere in its tree."""
    found: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            if "Index Name" in node:
                found.add(node["Index Name"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(plan) if isinstance(plan, str) else plan)
    return found


def explain(engine, statement: str):
    with engine.connect() as connection:
        return connection.exec_driver_sql(
            "EXPLAIN (FORMAT JSON) " + statement.replace("%", "%%")
        ).scalar_one()


@pytest.mark.postgresql
def test_the_tenant_listing_reads_the_tenant_index(busy_table) -> None:
    plan = explain(busy_table, str(transport._LIST_QUEUED_TENANTS_POSTGRESQL))

    assert INDEX in index_names(plan)


@pytest.mark.postgresql
def test_the_window_query_of_one_tenant_reads_the_tenant_index(busy_table) -> None:
    cap = DispatchCapSnapshot(
        policy=DispatchPolicy(limit=30, interactive_reserve=3),
        active_ids=frozenset(),
    )
    with sa.orm.Session(busy_table) as db:
        query = transport._in_tenant_window(
            transport._dispatcher_candidate_query(
                db,
                runner_id="me",
                command_db_id=None,
                shared=True,
                cap=cap,
                now=datetime(2026, 10, 23, tzinfo=timezone.utc),
            ),
            "user:5",
            "batch",
        )
        statement = str(
            query.statement.compile(
                dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
            )
        )

    assert "fairness_key >=" in statement and "fairness_key <=" in statement
    assert INDEX in index_names(explain(busy_table, statement))
