"""Partial index on task_execution_commands by tenant key, dispatch class and id."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, event, inspect, text

from tests.alembic.lock_retry_support import (
    driver_engines,
    held_lock,
    release_before_nth,
)
from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config
from xagent.web.models.database import Base
from xagent.web.models.task_command import TaskExecutionCommand

REVISION = "20261022_task_cmd_fair_index"
DOWN_REVISION = "20261021_task_cmd_fairness_key"
TABLE = "task_execution_commands"
INDEX = "ix_task_commands_claimable_fair"
INDEX_COLUMNS = ("fairness_key", "dispatch_class", "id")
MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20261022_task_command_claimable_fair_index.py"
)
FAST = {"_LOCK_TIMEOUT": "200ms", "_LOCK_RETRY_DELAYS": (0.01, 0.01)}


def _table_sql(dialect: str, *, with_key: bool = True) -> str:
    serial = "SERIAL" if dialect == "postgresql" else "INTEGER"
    key = "fairness_key VARCHAR(64) NOT NULL DEFAULT '', " if with_key else ""
    return (
        f"CREATE TABLE {TABLE} (id {serial} PRIMARY KEY, task_id INTEGER NOT NULL, "
        f"kind VARCHAR(32) NOT NULL, status VARCHAR(16) NOT NULL, {key}"
        "dispatch_class VARCHAR(16) NOT NULL DEFAULT 'batch')"
    )


def _stamp(connection) -> None:
    connection.execute(
        text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
    )
    connection.execute(
        text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
        {"revision": DOWN_REVISION},
    )


def _index_names(connection) -> set[str]:
    return {index["name"] for index in inspect(connection).get_indexes(TABLE)}


def _sqlite_index_sql(connection) -> str:
    return " ".join(
        connection.execute(
            text("SELECT sql FROM sqlite_master WHERE name = :name"), {"name": INDEX}
        )
        .scalar_one()
        .split()
    )


def test_sqlite_upgrade_builds_the_partial_index_and_downgrade_drops_it() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        _stamp(connection)
        connection.execute(text(_table_sql("sqlite")))
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        columns = {
            i["name"]: i["column_names"] for i in inspect(connection).get_indexes(TABLE)
        }
        assert tuple(columns[INDEX]) == INDEX_COLUMNS
        assert "WHERE status IN ('pending', 'processing')" in _sqlite_index_sql(
            connection
        )

        # Running the revision body again changes nothing.
        migration = load_migration_module(MIGRATION_PATH, "fair_index_twice")
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
        assert INDEX in _index_names(connection)

        command.downgrade(config, DOWN_REVISION)
        assert INDEX not in _index_names(connection)


def test_sqlite_upgrade_skips_without_the_table_or_without_a_column() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)
    with engine.begin() as connection:
        _stamp(connection)
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)
        assert TABLE not in inspect(connection).get_table_names()

    with engine.begin() as connection:
        connection.execute(text("DELETE FROM alembic_version"))
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:r)"),
            {"r": DOWN_REVISION},
        )
        connection.execute(text(_table_sql("sqlite", with_key=False)))
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)
        assert INDEX not in _index_names(connection)


def test_create_all_builds_the_same_index_and_the_migration_is_then_a_noop() -> None:
    migration = load_migration_module(MIGRATION_PATH, "fair_index_parity")
    model_index = next(
        i for i in TaskExecutionCommand.__table__.indexes if i.name == INDEX
    )
    assert tuple(c.name for c in model_index.columns) == INDEX_COLUMNS
    assert migration.INDEX_COLUMNS == INDEX_COLUMNS and migration.INDEX == INDEX
    for dialect in ("postgresql", "sqlite"):
        predicate = model_index.dialect_options[dialect]["where"]
        assert str(predicate) == str(migration.PREDICATE)

    built = create_engine("sqlite:///:memory:")
    with built.begin() as connection:
        Base.metadata.create_all(connection)
        from_model = _sqlite_index_sql(connection).replace(" ", "").lower()
    migrated = create_engine("sqlite:///:memory:")
    config = create_alembic_config(migrated)
    with migrated.begin() as connection:
        _stamp(connection)
        connection.execute(text(_table_sql("sqlite")))
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)
        from_migration = _sqlite_index_sql(connection).replace(" ", "").lower()
    assert from_model == from_migration

    with built.begin() as connection:
        _stamp(connection)
        config = create_alembic_config(built)
        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)
        assert INDEX in _index_names(connection)


def _offline_sql(dialect: str, operation: str) -> str:
    migration = load_migration_module(MIGRATION_PATH, f"fair_index_off_{dialect}")
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


def test_offline_postgresql_builds_the_index_between_set_and_reset_lock_timeout() -> (
    None
):
    sql = _offline_sql("postgresql", "upgrade")
    parts = [
        sql.index("COMMIT"),
        sql.index("SET lock_timeout = '5s'"),
        sql.index(
            f"CREATE INDEX {INDEX} ON {TABLE} (fairness_key, dispatch_class, id)"
        ),
        sql.index("RESET lock_timeout"),
    ]
    assert parts == sorted(parts)
    assert "WHERE status IN ('pending', 'processing')" in sql
    assert "CONCURRENTLY" not in sql
    down = _offline_sql("postgresql", "downgrade")
    assert f"DROP INDEX IF EXISTS {INDEX}" in down
    assert down.index("SET lock_timeout") < down.index("DROP INDEX")


def test_offline_sqlite_builds_the_partial_index() -> None:
    sql = _offline_sql("sqlite", "upgrade")
    assert f"CREATE INDEX {INDEX} ON {TABLE} (fairness_key, dispatch_class, id)" in sql
    assert "WHERE status IN ('pending', 'processing')" in sql
    assert "lock_timeout" not in sql


# --- PostgreSQL ---------------------------------------------------------------


@pytest.fixture
def pg_factory():
    with disposable_database_factory("xagent_fair_index") as make:
        yield make


@pytest.fixture(params=["psycopg2", "psycopg"])
def pg(request, pg_factory):
    """(engine for the driver under test, psycopg2 engine for the lock holder)."""
    engines = driver_engines(pg_factory, request.param)
    with engines["psycopg2"].begin() as setup:
        setup.execute(text(_table_sql("postgresql")))
    try:
        yield engines[request.param], engines["psycopg2"]
    finally:
        if request.param != "psycopg2":
            engines[request.param].dispose()


def _migration(monkeypatch, name: str, **overrides):
    migration = load_migration_module(MIGRATION_PATH, name)
    for attribute, value in overrides.items():
        monkeypatch.setattr(migration, attribute, value)
    return migration


def _run(migration, connection, operation: str) -> None:
    connection.commit()
    context = MigrationContext.configure(
        connection, opts={"transaction_per_migration": True}
    )
    with Operations.context(context):
        with context.begin_transaction(_per_migration=True):
            getattr(migration, operation)()


def _pg_state(connection):
    row = connection.execute(
        text(
            "SELECT i.indisvalid, pg_catalog.pg_get_expr(i.indpred, i.indrelid), "
            "(SELECT array_agg(a.attname ORDER BY k.n) FROM unnest(i.indkey) "
            "WITH ORDINALITY AS k(attnum, n) JOIN pg_attribute a "
            "ON a.attrelid = i.indrelid AND a.attnum = k.attnum) "
            "FROM pg_catalog.pg_index AS i "
            "WHERE i.indexrelid = pg_catalog.to_regclass(:name)"
        ),
        {"name": INDEX},
    ).one_or_none()
    connection.commit()
    return None if row is None else (row[0], row[1], tuple(row[2]))


@contextmanager
def _counting(engine, marker: str) -> Iterator[list[str]]:
    seen: list[str] = []

    def record(_conn, _cursor, statement, *_args) -> None:
        if marker in statement:
            seen.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        yield seen
    finally:
        event.remove(engine, "before_cursor_execute", record)


@pytest.mark.postgresql
def test_postgresql_upgrade_repeat_and_downgrade(pg, monkeypatch) -> None:
    engine, _ = pg
    migration = _migration(monkeypatch, "fair_index_pg")
    with engine.connect() as connection:
        _run(migration, connection, "upgrade")
        valid, predicate, columns = _pg_state(connection)
        assert valid is True and columns == INDEX_COLUMNS
        assert migration._postgres_predicate_matches_expected(predicate), predicate
        assert "'pending'" in predicate and "'processing'" in predicate

        with _counting(engine, "CREATE INDEX") as creates:
            _run(migration, connection, "upgrade")
        assert creates == []

        _run(migration, connection, "downgrade")
        assert _pg_state(connection) is None


@pytest.mark.postgresql
def test_postgresql_upgrade_replaces_an_index_with_another_predicate(
    pg, monkeypatch
) -> None:
    engine, _ = pg
    migration = _migration(monkeypatch, "fair_index_pg_stale")
    with engine.connect() as connection:
        connection.execute(
            text(
                f"CREATE INDEX {INDEX} ON {TABLE} "
                "(fairness_key, dispatch_class, id) WHERE status = 'pending'"
            )
        )
        connection.commit()

        _run(migration, connection, "upgrade")

        _, predicate, _ = _pg_state(connection)
        assert migration._postgres_predicate_matches_expected(predicate), predicate


@pytest.mark.postgresql
def test_postgresql_upgrade_is_retried_while_a_write_is_uncommitted(
    pg, monkeypatch
) -> None:
    engine, holder_engine = pg
    migration = _migration(monkeypatch, "fair_index_pg_lock", **FAST)
    with engine.connect() as connection:
        with held_lock(holder_engine, TABLE) as held:
            with _counting(engine, "CREATE INDEX") as attempts:
                with pytest.raises(RuntimeError, match=f"could not lock {TABLE}"):
                    _run(migration, connection, "upgrade")
            assert len(attempts) == 3
            assert _pg_state(connection) is None

            with release_before_nth(engine, held, "CREATE INDEX", 2):
                _run(migration, connection, "upgrade")
        assert _pg_state(connection) is not None


@pytest.mark.postgresql
def test_postgresql_upgrade_builds_the_index_while_readers_hold_the_table(
    pg, monkeypatch
) -> None:
    engine, holder_engine = pg
    migration = _migration(monkeypatch, "fair_index_pg_readers", **FAST)
    with engine.connect() as connection:
        with held_lock(holder_engine, TABLE, "ACCESS SHARE"):
            with _counting(engine, "CREATE INDEX") as attempts:
                _run(migration, connection, "upgrade")
        assert len(attempts) == 1
        assert _pg_state(connection) is not None


@pytest.mark.postgresql
def test_postgresql_upgrade_skips_a_correct_index_while_readers_hold_the_table(
    pg, monkeypatch
) -> None:
    engine, holder_engine = pg
    migration = _migration(monkeypatch, "fair_index_pg_present", **FAST)
    with engine.connect() as connection:
        _run(migration, connection, "upgrade")
        with held_lock(holder_engine, TABLE, "ACCESS SHARE"):
            # A drop would wait for the reader and give up.
            _run(migration, connection, "upgrade")
        assert _pg_state(connection) is not None
