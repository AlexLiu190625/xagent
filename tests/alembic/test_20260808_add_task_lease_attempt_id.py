"""Tests for the tasks.lease_attempt_id migration (lease attempt identity)."""

import importlib.util
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260808_add_task_lease_attempt_id.py"
)
TABLE = "tasks"
COLUMN = "lease_attempt_id"


def _migration_module():
    spec = importlib.util.spec_from_file_location(
        "task_lease_attempt_id_migration", MIGRATION_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operations(connection: sa.Connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _offline_sql(migration, dialect_name: str, operation: str) -> str:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


def _legacy_tasks_table(engine) -> None:
    """Migration-only schema: just the columns this migration reads."""
    metadata = sa.MetaData()
    sa.Table(
        TABLE,
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("runner_id", sa.String(255)),
        sa.Column("run_id", sa.String(64)),
    )
    metadata.create_all(engine)


def _column_names(connection) -> set[str]:
    return {column["name"] for column in sa.inspect(connection).get_columns(TABLE)}


def test_online_upgrade_adds_a_nullable_column_and_downgrade_removes_it() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _legacy_tasks_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

            columns = {c["name"]: c for c in sa.inspect(connection).get_columns(TABLE)}
            assert COLUMN in columns
            assert columns[COLUMN]["nullable"] is True

            migration.downgrade()
            assert COLUMN not in _column_names(connection)


def test_online_upgrade_is_idempotent() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _legacy_tasks_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

            columns = [c["name"] for c in sa.inspect(connection).get_columns(TABLE)]
            assert columns.count(COLUMN) == 1


def test_online_downgrade_is_idempotent_when_column_is_absent() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _legacy_tasks_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()

            assert COLUMN not in _column_names(connection)


def test_online_upgrade_and_downgrade_noop_without_the_tasks_table() -> None:
    migration = _migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        assert TABLE not in sa.inspect(connection).get_table_names()


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_offline_upgrade_emits_plain_add_column_on_both_dialects(dialect_name) -> None:
    migration = _migration_module()

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        upgrade_sql = _offline_sql(migration, dialect_name, "upgrade")

    assert f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(64)" in upgrade_sql
    assert "CONCURRENTLY" not in upgrade_sql


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_offline_downgrade_emits_plain_drop_column_on_both_dialects(
    dialect_name,
) -> None:
    migration = _migration_module()

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        downgrade_sql = _offline_sql(migration, dialect_name, "downgrade")

    assert f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}" in downgrade_sql
    assert "CONCURRENTLY" not in downgrade_sql


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_offline_sql_carries_no_bind_parameters(dialect_name) -> None:
    migration = _migration_module()

    upgrade_sql = _offline_sql(migration, dialect_name, "upgrade")
    downgrade_sql = _offline_sql(migration, dialect_name, "downgrade")

    for sql in (upgrade_sql, downgrade_sql):
        assert "%(" not in sql
        assert ":table_name" not in sql
