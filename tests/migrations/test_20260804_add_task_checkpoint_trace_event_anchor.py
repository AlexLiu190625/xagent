"""Tests for migration 20260804_add_task_checkpoint_trace_event_anchor.

Following this repo's existing migration-test convention (see
tests/migrations/test_20260728_add_agent_template_id_and_name_uniqueness.py):
``tasks``/``trace_events`` are create_all-only in production (never created
by a migration -- see fab71cf4b1ad_add_sdk_fields_to_tasks.py's guard), so
the pre-migration schema is built directly here with SQLAlchemy Core table
objects, stamped to this migration's parent revision, and only the
migration under test is run against it.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy import create_engine, inspect, text

from xagent.db.config import create_alembic_config

PARENT_REVISION = "20260729_add_gmail_audience_grace"
TARGET_REVISION = "20260804_add_task_checkpoint_trace_event_anchor"
COLUMN = "last_checkpoint_trace_event_id"
FK_NAME = "fk_tasks_last_checkpoint_trace_event_id"


def _migration_module() -> ModuleType:
    """Load the migration file directly so its BACKFILL_SQL can be re-run
    against an already-upgraded database, independent of Alembic's own
    revision bookkeeping (which short-circuits command.upgrade() to a no-op
    once a database is already stamped at the target revision -- calling it
    twice does not actually exercise the backfill SQL a second time)."""
    import xagent.migrations as migrations_pkg

    # xagent.migrations is a namespace package (no __init__.py), so it has
    # no __file__ -- __path__ is the only way to locate it.
    migrations_dir = Path(next(iter(migrations_pkg.__path__)))
    path = migrations_dir / "versions" / f"{TARGET_REVISION}.py"
    spec = importlib.util.spec_from_file_location(TARGET_REVISION, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pre_migration_metadata() -> sa.MetaData:
    """The subset of tasks/trace_events columns this migration reads or
    writes -- not the full production schema (see the module docstring)."""
    metadata = sa.MetaData()
    sa.Table(
        "tasks",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("last_checkpoint_event_id", sa.String(255), nullable=True),
    )
    sa.Table(
        "trace_events",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("task_id", sa.Integer, nullable=False),
        sa.Column("build_id", sa.String(255), nullable=True),
        sa.Column("event_id", sa.String(255), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("data", sa.Text, nullable=False),
    )
    return metadata


def _stamp_parent_revision(engine: sa.engine.Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        conn.execute(
            text(
                "INSERT INTO alembic_version (version_num) VALUES "
                f"('{PARENT_REVISION}')"
            )
        )


def _insert_task(
    conn: sa.engine.Connection,
    *,
    task_id: int,
    last_checkpoint_event_id: str | None = None,
) -> None:
    conn.execute(
        text(
            "INSERT INTO tasks (id, user_id, title, status, "
            "last_checkpoint_event_id) VALUES "
            "(:id, 1, 'Task', 'pending', :legacy)"
        ),
        {"id": task_id, "legacy": last_checkpoint_event_id},
    )


def _insert_trace_event(
    conn: sa.engine.Connection,
    *,
    row_id: int,
    task_id: int,
    event_id: str,
    build_id: str | None = None,
    event_type: str = "system_update_general",
) -> None:
    conn.execute(
        text(
            "INSERT INTO trace_events "
            "(id, task_id, build_id, event_id, event_type, timestamp, data) "
            "VALUES (:id, :task_id, :build_id, :event_id, :event_type, "
            ":timestamp, :data)"
        ),
        {
            "id": row_id,
            "task_id": task_id,
            "build_id": build_id,
            "event_id": event_id,
            "event_type": event_type,
            "timestamp": "2026-08-04 00:00:00",
            "data": "{}",
        },
    )


@pytest.fixture
def sqlite_engine(tmp_path: Path) -> sa.engine.Engine:
    engine = create_engine(f"sqlite:///{tmp_path / 'anchor.db'}")
    _pre_migration_metadata().create_all(bind=engine)
    _stamp_parent_revision(engine)
    return engine


def _alembic_config(engine: sa.engine.Engine):
    # create_alembic_config() stores str(engine.url), which SQLAlchemy
    # masks to "***" for display -- fine for logging, fatal for a real
    # connection. Overwrite it with the unmasked form before use.
    config = create_alembic_config(engine)
    config.set_main_option(
        "sqlalchemy.url", engine.url.render_as_string(hide_password=False)
    )
    return config


def _upgrade(engine: sa.engine.Engine, revision: str = TARGET_REVISION) -> None:
    command.upgrade(_alembic_config(engine), revision)


def _downgrade(engine: sa.engine.Engine, revision: str = PARENT_REVISION) -> None:
    command.downgrade(_alembic_config(engine), revision)


class TestUpgradeSqlite:
    def test_adds_nullable_anchor_column(self, sqlite_engine: sa.engine.Engine) -> None:
        _upgrade(sqlite_engine)

        columns = {c["name"]: c for c in inspect(sqlite_engine).get_columns("tasks")}
        assert COLUMN in columns
        assert columns[COLUMN]["nullable"]

    def test_upgraded_sqlite_has_no_db_level_fk(
        self, sqlite_engine: sa.engine.Engine
    ) -> None:
        """D1 asymmetry as a tested fact: Alembic's SQLite batch mode cannot
        add a FK without a full table rebuild, so an upgraded (not freshly
        create_all'd) SQLite database has no DB-level constraint here --
        only the application's NULL-first delete ordering protects it."""
        _upgrade(sqlite_engine)

        fk_names = {
            fk["name"] for fk in inspect(sqlite_engine).get_foreign_keys("tasks")
        }
        assert FK_NAME not in fk_names

    def test_backfills_unambiguous_legacy_pointer(
        self, sqlite_engine: sa.engine.Engine
    ) -> None:
        with sqlite_engine.begin() as conn:
            _insert_task(conn, task_id=1, last_checkpoint_event_id="evt-1")
            _insert_trace_event(conn, row_id=100, task_id=1, event_id="evt-1")

        _upgrade(sqlite_engine)

        with sqlite_engine.begin() as conn:
            value = conn.execute(
                text(f"SELECT {COLUMN} FROM tasks WHERE id = 1")
            ).scalar_one()
        assert value == 100

    def test_ambiguous_match_backfills_to_null(
        self, sqlite_engine: sa.engine.Engine
    ) -> None:
        """Two rows share the legacy event_id within the same task/partition
        -- the backfill must not guess which one is authoritative."""
        with sqlite_engine.begin() as conn:
            _insert_task(conn, task_id=1, last_checkpoint_event_id="evt-dup")
            _insert_trace_event(conn, row_id=100, task_id=1, event_id="evt-dup")
            _insert_trace_event(conn, row_id=101, task_id=1, event_id="evt-dup")

        _upgrade(sqlite_engine)

        with sqlite_engine.begin() as conn:
            value = conn.execute(
                text(f"SELECT {COLUMN} FROM tasks WHERE id = 1")
            ).scalar_one()
        assert value is None

    def test_missing_match_backfills_to_null(
        self, sqlite_engine: sa.engine.Engine
    ) -> None:
        """A legacy pointer with zero matching rows (e.g. the target row was
        concurrently deleted) resolves to NULL, not an aborted migration."""
        with sqlite_engine.begin() as conn:
            _insert_task(conn, task_id=1, last_checkpoint_event_id="evt-gone")

        _upgrade(sqlite_engine)

        with sqlite_engine.begin() as conn:
            value = conn.execute(
                text(f"SELECT {COLUMN} FROM tasks WHERE id = 1")
            ).scalar_one()
        assert value is None

    def test_build_scoped_row_is_not_treated_as_the_root_anchor(
        self, sqlite_engine: sa.engine.Engine
    ) -> None:
        """A worker-build checkpoint sharing the legacy event_id string must
        not resolve the root pointer -- only build_id IS NULL rows do."""
        with sqlite_engine.begin() as conn:
            _insert_task(conn, task_id=1, last_checkpoint_event_id="evt-1")
            _insert_trace_event(
                conn,
                row_id=100,
                task_id=1,
                event_id="evt-1",
                build_id="agent_123_abcd",
            )

        _upgrade(sqlite_engine)

        with sqlite_engine.begin() as conn:
            value = conn.execute(
                text(f"SELECT {COLUMN} FROM tasks WHERE id = 1")
            ).scalar_one()
        assert value is None

    def test_upgrade_is_idempotent_when_rerun(
        self, sqlite_engine: sa.engine.Engine
    ) -> None:
        """command.upgrade() a second time at the same revision is a no-op
        at the Alembic bookkeeping layer -- it never re-executes this
        migration's upgrade() function -- so the backfill SQL itself is
        re-run directly here to exercise its own idempotency guard."""
        with sqlite_engine.begin() as conn:
            _insert_task(conn, task_id=1, last_checkpoint_event_id="evt-1")
            _insert_trace_event(conn, row_id=100, task_id=1, event_id="evt-1")

        _upgrade(sqlite_engine)
        module = _migration_module()
        with sqlite_engine.begin() as conn:
            conn.execute(module.BACKFILL_SQL)

        with sqlite_engine.begin() as conn:
            value = conn.execute(
                text(f"SELECT {COLUMN} FROM tasks WHERE id = 1")
            ).scalar_one()
        assert value == 100

    def test_rerun_does_not_clobber_a_resolved_pointer(
        self, sqlite_engine: sa.engine.Engine
    ) -> None:
        """Idempotency guard: WHERE last_checkpoint_trace_event_id IS NULL
        must stop a second run from re-deriving (or blanking) an already
        resolved pointer, even if the legacy row set changed in between.
        See test_upgrade_is_idempotent_when_rerun above for why the
        backfill SQL is re-run directly rather than through command.upgrade()
        again."""
        with sqlite_engine.begin() as conn:
            _insert_task(conn, task_id=1, last_checkpoint_event_id="evt-1")
            _insert_trace_event(conn, row_id=100, task_id=1, event_id="evt-1")

        _upgrade(sqlite_engine)

        with sqlite_engine.begin() as conn:
            # Simulate a second matching row appearing after the first
            # backfill -- if the WHERE guard were missing, a rerun would
            # flip an already-resolved pointer back to NULL.
            _insert_trace_event(conn, row_id=101, task_id=1, event_id="evt-1")

        module = _migration_module()
        with sqlite_engine.begin() as conn:
            conn.execute(module.BACKFILL_SQL)

        with sqlite_engine.begin() as conn:
            value = conn.execute(
                text(f"SELECT {COLUMN} FROM tasks WHERE id = 1")
            ).scalar_one()
        assert value == 100


class TestDowngradeSqlite:
    def test_removes_the_anchor_column(self, sqlite_engine: sa.engine.Engine) -> None:
        with sqlite_engine.begin() as conn:
            _insert_task(conn, task_id=1, last_checkpoint_event_id="evt-1")
            _insert_trace_event(conn, row_id=100, task_id=1, event_id="evt-1")

        _upgrade(sqlite_engine)
        _downgrade(sqlite_engine)

        columns = {c["name"] for c in inspect(sqlite_engine).get_columns("tasks")}
        assert COLUMN not in columns

    def test_downgrade_leaves_the_legacy_column_untouched(
        self, sqlite_engine: sa.engine.Engine
    ) -> None:
        """The legacy string column is never written by this migration, so
        downgrading (dropping only the new column) trivially restores the
        exact pre-migration readable state."""
        with sqlite_engine.begin() as conn:
            _insert_task(conn, task_id=1, last_checkpoint_event_id="evt-1")
            _insert_trace_event(conn, row_id=100, task_id=1, event_id="evt-1")

        _upgrade(sqlite_engine)
        _downgrade(sqlite_engine)

        with sqlite_engine.begin() as conn:
            value = conn.execute(
                text("SELECT last_checkpoint_event_id FROM tasks WHERE id = 1")
            ).scalar_one()
        assert value == "evt-1"

    def test_upgrade_downgrade_upgrade_is_idempotent(
        self, sqlite_engine: sa.engine.Engine
    ) -> None:
        _upgrade(sqlite_engine)
        _downgrade(sqlite_engine)
        _upgrade(sqlite_engine)

        columns = {c["name"] for c in inspect(sqlite_engine).get_columns("tasks")}
        assert COLUMN in columns


def _postgres_url() -> str | None:
    return os.getenv("XAGENT_TEST_POSTGRES_URL") or os.getenv(
        "POSTGRES_TEST_DATABASE_URL"
    )


@pytest.fixture
def postgres_engine():
    url = _postgres_url()
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    engine = create_engine(url)
    # tasks may carry the anchor FK to trace_events from a previous run --
    # drop tasks (and its constraint) before trace_events, or CASCADE.
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        conn.execute(text("DROP TABLE IF EXISTS tasks CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS trace_events CASCADE"))
    _pre_migration_metadata().create_all(bind=engine)
    _stamp_parent_revision(engine)
    yield engine
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        conn.execute(text("DROP TABLE IF EXISTS tasks CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS trace_events CASCADE"))
    engine.dispose()


@pytest.mark.postgresql
class TestUpgradePostgres:
    def test_upgrade_adds_a_named_foreign_key(self, postgres_engine) -> None:
        _upgrade(postgres_engine)

        foreign_keys = inspect(postgres_engine).get_foreign_keys("tasks")
        anchor_fk = next(fk for fk in foreign_keys if fk["name"] == FK_NAME)
        assert anchor_fk["referred_table"] == "trace_events"
        assert anchor_fk["referred_columns"] == ["id"]
        assert anchor_fk["constrained_columns"] == [COLUMN]

    def test_upgrade_is_idempotent_when_rerun(self, postgres_engine) -> None:
        _upgrade(postgres_engine)
        _upgrade(postgres_engine)

        foreign_keys = {
            fk["name"] for fk in inspect(postgres_engine).get_foreign_keys("tasks")
        }
        assert FK_NAME in foreign_keys

    def test_backfills_unambiguous_legacy_pointer(self, postgres_engine) -> None:
        with postgres_engine.begin() as conn:
            _insert_task(conn, task_id=1, last_checkpoint_event_id="evt-1")
            _insert_trace_event(conn, row_id=100, task_id=1, event_id="evt-1")

        _upgrade(postgres_engine)

        with postgres_engine.begin() as conn:
            value = conn.execute(
                text(f"SELECT {COLUMN} FROM tasks WHERE id = 1")
            ).scalar_one()
        assert value == 100

    def test_downgrade_drops_the_named_foreign_key(self, postgres_engine) -> None:
        _upgrade(postgres_engine)
        _downgrade(postgres_engine)

        foreign_keys = {
            fk["name"] for fk in inspect(postgres_engine).get_foreign_keys("tasks")
        }
        assert FK_NAME not in foreign_keys
        columns = {c["name"] for c in inspect(postgres_engine).get_columns("tasks")}
        assert COLUMN not in columns
