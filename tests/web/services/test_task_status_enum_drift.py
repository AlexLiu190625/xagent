"""``check_task_status_enum_drift`` (``models/task.py``).

Companion to test_task_status_storage_postgresql.py's
``test_pg_enum_reflects_exactly_the_taskstatus_members``, which pins that a
*fresh* ``create_all`` schema has no drift -- it cannot observe a deployed
database whose enum type predates a later addition to ``TaskStatus``. This
file is that missing half: it builds the ``taskstatus`` type by hand, with a
label set the test controls directly, so it can put the check in front of a
type that has actually drifted rather than one ``create_all`` always gets
right.

Disposable-database plumbing is the one this repository already centralizes
for this exact need (``tests/shared/postgres_disposable.py``, extracted from
two near-identical fixtures for the same reason this file would otherwise
duplicate a third).
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy import text

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.database import Base
from xagent.web.models.task import (
    TaskStatus,
    TaskStatusEnumDriftError,
    check_task_status_enum_drift,
)

_ALL_LABELS = [member.name for member in TaskStatus]


def _label_list_sql(labels: list[str]) -> str:
    return ", ".join(f"'{label}'" for label in labels)


def _create_taskstatus_type_and_table(conn: sa.Connection, labels: list[str]) -> None:
    """Minimal ``taskstatus`` enum plus a ``tasks`` table using it -- enough
    for ``check_task_status_enum_drift`` to see (it only queries
    ``pg_catalog`` for a type literally named ``taskstatus`` and calls
    ``has_table("tasks")``), without going through the full ``Task`` ORM
    schema this check has no other dependency on."""
    conn.execute(text(f"CREATE TYPE taskstatus AS ENUM ({_label_list_sql(labels)})"))
    conn.execute(
        text(
            "CREATE TABLE tasks (id SERIAL PRIMARY KEY, "
            f"status taskstatus NOT NULL DEFAULT '{labels[0]}')"
        )
    )


@pytest.fixture()
def postgresql_engine_factory():
    with disposable_database_factory("xagent_taskstatus_drift") as make:
        yield make


@pytest.mark.postgresql
def test_pg_enum_labels_match_passes(postgresql_engine_factory) -> None:
    """A correct schema must pass. Both label sets here come from the same
    ``TaskStatus`` class (the live one via ``create_all``, expected via
    iterating the enum), so this cell cannot fail on a genuine label
    mismatch -- that direction is covered by the failure cells below. What
    it does pin is the opposite direction: the check must not reject a
    deployment whose enum is correct, which is the regression a stricter
    query (an unqualified type name, a missing visibility predicate) would
    introduce.
    """
    engine = postgresql_engine_factory("correct")
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        check_task_status_enum_drift(conn)


@pytest.mark.postgresql
def test_pg_enum_missing_label_raises(postgresql_engine_factory) -> None:
    engine = postgresql_engine_factory("missing")
    missing_label = _ALL_LABELS[-1]
    present_labels = _ALL_LABELS[:-1]
    with engine.begin() as conn:
        _create_taskstatus_type_and_table(conn, present_labels)

    with engine.connect() as conn:
        with pytest.raises(TaskStatusEnumDriftError) as exc_info:
            check_task_status_enum_drift(conn)

    message = str(exc_info.value)
    assert missing_label in message
    assert "pending migration" in message
    assert "unexpected" not in message.split("missing labels")[0]


@pytest.mark.postgresql
def test_pg_enum_extra_label_raises(postgresql_engine_factory) -> None:
    engine = postgresql_engine_factory("extra")
    extra_label = "ARCHIVED_LEGACY"
    with engine.begin() as conn:
        _create_taskstatus_type_and_table(conn, [*_ALL_LABELS, extra_label])

    with engine.connect() as conn:
        with pytest.raises(TaskStatusEnumDriftError) as exc_info:
            check_task_status_enum_drift(conn)

    message = str(exc_info.value)
    assert extra_label in message
    assert "ALTER TYPE" in message
    assert "DROP VALUE" in message


@pytest.mark.postgresql
def test_pg_enum_missing_and_extra_labels_name_both_in_the_message(
    postgresql_engine_factory,
) -> None:
    """Both directions can be true at once -- a process older than the
    database (unexpected labels present) that is also missing a label a
    newer process added (a concurrent-deployment window, not just a stale
    one). The remediation sentence has to say both things are true rather
    than picking one, since only one of them has a migration that fixes it.
    """
    engine = postgresql_engine_factory("both")
    extra_label = "ARCHIVED_LEGACY"
    with engine.begin() as conn:
        _create_taskstatus_type_and_table(conn, [*_ALL_LABELS[:-1], extra_label])

    with engine.connect() as conn:
        with pytest.raises(TaskStatusEnumDriftError) as exc_info:
            check_task_status_enum_drift(conn)

    message = str(exc_info.value)
    assert _ALL_LABELS[-1] in message
    assert extra_label in message
    assert "cannot be reconciled by a migration" in message


@pytest.mark.postgresql
def test_missing_tasks_table_is_a_noop(postgresql_engine_factory) -> None:
    """A bind whose schema has not been created yet -- ``has_table("tasks")``
    is false, so the check has nothing to compare and must not raise. The
    startup path itself never reaches this: it runs after schema creation.
    This covers a caller that doesn't.
    """
    engine = postgresql_engine_factory("empty")
    with engine.connect() as conn:
        check_task_status_enum_drift(conn)  # must not raise


def test_sqlite_backend_is_a_noop() -> None:
    """Any non-PostgreSQL backend is a no-op before the check ever touches
    ``pg_catalog`` -- which doesn't exist on SQLite, so if the dialect guard
    didn't fire, this would raise ``OperationalError`` rather than pass
    silently.

    Builds the real schema first (``tasks`` present) rather than using an
    empty database: an empty database returns early through the
    schema-not-created-yet guard regardless of dialect, which would let a
    missing or deleted dialect guard pass this test by accident. With
    ``tasks`` present, the dialect guard is the only thing standing between
    this call and a query ``pg_catalog`` doesn't have.
    """
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        check_task_status_enum_drift(conn)  # must not raise, must not query


def test_sqlite_backend_never_reaches_the_catalog_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same claim as above, proven the other way: replace ``text`` in
    ``models/task.py`` with a spy that raises if called, so a failure to
    short-circuit on ``bind.dialect.name`` shows up as an assertion failure
    naming the exact reason, not a generic ``OperationalError`` fifteen
    frames down in the sqlite3 driver. Same schema requirement as the cell
    above, for the same reason: an empty database would pass this test
    whether or not the dialect guard exists.
    """
    import xagent.web.models.task as task_module

    def _spy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "check_task_status_enum_drift queried the catalog on a "
            "non-PostgreSQL bind instead of returning early"
        )

    monkeypatch.setattr(task_module, "text", _spy)
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        check_task_status_enum_drift(conn)


@pytest.mark.postgresql
def test_search_path_schema_with_extra_labels_does_not_reject_correct_deployment(
    postgresql_engine_factory,
) -> None:
    """A same-named ``taskstatus`` type in a schema later on the search path,
    carrying labels this database's real type does not have, must not make
    a correct deployment look drifted. Without the ``pg_type_is_visible``
    narrowing, the raw label join across both schemas would surface the
    other schema's extra label as if it belonged to the resolved type.
    """
    engine = postgresql_engine_factory("visible_extra")
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA other_app"))
        conn.execute(text("SET search_path TO other_app"))
        conn.execute(
            text(
                "CREATE TYPE taskstatus AS ENUM "
                f"({_label_list_sql([*_ALL_LABELS, 'SHADOWED_EXTRA'])})"
            )
        )
        conn.execute(text("SET search_path TO public"))
        _create_taskstatus_type_and_table(conn, _ALL_LABELS)

    with engine.connect() as conn:
        conn.execute(text("SET search_path TO public"))
        check_task_status_enum_drift(conn)  # must not raise


@pytest.mark.postgresql
def test_search_path_schema_with_complete_copy_does_not_mask_missing_label_here(
    postgresql_engine_factory,
) -> None:
    """The inverse: a complete, correct copy of the type sitting in a schema
    that is *not* first on the search path must not hide a genuinely
    incomplete type in the schema that is. Without the visibility
    narrowing, the join across both schemas would produce the full label
    set and mask the drift this check exists to catch.
    """
    engine = postgresql_engine_factory("visible_masked")
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA complete_copy"))
        conn.execute(text("SET search_path TO complete_copy"))
        _create_taskstatus_type_and_table(conn, _ALL_LABELS)
        conn.execute(text("SET search_path TO public"))
        _create_taskstatus_type_and_table(conn, _ALL_LABELS[:-1])

    with engine.connect() as conn:
        conn.execute(text("SET search_path TO public"))
        with pytest.raises(TaskStatusEnumDriftError) as exc_info:
            check_task_status_enum_drift(conn)

    assert _ALL_LABELS[-1] in str(exc_info.value)
