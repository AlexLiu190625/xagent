"""add partial index on task_execution_commands for claimable commands by kind

The dispatcher looks for the oldest claimable ``resume_input`` command before
it falls back to the oldest claimable command of any kind. That lookup filters
on ``kind`` and orders by ``id``, and only ever concerns commands that are
still ``pending`` or ``processing``; every other row is settled history. A
partial index scoped to the claimable statuses keeps the lookup cheap however
large the history grows.

On PostgreSQL, ``CREATE INDEX`` alone takes a SHARE lock that blocks writes to
the table for as long as the build runs, and every command enqueue and every
status change writes to it. This mirrors 20260927_pending_delivery_index.py:
build the index with ``CONCURRENTLY`` outside the migration's own transaction
(``autocommit_block()``), detect and rebuild a leftover ``INVALID`` index from
an interrupted attempt, rebuild an index whose columns or predicate differ, and
keep the operation idempotent. Non-PostgreSQL dialects keep the plain,
transactional ``CREATE INDEX`` path.

Accepted cost: a partial index whose predicate reads ``status`` disables
PostgreSQL's HOT (heap-only tuple) update optimization for a command row whose
``status`` changes, so each transition (pending, processing, completed or
failed) costs one extra index maintenance for that row.

Revision ID: 20261014_task_cmd_claimable_kind
Revises: 20261012_zendesk_oauth
Create Date: 2026-10-14

"""

import re
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261014_task_cmd_claimable_kind"
down_revision: Union[str, None] = "20261012_zendesk_oauth"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "task_execution_commands"
INDEX = "ix_task_commands_claimable_kind"
INDEX_COLUMNS = ("kind", "id")
# The predicate column itself, so a table that lacks ``status`` is recognized
# as not ready for this index, same as the two index columns.
REQUIRED_COLUMNS = (*INDEX_COLUMNS, "status")
PREDICATE = sa.text("status IN ('pending', 'processing')")

POSTGRES_INDEX_VALIDITY_SQL = sa.text(
    """
    SELECT i.indisvalid
    FROM pg_catalog.pg_index AS i
    WHERE i.indexrelid = pg_catalog.to_regclass(:index_name)
    """
)

# A same-named (kind, id) index that lacks -- or has a stale -- partial
# predicate is column-list-identical to a correct one, so the column check
# cannot tell them apart. ``pg_get_expr(indpred, indrelid)`` deparses the
# stored predicate back into SQL text (NULL when the index is not partial at
# all); comparing that against our own PREDICATE catches both a plain,
# non-partial (kind, id) index and one scoped to the wrong statuses.
POSTGRES_INDEX_PREDICATE_SQL = sa.text(
    """
    SELECT pg_catalog.pg_get_expr(i.indpred, i.indrelid)
    FROM pg_catalog.pg_index AS i
    WHERE i.indexrelid = pg_catalog.to_regclass(:index_name)
    """
)


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _columns(inspector: sa.Inspector) -> set[str]:
    if TABLE not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(TABLE)}


def _indexes(inspector: sa.Inspector) -> set[str]:
    if TABLE not in inspector.get_table_names():
        return set()
    return {
        name
        for item in inspector.get_indexes(TABLE)
        if (name := item.get("name")) is not None
    }


def _index_columns(inspector: sa.Inspector, index_name: str) -> tuple[str, ...] | None:
    if TABLE not in inspector.get_table_names():
        return None
    for item in inspector.get_indexes(TABLE):
        if item.get("name") == index_name:
            return tuple(str(name) for name in item.get("column_names") or ())
    return None


def _postgres_index_validity() -> bool | None:
    """Return whether the current-schema index exists and is usable."""

    return (
        op.get_bind()
        .execute(
            POSTGRES_INDEX_VALIDITY_SQL,
            {"index_name": INDEX},
        )
        .scalar_one_or_none()
    )


def _postgres_index_predicate() -> str | None:
    """Return the current-schema index's deparsed predicate, if any."""

    return (
        op.get_bind()
        .execute(
            POSTGRES_INDEX_PREDICATE_SQL,
            {"index_name": INDEX},
        )
        .scalar_one_or_none()
    )


def _normalize_pg_predicate(expression: str) -> str:
    """Normalize a Postgres-deparsed predicate for structural comparison.

    ``pg_get_expr()`` reformats the predicate it stored: it wraps the column
    reference in parentheses, casts the column and the literals to the
    operator's resolved type, rewrites an ``IN`` list into
    ``= ANY (ARRAY[...])`` (this migration's own predicate comes back as
    ``((status)::text = ANY ((ARRAY['pending'::character varying,
    'processing'::character varying])::text[]))``), and does not guarantee
    whitespace. None of that changes what the predicate selects, so both
    sides are reduced to the column and its sorted set of literal values.
    """

    normalized = expression.lower()
    # A cast to an array type, then any scalar cast, are dropped.
    normalized = re.sub(r"::[a-z_ ]+\[\]", "", normalized)
    normalized = re.sub(r"::[a-z_ ]+(\([0-9]+\))?", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    column = re.match(r"^[\s(]*([a-z_]+)", normalized)
    values = sorted(re.findall(r"'([^']*)'", normalized))
    if column is None:
        return normalized
    return f"{column.group(1)}:{','.join(values)}"


def _postgres_predicate_matches_expected(expression: str | None) -> bool:
    """Whether ``expression`` is our own PREDICATE (a non-partial index,
    ``expression is None``, never matches)."""

    if expression is None:
        return False
    return _normalize_pg_predicate(expression) == _normalize_pg_predicate(
        str(PREDICATE)
    )


def _upgrade_postgresql() -> None:
    """Build the claimable-kind index without blocking table writes."""

    inspector = _inspector()
    if not set(REQUIRED_COLUMNS) <= _columns(inspector):
        return

    validity = _postgres_index_validity()
    existing_columns = _index_columns(inspector, INDEX)
    is_up_to_date = (
        validity is True
        and existing_columns == INDEX_COLUMNS
        and _postgres_predicate_matches_expected(_postgres_index_predicate())
    )
    if is_up_to_date:
        return

    with op.get_context().autocommit_block():
        if validity is not None or existing_columns is not None:
            op.drop_index(
                INDEX,
                table_name=TABLE,
                if_exists=True,
                postgresql_concurrently=True,
            )
        op.create_index(
            INDEX,
            TABLE,
            list(INDEX_COLUMNS),
            if_not_exists=True,
            postgresql_concurrently=True,
            postgresql_where=PREDICATE,
        )


def upgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            with context.autocommit_block():
                op.create_index(
                    INDEX,
                    TABLE,
                    list(INDEX_COLUMNS),
                    postgresql_concurrently=True,
                    postgresql_where=PREDICATE,
                )
        elif context.dialect.name == "sqlite":
            op.create_index(INDEX, TABLE, list(INDEX_COLUMNS), sqlite_where=PREDICATE)
        else:
            op.create_index(INDEX, TABLE, list(INDEX_COLUMNS))
        return

    if context.dialect.name == "postgresql":
        _upgrade_postgresql()
        return

    inspector = _inspector()
    # A database whose table lacks one of these columns (an Alembic-only
    # install stopped short of the migration that added it, or a legacy
    # fixture in a test) cannot support this predicate or these index columns
    # yet; skip rather than fail the whole chain.
    if not set(REQUIRED_COLUMNS) <= _columns(inspector):
        return
    if INDEX in _indexes(inspector):
        return
    kwargs: dict[str, object] = {}
    if context.dialect.name == "sqlite":
        kwargs["sqlite_where"] = PREDICATE
    op.create_index(INDEX, TABLE, list(INDEX_COLUMNS), **kwargs)


def downgrade() -> None:
    context = op.get_context()
    is_postgresql = context.dialect.name == "postgresql"
    if context.as_sql:
        if is_postgresql:
            with context.autocommit_block():
                op.drop_index(
                    INDEX,
                    table_name=TABLE,
                    postgresql_concurrently=True,
                )
        else:
            op.drop_index(INDEX, table_name=TABLE)
        return

    if is_postgresql:
        with context.autocommit_block():
            op.drop_index(
                INDEX,
                table_name=TABLE,
                if_exists=True,
                postgresql_concurrently=True,
            )
        return

    inspector = _inspector()
    if INDEX in _indexes(inspector):
        op.drop_index(INDEX, table_name=TABLE)
