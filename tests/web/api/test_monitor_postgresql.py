"""Monitoring endpoints against a real PostgreSQL database.

``monitor.py`` reads fields out of ``TraceEvent.data`` (a ``Column(JSON)``)
through a per-dialect helper. Its PostgreSQL branch is the one branch the
SQLite suite never executes, and it shipped ``~?`` -- an operator PostgreSQL
does not have for ``json``, ``jsonb`` or ``text``. Every query through that
branch therefore failed on PostgreSQL while SQLite CI stayed green, and each
call site swallowed the error into an empty result (#1149).

The two symptoms differ by endpoint, which is why all three are covered here:

- ``/monitor/stats`` raised. Its ``except`` substitutes ``active_models = 0``,
  but the failed statement leaves the transaction aborted, so the *next*
  query in the same session (the ``llm_call_end`` scan) raised
  ``PendingRollbackError`` outside any local handler and the request became a
  500.
- ``/monitor/popular-tools`` and ``/monitor/model-stats`` issue no further
  query after their handler, so they returned HTTP 200 with an empty list --
  a dashboard of zeros with nothing but a log line to show for it.

The seed data also pins which payloads the dialect guard drops. PostgreSQL
accepts several escape sequences into a ``json`` column that ``->>`` then
refuses to convert -- the NUL escape and either half of an unpaired UTF-16
surrogate -- and one such row fails the whole query. A *valid* surrogate pair
is not a hazard and must survive, so a non-BMP payload is seeded alongside.

Fixture pattern copied from
``tests/web/services/test_task_status_storage_postgresql.py`` (skip-if-unset
via ``XAGENT_TEST_POSTGRES_URL``; CI provides it in the PostgreSQL job).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Iterator

import pytest
from sqlalchemy.orm import Session

from xagent.web.api.monitor import (
    get_model_stats,
    get_monitoring_stats,
    get_popular_tools,
)
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.user import User

# String values PostgreSQL accepts into a ``json`` column but then refuses to
# convert to text, so ``json ->> key`` raises on them and takes the whole query
# down with it. These are the payload shapes the dialect branch's guard exists
# to keep away from ``->>``. Built with ``chr`` because an editor will happily
# turn an escape sequence into the character it names, and a lone surrogate
# character is not encodable as UTF-8.
NUL_PAYLOAD = chr(0x0000)  # -> ``unsupported Unicode escape sequence``
LONE_HIGH_SURROGATE = chr(0xD800)  # -> ``invalid input syntax for type json``
LONE_LOW_SURROGATE = chr(0xDC00)  # same, from the other side of the pair

# A non-BMP character, which ``json.dumps`` writes as a *valid* surrogate
# pair. It must NOT be dropped: emoji in an LLM payload are ordinary.
NON_BMP_CHAR = chr(0x1F600)

# Text that merely looks like an escape: the JSON carries a doubled backslash,
# so ``->>`` reads it back without complaint. Dropping it would cost a real
# monitoring row for nothing.
BACKSLASH = chr(92)
LITERAL_ESCAPE_TEXT = BACKSLASH + "u0000"

# The nasty one: literal text shaped like a high surrogate escape, immediately
# followed by a genuinely unpaired low surrogate. Read naively the two look
# like a valid pair, the row slips through, and ``->>`` fails the whole query.
SURROGATE_BEHIND_LITERAL = BACKSLASH + "ud83d" + LONE_LOW_SURROGATE


@pytest.fixture()
def pg_session() -> Iterator[Session]:
    """Session against a real PostgreSQL, where the json operators are real.

    SQLite resolves the helper's other branch, so a dialect-specific operator
    error can only surface here.
    """
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    init_db(db_url=url)
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def _seed_admin_with_trace_events(db: Session) -> User:
    """One admin, one task, and the trace events the three endpoints count.

    Every payload carrying an unconvertible escape is one the guard must drop
    without failing the surrounding query; the paired-surrogate payload is one
    it must leave alone.
    """
    admin = User(username="monitor-pg-admin", password_hash="hash", is_admin=True)
    db.add(admin)
    db.commit()
    db.refresh(admin)

    task = Task(user_id=admin.id, title="monitor-pg-task")
    db.add(task)
    db.commit()
    db.refresh(task)

    now = datetime.now()
    emoji_model = f"emoji-model {NON_BMP_CHAR}"
    payloads: list[tuple[str, dict[str, Any]]] = [
        ("llm_call_start", {"model_name": "gpt-4o", "step_id": "s1", "attempt": 1}),
        ("llm_call_start", {"model_name": "gpt-4o", "step_id": "s2", "attempt": 1}),
        ("llm_call_start", {"model_name": "claude-opus", "step_id": "s3"}),
        # Kept: a valid surrogate pair is not a hazard and must still count.
        ("llm_call_start", {"model_name": emoji_model, "step_id": "s4"}),
        # Dropped: one row per unconvertible escape class.
        ("llm_call_start", {"model_name": "nul-model", "note": NUL_PAYLOAD}),
        ("llm_call_start", {"model_name": "high-model", "note": LONE_HIGH_SURROGATE}),
        ("llm_call_start", {"model_name": "low-model", "note": LONE_LOW_SURROGATE}),
        # Dropped, with the escape in the extracted field rather than beside
        # it: the guard reads the whole payload, so position must not matter.
        ("llm_call_start", {"model_name": f"tainted-{NUL_PAYLOAD}"}),
        # Dropped: a real unpaired surrogate hiding behind text that imitates
        # the other half of a pair.
        (
            "llm_call_start",
            {"model_name": "hidden-model", "note": SURROGATE_BEHIND_LITERAL},
        ),
        # Kept: text that only looks like an escape extracts fine, so the
        # guard must not treat it as a hazard.
        (
            "llm_call_start",
            {"model_name": "literal-escape-model", "note": LITERAL_ESCAPE_TEXT},
        ),
        ("tool_execution_start", {"tool_name": "calculator"}),
        ("tool_execution_start", {"tool_name": "calculator"}),
        ("tool_execution_start", {"tool_name": "web_search"}),
        ("tool_execution_start", {"tool_name": "nul-tool", "note": NUL_PAYLOAD}),
        (
            "tool_execution_start",
            {"tool_name": "surrogate-tool", "note": LONE_HIGH_SURROGATE},
        ),
    ]
    for index, (event_type, data) in enumerate(payloads):
        db.add(
            TraceEvent(
                task_id=task.id,
                event_id=f"monitor-pg-{index}",
                event_type=event_type,
                timestamp=now,
                data=data,
            )
        )
    db.commit()
    return admin


@pytest.mark.postgresql
async def test_monitoring_stats_counts_active_models_on_postgresql(
    pg_session: Session,
) -> None:
    """/monitor/stats reports the real model count instead of failing.

    Before the fix this raised a 500: the rejected operator aborted the
    transaction and the next query in the handler could not run.
    """
    admin = _seed_admin_with_trace_events(pg_session)

    stats = await get_monitoring_stats(db=pg_session, current_user=admin)

    # gpt-4o, claude-opus, the emoji model and the literal-escape model. The
    # five payloads carrying an escape PostgreSQL cannot convert are dropped;
    # a valid surrogate pair and text that merely looks like an escape are not.
    assert stats["activeModels"] == 4


@pytest.mark.postgresql
async def test_popular_tools_returns_usage_counts_on_postgresql(
    pg_session: Session,
) -> None:
    """/monitor/popular-tools returns real rows instead of an empty list."""
    admin = _seed_admin_with_trace_events(pg_session)

    tools = await get_popular_tools(db=pg_session, current_user=admin)

    assert [(entry["name"], entry["usage_count"]) for entry in tools] == [
        ("calculator", 2),
        ("web_search", 1),
    ]


@pytest.mark.postgresql
async def test_model_stats_returns_per_model_calls_on_postgresql(
    pg_session: Session,
) -> None:
    """/monitor/model-stats returns real rows instead of an empty list."""
    admin = _seed_admin_with_trace_events(pg_session)

    stats = await get_model_stats(db=pg_session, current_user=admin)

    assert {entry["name"]: entry["total_tasks"] for entry in stats} == {
        "gpt-4o": 2,
        "claude-opus": 1,
        f"emoji-model {NON_BMP_CHAR}": 1,
        "literal-escape-model": 1,
    }
