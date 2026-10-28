"""The admission bucket head follows command priority while fair-share is on."""

from __future__ import annotations

import re

import pytest
from sqlalchemy import select, true
from sqlalchemy.dialects import postgresql, sqlite

from tests.web.services.test_task_execution_admission import Execution, dispatch_next
from tests.web.services.test_task_execution_admission import engine as engine_fixture
from tests.web.services.test_task_execution_admission import host as host_fixture
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_admission import TaskAdmissionTicket
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_execution_admission as admission

host = host_fixture
engine = engine_fixture

FAIR_SHARE = "XAGENT_DISPATCHER_FAIR_SHARE"


@pytest.fixture(autouse=True)
def fresh_cache():
    admission.reset_admission_eligible_cache()
    yield
    admission.reset_admission_eligible_cache()


def enqueue(host, priority):
    with host.sessions() as db:
        task = Task(
            user_id=host.user,
            title="Work",
            status=TaskStatus.PENDING,
            dispatch_priority=priority,
        )
        db.add(task)
        db.flush()
        return transport.enqueue_task_command(
            db,
            task_id=task.id,
            actor_user_id=host.user,
            command_id=f"start-{task.id}",
            kind=transport.TaskCommandKind.START,
            payload={},
        )


def waiting(host, *commands):
    with host.sessions() as db:
        return [admission.waiting_for_capacity(db, c.command_id) for c in commands]


@pytest.mark.parametrize("fair_share", [True, False], ids=["on", "off"])
def test_bucket_head_order_follows_the_fair_share_switch(host, monkeypatch, fair_share):
    monkeypatch.setenv(FAIR_SHARE, str(fair_share).lower())
    old_low, high, equal_high = enqueue(host, 0), enqueue(host, 5), enqueue(host, 5)

    if fair_share:
        # Highest priority first, the oldest among equal priorities.
        assert waiting(host, old_low, high, equal_high) == [True, False, True]
    else:
        assert waiting(host, old_low, high, equal_high) == [False, True, True]


def test_switching_fair_share_rebuilds_the_cached_predicate(host, monkeypatch):
    old_low, high = enqueue(host, 0), enqueue(host, 5)
    monkeypatch.setenv(FAIR_SHARE, "false")
    by_age = admission.admission_eligible()
    assert waiting(host, old_low, high) == [False, True]

    monkeypatch.setenv(FAIR_SHARE, "true")
    assert admission.admission_eligible() is not by_age
    assert waiting(host, old_low, high) == [True, False]

    monkeypatch.setenv(FAIR_SHARE, "false")
    assert waiting(host, old_low, high) == [False, True]


async def test_higher_priority_waiter_takes_the_freed_slot_first(host, monkeypatch):
    monkeypatch.setenv(FAIR_SHARE, "true")
    holder = enqueue(host, 0)
    execute = Execution(host)
    await dispatch_next(execute)
    old_low, high = enqueue(host, 0), enqueue(host, 5)

    execute.finish.set()
    execute.cleanup.set()
    await dispatch_next(execute)

    assert execute.started == [holder.command_id, high.command_id]
    assert transport.load_task_command(old_low.command_id).status == "pending"


@pytest.mark.parametrize(
    ("fair_share", "priorities", "runs"),
    [
        (True, (0, 5), "newer"),
        (False, (0, 5), "older"),
        (True, (5, 5), "older"),
    ],
    ids=["on", "off", "on_tie"],
)
async def test_claim_recheck_keeps_the_waiter_behind_the_head(
    host, monkeypatch, fair_share, priorities, runs
):
    monkeypatch.setenv(FAIR_SHARE, str(fair_share).lower())
    older, newer = (enqueue(host, priority) for priority in priorities)
    head, behind = (newer, older) if runs == "newer" else (older, newer)
    # Let the advisory scan offer the waiter behind the head, so only the
    # locked recheck at claim time can keep it there.
    monkeypatch.setattr(admission, "admission_eligible", true)
    execute = Execution(host)

    assert not await transport.dispatch_one_task_command(
        execute, command_db_id=behind.command_id
    )
    assert await transport.dispatch_one_task_command(
        execute, command_db_id=head.command_id
    )
    assert execute.started == [head.command_id]
    execute.finish.set()
    execute.cleanup.set()


DIALECTS = {"sqlite": sqlite.dialect, "postgresql": postgresql.dialect}
# A reference to the command table itself rather than to an alias of it.
BARE_COMMAND_TABLE = re.compile(r"\btask_execution_commands\b(?!_\d| AS)")


def scan(by_priority):
    return select(1).where(admission._build_admission_eligible(by_priority=by_priority))


def recheck(by_priority):
    return select(admission._older_waiter(by_priority=by_priority)).where(
        TaskAdmissionTicket.command_id == 1
    )


@pytest.mark.parametrize("statement", [scan, recheck])
@pytest.mark.parametrize("dialect_name", sorted(DIALECTS))
def test_priority_order_reads_the_command_table_only_through_aliases(
    dialect_name, statement
):
    """Ordering by priority must not join the command table into either query."""

    def bare_references(by_priority):
        compiled = statement(by_priority).compile(dialect=DIALECTS[dialect_name]())
        return len(BARE_COMMAND_TABLE.findall(str(compiled)))

    assert bare_references(True) == bare_references(False)


@pytest.mark.parametrize("statement", [scan, recheck])
@pytest.mark.parametrize("dialect_name", sorted(DIALECTS))
def test_age_order_does_not_read_priority(dialect_name, statement):
    compiled = statement(False).compile(dialect=DIALECTS[dialect_name]())
    assert "priority" not in str(compiled)
