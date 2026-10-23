"""Fair-share dispatch: the switch, the cap snapshot's slot counts and the tenant rotation."""

import logging

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from tests.web.services.fair_selection_support import World
from xagent import config
from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_execution
from xagent.web.services.task_dispatch_cap import DispatchCapSnapshot, DispatchPolicy
from xagent.web.services.task_execution import BackgroundTaskManager

SWITCH = "XAGENT_DISPATCHER_FAIR_SHARE"
LARGE = DispatchPolicy(limit=10, interactive_reserve=0)
POLICY = DispatchPolicy(limit=2, interactive_reserve=1)


# --- the cap snapshot's slot counts ---------------------------------------------


def reserve(manager, task_id, key):
    token = manager.reserve_dispatch_slot(
        task_id, policy=LARGE, interactive=False, fairness_key=key
    )
    assert token is not None
    return token


async def test_the_snapshot_counts_the_slots_each_tenant_holds():
    manager = BackgroundTaskManager()
    for task_id, key in ((1, "a"), (2, "a"), (3, "b")):
        reserve(manager, task_id, key)

    snapshot = manager.dispatch_cap_snapshot(LARGE)

    assert dict(snapshot.occupancy) == {"a": 2, "b": 1}
    assert dict(snapshot.occupancy) == dict(
        manager._tenant_occupancy(snapshot.active_ids)
    )
    tenants, top = manager.slot_tenant_counts()
    assert (len(snapshot.occupancy), max(snapshot.occupancy.values())) == (tenants, top)


async def test_the_snapshot_drops_the_tenant_of_a_task_that_left_the_active_set():
    manager = BackgroundTaskManager()
    token = reserve(manager, 1, "a")
    reserve(manager, 2, "b")
    manager.release_dispatch_slot(token)

    snapshot = manager.dispatch_cap_snapshot(LARGE)

    assert dict(snapshot.occupancy) == {"b": 1}


async def test_the_snapshot_that_only_observes_the_cap_state_has_no_counts(monkeypatch):
    manager = BackgroundTaskManager()
    reserve(manager, 1, "a")
    built: list[DispatchCapSnapshot] = []

    def record(**fields):
        built.append(DispatchCapSnapshot(**fields))
        return built[-1]

    monkeypatch.setattr(task_execution, "DispatchCapSnapshot", record)

    manager._observe_current_cap_state(0.0)

    assert [dict(snapshot.occupancy) for snapshot in built] == [{}]


async def test_a_snapshot_can_be_hashed_and_its_counts_do_not_decide_equality():
    manager = BackgroundTaskManager()
    reserve(manager, 1, "a")
    snapshot = manager.dispatch_cap_snapshot(LARGE)
    same_ids = DispatchCapSnapshot(policy=LARGE, active_ids=snapshot.active_ids)

    assert hash(snapshot) == hash(same_ids)
    assert snapshot == same_ids


# --- the switch --------------------------------------------------------------------


@pytest.fixture
def switch(monkeypatch, caplog):
    """The dispatch-side switch with no memory from an earlier test."""
    monkeypatch.setattr(transport, "_fair_share_memo", None)
    monkeypatch.delenv(SWITCH, raising=False)
    caplog.set_level(logging.WARNING, logger="xagent.config")

    def warnings() -> list[str]:
        return [
            record.getMessage()
            for record in caplog.records
            if record.name == "xagent.config" and record.levelno == logging.WARNING
        ]

    return warnings


def test_a_wrong_value_warns_once_however_often_it_is_read(monkeypatch, switch):
    monkeypatch.setenv(SWITCH, "garbage")

    assert [transport._fair_share_enabled() for _ in range(100)] == [False] * 100

    assert len(switch()) == 1 and SWITCH in switch()[0]


def test_another_wrong_value_warns_again_and_a_right_one_takes_effect_at_once(
    monkeypatch, switch
):
    monkeypatch.setenv(SWITCH, "garbage")
    assert transport._fair_share_enabled() is False
    monkeypatch.setenv(SWITCH, "other garbage")
    assert transport._fair_share_enabled() is False
    assert len(switch()) == 2

    monkeypatch.setenv(SWITCH, "true")
    assert transport._fair_share_enabled() is True
    monkeypatch.delenv(SWITCH)
    assert transport._fair_share_enabled() is False
    assert len(switch()) == 2


def test_the_switch_reads_the_configured_value(monkeypatch, switch):
    assert config.DISPATCHER_FAIR_SHARE == SWITCH
    monkeypatch.setenv(SWITCH, "false")
    assert transport._fair_share_enabled() is False
    assert switch() == []


# --- a SQLite world of queued commands --------------------------------------------


@pytest.fixture
def world(tmp_path, monkeypatch):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'fair.db'}")
    Base.metadata.create_all(engine)
    built = World(engine)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(database_module, "_SessionLocal", built.sessions)
    monkeypatch.setattr(transport, "get_runner_id", lambda: "me")
    try:
        yield built
    finally:
        engine.dispose()


def statements_of(engine):
    seen: list[str] = []

    def record(_conn, _cursor, statement, *_args):
        seen.append(" ".join(statement.split()))

    sa.event.listen(engine, "before_cursor_execute", record)
    return seen, lambda: sa.event.remove(engine, "before_cursor_execute", record)


def test_listing_gives_the_oldest_queued_command_of_each_tenant_and_class(world):
    first = world.add("a", "batch")
    world.add("a", "batch", status="processing")
    third = world.add("a", "interactive")
    world.add("b", "batch", status="completed")
    world.add("b", "batch", status="failed")
    sixth = world.add("b", "interactive")
    seen, stop = statements_of(world.engine)
    try:
        with world.sessions() as db:
            tenants = transport._list_queued_tenants(db)
    finally:
        stop()

    assert tenants == {
        "a": {"batch": first, "interactive": third},
        "b": {"interactive": sixth},
    }
    assert len(seen) == 1


def test_listing_counts_a_command_that_cannot_be_claimed_as_queued(world):
    pinned = world.add("a", pinned=True)
    unavailable = world.add("b", unavailable=True)
    with world.sessions() as db:
        assert transport._list_queued_tenants(db) == {
            "a": {"batch": pinned},
            "b": {"batch": unavailable},
        }


def test_between_the_batch_line_and_the_limit_only_interactive_commands_count():
    tenants = {
        "a": {"batch": 1, "interactive": 3},
        "b": {"batch": 2},
        "c": {"interactive": 4},
    }
    open_cap = DispatchCapSnapshot(policy=POLICY, active_ids=frozenset())
    band = DispatchCapSnapshot(policy=POLICY, active_ids=frozenset({90}))

    assert transport._tenants_in_band(tenants, open_cap) == tenants
    assert transport._tenants_in_band(tenants, band) == {
        "a": {"interactive": 3},
        "c": {"interactive": 4},
    }
    assert transport._tenants_in_band(tenants, None) == tenants


def test_the_window_is_a_range_on_the_key_so_that_the_index_serves_it():
    sql = str(
        transport._queued_tenant_window("user:1", "batch").compile(
            dialect=postgresql.dialect()
        )
    )

    assert "fairness_key >=" in sql and "fairness_key <=" in sql
    assert "fairness_key =" not in sql
    assert "ORDER BY task_execution_commands.fairness_key, " in sql
    assert transport.FAIR_SHARE_WINDOW == 16
    assert f"LIMIT {transport.FAIR_SHARE_WINDOW}" in str(
        transport._queued_tenant_window("user:1", "batch").compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
