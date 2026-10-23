"""Fair-share dispatch: the switch, the cap snapshot's slot counts and the tenant rotation."""

import logging

import pytest

from xagent import config
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_execution
from xagent.web.services.task_dispatch_cap import DispatchCapSnapshot, DispatchPolicy
from xagent.web.services.task_execution import BackgroundTaskManager

SWITCH = "XAGENT_DISPATCHER_FAIR_SHARE"
LARGE = DispatchPolicy(limit=10, interactive_reserve=0)


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
