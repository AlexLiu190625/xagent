"""The tenant cap of fair-share dispatch: which tenant is held back, and where."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.web.services.test_dispatch_fair_selection import clock as clock_fixture
from tests.web.services.test_dispatch_fair_selection import on as on_fixture
from tests.web.services.test_dispatch_fair_selection import world as world_fixture
from xagent import config
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_execution
from xagent.web.services.task_dispatch_cap import (
    DispatchCapSnapshot,
    DispatchPolicy,
    TenantListingRequired,
)
from xagent.web.services.task_execution import BackgroundTaskManager

KINDS = transport.TaskCommandKind
BAND = DispatchPolicy(limit=4, interactive_reserve=2)

world = world_fixture
clock = clock_fixture
on = on_fixture


def band_cap(occupancy, *, active=2, held=False, policy=BAND):
    return DispatchCapSnapshot(
        policy=policy,
        active_ids=frozenset(range(900, 900 + active)),
        held=held,
        occupancy=occupancy,
    )


# --- which tenant is held back ---------------------------------------------------


@pytest.mark.parametrize(
    ("occupancy", "queued", "expected"),
    [
        ({"x": 2, "y": 2}, {}, None),
        ({"x": 2, "y": 2}, {"a": 1}, None),
        ({}, {"a": 1, "b": 2}, None),
        # The top holder has nothing queued, so nobody is held back.
        ({"x": 3, "y": 1}, {"y": 5, "z": 6}, None),
        # A tie: only the tied tenant with the oldest queued command is held back.
        ({"x": 2, "y": 2}, {"x": 7, "y": 3}, "y"),
        ({"x": 2, "y": 2}, {"x": 3, "y": 7}, "x"),
        ({"x": 2, "y": 2}, {"x": 9, "z": 1}, "x"),
        ({"a": 2, "b": 1}, {"a": 9, "b": 1}, "a"),
        # The empty key of a row from before tenant keys is an ordinary tenant.
        ({"": 3, "b": 1}, {"": 4, "b": 5}, ""),
        ({"a": 2, "b": 1}, None, None),
    ],
    ids=[
        "empty-listing",
        "one-tenant",
        "nothing-held",
        "top-holder-not-queued",
        "tie-held-back-is-the-earlier-queue",
        "tie-held-back-is-the-earlier-queue-other-side",
        "tie-only-one-queued",
        "unique-top",
        "empty-key",
        "no-listing",
    ],
)
def test_the_capped_tenant_rule(occupancy, queued, expected):
    assert band_cap(occupancy).capped_tenant(queued) == expected


@pytest.mark.parametrize(
    ("active", "held", "band", "expected"),
    [(2, False, True, "a"), (3, False, True, "a"), (2, True, False, None)],
    ids=["lower-edge", "upper-edge", "held"],
)
def test_the_cap_applies_only_between_the_batch_line_and_the_limit(
    active, held, band, expected
):
    cap = band_cap({"a": 2, "b": 1}, active=active, held=held)

    assert cap.in_reserve_band is band
    assert cap.capped_tenant({"a": 1, "b": 2}) == expected


# --- the reservation decides --------------------------------------------------------

BOTH = {"a": 10, "b": 20}
RESERVATION_ROWS = {
    # (policy, seeded (task, tenant) slots, [(task, tenant, listing, require)], outcomes)
    "a-stale-listing-cannot-overshoot": (
        BAND,
        [(1, "a"), (2, "b")],
        [(3, "b", BOTH, False), (4, "b", BOTH, False)],
        ["token", None],
    ),
    "b-no-listing-no-cap": (
        BAND,
        [(1, "a"), (2, "b")],
        [(3, "b", None, False), (4, "b", None, False)],
        ["token", "token"],
    ),
    "c-one-tenant-queued": (
        BAND,
        [(1, "a"), (2, "b")],
        [(3, "b", {"b": 20}, False)],
        ["token"],
    ),
    "d-an-active-task-passes": (
        BAND,
        [(1, "a"), (2, "b")],
        [(1, "a", BOTH, False)],
        ["token"],
    ),
    "e-band-without-listing": (
        BAND,
        [(1, "a"), (2, "b")],
        [(3, "b", None, True)],
        ["raises"],
    ),
    "f-below-the-batch-line": (BAND, [(1, "a")], [(2, "b", None, True)], ["token"]),
    "g-full": (
        BAND,
        [(1, "a"), (2, "b"), (3, "a"), (4, "b")],
        [(5, "a", None, True)],
        [None],
    ),
    "h-held": (BAND, [(1, "a"), (2, "b")], [(3, "b", None, True)], [None]),
    "i-no-reserve": (
        DispatchPolicy(limit=4, interactive_reserve=0),
        [(1, "a"), (2, "b"), (3, "a")],
        [(4, "b", None, True)],
        ["token"],
    ),
}


def slot_state(manager):
    return dict(manager._dispatch_reservations), dict(manager._slot_keys)


@pytest.mark.parametrize("row", list(RESERVATION_ROWS))
async def test_the_reservation_decision(row, monkeypatch):
    policy, seeded, calls, outcomes = RESERVATION_ROWS[row]
    manager = BackgroundTaskManager()
    for task_id, tenant in seeded:
        assert manager.reserve_dispatch_slot(
            task_id, policy=policy, interactive=True, fairness_key=tenant
        )
    if row == "h-held":
        monkeypatch.setattr(
            task_execution, "release_hold", SimpleNamespace(closed=True)
        )

    for (task_id, tenant, listing, require), outcome in zip(calls, outcomes):
        before = slot_state(manager)

        def reserve():
            return manager.reserve_dispatch_slot(
                task_id,
                policy=policy,
                interactive=True,
                fairness_key=tenant,
                queued_tenants=listing,
                require_listing=require,
            )

        if outcome == "raises":
            with pytest.raises(TenantListingRequired):
                reserve()
            assert slot_state(manager) == before
        elif outcome is None:
            assert reserve() is None
            assert slot_state(manager) == before
        else:
            assert reserve() is not None
            reservations, keys = slot_state(manager)
            assert reservations[task_id] == before[0].get(task_id, 0) + 1
            assert keys == {**before[1], task_id: keys[task_id]}


# --- one dispatch attempt -----------------------------------------------------------


@pytest.fixture
def dispatcher(monkeypatch):
    """A real slot manager and a coordinator that applies nothing."""
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv(config.DISPATCHER_MAX_INFLIGHT_PER_PROCESS, "4")
    monkeypatch.setenv(config.DISPATCHER_INTERACTIVE_RESERVE_PER_PROCESS, "2")
    manager = BackgroundTaskManager()
    monkeypatch.setattr(task_execution, "background_task_manager", manager)
    ensured: list[int] = []
    coordinator = SimpleNamespace(
        _command_tasks=set(),
        execute_command=AsyncMock(return_value=False),
        require_recovery=Mock(),
    )

    async def ensure(task_id):
        ensured.append(task_id)
        return coordinator

    registry = SimpleNamespace(busy_command_tasks=lambda: set(), ensure=ensure)
    monkeypatch.setattr(runtime, "get_task_coordinator_registry", lambda: registry)

    async def executor(command):
        raise AssertionError("no command should be applied")

    return SimpleNamespace(manager=manager, ensured=ensured, executor=executor)


def seed_slots(manager, *slots):
    for task_id, tenant in slots:
        assert manager.reserve_dispatch_slot(
            task_id, policy=BAND, interactive=True, fairness_key=tenant
        )


async def test_a_second_missing_listing_ends_the_attempt(
    world, on, dispatcher, monkeypatch
):
    # The reply of tenant a is found under a snapshot below the batch line, so its
    # candidate carries no listing; the process is already between the lines when
    # the slot is asked for. The attempt selects again once, and when the listing is
    # still missing it ends. With the code as it is, a selection that follows a
    # refusal reads the live snapshot and so is always between the lines; the
    # snapshot is faked below the batch line to keep the bound of one rescan from
    # depending on that.
    world.add("a", "interactive", kind="resume_input")
    seed_slots(dispatcher.manager, (900, "a"), (901, "a"))
    below = DispatchCapSnapshot(
        policy=BAND, active_ids=frozenset({900}), occupancy={"a": 1}
    )
    monkeypatch.setattr(transport, "build_dispatch_cap_snapshot", lambda: below)
    real_find = transport._find_command_candidate
    scans: list[int] = []

    def spy(*args, **kwargs):
        scans.append(len(scans))
        assert len(scans) <= 2, "a third selection was started"
        return real_find(*args, **kwargs)

    monkeypatch.setattr(transport, "_find_command_candidate", spy)
    held = slot_state(dispatcher.manager)

    assert await transport.dispatch_one_task_command(dispatcher.executor) is False

    assert len(scans) == 2
    assert slot_state(dispatcher.manager) == held
    assert dispatcher.ensured == []


@pytest.mark.parametrize(
    ("switch", "directed", "listing", "require"),
    [("true", None, {"team:3": 11, "team:4": 12}, True), ("true", 11, None, False)]
    + [("false", None, None, False)],
    ids=["on", "directed", "off"],
)
async def test_dispatch_passes_the_listing_and_the_listing_requirement(
    switch, directed, listing, require, dispatcher, monkeypatch
):
    monkeypatch.setenv("XAGENT_DISPATCHER_FAIR_SHARE", switch)
    monkeypatch.setattr(transport, "_fair_share_memo", None)
    candidate = transport.CommandCandidate(11, 5, KINDS.START, True, "team:3", listing)
    monkeypatch.setattr(
        transport, "_find_command_candidate", lambda *_a, **_k: candidate
    )
    seen: list[dict] = []

    def reserve(task_id, **kwargs):
        seen.append(kwargs)
        return None

    monkeypatch.setattr(dispatcher.manager, "reserve_dispatch_slot", reserve)

    assert (
        await transport.dispatch_one_task_command(
            dispatcher.executor, command_db_id=directed
        )
        is False
    )

    assert len(seen) == 1
    assert seen[0]["require_listing"] is require
    assert seen[0]["queued_tenants"] == listing


async def test_a_cap_refusal_ends_the_attempt_without_a_miss(
    clock, dispatcher, monkeypatch
):
    seed_slots(dispatcher.manager, (1, "a"), (2, "b"))
    listing = {"a": 10, "b": 20}
    scans: list[int] = []

    def find(*_args, **_kwargs):
        scans.append(1)
        return transport.CommandCandidate(30, 3, KINDS.START, True, "a", listing)

    monkeypatch.setattr(transport, "_find_command_candidate", find)
    before = slot_state(dispatcher.manager)

    assert await transport.dispatch_one_task_command(dispatcher.executor) is False

    assert len(scans) == 1
    assert slot_state(dispatcher.manager) == before
    assert 3 not in dispatcher.manager._slot_keys
    assert transport._fair_state.skips == {} and transport._fair_state.misses == 0
    assert dispatcher.ensured == []
