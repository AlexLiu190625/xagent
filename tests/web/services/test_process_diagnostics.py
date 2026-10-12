"""Per-process runtime diagnostics line and the execution counts behind it."""

from __future__ import annotations

import asyncio
import logging
import os

import pytest

from xagent.web.services import process_diagnostics
from xagent.web.services.task_execution import BackgroundTaskManager

LOGGER = process_diagnostics.__name__


@pytest.fixture
def manager(monkeypatch):
    fresh = BackgroundTaskManager()
    monkeypatch.setattr(
        "xagent.web.services.task_execution.background_task_manager", fresh
    )
    return fresh


def _counts(manager: BackgroundTaskManager) -> tuple[int, int, int, int]:
    return manager.drain_window_counts()


async def _settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
async def test_flush_logs_one_line_per_window(manager, monkeypatch, caplog, busy):
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "combined")
    pending = asyncio.create_task(asyncio.sleep(3600))
    done = asyncio.create_task(asyncio.sleep(0))
    await done
    if busy:
        manager.register_task(1, pending)
        manager.register_task(2, done)
        await _settle()
    clock = _FakeClock()
    flushes = 0

    async def fake_sleep(seconds: float) -> None:
        nonlocal flushes
        clock.now += seconds
        if flushes == 1:
            raise asyncio.CancelledError
        flushes += 1

    caplog.set_level(logging.INFO, logger=LOGGER)
    with pytest.raises(asyncio.CancelledError):
        await process_diagnostics.run_process_diagnostics(
            flush_interval_s=1.0, sample_interval_s=1.0, clock=clock, sleep=fake_sleep
        )
    pending.cancel()

    lines = [r for r in caplog.records if r.name == LOGGER]
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    message = lines[0].getMessage()
    assert message.startswith("runtime diagnostics role=combined ")
    assert f" pid={os.getpid()} window_s=1.0 " in message
    if busy:
        assert " inflight=1 " in message
        assert " started_1m=2 finished_1m=1 ok=1 failed=0 cancelled=0 " in message
    else:
        assert " inflight=0 " in message
        assert " started_1m=0 finished_1m=0 " in message
    assert " gate_held_s=0.0 " in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "bucket"),
    [
        ("ok", "ok"),
        ("raised", "failed"),
        ("flagged", "failed"),
        ("cancelled", "cancelled"),
        ("flagged_and_cancelled", "cancelled"),
        ("flag_raises", "failed"),
    ],
)
async def test_finish_outcome_bucket(manager, outcome, bucket):
    handled_failure = outcome.startswith("flagged")
    release = asyncio.Event()

    async def run() -> None:
        await release.wait()
        if outcome == "raised":
            raise RuntimeError("boom")

    def flag() -> bool:
        if outcome == "flag_raises":
            raise RuntimeError("flag broke")
        return handled_failure

    task = asyncio.create_task(run())
    manager.register_task(1, task, failure_flag=flag)
    await asyncio.sleep(0)
    if outcome.endswith("cancelled"):
        task.cancel()
    release.set()
    await asyncio.gather(task, return_exceptions=True)
    await _settle()

    started, ok, failed, cancelled = _counts(manager)
    assert started == 1
    assert {"ok": ok, "failed": failed, "cancelled": cancelled} == {
        name: int(name == bucket) for name in ("ok", "failed", "cancelled")
    }


@pytest.mark.asyncio
async def test_drain_resets_and_shutdown_keeps_counts(manager):
    task = asyncio.create_task(asyncio.sleep(3600))
    manager.register_task(1, task)
    await manager.shutdown()
    await _settle()

    assert _counts(manager) == (1, 0, 0, 1)
    assert _counts(manager) == (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_new_lifespan_starts_from_zero(manager):
    task = asyncio.create_task(asyncio.sleep(0))
    manager.register_task(1, task)
    await task
    await _settle()

    manager.start_accepting()

    assert _counts(manager) == (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_same_task_registered_twice_counted_once(manager):
    task = asyncio.create_task(asyncio.sleep(0))
    manager.register_task(1, task)
    manager.register_task(1, task)
    await task
    await _settle()

    assert _counts(manager) == (1, 1, 0, 0)


@pytest.mark.asyncio
async def test_replaced_registration_still_counts_inflight(manager):
    older = asyncio.create_task(asyncio.sleep(3600))
    newer = asyncio.create_task(asyncio.sleep(3600))
    manager.register_task(1, older)
    manager.register_task(1, newer)

    assert manager.inflight_counts() == (2, 0)
    for task in (older, newer):
        task.cancel()
    await asyncio.gather(older, newer, return_exceptions=True)


@pytest.mark.asyncio
async def test_promoted_resume_counted_and_fields_disjoint(manager):
    coordinator = asyncio.create_task(asyncio.sleep(3600))
    assert manager.reserve_resume(1)
    manager.register_reserved_resume(1, coordinator, run_id="run-1")

    assert manager.inflight_counts() == (0, 1)
    assert _counts(manager) == (0, 0, 0, 0)

    manager.promote_resume_task(1, coordinator)

    assert manager.inflight_counts() == (1, 0)
    coordinator.cancel()
    await asyncio.gather(coordinator, return_exceptions=True)
    await _settle()
    assert _counts(manager) == (1, 0, 0, 1)


@pytest.mark.asyncio
async def test_registration_rejected_while_shutting_down_is_not_counted(manager):
    await manager.shutdown()
    task = asyncio.create_task(asyncio.sleep(3600))

    with pytest.raises(RuntimeError, match="shutting down"):
        manager.register_task(1, task)
    await asyncio.gather(task, return_exceptions=True)
    await _settle()

    assert _counts(manager) == (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_lag_sampler_counts_overrun(manager, monkeypatch):
    clock = _FakeClock()
    sleeps = 0

    async def fake_sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 3:
            raise asyncio.CancelledError
        # The second sleep overruns by 0.8 s.
        clock.now += seconds + (0.8 if sleeps == 2 else 0.0)

    monkeypatch.setattr(process_diagnostics, "_window", process_diagnostics._Window())
    with pytest.raises(asyncio.CancelledError):
        await process_diagnostics.run_process_diagnostics(
            flush_interval_s=3600.0,
            sample_interval_s=1.0,
            clock=clock,
            sleep=fake_sleep,
        )

    counters = process_diagnostics.drain_execution_counters(now=clock.now)
    assert counters.loop_lag_samples == 1
    assert counters.loop_lag_max_ms == pytest.approx(800.0)


@pytest.mark.asyncio
async def test_flush_failure_keeps_loop_running(manager, monkeypatch, caplog):
    clock = _FakeClock()
    real_drain = process_diagnostics.drain_execution_counters
    calls = 0

    def flaky_drain(now=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("flush broke")
        return real_drain(now)

    async def fake_sleep(seconds: float) -> None:
        if calls == 2:
            raise asyncio.CancelledError
        clock.now += seconds

    monkeypatch.setattr(process_diagnostics, "drain_execution_counters", flaky_drain)
    caplog.set_level(logging.INFO, logger=LOGGER)
    with pytest.raises(asyncio.CancelledError):
        await process_diagnostics.run_process_diagnostics(
            flush_interval_s=1.0, sample_interval_s=1.0, clock=clock, sleep=fake_sleep
        )

    records = [r for r in caplog.records if r.name == LOGGER]
    assert [r.levelno for r in records] == [logging.ERROR, logging.INFO]
    assert records[1].getMessage().startswith("runtime diagnostics ")


@pytest.mark.asyncio
async def test_bad_role_still_starts_and_stop_logs_final_line(
    manager, monkeypatch, caplog
):
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "nonsense")
    caplog.set_level(logging.INFO, logger=LOGGER)

    task = process_diagnostics.start_process_diagnostics()
    assert task is not None and not task.done()
    execution = asyncio.create_task(asyncio.sleep(3600))
    manager.register_task(1, execution)
    await manager.shutdown()
    await process_diagnostics.stop_process_diagnostics(task)

    assert task.cancelled()
    lines = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert len(lines) == 1
    assert lines[0].startswith("runtime diagnostics role=unknown ")
    assert " started_1m=1 finished_1m=1 ok=0 failed=0 cancelled=1 " in lines[0]


@pytest.mark.asyncio
async def test_stop_keeps_the_callers_own_cancellation(manager):
    task = process_diagnostics.start_process_diagnostics()
    assert task is not None
    gate = asyncio.Event()

    async def caller() -> None:
        await gate.wait()
        await process_diagnostics.stop_process_diagnostics(task)

    outer = asyncio.create_task(caller())
    await asyncio.sleep(0)
    gate.set()
    await asyncio.sleep(0)
    outer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await outer


@pytest.mark.asyncio
async def test_app_lifespan_starts_one_diagnostics_task_and_stops_it(manager):
    from fastapi import FastAPI

    from xagent.web import app as app_module

    test_app = FastAPI()
    app_module.start_app_process_diagnostics(test_app)
    first = test_app.state.process_diagnostics_task
    app_module.start_app_process_diagnostics(test_app)

    assert first is not None and not first.done()
    assert test_app.state.process_diagnostics_task is first

    await app_module.stop_app_process_diagnostics(test_app)
    assert test_app.state.process_diagnostics_task is None
    assert first.cancelled()
