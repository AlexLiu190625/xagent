"""Tests for src/xagent/web/jobs/trigger_tasks.py.

PR #1060 review, F4: the stale-preview-run reaper (reap_stale_preview_workforce_runs)
must run from every trigger-scan entrypoint, not just the Celery Beat one
(scan_due_triggers) -- handle_trigger_scan is the BackgroundJob-driven variant of
the same scan, and had no reaper wiring at all before this fix.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from xagent.web.jobs import trigger_tasks
from xagent.web.models.background_job import BackgroundJob, BackgroundJobType
from xagent.web.models.database import get_session_local, init_db
from xagent.web.models.user import User
from xagent.web.services.workforce_runtime import WorkforceRunPauseTarget


def _init_test_db(path: Path):
    init_db(f"sqlite:///{path}")
    return get_session_local()


def _create_user(db, username: str = "trigger-scan-test") -> User:
    user = User(username=username, password_hash="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_handle_trigger_scan_reaps_stale_preview_workforce_runs(
    tmp_path, monkeypatch
) -> None:
    """handle_trigger_scan must call the reaper and dispatch PAUSE for any
    reaped run still RUNNING, the same as scan_due_triggers already does."""
    SessionLocal = _init_test_db(tmp_path / "trigger-scan.db")
    db = SessionLocal()
    user = _create_user(db)

    pause_target = WorkforceRunPauseTarget(run_id=1, task_id=2, actor_user_id=3)
    reap_mock = MagicMock(return_value=[pause_target])
    dispatch_calls: list[tuple[list[WorkforceRunPauseTarget], str]] = []

    async def fake_dispatch(pause_targets, *, reason="archive"):
        dispatch_calls.append((pause_targets, reason))

    monkeypatch.setattr(trigger_tasks, "reap_stale_preview_workforce_runs", reap_mock)
    monkeypatch.setattr(
        trigger_tasks, "pause_workforce_tasks_after_archive", fake_dispatch
    )
    monkeypatch.setattr(trigger_tasks, "requeue_stale_background_jobs", lambda _db: [])
    monkeypatch.setattr(trigger_tasks, "scan_due_scheduled_triggers", lambda _db: [])

    job = BackgroundJob(
        user_id=int(user.id),
        job_type=BackgroundJobType.TRIGGER_SCAN.value,
        payload={},
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    result = trigger_tasks.handle_trigger_scan(db, job)

    reap_mock.assert_called_once_with(db)
    assert dispatch_calls == [([pause_target], "preview-reap")]
    assert result["reaped_preview_run_pause_dispatches"] == 1


def test_handle_trigger_scan_skips_dispatch_when_nothing_reaped(
    tmp_path, monkeypatch
) -> None:
    SessionLocal = _init_test_db(tmp_path / "trigger-scan-empty.db")
    db = SessionLocal()
    user = _create_user(db)

    reap_mock = MagicMock(return_value=[])
    dispatch_mock = MagicMock()

    monkeypatch.setattr(trigger_tasks, "reap_stale_preview_workforce_runs", reap_mock)
    monkeypatch.setattr(
        trigger_tasks, "pause_workforce_tasks_after_archive", dispatch_mock
    )
    monkeypatch.setattr(trigger_tasks, "requeue_stale_background_jobs", lambda _db: [])
    monkeypatch.setattr(trigger_tasks, "scan_due_scheduled_triggers", lambda _db: [])

    job = BackgroundJob(
        user_id=int(user.id),
        job_type=BackgroundJobType.TRIGGER_SCAN.value,
        payload={},
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    result = trigger_tasks.handle_trigger_scan(db, job)

    reap_mock.assert_called_once_with(db)
    dispatch_mock.assert_not_called()
    assert result["reaped_preview_run_pause_dispatches"] == 0


def test_handle_trigger_scan_survives_stale_sweep_failure(
    tmp_path, monkeypatch
) -> None:
    """A stale-sweep failure must not stall the rest of the scan tick's work."""
    SessionLocal = _init_test_db(tmp_path / "trigger-scan-sweep-failure.db")
    db = SessionLocal()
    user = _create_user(db)

    def raise_sweep(_db):
        raise RuntimeError("simulated sweep failure")

    scan_calls: list[object] = []

    def fake_scan(_db):
        scan_calls.append(_db)
        return []

    monkeypatch.setattr(trigger_tasks, "requeue_stale_background_jobs", raise_sweep)
    monkeypatch.setattr(trigger_tasks, "scan_due_scheduled_triggers", fake_scan)
    monkeypatch.setattr(
        trigger_tasks, "reap_stale_preview_workforce_runs", MagicMock(return_value=[])
    )

    job = BackgroundJob(
        user_id=int(user.id),
        job_type=BackgroundJobType.TRIGGER_SCAN.value,
        payload={},
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    result = trigger_tasks.handle_trigger_scan(db, job)

    assert result["requeued_stale_jobs"] == 0
    assert scan_calls == [db]


def test_scan_due_triggers_survives_stale_sweep_failure(tmp_path, monkeypatch) -> None:
    """The Celery Beat entrypoint must survive a stale-sweep failure the same
    way the BackgroundJob-driven variant does."""
    _init_test_db(tmp_path / "scan-due-triggers-sweep-failure.db")

    def raise_sweep(_db):
        raise RuntimeError("simulated sweep failure")

    scan_calls: list[object] = []

    def fake_scan(_db):
        scan_calls.append(_db)
        return []

    monkeypatch.setattr(trigger_tasks, "requeue_stale_background_jobs", raise_sweep)
    monkeypatch.setattr(trigger_tasks, "scan_due_scheduled_triggers", fake_scan)
    monkeypatch.setattr(
        trigger_tasks, "reap_stale_preview_workforce_runs", MagicMock(return_value=[])
    )

    result = trigger_tasks.scan_due_triggers()

    assert result["requeued_stale_jobs"] == 0
    assert len(scan_calls) == 1
