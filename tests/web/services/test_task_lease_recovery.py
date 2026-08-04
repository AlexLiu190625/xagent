"""Tests for automatic recovery of expired task execution leases."""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Barrier

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Query, Session, sessionmaker

from xagent.core.agent.checkpoint import CHECKPOINT_TYPE
from xagent.web.models.agent import Agent
from xagent.web.models.database import (
    Base,
    get_db,
    get_engine,
    get_session_local,
    init_db,
)
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services import task_lease_recovery, task_lease_service
from xagent.web.services.task_lease_recovery import (
    TASK_LEASE_EXPIRED_ERROR,
    TASK_LEASE_PAUSED_TRIGGER_ERROR,
    recover_expired_task_leases_until_cutoff,
    recover_task_lease_candidate_isolated,
    recover_task_lease_candidate_no_commit,
    run_task_lease_recovery_loop,
)
from xagent.web.services.task_lease_service import (
    TASK_RUN_ID_TRACE_FIELD,
    CheckpointRecoveryVerdict,
    TaskLeaseRecoveryCandidate,
    get_expired_task_lease_candidates,
    resolve_checkpoint_recovery,
    utc_now,
)

ANCHOR_FK_NAME = "fk_tasks_last_checkpoint_trace_event_id"


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'task-lease-recovery.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        # FK enforcement is on for this engine's connections (see
        # apply_sqlite_concurrency_pragmas), and the checkpoint pointer's
        # FK forms a cycle with trace_events.task_id -> tasks.id. A test
        # that leaves a real cross-reference between the two tables (a
        # populated last_checkpoint_trace_event_id) makes drop_all's table
        # order irrelevant -- no order satisfies both directions of a
        # populated cycle under enforcement. Dropping the tables is the
        # only goal here, not preserving referential integrity of data
        # about to be discarded, so enforcement is turned off for this one
        # connection rather than for the fixture's tests.
        engine = get_engine()
        with engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
            Base.metadata.drop_all(bind=conn)


def _create_user(db, *, suffix: str) -> User:
    user = User(
        username=f"lease-recovery-{suffix}",
        password_hash="hash",
        is_admin=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_expired_task(
    db,
    *,
    user_id: int,
    suffix: str,
    with_checkpoint: bool = False,
) -> Task:
    task = Task(
        user_id=user_id,
        title=f"Expired lease {suffix}",
        description="lease recovery test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
        runner_id=f"dead-runner-{suffix}",
        run_id=f"run-{suffix}",
        lease_expires_at=utc_now() - timedelta(seconds=5),
        last_heartbeat_at=utc_now() - timedelta(seconds=10),
        state_version=3,
        control_state="running",
        output="stale output",
    )
    db.add(task)
    db.flush()
    if with_checkpoint:
        event_id = f"checkpoint-{suffix}"
        db.add(
            TraceEvent(
                task_id=task.id,
                event_id=event_id,
                event_type="system_update_general",
                timestamp=utc_now(),
                step_id=None,
                parent_event_id=None,
                data={
                    "checkpoint_type": CHECKPOINT_TYPE,
                    "snapshot": {"type": "checkpoint"},
                    TASK_RUN_ID_TRACE_FIELD: task.run_id,
                },
            )
        )
        task.last_checkpoint_event_id = event_id
    db.commit()
    db.refresh(task)
    return task


def _recover_expired_task(db, task: Task) -> TaskStatus | None:
    candidate = get_expired_task_lease_candidates(
        db,
        cutoff=utc_now(),
        limit=1,
    )[0]
    db.rollback()
    return recover_task_lease_candidate_isolated(
        candidate,
        recovered_at=utc_now(),
    )


def _candidate_for_task(task: Task) -> TaskLeaseRecoveryCandidate:
    """Build the recovery candidate snapshot resolve_checkpoint_recovery
    consumes, directly from a task's current column values -- for tests
    that exercise checkpoint pointer resolution without going through the
    full expired-lease scan query.
    """
    return TaskLeaseRecoveryCandidate(
        task_id=int(task.id),
        runner_id=task.runner_id,
        run_id=task.run_id,
        lease_expires_at=task.lease_expires_at,
        state_version=int(task.state_version or 0),
        last_checkpoint_event_id=task.last_checkpoint_event_id,
        last_checkpoint_trace_event_id=task.last_checkpoint_trace_event_id,
    )


@pytest.fixture()
def sqlite_no_anchor_fk_session(tmp_path):
    """A SQLite database shaped like the checkpoint-anchor migration's ADD
    COLUMN path applied to a table that already existed: full schema via
    create_all, then ``tasks`` rebuilt without the checkpoint pointer's FK
    clause. Alembic's SQLite batch mode cannot add that FK without a full
    table rebuild, so this asymmetric shape -- not the fresh create_all
    schema every other fixture in this module uses -- is the only one on
    which ``last_checkpoint_trace_event_id`` can point at a row that no
    longer exists. Mirrors
    tests/web/services/test_task_deletion_checkpoint_pointer.py's
    sqlite_upgraded_session for the same reason, kept as a separate
    module-local copy rather than a shared import so this module's
    fixtures stay self-contained.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'lease-recovery-no-fk.db'}")
    for constraint in Base.metadata.tables["tasks"].constraints:
        if getattr(constraint, "name", None) == ANCHOR_FK_NAME:
            constraint._create_rule = None
            break
    else:
        raise AssertionError(f"{ANCHOR_FK_NAME} constraint not found on tasks")
    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        original_sql = conn.execute(
            text(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
            )
        ).scalar_one()
        stripped_sql, count = re.subn(
            r",\s*CONSTRAINT "
            + re.escape(ANCHOR_FK_NAME)
            + r" FOREIGN KEY\([^)]*\) REFERENCES [^,]*",
            "",
            original_sql,
        )
        assert count == 1, (
            f"expected exactly one {ANCHOR_FK_NAME} clause in the tasks DDL"
        )
        conn.execute(text("ALTER TABLE tasks RENAME TO tasks_old"))
        conn.execute(text(stripped_sql))
        conn.execute(text("INSERT INTO tasks SELECT * FROM tasks_old"))
        conn.execute(text("DROP TABLE tasks_old"))

    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


def _attach_workforce_and_trigger(db, *, task: Task, user: User) -> tuple:
    manager = Agent(user_id=user.id, name="lease recovery manager")
    db.add(manager)
    db.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name=f"Recovery workforce {task.id}",
        manager_agent_id=manager.id,
        status="published",
    )
    db.add(workforce)
    db.flush()
    workforce_run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="running",
        snapshot={},
    )
    db.add(workforce_run)
    db.flush()
    task.agent_config = {"workforce_run_id": int(workforce_run.id)}

    trigger = AgentTrigger(
        user_id=user.id,
        workforce_id=workforce.id,
        type=TriggerType.SCHEDULED.value,
        name=f"Recovery trigger {task.id}",
        config={},
    )
    db.add(trigger)
    db.flush()
    trigger_run = TriggerRun(
        trigger_id=trigger.id,
        task_id=task.id,
        status=TriggerRunStatus.RUNNING.value,
        idempotency_key=f"lease-recovery-{task.id}",
    )
    db.add(trigger_run)
    db.commit()
    return workforce_run, trigger_run


def test_expired_lease_with_checkpoint_pauses_all_lifecycle_projections(
    db_session,
) -> None:
    user = _create_user(db_session, suffix="checkpoint")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="checkpoint",
        with_checkpoint=True,
    )
    workforce_run, trigger_run = _attach_workforce_and_trigger(
        db_session,
        task=task,
        user=user,
    )

    assert _recover_expired_task(db_session, task) == TaskStatus.PAUSED

    db_session.refresh(task)
    db_session.refresh(workforce_run)
    db_session.refresh(trigger_run)
    assert task.status == TaskStatus.PAUSED
    assert task.control_state == "paused"
    assert task.state_version == 4
    assert task.runner_id is None
    assert task.lease_expires_at is None
    assert task.run_id == "run-checkpoint"
    assert task.error_message is None
    assert workforce_run.status == "paused"
    assert workforce_run.completed_at is None
    assert trigger_run.status == TriggerRunStatus.FAILED.value
    assert trigger_run.error_message == TASK_LEASE_PAUSED_TRIGGER_ERROR
    assert trigger_run.finished_at is not None


def test_expired_lease_without_checkpoint_fails_and_clears_stale_output(
    db_session,
) -> None:
    user = _create_user(db_session, suffix="failed")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="failed",
    )
    workforce_run, trigger_run = _attach_workforce_and_trigger(
        db_session,
        task=task,
        user=user,
    )

    assert _recover_expired_task(db_session, task) == TaskStatus.FAILED

    db_session.refresh(task)
    db_session.refresh(workforce_run)
    db_session.refresh(trigger_run)
    assert task.status == TaskStatus.FAILED
    assert task.control_state == "failed"
    assert task.state_version == 4
    assert task.runner_id is None
    assert task.lease_expires_at is None
    assert task.output is None
    assert task.error_message == TASK_LEASE_EXPIRED_ERROR
    assert workforce_run.status == "failed"
    assert workforce_run.completed_at is not None
    assert trigger_run.status == TriggerRunStatus.FAILED.value
    assert trigger_run.error_message == TASK_LEASE_EXPIRED_ERROR


@pytest.mark.parametrize("checkpoint_run_id", [None, "previous-run"])
def test_recovery_rejects_checkpoint_without_current_run_provenance(
    db_session,
    checkpoint_run_id: str | None,
) -> None:
    user = _create_user(db_session, suffix=f"provenance-{checkpoint_run_id}")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix=f"provenance-{checkpoint_run_id}",
        with_checkpoint=True,
    )
    checkpoint = (
        db_session.query(TraceEvent)
        .filter(TraceEvent.event_id == task.last_checkpoint_event_id)
        .one()
    )
    data = dict(checkpoint.data)
    if checkpoint_run_id is None:
        data.pop(TASK_RUN_ID_TRACE_FIELD, None)
    else:
        data[TASK_RUN_ID_TRACE_FIELD] = checkpoint_run_id
    checkpoint.data = data
    # Also anchor the PK pointer at the same row: the PK-first resolution
    # path must reach the identical FAILED verdict the legacy scan does,
    # not just the legacy path exercised when this column is unset.
    task.last_checkpoint_trace_event_id = checkpoint.id
    db_session.commit()

    assert _recover_expired_task(db_session, task) == TaskStatus.FAILED

    db_session.expire_all()
    persisted = db_session.get(Task, int(task.id))
    assert persisted.status == TaskStatus.FAILED
    assert persisted.output is None


def test_exact_checkpoint_pointer_is_not_limited_to_latest_one_hundred_events(
    db_session,
) -> None:
    user = _create_user(db_session, suffix="exact-pointer")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="exact-pointer",
        with_checkpoint=True,
    )
    for index in range(110):
        db_session.add(
            TraceEvent(
                task_id=task.id,
                event_id=f"noise-{index}",
                event_type="system_update_general",
                timestamp=utc_now() + timedelta(microseconds=index + 1),
                step_id=None,
                parent_event_id=None,
                data={"message": f"noise-{index}"},
            )
        )
    db_session.commit()

    assert _recover_expired_task(db_session, task) == TaskStatus.PAUSED


def test_pk_anchor_resolves_without_a_usable_legacy_pointer(db_session) -> None:
    """The exact-row pointer is resolved before the legacy scan runs at
    all -- proven here by leaving the legacy event_id column unset, a
    state a legacy-only scan could never resolve to anything.
    """
    user = _create_user(db_session, suffix="pk-priority")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="pk-priority",
    )
    checkpoint = TraceEvent(
        task_id=task.id,
        event_id="pk-priority-checkpoint",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: task.run_id,
        },
    )
    db_session.add(checkpoint)
    db_session.flush()
    task.last_checkpoint_trace_event_id = checkpoint.id
    assert task.last_checkpoint_event_id is None
    db_session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(db_session, candidate)
        is CheckpointRecoveryVerdict.RECOVERABLE
    )
    assert _recover_expired_task(db_session, task) == TaskStatus.PAUSED


def test_pk_anchor_validation_failure_does_not_fall_back_to_the_legacy_scan(
    db_session,
) -> None:
    """A pointer that resolves to a row failing validation is corruption of
    that exact row, not a cue to go search other rows -- even when the
    legacy event_id column names a different, genuinely valid checkpoint
    that a legacy-only scan would have accepted.
    """
    user = _create_user(db_session, suffix="pk-no-fallback")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="pk-no-fallback",
    )
    invalid_anchor = TraceEvent(
        task_id=task.id,
        build_id="agent_child",  # fails the build_id IS NULL check
        event_id="pk-no-fallback-invalid",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: task.run_id,
        },
    )
    valid_legacy_row = TraceEvent(
        task_id=task.id,
        event_id="pk-no-fallback-valid",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: task.run_id,
        },
    )
    db_session.add_all([invalid_anchor, valid_legacy_row])
    db_session.flush()
    task.last_checkpoint_trace_event_id = invalid_anchor.id
    task.last_checkpoint_event_id = valid_legacy_row.event_id
    db_session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(db_session, candidate)
        is CheckpointRecoveryVerdict.NOT_RECOVERABLE
    )
    assert _recover_expired_task(db_session, task) == TaskStatus.FAILED


def test_dangling_pk_pointer_falls_back_to_the_legacy_scan(
    sqlite_no_anchor_fk_session,
) -> None:
    """A pointer whose row is gone is only reachable on a database upgraded
    without this column's FK (the D1 asymmetry -- see the migration and
    test_task_deletion_checkpoint_pointer.py). It is a compatibility-window
    state, not corruption: recovery must fall back to the legacy scan
    instead of failing the candidate outright.
    """
    session = sqlite_no_anchor_fk_session
    user = User(
        username="lease-recovery-dangling",
        password_hash="hash",
        is_admin=False,
    )
    session.add(user)
    session.flush()
    task = Task(
        user_id=int(user.id),
        title="dangling pointer",
        description="lease recovery test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
        runner_id="dead-runner-dangling",
        run_id="run-dangling",
        lease_expires_at=utc_now() - timedelta(seconds=5),
        last_heartbeat_at=utc_now() - timedelta(seconds=10),
        state_version=3,
        control_state="running",
    )
    session.add(task)
    session.flush()
    legacy_checkpoint = TraceEvent(
        task_id=task.id,
        event_id="dangling-legacy-checkpoint",
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: "run-dangling",
        },
    )
    session.add(legacy_checkpoint)
    session.flush()
    task.last_checkpoint_event_id = legacy_checkpoint.event_id
    # No DB-level FK on this schema form -- this id names no row.
    task.last_checkpoint_trace_event_id = legacy_checkpoint.id + 999
    session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(session, candidate)
        is CheckpointRecoveryVerdict.RECOVERABLE
    )


def test_ambiguous_legacy_checkpoint_skips_the_candidate_for_the_next_sweep(
    db_session,
) -> None:
    """Two trace_events rows share the same legacy event_id within one
    task's root partition -- the row's identity itself cannot be
    determined from that string alone. This is not folded into FAILED:
    the candidate is left untouched (lease and status unchanged, no
    recovery statement executes) so a later sweep gets another chance to
    resolve it, instead of permanently failing a task whose checkpoint may
    in fact be readable once the ambiguity is gone.
    """
    user = _create_user(db_session, suffix="ambiguous")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="ambiguous",
        with_checkpoint=True,
    )
    duplicate_checkpoint = TraceEvent(
        task_id=task.id,
        event_id=task.last_checkpoint_event_id,
        event_type="system_update_general",
        timestamp=utc_now(),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot": {"type": "checkpoint"},
            TASK_RUN_ID_TRACE_FIELD: task.run_id,
        },
    )
    db_session.add(duplicate_checkpoint)
    db_session.commit()

    candidate = _candidate_for_task(task)
    assert (
        resolve_checkpoint_recovery(db_session, candidate)
        is CheckpointRecoveryVerdict.INDETERMINATE
    )
    assert _recover_expired_task(db_session, task) is None

    db_session.expire_all()
    persisted = db_session.get(Task, int(task.id))
    assert persisted.status == TaskStatus.RUNNING
    assert persisted.runner_id is not None
    assert persisted.lease_expires_at is not None

    # Nothing about the skip removed the candidate from the expired-lease
    # scan -- a later sweep still selects it.
    rescanned = get_expired_task_lease_candidates(
        db_session,
        cutoff=utc_now(),
        limit=10,
    )
    assert any(c.task_id == int(task.id) for c in rescanned)


@pytest.mark.parametrize(
    "replacement",
    ["heartbeat", "runner", "run", "state_version", "checkpoint", "checkpoint_pk"],
)
def test_recovery_candidate_cannot_overwrite_newer_task_state(
    db_session,
    replacement: str,
) -> None:
    user = _create_user(db_session, suffix=f"race-{replacement}")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix=f"race-{replacement}",
    )
    candidates = get_expired_task_lease_candidates(
        db_session,
        cutoff=utc_now(),
        limit=10,
    )
    assert len(candidates) == 1

    if replacement == "heartbeat":
        task.lease_expires_at = utc_now() + timedelta(minutes=1)
    elif replacement == "runner":
        task.runner_id = "replacement-runner"
    elif replacement == "run":
        task.run_id = "replacement-run"
    elif replacement == "state_version":
        task.state_version += 1
    elif replacement == "checkpoint":
        task.last_checkpoint_event_id = "new-checkpoint"
    else:
        # checkpoint_pk: only the exact-row pointer moves, the legacy string
        # column is left alone -- a single-column drift the fence's
        # conjunction over both pointer columns must still catch.
        new_event = TraceEvent(
            task_id=task.id,
            event_id="new-checkpoint-pk",
            event_type="system_update_general",
            timestamp=utc_now(),
            step_id=None,
            parent_event_id=None,
            data={
                "checkpoint_type": CHECKPOINT_TYPE,
                "snapshot": {"type": "checkpoint"},
                TASK_RUN_ID_TRACE_FIELD: task.run_id,
            },
        )
        db_session.add(new_event)
        db_session.flush()
        task.last_checkpoint_trace_event_id = new_event.id
    db_session.commit()

    assert (
        recover_task_lease_candidate_isolated(
            candidates[0],
            recovered_at=utc_now(),
        )
        is None
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id is not None


def test_candidate_fence_allows_only_one_recovery(db_session) -> None:
    user = _create_user(db_session, suffix="single-winner")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="single-winner",
    )
    candidate = get_expired_task_lease_candidates(
        db_session,
        cutoff=utc_now(),
        limit=1,
    )[0]

    assert (
        recover_task_lease_candidate_isolated(
            candidate,
            recovered_at=utc_now(),
        )
        == TaskStatus.FAILED
    )
    assert (
        recover_task_lease_candidate_isolated(
            candidate,
            recovered_at=utc_now(),
        )
        is None
    )
    db_session.refresh(task)
    assert task.state_version == 4


def test_two_recovery_workers_have_one_atomic_winner(db_session) -> None:
    user = _create_user(db_session, suffix="concurrent-winner")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="concurrent-winner",
    )
    candidate = get_expired_task_lease_candidates(
        db_session,
        cutoff=utc_now(),
        limit=1,
    )[0]
    db_session.rollback()
    barrier = Barrier(2)

    def recover() -> TaskStatus | None:
        barrier.wait()
        return recover_task_lease_candidate_isolated(
            candidate,
            recovered_at=utc_now(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [
            future.result() for future in [executor.submit(recover) for _ in range(2)]
        ]

    assert results.count(TaskStatus.FAILED) == 1
    assert results.count(None) == 1
    db_session.expire_all()
    assert db_session.get(Task, int(task.id)).state_version == 4


def test_postgresql_candidate_query_partitions_workers_with_skip_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_sql: list[str] = []

    def capture_first(query: Query):
        captured_sql.append(
            str(
                query.limit(1).statement.compile(
                    dialect=postgresql.dialect(),
                    compile_kwargs={"literal_binds": True},
                )
            )
        )
        return None

    monkeypatch.setattr(Query, "first", capture_first)
    db = Session()
    try:
        candidate = task_lease_service.get_next_expired_task_lease_candidate_for_update(
            db,
            cutoff=utc_now(),
            after=None,
        )
    finally:
        db.close()

    assert candidate is None
    assert len(captured_sql) == 1
    normalized_sql = " ".join(captured_sql[0].split())
    assert "ORDER BY tasks.lease_expires_at ASC, tasks.id ASC" in normalized_sql
    assert "LIMIT 1" in normalized_sql
    assert "FOR UPDATE OF tasks SKIP LOCKED" in normalized_sql


def test_postgresql_batch_uses_one_short_locked_transaction_per_candidate(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _create_user(db_session, suffix="postgresql-partition")
    expired_ids = [
        int(
            _create_expired_task(
                db_session,
                user_id=int(user.id),
                suffix=f"postgresql-partition-{index}",
            ).id
        )
        for index in range(3)
    ]
    selected_sessions: list[Session] = []

    def select_next_candidate(
        db: Session,
        *,
        cutoff: datetime,
        after: tuple[datetime, int] | None,
    ):
        selected_sessions.append(db)
        candidates = task_lease_service.get_expired_task_lease_candidates(
            db,
            cutoff=cutoff,
            limit=1,
            after=after,
        )
        return candidates[0] if candidates else None

    def reject_legacy_page_scan(*args, **kwargs):
        raise AssertionError("PostgreSQL recovery must not scan an unlocked page")

    monkeypatch.setattr(
        task_lease_recovery,
        "_use_postgresql_recovery_partitioning",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        task_lease_recovery,
        "get_next_expired_task_lease_candidate_for_update",
        select_next_candidate,
        raising=False,
    )
    monkeypatch.setattr(
        task_lease_recovery,
        "get_expired_task_lease_candidates",
        reject_legacy_page_scan,
    )

    first = task_lease_recovery.recover_expired_task_leases_batch_isolated(
        cutoff=utc_now(),
        batch_size=2,
        after=None,
    )
    second = task_lease_recovery.recover_expired_task_leases_batch_isolated(
        cutoff=utc_now(),
        batch_size=2,
        after=first.next_cursor,
    )

    assert first.scanned == 2
    assert first.recovered == 2
    assert second.scanned == 1
    assert second.recovered == 1
    assert len(selected_sessions) == 4
    assert len({id(db) for db in selected_sessions}) == 4
    db_session.expire_all()
    assert {db_session.get(Task, task_id).status for task_id in expired_ids} == {
        TaskStatus.FAILED
    }


def test_postgresql_batch_does_not_count_a_failed_candidate_transaction(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _create_user(db_session, suffix="postgresql-commit-failure")
    first_task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="postgresql-commit-failure-first",
    )
    second_task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="postgresql-commit-failure-second",
    )
    commit_state = {"must_fail": True}

    class CommitFailingSession(Session):
        def commit(self) -> None:
            if commit_state["must_fail"]:
                commit_state["must_fail"] = False
                raise RuntimeError("simulated commit failure")
            super().commit()

    TestSessionLocal = sessionmaker(
        bind=db_session.get_bind(),
        class_=CommitFailingSession,
        autocommit=False,
        autoflush=False,
    )

    def select_next_candidate(
        db: Session,
        *,
        cutoff: datetime,
        after: tuple[datetime, int] | None,
    ):
        candidates = task_lease_service.get_expired_task_lease_candidates(
            db,
            cutoff=cutoff,
            limit=1,
            after=after,
        )
        return candidates[0] if candidates else None

    monkeypatch.setattr(
        task_lease_recovery,
        "get_next_expired_task_lease_candidate_for_update",
        select_next_candidate,
    )
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        lambda: TestSessionLocal,
    )

    batch = (
        task_lease_recovery._recover_expired_task_leases_batch_with_postgresql_locks(
            cutoff=utc_now(),
            batch_size=2,
            after=None,
        )
    )

    assert batch.scanned == 2
    assert batch.recovered == 1
    db_session.expire_all()
    assert db_session.get(Task, int(first_task.id)).status == TaskStatus.RUNNING
    assert db_session.get(Task, int(second_task.id)).status == TaskStatus.FAILED


def test_projection_failure_rolls_back_the_task_recovery_transaction(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _create_user(db_session, suffix="projection-rollback")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="projection-rollback",
    )

    def fail_projection(*args, **kwargs):
        raise RuntimeError("projection failed")

    monkeypatch.setattr(
        task_lease_recovery,
        "sync_workforce_run_status",
        fail_projection,
    )

    candidate = get_expired_task_lease_candidates(
        db_session,
        cutoff=utc_now(),
        limit=1,
    )[0]
    db_session.rollback()

    with pytest.raises(RuntimeError, match="projection failed"):
        recover_task_lease_candidate_isolated(
            candidate,
            recovered_at=utc_now(),
        )

    db_session.expire_all()
    persisted = db_session.get(Task, int(task.id))
    assert persisted.status == TaskStatus.RUNNING
    assert persisted.runner_id == "dead-runner-projection-rollback"


def test_recovery_core_leaves_commit_to_the_session_owner(db_session) -> None:
    user = _create_user(db_session, suffix="no-commit")
    task = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="no-commit",
    )
    candidate = get_expired_task_lease_candidates(
        db_session,
        cutoff=utc_now(),
        limit=1,
    )[0]
    pending_user = User(
        username="unrelated-pending-user",
        password_hash="hash",
        is_admin=False,
    )
    db_session.add(pending_user)

    assert (
        recover_task_lease_candidate_no_commit(
            db_session,
            candidate,
            recovered_at=utc_now(),
        )
        == TaskStatus.FAILED
    )

    SessionLocal = get_session_local()
    with SessionLocal() as observer:
        assert observer.get(Task, int(task.id)).status == TaskStatus.RUNNING
        assert (
            observer.query(User)
            .filter(User.username == "unrelated-pending-user")
            .first()
            is None
        )

    db_session.rollback()
    db_session.expire_all()
    assert db_session.get(Task, int(task.id)).status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_periodic_recovery_drains_more_than_one_batch_at_fixed_cutoff(
    db_session,
) -> None:
    user = _create_user(db_session, suffix="batch")
    expired_ids = [
        int(
            _create_expired_task(
                db_session,
                user_id=int(user.id),
                suffix=f"batch-{index}",
            ).id
        )
        for index in range(5)
    ]
    live = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="live",
    )
    live.lease_expires_at = utc_now() + timedelta(minutes=1)
    no_expiry = _create_expired_task(
        db_session,
        user_id=int(user.id),
        suffix="no-expiry",
    )
    no_expiry.lease_expires_at = None
    db_session.commit()

    recovered = await recover_expired_task_leases_until_cutoff(
        cutoff=utc_now(),
        batch_size=2,
    )

    assert recovered == 5
    db_session.expire_all()
    statuses = {
        int(task.id): task.status
        for task in db_session.query(Task).filter(Task.id.in_(expired_ids)).all()
    }
    assert statuses == {task_id: TaskStatus.FAILED for task_id in expired_ids}
    assert db_session.get(Task, int(live.id)).status == TaskStatus.RUNNING
    assert db_session.get(Task, int(no_expiry.id)).status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_recovery_loop_survives_pool_timeout_and_waits_for_next_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    async def fake_recover(*, cutoff: datetime, batch_size: int) -> int:
        nonlocal calls
        calls += 1
        assert cutoff.tzinfo is not None
        assert batch_size == 7
        if calls == 1:
            raise SQLAlchemyTimeoutError("pool checkout timed out")
        raise asyncio.CancelledError

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        task_lease_recovery,
        "recover_expired_task_leases_until_cutoff",
        fake_recover,
    )
    monkeypatch.setattr(task_lease_recovery.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await run_task_lease_recovery_loop(
            poll_interval_seconds=11,
            batch_size=7,
        )

    assert calls == 2
    assert sleeps == [11]
