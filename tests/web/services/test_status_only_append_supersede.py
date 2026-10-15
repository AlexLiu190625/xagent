"""A status-only caller's append supersedes the automatic resume its task waits for.

In shared execution an append is accepted by staging a START at the version
the acceptance read; an automatic RESUME queued for the same paused run moves
that version first and the START would be rejected, losing the message.
``_accept_turn_no_commit`` therefore replaces a pending automatic resume of a
status-only (SDK) task in the acceptance transaction. Every other source, and
combined execution, accepts exactly as before. Runs on SQLite and, when
``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL.
"""

from __future__ import annotations

import ast
import pathlib
import uuid
from typing import Any
from unittest.mock import Mock

import pytest
import sqlalchemy as sa

from tests.web.services.coordinator_command_shared import claim_for_owner
from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.test_channel_reply_hold import (
    CID,
    PV,
    RUN,
    _recovery_row,
)
from tests.web.services.test_channel_reply_hold import factory as factory_fixture
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import get_engine
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import (
    task_auto_resume,
    task_coordinator_service,
    task_event_bridge,
    task_orchestrator,
    task_start_consumer,
)
from xagent.web.services.task_auto_resume import (
    check_auto_resume_claim_sync,
    run_auto_resume_tick,
)
from xagent.web.services.task_execution_controller import TaskControlState
from xagent.web.services.task_orchestrator import (
    TaskTurnError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
)

engine = engine_fixture
factory = factory_fixture

SERVICES = pathlib.Path(task_auto_resume.__file__).parent


@pytest.fixture(autouse=True)
def _shared_acceptance(monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_event_bridge, "get_task_event_bridge", lambda: Mock())
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: None
    )


def _seed(
    factory,
    *,
    source: str,
    row: str | None = "scheduled",
    command_status: str | None = None,
    version: int = PV,
    control_state: str = "paused",
    status: TaskStatus = TaskStatus.PAUSED,
    row_version: int = PV,
) -> int:
    """A task of ``source`` resting on ``RUN`` at ``version`` and, unless
    ``row`` is None, its recovery row fenced at ``row_version``."""
    with factory() as db:
        user = User(username=f"append-{uuid.uuid4().hex[:8]}", password_hash="x")
        db.add(user)
        db.flush()
        task = Task(
            user_id=user.id,
            title="Append",
            description="append test",
            execution_mode="balanced",
            source=source,
            status=status,
            control_state=control_state,
            run_id=RUN,
            state_version=version,
        )
        db.add(task)
        db.flush()
        if row is not None:
            _recovery_row(
                db,
                int(task.id),
                RUN,
                row_version,
                state=row,
                interrupted_ago=60,
                command_status=command_status,
            )
        db.commit()
        return int(task.id)


def _accept(factory, tid: int, *, turn_id: str = "append-turn") -> Any:
    """The caller's transaction: accept an append, commit; roll back on error."""
    with factory() as db:
        owner = db.get(Task, tid).user_id
        try:
            prepared = TaskTurnOrchestrator.claim_append_turn_no_commit(
                db,
                task_id=tid,
                task_owner_user_id=owner,
                payload=TaskTurnPayload("message M", turn_id=turn_id),
            )
        except BaseException:
            db.rollback()
            raise
        db.commit()
        return prepared


def _snapshot(factory, tid: int) -> dict[str, Any]:
    """Everything the acceptance can change, without ids and timestamps."""
    with factory() as db:
        task = db.get(Task, tid)
        row = db.get(TaskAutoRecovery, tid)
        events = db.scalars(
            sa.select(TaskRecoveryEvent)
            .where(TaskRecoveryEvent.task_id == tid)
            .order_by(TaskRecoveryEvent.id)
        ).all()
        commands = db.scalars(
            sa.select(TaskExecutionCommand)
            .where(TaskExecutionCommand.task_id == tid)
            .order_by(TaskExecutionCommand.id)
        ).all()
        messages = db.scalars(
            sa.select(TaskChatMessage.turn_id).where(TaskChatMessage.task_id == tid)
        ).all()
        return {
            "task": (
                task.status,
                task.control_state,
                task.run_id if task.run_id == RUN else "<new run>",
                int(task.state_version),
            ),
            "row": None
            if row is None
            else (
                row.state,
                row.state_detail,
                int(row.paused_state_version),
                row.last_command_id,
            ),
            "events": [(e.event, e.detail) for e in events],
            "commands": [
                (c.kind, c.status, c.command_id if c.kind == "resume" else None)
                for c in commands
            ],
            "messages": list(messages),
        }


def _count_statements(factory, tid: int, fn) -> int:
    statements: list[str] = []

    def count(_conn, _cursor, statement, *_args):
        statements.append(statement)

    target = get_engine()
    sa.event.listen(target, "before_cursor_execute", count)
    try:
        fn()
    finally:
        sa.event.remove(target, "before_cursor_execute", count)
    return len(statements)


def _hand_off_start(factory, tid: int, command_db_id: int) -> Any:
    """Claim the START as the task's owner and run the consumer's handoff."""
    with factory() as db:
        lease = task_coordinator_service.acquire_task_lease_no_commit(
            db, tid, runner_id="worker-1"
        )
        assert lease is not None
        db.commit()
        command = claim_for_owner(db, lease, command_db_id)
    return task_start_consumer._commit_handoff(command, lease)


PENDING_ROWS = {
    "scheduled": dict(row="scheduled"),
    "dispatched_pending": dict(row="dispatched", command_status="pending"),
    "dispatched_rolled_back": dict(
        row="dispatched", command_status="processing", version=PV + 2
    ),
}


# --------------------------------------------------------------------------
# A pending automatic resume is replaced
# --------------------------------------------------------------------------


@pytest.mark.parametrize("row_state", sorted(PENDING_ROWS))
def test_append_supersedes_queued_auto_resume(factory, row_state):
    seed = PENDING_ROWS[row_state]
    tid = _seed(factory, source="sdk", **seed)
    version = seed.get("version", PV)

    prepared = _accept(factory, tid)

    after = _snapshot(factory, tid)
    assert after["task"] == (TaskStatus.PAUSED, "paused", RUN, version + 1)
    assert after["row"][:3] == ("manual", "new_turn", version + 1)
    # The event stays ``stale``, as a channel turn's supersede writes it, so
    # one timeline query reads both; the row says why it rests in ``manual``.
    superseded = [d for e, d in after["events"] if e == "stale"]
    assert len(superseded) == 1
    assert (
        superseded[0].items()
        >= {
            "at": "sdk_append",
            "state": "manual",
            "state_detail": "new_turn",
            "mismatch": "new_turn",
        }.items()
    )
    assert after["messages"] == ["append-turn"]
    start = [c for c in after["commands"] if c[0] == "start"]
    assert start == [("start", "pending", None)]
    # The START is staged at the version that moved, against the same run.
    assert prepared.command_payload["state_version"] == version + 1
    assert prepared.command_payload["expected_run_id"] == RUN

    if seed.get("command_status"):
        # The RESUME already queued is skipped by its claim guard.
        decision = check_auto_resume_claim_sync(
            task_id=tid,
            command_id=CID if version == PV else after["row"][3],
            auto_resume={"expected_run_id": RUN, "expected_state_version": PV},
            status=TaskStatus.PAUSED,
            control_state=TaskControlState.PAUSED.value,
            run_id=RUN,
            state_version=version + 1,
            attempt_count=1,
        )
        assert decision.proceed is False
    else:
        # Nothing is left for the sweeper to dispatch.
        assert run_auto_resume_tick().dispatched == 0
        assert [
            c for c in _snapshot(factory, tid)["commands"] if c[0] == "resume"
        ] == []


@pytest.mark.parametrize("row_state", sorted(PENDING_ROWS))
def test_superseded_resume_cannot_reject_the_start(factory, row_state):
    """Message M runs: the queued RESUME, fenced on the version it admitted
    from, can no longer move the task, so the START is accepted by the
    consumer and the task moves to a new run."""
    seed = PENDING_ROWS[row_state]
    tid = _seed(factory, source="sdk", **seed)
    version = seed.get("version", PV)
    prepared = _accept(factory, tid)

    with factory() as db:
        moved = db.execute(
            sa.update(Task)
            .where(
                Task.id == tid,
                Task.state_version == version,
                Task.control_state == TaskControlState.PAUSED.value,
            )
            .values(
                control_state=TaskControlState.RESUME_REQUESTED.value,
                state_version=Task.state_version + 1,
            )
        ).rowcount
        db.commit()
    assert moved == 0
    # The guard skips the RESUME, which ends and lets the START be claimed.
    with factory() as db:
        db.execute(
            sa.update(TaskExecutionCommand)
            .where(
                TaskExecutionCommand.task_id == tid,
                TaskExecutionCommand.kind == "resume",
            )
            .values(status="completed")
        )
        db.commit()

    outcome = _hand_off_start(factory, tid, prepared.command_db_id)
    assert isinstance(outcome, task_start_consumer._StartHandoff)
    assert outcome.start.message == "message M"
    with factory() as db:
        task = db.get(Task, tid)
        assert (task.status, task.run_id) == (TaskStatus.RUNNING, outcome.start.run_id)
        assert task.run_id != RUN


# --------------------------------------------------------------------------
# Scope: only a pending resume of a status-only source, in shared mode
# --------------------------------------------------------------------------

_SCOPE_ROWS = {
    "none": dict(row=None),
    "scheduled": dict(row="scheduled"),
    "dispatched": dict(row="dispatched", command_status="pending"),
    "exhausted": dict(row="exhausted"),
    # A user pause moved the task three versions past the row's fence.
    "scheduled_at_older_fence": dict(row="scheduled", version=PV + 3),
}


@pytest.mark.parametrize("mode", ["shared", "combined"])
@pytest.mark.parametrize("row_state", sorted(_SCOPE_ROWS))
@pytest.mark.parametrize("source", ["sdk", "a2a", "external"])
def test_supersede_scope(factory, monkeypatch, source, row_state, mode):
    if mode == "combined":
        monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    writes = (
        source == "sdk"
        and mode == "shared"
        and row_state
        in {
            "scheduled",
            "dispatched",
        }
    )

    def accept_with_count() -> tuple[int, dict[str, Any], dict[str, Any]]:
        tid = _seed(factory, source=source, **_SCOPE_ROWS[row_state])
        before = _snapshot(factory, tid)
        used = _count_statements(factory, tid, lambda: _accept(factory, tid))
        return used, before, _snapshot(factory, tid)

    # The same acceptance with the supersede switched off is the baseline.
    with monkeypatch.context() as m:
        m.setattr(
            task_orchestrator, "_lock_status_only_append_no_commit", lambda *_a: None
        )
        baseline_count, baseline_before, baseline_after = accept_with_count()
    count, before, after = accept_with_count()

    if writes:
        assert after["row"][:2] == ("manual", "new_turn")
        assert after["task"][3] == before["task"][3] + 1
        assert after != baseline_after
        return
    assert after == baseline_after
    if source != "sdk":
        # Not the status-only caller: the very statements main issues.
        assert count == baseline_count


# --------------------------------------------------------------------------
# Busy and zero-row paths leave nothing behind
# --------------------------------------------------------------------------


def _zero_rowcount_on(monkeypatch, call: int) -> None:
    real = task_auto_resume._rowcount
    seen = {"n": 0}

    def rowcount(result):
        seen["n"] += 1
        return 0 if seen["n"] == call else real(result)

    monkeypatch.setattr(task_auto_resume, "_rowcount", rowcount)


@pytest.mark.parametrize(
    "case",
    [
        "resume_requested_with_live_resume",
        "version_cas_loses",
        "refence_loses",
        "give_up_cas_loses",
        "stranded_hand_off",
    ],
)
def test_append_busy(factory, monkeypatch, case):
    if case in {"resume_requested_with_live_resume", "stranded_hand_off"}:
        tid = _seed(
            factory,
            source="sdk",
            row="dispatched",
            # A hand-off with no RESUME left to finish is stranded, not busy.
            command_status=(
                "processing"
                if case == "resume_requested_with_live_resume"
                else "completed"
            ),
            version=PV + 1,
            control_state="resume_requested",
        )
    else:
        tid = _seed(factory, source="sdk")
        _zero_rowcount_on(
            monkeypatch,
            {"version_cas_loses": 1, "refence_loses": 2, "give_up_cas_loses": 3}[case],
        )
    before = _snapshot(factory, tid)

    if case == "stranded_hand_off":
        _accept(factory, tid)
        after = _snapshot(factory, tid)
        assert after["messages"] == ["append-turn"]
        assert after["row"][:2] == ("dispatched", None)
        return
    with pytest.raises(TaskTurnError, match="busy"):
        _accept(factory, tid)

    assert _snapshot(factory, tid) == before


# --------------------------------------------------------------------------
# Writers and exits stay enumerable
# --------------------------------------------------------------------------


def test_give_up_call_sites():
    source = (SERVICES / "task_auto_resume.py").read_text()
    # The definition plus one call per exit.
    assert source.count("_give_up_no_commit(") - 1 == 15


def _is_state_writer_call(node: ast.Call) -> bool:
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "values"
        and any(keyword.arg == "state" for keyword in node.keywords)
    ):
        return False
    receiver: ast.AST = node.func.value
    while isinstance(receiver, (ast.Call, ast.Attribute)):
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id == "update"
            and receiver.args
            and isinstance(receiver.args[0], ast.Name)
            and receiver.args[0].id == "TaskAutoRecovery"
        ):
            return True
        receiver = receiver.func if isinstance(receiver, ast.Call) else receiver.value
        if isinstance(receiver, ast.Attribute):
            receiver = receiver.value
    return False


def test_recovery_state_writers():
    writers: set[str] = set()
    for path in sorted(SERVICES.glob("*.py")):
        for function in ast.walk(ast.parse(path.read_text())):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                direct = isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Attribute)
                    and target.attr == "state"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "row"
                    for target in node.targets
                )
                if direct or (
                    isinstance(node, ast.Call) and _is_state_writer_call(node)
                ):
                    writers.add(function.name)
    assert writers == {
        "_give_up_no_commit",
        "_process_due_candidate_no_commit",
        "_reschedule_dispatched_no_commit",
        "record_interruption_no_commit",
    }
