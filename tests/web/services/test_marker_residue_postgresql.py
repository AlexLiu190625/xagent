"""The marker and active-row pairing, checked on PostgreSQL after the commits
of the legacy resume close paths and the resume-abandonment compensation
paths.

When each test samples:

- The online chat close is sampled after ``handle_task_message`` returns.
  The background resume it hands off to is stubbed, so nothing writes to
  the task or the interaction row after the close commits.
- The deferred close is sampled when the agent's resume call is reached,
  which is after the close committed; the task is RUNNING at that moment.
- The A2A close is sampled after ``_update_a2a_resume_input_sync`` commits.

Every test writes the marker and the active row itself and relies on no
production code to set the marker.

The compensation path, ``clear_interaction_marker_if_unpaired``, has three
callers: the deferred resume's restore to the prior status, the A2A
prelease restore and the v1 reply prelease restore. The online chat path
has no compensation path, only the close statement.

None of the three callers takes a lock read first: the first statement
their transaction directs at ``tasks`` is the UPDATE issued by
``release_task_lease_no_commit``, which writes only non-key columns.

Each parameter id of the compensation test is one route by which an
abandoned resume reaches a compensation caller, written out route by route
rather than merged by group.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from tests.web.services.interaction_marker_shared import (
    assert_interaction_marker_invariant,
    assert_marker_residue,
)
from tests.web.services.task_interaction_schema_shared import (
    seed_active_row,
    seed_task_with_run,
)
from tests.web.services.task_lease_shared import (
    live_task_lease as live_task_lease_fixture,
)
from xagent.core.agent.checkpoint import CheckpointUnavailableError
from xagent.core.agent.runner import (
    UserMessageInjectionOutcome,
    UserMessageInjectionRejectedError,
    _record_injection_outcome,
)
from xagent.web.api.websocket import _make_command_reply
from xagent.web.models import database
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.user import User
from xagent.web.services import task_interaction_close
from xagent.web.services.task_command_execution import handle_task_message
from xagent.web.services.task_execution import (
    ResumeReservationOutcome,
    execute_resume_background,
)
from xagent.web.services.task_interaction_close import (
    clear_interaction_marker_if_unpaired,
    close_legacy_resume_interaction_sync,
)
from xagent.web.services.task_lease_service import TaskLease
from xagent.web.services.task_resume import (
    TaskReplyInput,
    TaskResumeNotAcceptedError,
    TaskResumeNotResumableError,
    TaskResumeRetryableError,
    _restore_a2a_resume_prelease_sync,
    _restore_reply_prelease_sync,
    _update_a2a_resume_input_sync,
    resume_a2a_task,
    resume_task_reply,
)

pytestmark = pytest.mark.postgresql

live_task_lease = live_task_lease_fixture

_RUN = "run-abandoned"


@pytest.fixture()
def pg_sessions(monkeypatch) -> Iterator[sessionmaker[Session]]:
    with disposable_database_factory("marker_residue") as make_database:
        engine = make_database("t")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        # Production code resolves its Session factory and engine globally.
        monkeypatch.setattr(database, "_engine", engine)
        monkeypatch.setattr(database, "_SessionLocal", sessions)
        yield sessions


def _seed(
    pg_sessions: sessionmaker[Session],
    *,
    run_id: str,
    status: TaskStatus,
    source: str,
    with_agent: bool = False,
    lease: TaskLease | None = None,
) -> tuple[int, int, int | None, int]:
    """Seed a task on ``run_id`` with marker 1 and one active row on that run.

    Returns ``(task_id, owner_id, agent_id, row_id)``. ``lease`` writes the
    lease columns onto the task row.
    """
    with pg_sessions() as db:
        task_id = seed_task_with_run(db, run_id=run_id, marker=1)
        owner_id = int(db.get(Task, task_id).user_id)
        agent_id: int | None = None
        if with_agent:
            agent = Agent(user_id=owner_id, name="marker residue agent")
            db.add(agent)
            db.flush()
            agent_id = int(agent.id)
        values: dict[Any, Any] = {
            Task.status: status,
            Task.source: source,
            Task.description: "d",
            Task.execution_mode: "balanced",
            Task.agent_id: agent_id,
        }
        if lease is not None:
            values.update(
                {
                    Task.runner_id: lease.runner_id,
                    Task.lease_attempt_id: lease.attempt_id,
                    Task.lease_expires_at: datetime.now(timezone.utc)
                    + timedelta(minutes=5),
                }
            )
        db.query(Task).filter(Task.id == task_id).update(values)
        db.commit()
        row_id = seed_active_row(db, task_id=task_id, run_id=run_id)
    return task_id, owner_id, agent_id, row_id


def _owner(pg_sessions: sessionmaker[Session], owner_id: int) -> User:
    with pg_sessions() as db:
        owner = db.get(User, owner_id)
        assert owner is not None
        db.refresh(owner)
        db.expunge(owner)
        return owner


def _task_state(
    observer: Session, task_id: int
) -> tuple[TaskStatus, int | None, str | None]:
    """``(status, interaction_protocol_version, input)`` of one task."""
    status, marker, text = observer.execute(
        sa.select(Task.status, Task.interaction_protocol_version, Task.input).where(
            Task.id == task_id
        )
    ).one()
    return status, marker, text


def _row_state(observer: Session, row_id: int) -> tuple[str, str | None]:
    """``(status, terminal_reason)`` of one interaction row."""
    status, reason = observer.execute(
        sa.select(
            TaskInteractionRequest.status, TaskInteractionRequest.terminal_reason
        ).where(TaskInteractionRequest.id == row_id)
    ).one()
    return status, reason


def _ws_manager() -> MagicMock:
    return MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )


def _completed_result() -> dict[str, Any]:
    context = SimpleNamespace(
        messages=[SimpleNamespace(role="user", metadata={"turn_id": "deferred-turn"})]
    )
    return {
        "status": "completed",
        "success": True,
        "output": "Applied",
        "agent_result": {"context": context},
    }


# --------------------------------------------------------------------------
# Close paths
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_online_chat_close_leaves_marker_and_row_paired(
    pg_sessions: sessionmaker[Session], live_task_lease
) -> None:
    task_id, owner_id, _agent_id, row_id = _seed(
        pg_sessions,
        run_id="run-online",
        status=TaskStatus.RUNNING,
        source="sdk",
    )
    with pg_sessions() as db:
        task = db.get(Task, task_id)
        task.runner_id = "online-runner"
        db.commit()
        live_task_lease(db, task)
    owner = _owner(pg_sessions, owner_id)
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(
        return_value=UserMessageInjectionOutcome.POSTED_FRESH
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None

    with (
        patch(
            "xagent.web.services.agent_service_manager.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch("xagent.web.api.websocket.manager", _ws_manager()),
        patch("xagent.web.services.task_execution.background_task_manager", bg_mgr),
        patch(
            "xagent.web.services.task_execution.execute_resume_background", AsyncMock()
        ),
    ):
        await handle_task_message(
            _make_command_reply(MagicMock()),
            task_id,
            {
                "message": "Here is the answer",
                "client_message_id": "online-turn",
                "user": owner,
                "files": [],
            },
        )

    with pg_sessions() as observer:
        assert_marker_residue(observer, task_id=task_id, run_id="run-online")
        assert _task_state(observer, task_id)[0] == TaskStatus.RUNNING
        assert _row_state(observer, row_id) == (
            "terminated",
            "answered_via_legacy_resume",
        )
    assert agent.post_user_message.await_count == 1


@pytest.mark.asyncio
async def test_deferred_injection_close_leaves_marker_and_row_paired_while_running(
    pg_sessions: sessionmaker[Session],
) -> None:
    task_id, owner_id, _agent_id, row_id = _seed(
        pg_sessions,
        run_id="run-deferred",
        status=TaskStatus.PAUSED,
        source="sdk",
    )
    samples: list[tuple[TaskStatus, AssertionError | None]] = []

    async def sample_then_complete(*_args: object, **_kwargs: object) -> Any:
        # Raising here would be swallowed by execute_resume_background's own
        # error handling, so the failure is recorded instead.
        with pg_sessions() as observer:
            failure: AssertionError | None = None
            try:
                assert_marker_residue(observer, task_id=task_id, run_id="run-deferred")
            except AssertionError as exc:
                failure = exc
            samples.append((_task_state(observer, task_id)[0], failure))
        return _completed_result()

    agent = MagicMock(
        post_user_message=AsyncMock(
            return_value=UserMessageInjectionOutcome.POSTED_FRESH
        ),
        resume_execution_by_id=AsyncMock(side_effect=sample_then_complete),
    )

    with (
        patch("xagent.web.api.websocket.manager", _ws_manager()),
        patch(
            "xagent.web.services.task_execution.background_task_manager.promote_resume_task"
        ),
    ):
        await execute_resume_background(
            task_id=task_id,
            agent_service=agent,
            task_owner_user_id=owner_id,
            pending_user_message={
                "execution_message": "Here is the answer",
                "display_message": "Here is the answer",
                "files": [],
                "turn_id": "deferred-turn",
                "interaction_id": row_id,
            },
            expected_run_id="run-deferred",
        )

    assert samples == [(TaskStatus.RUNNING, None)]
    with pg_sessions() as observer:
        assert _row_state(observer, row_id) == (
            "terminated",
            "answered_via_legacy_resume",
        )
    assert agent.post_user_message.await_count == 1


def test_a2a_resume_input_close_leaves_marker_and_row_paired(
    pg_sessions: sessionmaker[Session],
) -> None:
    lease = TaskLease(
        task_id=0, runner_id="a2a-runner", run_id="run-a2a", attempt_id="a2a-attempt"
    )
    task_id, _owner_id, _agent_id, row_id = _seed(
        pg_sessions,
        run_id="run-a2a",
        status=TaskStatus.RUNNING,
        source="a2a",
        lease=lease,
    )
    lease = TaskLease(
        task_id=task_id,
        runner_id=lease.runner_id,
        run_id=lease.run_id,
        attempt_id=lease.attempt_id,
    )

    assert (
        _update_a2a_resume_input_sync(
            lease, "answer", row_id, UserMessageInjectionOutcome.POSTED_FRESH
        )
        is True
    )

    with pg_sessions() as observer:
        assert_marker_residue(observer, task_id=task_id, run_id="run-a2a")
        status, _marker, text = _task_state(observer, task_id)
        assert (status, text) == (TaskStatus.RUNNING, "answer")
        assert _row_state(observer, row_id) == (
            "terminated",
            "answered_via_legacy_resume",
        )


class _SimulatedCrash(Exception):
    pass


def test_close_and_marker_clear_commit_together(
    pg_sessions: sessionmaker[Session],
) -> None:
    """A failure after the close statement and before the marker clear
    leaves both halves as they were: they are one transaction."""
    with pg_sessions() as db:
        task_id = seed_task_with_run(db, run_id="run-crash", marker=1)
        row_id = seed_active_row(db, task_id=task_id, run_id="run-crash")

    def crash_before_the_marker_clear(**_kwargs: object) -> sa.Exists:
        raise _SimulatedCrash("process died before the marker clear")

    with pytest.MonkeyPatch.context() as crash:
        # Both marker clears in the module build their condition from this
        # function; the close path's clear is the one reached here, after
        # its close statement has run.
        crash.setattr(
            task_interaction_close,
            "_active_row_exists",
            crash_before_the_marker_clear,
        )
        with pytest.raises(_SimulatedCrash):
            close_legacy_resume_interaction_sync(
                task_id=task_id, run_id="run-crash", interaction_id=row_id
            )

    with pg_sessions() as observer:
        assert_marker_residue(observer, task_id=task_id, run_id="run-crash")
        assert _row_state(observer, row_id) == ("active", None)
        assert _task_state(observer, task_id)[1] == 1


# --------------------------------------------------------------------------
# Compensation paths
# --------------------------------------------------------------------------

_SETTLEMENT_LEASE = TaskLease(
    task_id=0,
    runner_id="settlement-runner",
    run_id=_RUN,
    attempt_id="settlement-attempt",
)


def _failure(trigger: str) -> BaseException | None:
    """The exception a route's stub raises, or ``None`` for a stub that
    returns."""
    if trigger == "rejected_error":
        return UserMessageInjectionRejectedError("not written")
    if trigger in {"checkpoint_unavailable", "resume_checkpoint_unavailable"}:
        return CheckpointUnavailableError("checkpoint read failed")
    if trigger == "agent_lookup_error":
        return RuntimeError("agent lookup failed")
    return None


def _input_agent_manager(trigger: str, failure: BaseException | None) -> MagicMock:
    """An agent manager whose agent injects according to ``trigger``."""
    agent = MagicMock()
    if trigger == "not_posted":
        agent.post_user_message = AsyncMock(
            return_value=UserMessageInjectionOutcome.NOT_POSTED
        )
    elif trigger == "rejected_retryable":
        agent.post_user_message = AsyncMock(
            return_value=UserMessageInjectionOutcome.REJECTED_RETRYABLE
        )
    elif trigger in {"rejected_error", "checkpoint_unavailable"}:
        agent.post_user_message = AsyncMock(side_effect=failure)
    elif trigger == "accepted_then_cancelled":

        async def accept_then_cancel(*_args: object, **_kwargs: object) -> None:
            _record_injection_outcome(UserMessageInjectionOutcome.POSTED_FRESH)
            raise asyncio.CancelledError()

        agent.post_user_message = AsyncMock(side_effect=accept_then_cancel)
    if trigger == "agent_lookup_error":
        return MagicMock(get_agent_for_task=AsyncMock(side_effect=failure))
    return MagicMock(get_agent_for_task=AsyncMock(return_value=agent))


async def _drive_deferred(
    trigger: str,
    failure: BaseException | None,
    *,
    task_id: int,
    owner_id: int,
    row_id: int,
) -> object:
    agent = MagicMock()
    if trigger == "resume_checkpoint_unavailable":
        agent.resume_execution_by_id = AsyncMock(side_effect=failure)
        pending: dict[str, Any] | None = None
    else:
        agent.post_user_message = AsyncMock(side_effect=failure)
        pending = {
            "execution_message": "Here is the answer",
            "display_message": "Here is the answer",
            "files": [],
            "turn_id": "deferred-turn",
            "interaction_id": row_id,
        }
    with (
        patch("xagent.web.api.websocket.manager", _ws_manager()),
        patch(
            "xagent.web.services.task_execution.background_task_manager.promote_resume_task"
        ),
    ):
        return await execute_resume_background(
            task_id=task_id,
            agent_service=agent,
            task_owner_user_id=owner_id,
            pending_user_message=pending,
            expected_run_id=_RUN,
        )


async def _drive(
    site: str,
    trigger: str,
    failure: BaseException | None,
    *,
    task_id: int,
    owner_id: int,
    agent_id: int | None,
    row_id: int,
) -> object:
    if site == "deferred":
        return await _drive_deferred(
            trigger, failure, task_id=task_id, owner_id=owner_id, row_id=row_id
        )
    if trigger == "settlement_callback":
        lease = TaskLease(
            task_id=task_id,
            runner_id=_SETTLEMENT_LEASE.runner_id,
            run_id=_SETTLEMENT_LEASE.run_id,
            attempt_id=_SETTLEMENT_LEASE.attempt_id,
        )
        if site == "a2a":
            return _restore_a2a_resume_prelease_sync(
                lease, status=TaskStatus.WAITING_FOR_USER
            )
        return _restore_reply_prelease_sync(lease)
    assert agent_id is not None
    with patch(
        "xagent.web.services.agent_service_manager.get_agent_manager",
        return_value=_input_agent_manager(trigger, failure),
    ):
        if site == "a2a":
            return await resume_a2a_task(
                agent_id=agent_id,
                task_owner_user_id=owner_id,
                task_id=task_id,
                previous_run_id=_RUN,
                resumable_status=TaskStatus.WAITING_FOR_USER,
                text="answer",
                message_id="abandoned-message",
            )
        return await resume_task_reply(
            TaskReplyInput(
                task_id=task_id,
                agent_id=agent_id,
                task_owner_user_id=owner_id,
                run_id=_RUN,
                status=TaskStatus.WAITING_FOR_USER,
                text="answer",
                actor_user_id=owner_id,
            )
        )


_RETURNS_NONE = ("returns", None)
_RETURNS_TRUE = ("returns", True)
_RAISES_SAME = ("raises_the_injected_error", None)


def _raises(error: type[BaseException]) -> tuple[str, type[BaseException]]:
    return ("raises", error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("site", "trigger", "expected_status", "outcome"),
    [
        pytest.param(
            "deferred",
            "resume_checkpoint_unavailable",
            TaskStatus.PAUSED,
            _RETURNS_NONE,
            id="not_injected-deferred-resume_checkpoint_unavailable",
        ),
        pytest.param(
            "a2a",
            "settlement_callback",
            TaskStatus.WAITING_FOR_USER,
            _RETURNS_TRUE,
            id="acquisition_cancelled-a2a-settlement_callback",
        ),
        pytest.param(
            "reply",
            "settlement_callback",
            TaskStatus.WAITING_FOR_USER,
            _RETURNS_TRUE,
            id="acquisition_cancelled-reply-settlement_callback",
        ),
        pytest.param(
            "deferred",
            "rejected_error",
            TaskStatus.PAUSED,
            _RETURNS_NONE,
            id="not_written-deferred-rejected_error",
        ),
        pytest.param(
            "a2a",
            "not_posted",
            TaskStatus.WAITING_FOR_USER,
            _raises(TaskResumeNotResumableError),
            id="not_written-a2a-not_posted",
        ),
        pytest.param(
            "a2a",
            "rejected_retryable",
            TaskStatus.WAITING_FOR_USER,
            _raises(TaskResumeNotAcceptedError),
            id="not_written-a2a-rejected_retryable",
        ),
        pytest.param(
            "a2a",
            "rejected_error",
            TaskStatus.WAITING_FOR_USER,
            _raises(TaskResumeNotAcceptedError),
            id="not_written-a2a-rejected_error",
        ),
        pytest.param(
            "reply",
            "not_posted",
            TaskStatus.WAITING_FOR_USER,
            _raises(TaskResumeNotResumableError),
            id="not_written-reply-not_posted",
        ),
        pytest.param(
            "reply",
            "rejected_retryable",
            TaskStatus.WAITING_FOR_USER,
            _raises(TaskResumeNotAcceptedError),
            id="not_written-reply-rejected_retryable",
        ),
        pytest.param(
            "reply",
            "rejected_error",
            TaskStatus.WAITING_FOR_USER,
            _raises(TaskResumeNotAcceptedError),
            id="not_written-reply-rejected_error",
        ),
        pytest.param(
            "deferred",
            "checkpoint_unavailable",
            TaskStatus.PAUSED,
            _RETURNS_NONE,
            id="failed_before_write-deferred-checkpoint_unavailable",
        ),
        pytest.param(
            "a2a",
            "checkpoint_unavailable",
            TaskStatus.WAITING_FOR_USER,
            _raises(TaskResumeRetryableError),
            id="failed_before_write-a2a-checkpoint_unavailable",
        ),
        pytest.param(
            "a2a",
            "agent_lookup_error",
            TaskStatus.WAITING_FOR_USER,
            _RAISES_SAME,
            id="failed_before_write-a2a-agent_lookup_error",
        ),
        pytest.param(
            "reply",
            "checkpoint_unavailable",
            TaskStatus.WAITING_FOR_USER,
            _raises(TaskResumeRetryableError),
            id="failed_before_write-reply-checkpoint_unavailable",
        ),
        pytest.param(
            "reply",
            "agent_lookup_error",
            TaskStatus.WAITING_FOR_USER,
            _RAISES_SAME,
            id="failed_before_write-reply-agent_lookup_error",
        ),
        pytest.param(
            "a2a",
            "accepted_then_cancelled",
            TaskStatus.WAITING_FOR_USER,
            _raises(asyncio.CancelledError),
            id="accepted_then_cancelled-a2a",
        ),
        pytest.param(
            "reply",
            "accepted_then_cancelled",
            TaskStatus.WAITING_FOR_USER,
            _raises(asyncio.CancelledError),
            id="accepted_then_cancelled-reply",
        ),
    ],
)
async def test_abandoned_resume_keeps_a_marker_that_names_an_active_row(
    pg_sessions: sessionmaker[Session],
    site: str,
    trigger: str,
    expected_status: TaskStatus,
    outcome: tuple[str, object],
) -> None:
    if site == "deferred":
        seeded = _seed(pg_sessions, run_id=_RUN, status=TaskStatus.PAUSED, source="sdk")
    elif trigger == "settlement_callback":
        seeded = _seed(
            pg_sessions,
            run_id=_RUN,
            status=TaskStatus.RUNNING,
            source="a2a" if site == "a2a" else "sdk",
            lease=_SETTLEMENT_LEASE,
        )
    else:
        seeded = _seed(
            pg_sessions,
            run_id=_RUN,
            status=TaskStatus.WAITING_FOR_USER,
            source="a2a" if site == "a2a" else "sdk",
            with_agent=True,
        )
    task_id, owner_id, agent_id, row_id = seeded
    failure = _failure(trigger)
    caller_module = (
        "xagent.web.services.task_execution"
        if site == "deferred"
        else "xagent.web.services.task_resume"
    )

    with patch(
        f"{caller_module}.clear_interaction_marker_if_unpaired",
        wraps=clear_interaction_marker_if_unpaired,
    ) as spy:
        drive = _drive(
            site,
            trigger,
            failure,
            task_id=task_id,
            owner_id=owner_id,
            agent_id=agent_id,
            row_id=row_id,
        )
        kind, expected = outcome
        if kind == "returns":
            assert await drive is expected
        elif kind == "raises":
            assert isinstance(expected, type)
            with pytest.raises(expected):
                await drive
        else:
            with pytest.raises(RuntimeError) as excinfo:
                await drive
            assert excinfo.value is failure

    with pg_sessions() as observer:
        assert_marker_residue(observer, task_id=task_id, run_id=_RUN)
        status, marker, _text = _task_state(observer, task_id)
        assert marker == 1
        assert _row_state(observer, row_id) == ("active", None)
        assert status == expected_status
    assert spy.call_args_list == [call(ANY, task_id=task_id, run_id=_RUN)]


@pytest.mark.asyncio
async def test_abandoned_resume_after_a_completed_answer_leaves_nothing_behind(
    pg_sessions: sessionmaker[Session],
) -> None:
    """The A2A close commits, then scheduling the background resume fails,
    so the prelease restore and its compensation run after the row was
    retired and the marker cleared."""
    task_id, owner_id, agent_id, row_id = _seed(
        pg_sessions,
        run_id=_RUN,
        status=TaskStatus.WAITING_FOR_USER,
        source="a2a",
        with_agent=True,
    )
    assert agent_id is not None
    agent = MagicMock()
    agent.post_user_message = AsyncMock(
        return_value=UserMessageInjectionOutcome.POSTED_FRESH
    )

    with (
        patch(
            "xagent.web.services.agent_service_manager.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch(
            "xagent.web.services.task_execution.background_task_manager.reserve_resume",
            return_value=False,
        ),
        patch(
            "xagent.web.services.task_resume.clear_interaction_marker_if_unpaired",
            wraps=clear_interaction_marker_if_unpaired,
        ) as spy,
        pytest.raises(RuntimeError) as excinfo,
    ):
        await resume_a2a_task(
            agent_id=agent_id,
            task_owner_user_id=owner_id,
            task_id=task_id,
            previous_run_id=_RUN,
            resumable_status=TaskStatus.WAITING_FOR_USER,
            text="answer",
            message_id="answered-message",
        )

    assert str(excinfo.value) == f"Task {task_id} already has a resume in progress"
    with pg_sessions() as observer:
        assert_marker_residue(observer, task_id=task_id, run_id=_RUN)
        assert _task_state(observer, task_id)[0] == TaskStatus.WAITING_FOR_USER
        assert_interaction_marker_invariant(observer, task_id=task_id)
        active_rows = observer.execute(
            sa.select(sa.func.count())
            .select_from(TaskInteractionRequest)
            .where(
                TaskInteractionRequest.task_id == task_id,
                TaskInteractionRequest.run_id == _RUN,
                TaskInteractionRequest.status == "active",
            )
        ).scalar_one()
        assert active_rows == 0
        assert _row_state(observer, row_id)[1] == "answered_via_legacy_resume"
    assert spy.call_args_list == [call(ANY, task_id=task_id, run_id=_RUN)]
