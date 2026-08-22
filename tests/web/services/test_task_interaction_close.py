"""Pin the retirement and marker-clear statements a legacy resume path issues.

``close_legacy_resume_interaction`` (and its short-transaction wrapper
``close_legacy_resume_interaction_sync``) and
``clear_interaction_marker_if_unpaired`` are exercised directly here at the
database level: rowcount classification across every input shape the close
statement can see, the no-op behavior on a deployment without the
interaction table, the ``NOT EXISTS`` guard the two marker-clear-only call
sites depend on, and a staging-primitive interaction proving the close is a
real behavior change, not a no-op. The production call sites that wire
these functions into the WebSocket and A2A resume paths are covered
separately in tests/web/api/test_websocket_owner_actor.py and
tests/web/api/test_a2a_api.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa

from tests.web.services.task_interaction_schema_shared import (
    make_row,
    make_task,
    make_trace_event,
    make_user,
    row_state,
    seed_active_row,
    seed_task_with_run,
    tables_excluding_interaction_requests,
    task_marker,
)
from xagent.web.models import database as database_module
from xagent.web.models.database import (
    Base,
    configure_db,
    get_db,
    get_engine,
    get_session_local,
    init_db,
)
from xagent.web.models.task import Task
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import ops_signals
from xagent.web.services.task_interaction_close import (
    _classify_close_rowcount,
    active_interaction_id_sync,
    clear_interaction_marker_if_unpaired,
    close_legacy_resume_interaction,
    close_legacy_resume_interaction_sync,
)
from xagent.web.services.task_interaction_staging import (
    InteractionAnchor,
    InteractionSlotTaken,
    stage_interaction_request,
)

_CLOSE_MODULE_NAME = "xagent.web.services.task_interaction_close"


@pytest.fixture(autouse=True)
def _reset_ops_signals():
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)
    yield
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)


@pytest.fixture()
def db(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'interaction_close.db'}")
    session = next(get_db())
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=get_engine())


# --------------------------------------------------------------------------
# _classify_close_rowcount -- the one place every rowcount the close
# statement can produce gets classified, called directly, no database
# involved.
# --------------------------------------------------------------------------


def test_classify_close_rowcount_logs_info_for_the_expected_single_row_case(
    caplog,
) -> None:
    with caplog.at_level(logging.INFO, logger=_CLOSE_MODULE_NAME):
        _classify_close_rowcount(1, task_id=1, run_id="run-a")

    assert [record.levelno for record in caplog.records] == [logging.INFO]
    assert ops_signals.active_degradations() == {}


def test_classify_close_rowcount_logs_debug_for_the_common_no_op_case(
    caplog,
) -> None:
    with caplog.at_level(logging.DEBUG, logger=_CLOSE_MODULE_NAME):
        _classify_close_rowcount(0, task_id=1, run_id="run-a")

    assert [record.levelno for record in caplog.records] == [logging.DEBUG]
    assert ops_signals.active_degradations() == {}


def test_classify_close_rowcount_logs_error_and_registers_a_signal_for_an_impossible_rowcount(
    caplog,
) -> None:
    """rowcount > 1 is impossible under uq_task_interaction_active_slot
    unless that constraint has already been violated -- see this module's
    docstring. Logged at error and surfaced on /health, not raised."""
    with caplog.at_level(logging.ERROR, logger=_CLOSE_MODULE_NAME):
        _classify_close_rowcount(2, task_id=7, run_id="run-b")

    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert (
        ops_signals.INTERACTION_LEGACY_RESUME_CLOSE_ROWCOUNT_ANOMALY
        in ops_signals.active_degradations()
    )


# --------------------------------------------------------------------------
# Rowcount grid -- every input shape the close statement's WHERE fence sees.
# --------------------------------------------------------------------------


def test_close_retires_the_active_row_for_its_own_run(db) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=row_id
    )
    db.commit()

    assert rowcount == 1
    row = row_state(db, row_id)
    assert row.status == "terminated"
    assert row.active_slot is None
    assert row.terminal_reason == "answered_via_legacy_resume"
    assert row.terminated_at is not None
    assert task_marker(db, task_id) is None


def test_close_is_a_no_op_replaying_an_already_terminated_row(db) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    anchor_id = make_trace_event(db, task_id=task_id)
    row = TaskInteractionRequest(
        **make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            run_id="run-a",
            status="terminated",
        )
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    row_id = int(row.id)
    original_terminal_reason = row.terminal_reason

    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=row_id
    )
    db.commit()

    assert rowcount == 0
    row = row_state(db, row_id)
    assert row.status == "terminated"
    assert row.terminal_reason == original_terminal_reason
    # The clear runs unconditionally: a marker left dangling by an earlier,
    # incomplete write still gets zeroed even though this close matched no
    # row of its own.
    assert task_marker(db, task_id) is None


def test_close_is_a_no_op_with_no_interaction_rows_at_all(db) -> None:
    """Today's 100% case: the table has no production writer yet."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=None)

    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=None
    )
    db.commit()

    assert rowcount == 0
    assert task_marker(db, task_id) is None


def test_close_does_not_touch_a_different_runs_active_row(db) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    orphan_row_id = seed_active_row(db, task_id=task_id, run_id="run-b")

    # The orphan's own id is passed in deliberately, so the run predicate
    # is the only thing left that can reject it.
    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=orphan_row_id
    )
    db.commit()

    assert rowcount == 0
    orphan = row_state(db, orphan_row_id)
    assert orphan.status == "active"
    # This call's own run still gets its marker cleared -- the orphan row
    # belongs to a different run's marker, which this call never touches.
    assert task_marker(db, task_id) is None


def test_close_does_not_overwrite_a_row_already_recycled_by_another_terminal_reason(
    db,
) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=None)
    anchor_id = make_trace_event(db, task_id=task_id)
    row = TaskInteractionRequest(
        **make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            run_id="run-a",
            status="terminated",
            terminal_reason="run_superseded",
        )
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    row_id = int(row.id)

    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=row_id
    )
    db.commit()

    assert rowcount == 0
    assert row_state(db, row_id).terminal_reason == "run_superseded"


# What the primary-key predicate does on its own, holding everything else
# constant: one active row for this task and run, and only the id handed to
# the close varies.
@pytest.mark.parametrize(
    ("id_to_pass", "expected_rowcount"),
    [
        pytest.param("the_active_row", 1, id="the_row_observed_before_injection"),
        pytest.param(None, 0, id="no_row_was_active_at_injection_time"),
        pytest.param("another_row", 0, id="a_row_that_is_not_this_tasks_active_one"),
    ],
)
def test_close_retires_only_the_row_it_was_given(
    db, id_to_pass: str | None, expected_rowcount: int
) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    other_task_id = seed_task_with_run(db, run_id="run-z", marker=1)
    other_row_id = seed_active_row(db, task_id=other_task_id, run_id="run-z")

    interaction_id = {
        "the_active_row": row_id,
        "another_row": other_row_id,
        None: None,
    }[id_to_pass]
    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=interaction_id
    )
    db.commit()

    assert rowcount == expected_rowcount
    assert (row_state(db, row_id).status == "terminated") is (expected_rowcount == 1)
    # The other task's row is out of range of this close regardless.
    assert row_state(db, other_row_id).status == "active"
    # The marker clear is not conditioned on the close matching anything: a
    # run that had no active row at injection time still needs its marker
    # zeroed.
    assert task_marker(db, task_id) is None


def test_close_sync_opens_its_own_transaction_and_commits(db) -> None:
    """The short-transaction wrapper the two WebSocket injection sites
    share: no caller-held session."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    rowcount = close_legacy_resume_interaction_sync(task_id, "run-a", row_id)

    assert rowcount == 1
    assert row_state(db, row_id).status == "terminated"
    assert task_marker(db, task_id) is None


# --------------------------------------------------------------------------
# Table absent -- a deployment not yet migrated to task_interaction_requests.
# --------------------------------------------------------------------------


@pytest.fixture()
def db_without_interaction_table(tmp_path):
    """A deployment shape missing task_interaction_requests -- bound as the
    *global* engine/session factory, not a private one.

    close_legacy_resume_interaction_sync (unlike the other functions this
    module tests) takes no db argument of its own: it opens its own session
    through get_session_local(), which reads the process-global factory.
    A fixture that built its own private engine here and handed back a
    session from it would leave that global factory pointed wherever the
    previous test left it, so close_legacy_resume_interaction_sync would run
    against a different database than the one this fixture seeds and
    asserts against -- the table-absence gate it is supposed to exercise
    would never actually see this fixture's schema. configure_db() only
    binds the engine and session factory; it does not create any tables
    (unlike init_db()), so the subset schema below is still built by hand.
    """
    previous_engine = database_module._engine
    previous_session_local = database_module._SessionLocal
    configure_db(db_url=f"sqlite:///{tmp_path / 'no_interaction_table.db'}")
    Base.metadata.create_all(
        bind=get_engine(), tables=tables_excluding_interaction_requests()
    )
    session = get_session_local()()
    try:
        yield session
    finally:
        session.close()
        # Restore the prior global factory so this fixture's rebinding does
        # not leak into whatever test runs next in this file (or module).
        database_module._engine = previous_engine
        database_module._SessionLocal = previous_session_local


def test_close_no_ops_when_the_interaction_table_does_not_exist(
    db_without_interaction_table,
) -> None:
    db = db_without_interaction_table
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    db.query(Task).filter(Task.id == task_id).update(
        {Task.run_id: "run-a", Task.interaction_protocol_version: 1}
    )
    db.commit()

    rowcount = close_legacy_resume_interaction_sync(task_id, "run-a", None)

    assert rowcount == 0
    # The gate is checked before the marker clear too: close_legacy_resume_
    # interaction_sync returns before opening the lock read or the clear
    # statement, so a deployment without the table pays for neither.
    db.expire_all()
    assert (
        db.query(Task).filter(Task.id == task_id).one().interaction_protocol_version
        == 1
    )


def test_clear_marker_if_unpaired_no_ops_when_the_interaction_table_does_not_exist(
    db_without_interaction_table,
) -> None:
    db = db_without_interaction_table
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    db.query(Task).filter(Task.id == task_id).update(
        {Task.run_id: "run-a", Task.interaction_protocol_version: 1}
    )
    db.commit()

    clear_interaction_marker_if_unpaired(db, task_id=task_id, run_id="run-a")
    db.commit()

    db.expire_all()
    assert (
        db.query(Task).filter(Task.id == task_id).one().interaction_protocol_version
        == 1
    )


# --------------------------------------------------------------------------
# active_interaction_id_sync -- the pre-injection read whose result the close
# binds to. Every caller reaches it through a patched name, so the body is
# exercised here: the id it returns for a live row, and the two shapes that
# must degrade to None rather than to a wrong id.
# --------------------------------------------------------------------------


def test_active_interaction_id_sync_returns_the_live_rows_id(db) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    assert active_interaction_id_sync(task_id) == row_id


def test_active_interaction_id_sync_returns_none_without_the_interaction_table(
    db_without_interaction_table,
) -> None:
    db = db_without_interaction_table
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    db.query(Task).filter(Task.id == task_id).update(
        {Task.run_id: "run-a", Task.interaction_protocol_version: 1}
    )
    db.commit()

    assert active_interaction_id_sync(task_id) is None


def test_active_interaction_id_sync_returns_none_when_the_read_fails(
    db, monkeypatch, caplog
) -> None:
    """A failing read must return None, not raise: the caller is on the
    injection path, and None closes nothing, while an exception would take
    the injection down with it."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    seed_active_row(db, task_id=task_id, run_id="run-a")

    def _fail():
        raise sa.exc.OperationalError("SELECT 1", {}, Exception("database is locked"))

    monkeypatch.setattr(
        "xagent.web.services.task_interaction_close.get_session_local", _fail
    )

    with caplog.at_level(logging.WARNING, logger=_CLOSE_MODULE_NAME):
        assert active_interaction_id_sync(task_id) is None

    assert [record.levelno for record in caplog.records] == [logging.WARNING]


# --------------------------------------------------------------------------
# Compensation clear -- the NOT EXISTS guard.
# --------------------------------------------------------------------------


def test_clear_marker_if_unpaired_zeroes_a_marker_with_no_active_row(db) -> None:
    """Sequence 'close already committed, then a compensation path runs'.

    The row is already terminated (the close already ran); a marker left
    at 1 -- however that happened -- has nothing to protect and is zeroed.
    """
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    anchor_id = make_trace_event(db, task_id=task_id)
    row = TaskInteractionRequest(
        **make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            run_id="run-a",
            status="terminated",
        )
    )
    db.add(row)
    db.commit()

    clear_interaction_marker_if_unpaired(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert task_marker(db, task_id) is None


def test_clear_marker_if_unpaired_leaves_a_still_active_row_untouched(db) -> None:
    """This is the mutation-testable half: removing the NOT EXISTS guard
    would zero a marker that still names a live question."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    clear_interaction_marker_if_unpaired(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert task_marker(db, task_id) == 1
    assert row_state(db, row_id).status == "active"


# --------------------------------------------------------------------------
# The close statement is a real behavior change, not a no-op: staging a
# second question on the same run behaves differently depending on whether
# it ran.
# --------------------------------------------------------------------------


def _stage(
    db,
    *,
    task_id: int,
    run_id: str,
    anchor_id: int,
    key: str,
):
    now = datetime.now(timezone.utc)
    return stage_interaction_request(
        db,
        task_id=task_id,
        run_id=run_id,
        anchor=InteractionAnchor(
            trace_event_id=anchor_id,
            resume_event_id="resume-event-1",
            resume_execution_id="resume-exec-1",
            resume_run_partition=run_id,
        ),
        kind="clarification",
        protocol_version=1,
        origin="internal",
        request_payload={"prompt": key},
        request_idempotency_key=key,
        expires_at=now + timedelta(minutes=15),
        now=now,
    )


def _stage_a_replacement_question_the_way_a_resumed_agent_would(
    db, *, task_id: int, run_id: str, anchor_id: int
):
    """Put a second question on the same run into the active slot, the way
    a resumed agent's own ``stage_interaction_request`` call does.

    The first question has to be reclaimable for that INSERT to land at
    all -- ``uq_task_interaction_active_slot`` allows one active row per
    task -- so its deadline is moved into the past first, standing in for
    the time that elapses while a resumed agent works. The second call
    then takes ``_reclaim_stale_slot_stmt``'s expired-row branch on its
    own; nothing here reclaims by hand. Returns ``(first_id, second_id)``.
    """
    first = _stage(db, task_id=task_id, run_id=run_id, anchor_id=anchor_id, key="q1")
    db.commit()
    # Both columns move together: ck_task_interaction_requests_expiry_
    # after_creation requires expires_at > created_at, so an already-lapsed
    # deadline has to belong to a row that was also created earlier.
    staged_at = datetime.now(timezone.utc) - timedelta(hours=2)
    db.execute(
        sa.update(TaskInteractionRequest)
        .where(TaskInteractionRequest.id == first.staged_db_id)
        .values(created_at=staged_at, expires_at=staged_at + timedelta(minutes=15))
    )
    db.commit()

    second = _stage(db, task_id=task_id, run_id=run_id, anchor_id=anchor_id, key="q2")
    db.commit()
    assert second.created is True
    return int(first.staged_db_id), int(second.staged_db_id)


def test_close_leaves_a_question_staged_after_the_injection_alone(db) -> None:
    """The window this close is keyed against. Injecting the user message
    is what resumes the agent, so between the observation and the close the
    resumed agent can ask something new. Retiring that new question as
    "answered via legacy resume" would silently discard a question nobody
    ever saw."""

    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    anchor_id = make_trace_event(db, task_id=task_id)

    observed_id, staged_after_injection_id = (
        _stage_a_replacement_question_the_way_a_resumed_agent_would(
            db, task_id=task_id, run_id="run-a", anchor_id=anchor_id
        )
    )

    rowcount = close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=observed_id
    )
    db.commit()

    assert rowcount == 0
    survivor = row_state(db, staged_after_injection_id)
    assert survivor.status == "active"
    assert survivor.terminal_reason is None
    # The row that was observed before injection is terminal either way --
    # the reclaim retired it as expired when the new question took the slot.
    assert row_state(db, observed_id).status == "terminated"


def test_close_lets_a_second_question_on_the_same_run_become_active(db) -> None:
    """Deleting the close call must turn this red: without it, the second
    stage attempt collides with the first question's still-active slot and
    raises InteractionSlotTaken instead of ever becoming the active row."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    anchor_id = make_trace_event(db, task_id=task_id)

    first = _stage(db, task_id=task_id, run_id="run-a", anchor_id=anchor_id, key="q1")
    db.commit()
    assert first.created is True

    close_legacy_resume_interaction(
        db, task_id=task_id, run_id="run-a", interaction_id=first.staged_db_id
    )
    db.commit()

    second = _stage(db, task_id=task_id, run_id="run-a", anchor_id=anchor_id, key="q2")
    db.commit()

    assert second.created is True
    active = (
        db.query(TaskInteractionRequest)
        .filter(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.status == "active",
        )
        .one()
    )
    assert active.id == second.staged_db_id
    assert active.request_idempotency_key == "q2"


def test_without_the_close_call_a_second_question_cannot_become_active(db) -> None:
    """The 'delete the close call' mutation, run for real: with no close in
    between, the first question's active row is still fresh (not expired,
    same run), so the second stage attempt's INSERT collides with the
    unique active-slot constraint and raises InteractionSlotTaken -- the
    first question remains the only active row."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    anchor_id = make_trace_event(db, task_id=task_id)

    first = _stage(db, task_id=task_id, run_id="run-a", anchor_id=anchor_id, key="q1")
    db.commit()

    with pytest.raises(InteractionSlotTaken):
        _stage(db, task_id=task_id, run_id="run-a", anchor_id=anchor_id, key="q2")
    db.rollback()

    active = (
        db.query(TaskInteractionRequest)
        .filter(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.status == "active",
        )
        .one()
    )
    assert active.id == first.staged_db_id
    assert active.request_idempotency_key == "q1"
