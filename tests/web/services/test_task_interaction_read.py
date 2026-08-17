"""Coverage for ``task_interaction_read.get_pending_interaction_question``:
the tuple adapter over ``materialize_compatibility_view``.

Organized by the seven-tier projection table in the module's own
docstring (T0 through T3'''), plus the two hard shape requirements: step 0
costs no query into the interaction table (A5), and the adapter never
filters or reshapes what it receives (A14).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_interaction_schema_shared import make_task, make_user
from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import task_interaction_read as read_surface
from xagent.web.services.chat_history_service import persist_assistant_message
from xagent.web.services.ops_signals import (
    CHECKPOINT_LOAD_UNAVAILABLE,
    CHECKPOINT_PK_ANCHOR_DANGLING,
    INTERACTION_READ_PAYLOAD_UNREADABLE,
    INTERACTION_READ_PROTOCOL_UNRECOGNIZED,
    clear_degradation,
)
from xagent.web.services.task_lease_service import TASK_RUN_ID_TRACE_FIELD

_DEGRADATION_SIGNALS_UNDER_TEST = (
    CHECKPOINT_PK_ANCHOR_DANGLING,
    CHECKPOINT_LOAD_UNAVAILABLE,
    INTERACTION_READ_PROTOCOL_UNRECOGNIZED,
    INTERACTION_READ_PAYLOAD_UNREADABLE,
)


@pytest.fixture(autouse=True)
def _clean_degradation_registry():
    """A10-A12 exercise materialize_compatibility_view's failure branches,
    which register process-global degradation signals; clear them around
    every test so they cannot leak into tests that read the shared
    registry (the /health suite asserts exact payloads and fails on any
    leftover entry). Same fixture as test_task_interaction_service.py's,
    duplicated rather than shared because pytest fixtures are file-local
    unless hoisted to a conftest.py, which is out of this delivery's
    scope."""
    for signal in _DEGRADATION_SIGNALS_UNDER_TEST:
        clear_degradation(signal)
    yield
    for signal in _DEGRADATION_SIGNALS_UNDER_TEST:
        clear_degradation(signal)


@pytest.fixture
def _engine(tmp_path: Path):
    db_path = tmp_path / "task_interaction_read.db"
    engine = create_engine(f"sqlite:///{db_path}")
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def _session_factory(_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=_engine)


@pytest.fixture
def _db(_session_factory) -> Session:
    db = _session_factory()
    try:
        yield db
    finally:
        db.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_task(db: Session, *, marker: int | None) -> Task:
    """A loaded, committed Task row with the given
    interaction_protocol_version -- the object this adapter's own
    signature requires (a row, not an id)."""

    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    task = db.query(Task).filter(Task.id == task_id).first()
    task.run_id = "run-a"
    task.interaction_protocol_version = marker
    db.commit()
    db.refresh(task)
    return task


def _make_trace_event(
    db: Session,
    *,
    task_id: int,
    run_partition: str = "run-a",
    execution_id: str = "exec-1",
    checkpoint_type: str = "agent_execution_checkpoint",
) -> int:
    event = TraceEvent(
        task_id=task_id,
        event_id=f"read-trace-event-{task_id}",
        event_type=str(CHECKPOINT_EVENT_TYPE),
        timestamp=_now(),
        build_id=None,
        data={
            TASK_RUN_ID_TRACE_FIELD: run_partition,
            "checkpoint_type": checkpoint_type,
            "execution_id": execution_id,
        },
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return int(event.id)


def _make_active_row(
    db: Session,
    *,
    task_id: int,
    resume_trace_event_id: int | None,
    run_id: str = "run-a",
    resume_run_partition: str = "run-a",
    resume_execution_id: str = "exec-1",
    protocol_version: int = 1,
    request_payload: dict[str, Any] | None = None,
) -> TaskInteractionRequest:
    now = _now()
    row = TaskInteractionRequest(
        task_id=task_id,
        run_id=run_id,
        kind="clarification",
        protocol_version=protocol_version,
        status="active",
        active_slot=1,
        origin="internal",
        request_payload=request_payload
        if request_payload is not None
        else {
            "message": "Which environment?",
            "interactions": [
                {"type": "text_input", "field": "env", "label": "Environment"}
            ],
        },
        request_idempotency_key=f"read-key-{task_id}",
        resume_trace_event_id=resume_trace_event_id,
        resume_event_id="resume-event-1",
        resume_execution_id=resume_execution_id,
        resume_locator_format="trace_event_pk_v1",
        resume_checkpoint_type="agent_execution_checkpoint",
        resume_run_partition=resume_run_partition,
        created_at=now,
        expires_at=now + timedelta(minutes=15),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ---------------------------------------------------------------------------
# T0 -- the protocol marker fast path. Every cell here must not depend on
# the task_interaction_requests table's contents at all.
# ---------------------------------------------------------------------------


def test_a1_marker_null_reads_the_live_legacy_question(_db: Session) -> None:
    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("A live question")
    assert interactions == [{"type": "text_input", "label": "Live"}]


def test_a2_marker_null_opens_the_superseded_gate(_db: Session) -> None:
    """T0's NULL-marker cell is the one legacy fallback that opens the
    supersede gate: no structured row was ever published for this task's
    current wait, so a transcript question a later publication superseded
    is still the honest answer. Mutation: change the gate condition to
    ``allow_superseded=False`` and this test turns red."""

    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "An old question",
        message_type="question_superseded",
        interactions=[{"type": "text_input", "label": "Old"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("An old question")
    assert interactions == [{"type": "text_input", "label": "Old"}]


def test_a3_marker_null_prefers_the_live_row_over_a_higher_id_superseded_row(
    _db: Session,
) -> None:
    """Priority must not invert even with the gate open: a live question
    always wins over a superseded row, regardless of id order. Mutation:
    collapse the reader's two passes into one ``message_type.in_(...)``
    predicate and this test turns red (see get_latest_waiting_question's
    own docstring for why)."""

    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A newer superseded row",
        message_type="question_superseded",
        interactions=[{"type": "text_input", "label": "Superseded"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("A live question")
    assert interactions == [{"type": "text_input", "label": "Live"}]


def test_a4_unrecognized_marker_stays_closed(_db: Session) -> None:
    """A marker that is neither 1 nor NULL means the structured side's
    state is unknown, so the second pass stays shut -- unlike the NULL
    cell above. The marker is set on the in-memory ORM object without a
    commit: ck_tasks_interaction_protocol_version pins the column to
    NULL-or-1 in the database, so persisting 2 would fail the CHECK; this
    adapter only ever reads the attribute, never writes it, so exercising
    the branch this way is faithful to what the adapter actually does.
    Mutation: change the gate condition to ``marker != 1`` and this test
    turns red (a superseded-only row would then be returned)."""

    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "An old question",
        message_type="question_superseded",
        interactions=[{"type": "text_input", "label": "Old"}],
    )
    task.interaction_protocol_version = 2

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert (question, interactions) == (None, None)


def test_a5_marker_fast_path_never_calls_the_rich_view(
    _db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 0 must cost zero queries into the interaction table -- the
    module docstring's central shape claim. Proven by making the rich
    view explode if it is ever called, then running A1's own scenario
    and confirming it still passes. Mutation: delete the marker check (so
    every task always calls the rich view) and this test turns red."""

    def _explode(db: Session, task_id: int, *, allow_superseded: bool = False):
        raise AssertionError("materialize_compatibility_view must not be called")

    monkeypatch.setattr(read_surface, "materialize_compatibility_view", _explode)

    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("A live question")
    assert interactions == [{"type": "text_input", "label": "Live"}]


# ---------------------------------------------------------------------------
# T1 -- marker == 1, but the rich view itself falls back to the legacy
# transcript (table absent, or no active native row on this run).
# ---------------------------------------------------------------------------


def test_a6_marker_matches_no_active_row_reads_the_legacy_transcript(
    _db: Session,
) -> None:
    task = _make_task(_db, marker=1)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("A live question")
    assert interactions == [{"type": "text_input", "label": "Live"}]


def test_a7_marker_matches_table_absent_reads_the_legacy_transcript(
    _db: Session,
) -> None:
    task = _make_task(_db, marker=1)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )
    TaskInteractionRequest.__table__.drop(bind=_db.get_bind())

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("A live question")
    assert interactions == [{"type": "text_input", "label": "Live"}]


def test_a8_marker_matches_no_active_row_opens_the_supersede_gate_too(
    _db: Session,
) -> None:
    """T1's gate is opened unconditionally by the adapter (marker == 1
    always calls the rich view with allow_superseded=True), not only on
    the NULL-marker T0 cell -- a structured row was expected for this
    wait (the marker says so) but none is active, so a transcript
    question a publication superseded is still the honest fallback.
    Mutation: have the adapter call materialize_compatibility_view
    without allow_superseded=True and this test turns red."""

    task = _make_task(_db, marker=1)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "An old question",
        message_type="question_superseded",
        interactions=[{"type": "text_input", "label": "Old"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question is not None
    assert question.startswith("An old question")
    assert interactions == [{"type": "text_input", "label": "Old"}]


# ---------------------------------------------------------------------------
# T2 -- an active native row whose anchor resolves.
# ---------------------------------------------------------------------------


def test_a9_native_projection(_db: Session) -> None:
    task = _make_task(_db, marker=1)
    trace_event_id = _make_trace_event(_db, task_id=int(task.id))
    _make_active_row(_db, task_id=int(task.id), resume_trace_event_id=trace_event_id)

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question == "Which environment?"
    assert interactions == [
        {
            "type": "text_input",
            "field": "env",
            "label": "Environment",
            "options": None,
            "placeholder": None,
            "multiline": False,
            "min": None,
            "max": None,
            "default_value": None,
            "accept": None,
            "multiple": False,
        }
    ]


# ---------------------------------------------------------------------------
# T3 family -- an active native row this reader cannot answer with.
# ---------------------------------------------------------------------------


def test_a10_anchor_dangling_keeps_the_question_text_drops_controls(
    _db: Session,
) -> None:
    task = _make_task(_db, marker=1)
    # A trace event on a different run partition than the interaction
    # row's own resume_run_partition breaks the anchor's row-validity
    # judgment -- one of _resolve_read_direction_anchor's six conditions
    # -- producing "anchor_dangling" through a real write, no monkeypatch.
    trace_event_id = _make_trace_event(
        _db, task_id=int(task.id), run_partition="a-different-run"
    )
    _make_active_row(_db, task_id=int(task.id), resume_trace_event_id=trace_event_id)

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question == "Which environment?"
    assert interactions is None


def test_a11_checkpoint_unavailable_keeps_the_question_text_drops_controls(
    _db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(_db, marker=1)
    trace_event_id = _make_trace_event(_db, task_id=int(task.id))
    _make_active_row(_db, task_id=int(task.id), resume_trace_event_id=trace_event_id)

    real_get = _db.get

    def _raising_get(model: Any, pk: Any, *args: Any, **kwargs: Any) -> Any:
        if model is TraceEvent:
            import sqlalchemy as sa

            raise sa.exc.OperationalError(
                "SELECT 1", {}, Exception("simulated session failure")
            )
        return real_get(model, pk, *args, **kwargs)

    monkeypatch.setattr(_db, "get", _raising_get)

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert question == "Which environment?"
    assert interactions is None


def test_a12_unrecognized_protocol_version_drops_both_slots(
    _db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(_db, marker=1)
    trace_event_id = _make_trace_event(_db, task_id=int(task.id))
    row = _make_active_row(
        _db, task_id=int(task.id), resume_trace_event_id=trace_event_id
    )
    row.protocol_version = 2

    import xagent.web.services.task_interaction_service as svc

    def _fake_active_row(db: Session, task_id: int) -> TaskInteractionRequest:
        return row

    monkeypatch.setattr(svc, "_active_native_row", _fake_active_row)

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert (question, interactions) == (None, None)


def test_a13_unparseable_payload_drops_both_slots(_db: Session) -> None:
    task = _make_task(_db, marker=1)
    trace_event_id = _make_trace_event(_db, task_id=int(task.id))
    _make_active_row(
        _db,
        task_id=int(task.id),
        resume_trace_event_id=trace_event_id,
        request_payload={"not": "a valid v1 payload"},
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert (question, interactions) == (None, None)


# ---------------------------------------------------------------------------
# Shape invariants that hold across tiers.
# ---------------------------------------------------------------------------


def test_a14_adapter_does_not_filter_non_dict_interaction_elements(
    _db: Session,
) -> None:
    """C-4: the non-dict element filter belongs to the v1 layer, not this
    adapter or the shared reader -- see _filter_interaction_descriptors'
    own docstring for why it does not sink down. Mutation: have this
    adapter filter non-dict elements out of what it returns and this test
    turns red."""

    task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(task.id),
        int(task.user_id),
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}, "not-a-dict"],
    )

    question, interactions = read_surface.get_pending_interaction_question(_db, task)

    assert interactions == [{"type": "text_input", "label": "Live"}, "not-a-dict"]


def test_a15_interaction_element_key_sets_differ_between_legacy_and_native_tiers(
    _db: Session,
) -> None:
    """B-12, the known gap this delivery does not converge (see the
    adapter's own module docstring): pins the two key sets so a future
    change that accidentally aligns or further diverges them is visible
    here first."""

    legacy_task = _make_task(_db, marker=None)
    persist_assistant_message(
        _db,
        int(legacy_task.id),
        int(legacy_task.user_id),
        "Which environment?",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Environment"}],
    )
    _, legacy_interactions = read_surface.get_pending_interaction_question(
        _db, legacy_task
    )
    assert legacy_interactions is not None
    assert set(legacy_interactions[0].keys()) == {"type", "label"}

    native_task = _make_task(_db, marker=1)
    trace_event_id = _make_trace_event(_db, task_id=int(native_task.id))
    _make_active_row(
        _db, task_id=int(native_task.id), resume_trace_event_id=trace_event_id
    )
    _, native_interactions = read_surface.get_pending_interaction_question(
        _db, native_task
    )
    assert native_interactions is not None
    assert set(native_interactions[0].keys()) == {
        "type",
        "field",
        "label",
        "options",
        "placeholder",
        "multiline",
        "min",
        "max",
        "default_value",
        "accept",
        "multiple",
    }
