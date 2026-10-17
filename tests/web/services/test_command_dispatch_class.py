"""The dispatch class of a command: classifier result, upstream rule, persistence."""

from __future__ import annotations

import itertools

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_interaction import INTERACTION_ORIGIN_VOCABULARY
from xagent.web.models.user import User
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_execution_admission as admission
from xagent.web.services.task_dispatch_cap import (
    DISPATCH_CLASS_BATCH,
    DISPATCH_CLASS_INTERACTIVE,
)

KINDS = transport.TaskCommandKind
I = DISPATCH_CLASS_INTERACTIVE  # noqa: E741
B = DISPATCH_CLASS_BATCH


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'dispatch-class.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(database_module, "_SessionLocal", sessions)
    with sessions() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.commit()
        user_id = user.id

    class Env:
        def __init__(self):
            self.sessions = sessions
            self.engine = engine
            self.user = user_id
            self.counter = itertools.count(1)

        def task(self, source="internal", channel_name=None):
            with sessions() as db:
                task = Task(
                    user_id=user_id,
                    title="Work",
                    status=TaskStatus.PENDING,
                    source=source,
                    channel_name=channel_name,
                )
                db.add(task)
                db.commit()
                return task.id

        def seed(self, task_id, kind, dispatch_class):
            with sessions() as db:
                row = TaskExecutionCommand(
                    task_id=task_id,
                    command_id=f"seed-{next(self.counter)}",
                    kind=kind,
                    payload={},
                    status="completed",
                    dispatch_class=dispatch_class,
                )
                db.add(row)
                db.commit()

        def stage(self, kind, *, task_id, command_id=None, reserved=False):
            command_id = command_id or (
                f"auto-resume:{next(self.counter)}"
                if reserved
                else f"{kind.value}-{next(self.counter)}"
            )
            with sessions() as db:
                staged = transport.stage_task_command(
                    db,
                    task_id=task_id,
                    actor_user_id=user_id,
                    command_id=command_id,
                    kind=kind,
                    payload={},
                    reserved=reserved,
                )
                db.commit()
                return staged

        def klass(self, db_id):
            with sessions() as db:
                return db.execute(
                    sa.select(TaskExecutionCommand.dispatch_class).where(
                        TaskExecutionCommand.id == db_id
                    )
                ).scalar_one()

    try:
        yield Env()
    finally:
        engine.dispose()


# ---------------------------------------------------------------- I-4

_SOURCES = sorted(
    INTERACTION_ORIGIN_VOCABULARY | {"external", None, "", "SDK", "future_unknown"},
    key=lambda value: (value is not None, value or ""),
)
# Hand-written expectations, one literal per source.
_SOURCE_CLASS = {
    "internal": I,
    "widget": I,
    "shared_link": I,
    "sdk": B,
    "trigger": B,
    "a2a": B,
    "external": None,
    None: None,
    "": None,
    "SDK": None,
    "future_unknown": None,
}
_FOREGROUND = {
    "internal": {"resume_input": I, "other": I},
    "widget": {"resume_input": I, "other": I},
    "shared_link": {"resume_input": I, "other": I},
    "sdk": {"resume_input": I, "other": B},
    "trigger": {"resume_input": I, "other": B},
    "a2a": {"resume_input": I, "other": B},
    "external": {"resume_input": I, "other": B},
    None: {"resume_input": I, "other": B},
    "": {"resume_input": I, "other": B},
    "SDK": {"resume_input": I, "other": B},
    "future_unknown": {"resume_input": I, "other": B},
}
_BACKGROUND_UNKNOWN_SOURCE = {None: B, I: I, B: B}


def _view(source, *, background, previous):
    return admission.TaskClassificationView(
        task_id=1,
        source=source,
        channel_name=None,
        owner_user_id=1,
        background=background,
        previous_dispatch_class=previous,
    )


def _expected_table():
    table = {}
    for source, kind, background, previous in itertools.product(
        _SOURCES, KINDS, (False, True), (None, I, B)
    ):
        if background and _SOURCE_CLASS[source] is not None:
            expected = _SOURCE_CLASS[source]
        elif background:
            expected = _BACKGROUND_UNKNOWN_SOURCE[previous]
        else:
            expected = _FOREGROUND[source][
                "resume_input" if kind is KINDS.RESUME_INPUT else "other"
            ]
        table[(source, kind, background, previous)] = expected
    return table


def test_upstream_rule_table_covers_the_whole_input_space():
    table = _expected_table()
    assert set(table) == set(
        itertools.product(_SOURCES, KINDS, (False, True), (None, I, B))
    )
    assert len(table) == 11 * 6 * 2 * 3
    for (source, kind, background, previous), expected in table.items():
        view = _view(source, background=background, previous=previous)
        assert admission.upstream_dispatch_class(view, kind.value) == expected, (
            source,
            kind,
            background,
            previous,
        )


def test_upstream_source_class_is_exact_match_only():
    for source, expected in _SOURCE_CLASS.items():
        assert admission.upstream_source_class(source) == expected


# ---------------------------------------------------------------- I-25 / result shape


def test_classification_dispatch_class_defaults_to_batch_and_is_validated():
    assert admission.CommandClassification().dispatch_class == B
    assert admission.CommandClassification(admission=None).dispatch_class == B
    assert admission.CommandClassification(dispatch_class=I).dispatch_class == I
    with pytest.raises(ValueError):
        admission.CommandClassification(dispatch_class="urgent")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        admission.CommandClassification(dispatch_class=None)  # type: ignore[arg-type]


def test_upstream_classifier_applies_the_upstream_rule():
    command = TaskExecutionCommand(kind="resume_input")
    result = admission.upstream_command_classifier(
        None, command, _view("sdk", background=False, previous=None)
    )
    assert result == admission.CommandClassification(dispatch_class=I)
    result = admission.upstream_command_classifier(
        None,
        TaskExecutionCommand(kind="start"),
        _view("sdk", background=False, previous=None),
    )
    assert result == admission.CommandClassification(dispatch_class=B)


# ---------------------------------------------------------------- I-24 and the view


def test_classifier_receives_the_task_view(env):
    seen = []

    def classify(db, command, view):
        seen.append(view)
        return admission.CommandClassification()

    admission.set_command_classifier(classify, admission_enabled=False)
    task_id = env.task(source="widget", channel_name="support")
    env.stage(KINDS.START, task_id=task_id)
    assert seen == [
        admission.TaskClassificationView(
            task_id=task_id,
            source="widget",
            channel_name="support",
            owner_user_id=env.user,
            background=False,
            previous_dispatch_class=None,
        )
    ]


def _previous_queries(engine):
    statements = []

    def record(_conn, _cursor, statement, *_args):
        if "task_execution_commands.dispatch_class" in statement and (
            "LIMIT" in statement and "ORDER BY" in statement
        ):
            statements.append(statement)

    return statements, record


def test_background_resume_is_classified_by_source_not_by_the_previous_command(env):
    # An SDK task: start, then a reply (interactive by kind), then a background
    # resume. The reply must not make the resume interactive.
    task_id = env.task(source="sdk")
    start = env.stage(KINDS.START, task_id=task_id)
    assert env.klass(start.staged_db_id) == B
    reply = env.stage(KINDS.RESUME_INPUT, task_id=task_id)
    assert env.klass(reply.staged_db_id) == I
    resume = env.stage(KINDS.RESUME, task_id=task_id, reserved=True)
    assert env.klass(resume.staged_db_id) == B


def test_background_resume_inherits_interactive_only_for_an_unknown_source(env):
    task_id = env.task(source="external")
    env.seed(task_id, "start", I)
    resume = env.stage(KINDS.RESUME, task_id=task_id, reserved=True)
    assert env.klass(resume.staged_db_id) == I


@pytest.mark.parametrize(
    "previous",
    [
        pytest.param(B, id="previous-batch-or-pre-migration-default"),
        pytest.param(None, id="no-previous-command"),
    ],
)
def test_background_resume_of_an_unknown_source_is_batch_otherwise(env, previous):
    task_id = env.task(source="external")
    if previous is not None:
        env.seed(task_id, "start", previous)
    resume = env.stage(KINDS.RESUME, task_id=task_id, reserved=True)
    assert env.klass(resume.staged_db_id) == B


def test_background_resume_sees_the_previous_class_in_its_view(env):
    seen = []

    def classify(db, command, view):
        seen.append((view.background, view.previous_dispatch_class))
        return admission.CommandClassification()

    admission.set_command_classifier(classify, admission_enabled=False)
    task_id = env.task(source="external")
    env.seed(task_id, "start", I)
    env.seed(task_id, "message", B)
    env.stage(KINDS.RESUME, task_id=task_id, reserved=True)
    assert seen == [(True, B)]


def test_a_foreground_resume_does_not_inherit(env):
    task_id = env.task(source="external")
    env.seed(task_id, "start", I)
    resume = env.stage(KINDS.RESUME, task_id=task_id)
    assert env.klass(resume.staged_db_id) == B


def test_the_previous_command_is_read_only_when_it_can_matter(env):
    statements, record = _previous_queries(env.engine)
    sa.event.listen(env.engine, "before_cursor_execute", record)
    try:
        sdk = env.task(source="sdk")
        env.stage(KINDS.START, task_id=sdk)
        env.stage(KINDS.RESUME, task_id=sdk, reserved=True)
        assert statements == []
        external = env.task(source="external")
        env.stage(KINDS.START, task_id=external)
        assert statements == []
        env.stage(KINDS.RESUME, task_id=external, reserved=True)
        assert len(statements) == 1
    finally:
        sa.event.remove(env.engine, "before_cursor_execute", record)


# ---------------------------------------------------------------- I-33


@pytest.fixture
def no_autoflush(env):
    """A session built the way production builds them (models/database.py)."""
    factory = sessionmaker(env.engine, autoflush=False, expire_on_commit=False)
    with factory() as probe:
        assert probe.autoflush is False
    return factory


def _install_by_kind(admission_enabled=False):
    def classify(db, command, view):
        klass = I if command.kind in ("message", "resume_input") else B
        policy = (
            admission.AdmissionPolicy("tenant:test", 5, 20)
            if admission_enabled
            else None
        )
        return admission.CommandClassification(dispatch_class=klass, admission=policy)

    admission.set_command_classifier(classify, admission_enabled=admission_enabled)


def _count_updates(engine):
    updates = []

    def record(_conn, _cursor, statement, *_args):
        if statement.lstrip().upper().startswith("UPDATE TASK_EXECUTION_COMMANDS"):
            updates.append(statement)

    return updates, record


@pytest.mark.parametrize("with_ticket", [False, True], ids=["no-ticket", "ticket"])
def test_dispatch_class_is_written_before_commit(env, no_autoflush, with_ticket):
    _install_by_kind(admission_enabled=with_ticket)
    task_id = env.task(source="internal")
    with no_autoflush() as db:
        staged = transport.stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=env.user,
            command_id="interactive-1",
            kind=KINDS.MESSAGE,
            payload={},
        )
        # Before commit, read through the connection: no session autoflush.
        assert (
            db.connection()
            .execute(
                sa.text(
                    "SELECT dispatch_class FROM task_execution_commands WHERE id = :id"
                ),
                {"id": staged.staged_db_id},
            )
            .scalar_one()
            == I
        )
        db.commit()
    with env.sessions() as fresh:
        assert fresh.get(TaskExecutionCommand, staged.staged_db_id).dispatch_class == I


def test_batch_classification_emits_no_update(env):
    _install_by_kind()
    updates, record = _count_updates(env.engine)
    sa.event.listen(env.engine, "before_cursor_execute", record)
    try:
        task_id = env.task(source="sdk")
        staged = env.stage(KINDS.START, task_id=task_id)
    finally:
        sa.event.remove(env.engine, "before_cursor_execute", record)
    assert env.klass(staged.staged_db_id) == B
    assert updates == []


def test_interactive_classification_emits_one_update(env):
    _install_by_kind()
    updates, record = _count_updates(env.engine)
    sa.event.listen(env.engine, "before_cursor_execute", record)
    try:
        task_id = env.task(source="sdk")
        staged = env.stage(KINDS.MESSAGE, task_id=task_id)
    finally:
        sa.event.remove(env.engine, "before_cursor_execute", record)
    assert env.klass(staged.staged_db_id) == I
    assert len(updates) == 1


def test_control_commands_are_not_classified_and_stay_batch(env):
    calls = []

    def classify(db, command, view):
        calls.append(command.kind)
        return admission.CommandClassification(dispatch_class=I)

    admission.set_command_classifier(classify, admission_enabled=False)
    task_id = env.task(source="internal")
    pause = env.stage(KINDS.PAUSE, task_id=task_id)
    cancel = env.stage(KINDS.CANCEL, task_id=task_id)
    assert calls == []
    assert env.klass(pause.staged_db_id) == B
    assert env.klass(cancel.staged_db_id) == B


def test_an_idempotent_replay_neither_classifies_nor_rewrites_the_class(env):
    _install_by_kind()
    task_id = env.task(source="sdk")
    first = env.stage(KINDS.MESSAGE, task_id=task_id, command_id="same")
    assert env.klass(first.staged_db_id) == I
    calls = []

    def always_batch(db, command, view):
        calls.append(command.id)
        return admission.CommandClassification(dispatch_class=B)

    admission.set_command_classifier(always_batch, admission_enabled=False)
    replay = env.stage(KINDS.MESSAGE, task_id=task_id, command_id="same")
    assert replay.created is False
    assert replay.staged_db_id == first.staged_db_id
    assert calls == []
    assert env.klass(first.staged_db_id) == I


def test_a_classifier_exception_rolls_the_acceptance_back(env):
    class Refused(Exception):
        pass

    def refuse(db, command, view):
        raise Refused

    admission.set_command_classifier(refuse, admission_enabled=False)
    task_id = env.task(source="sdk")
    with env.sessions() as db:
        with pytest.raises(Refused):
            transport.stage_task_command(
                db,
                task_id=task_id,
                actor_user_id=env.user,
                command_id="refused",
                kind=KINDS.MESSAGE,
                payload={},
            )
        db.rollback()
    with env.sessions() as db:
        assert db.query(TaskExecutionCommand).count() == 0


# ---------------------------------------------------------------- I-19


def test_the_class_survives_defer_failure_and_retry(env, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    monkeypatch.setattr(transport, "get_runner_id", lambda: "runner-a")
    _install_by_kind()
    task_id = env.task(source="sdk")
    staged = env.stage(KINDS.MESSAGE, task_id=task_id)
    command_id = staged.staged_db_id
    assert env.klass(command_id) == I
    # The deployment now classifies everything as batch.
    admission.set_command_classifier(
        lambda db, command, view: admission.CommandClassification(dispatch_class=B),
        admission_enabled=False,
    )

    with env.sessions() as db:
        claimed = transport.claim_task_command(
            db, runner_id="runner-a", command_db_id=command_id
        )
    assert claimed is not None
    assert env.klass(command_id) == I

    assert transport.defer_task_command(command_id, "runner-a", "later")
    assert env.klass(command_id) == I

    with env.sessions() as db:
        row = db.get(TaskExecutionCommand, command_id)
        row.claim_expires_at = None
        row.retry_available_at = None
        db.commit()
        claimed = transport.claim_task_command(
            db, runner_id="runner-a", command_db_id=command_id
        )
    assert claimed is not None
    assert transport.fail_task_command(
        command_id, "runner-a", "boom", force_terminal=True
    )
    assert env.klass(command_id) == I
    with env.sessions() as db:
        assert db.get(TaskExecutionCommand, command_id).status == "failed"
        assert transport.retry_failed_task_command(db, command_id)
    assert env.klass(command_id) == I
