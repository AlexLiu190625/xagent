"""A caller's declared dispatch class, judged afresh as each command is enqueued."""

import logging

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
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
ACCEPT = "XAGENT_DISPATCH_ACCEPT_REQUESTED_CLASS"
FAIR = "XAGENT_DISPATCHER_FAIR_SHARE"


@pytest.fixture(autouse=True)
def _clean_classifier():
    admission.reset_command_classifier()
    yield
    admission.reset_command_classifier()


@pytest.fixture
def host(tmp_path, monkeypatch):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'declared.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.delenv(ACCEPT, raising=False)
    monkeypatch.delenv(FAIR, raising=False)
    monkeypatch.setattr(database_module, "_SessionLocal", sessions)
    with sessions() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.commit()
        user_id = user.id

    class Host:
        sessions_factory = sessions
        engine_ = engine
        counter = 0

        def task(self, declared, source="sdk"):
            with sessions() as db:
                task = Task(
                    user_id=user_id,
                    title="Work",
                    status=TaskStatus.PENDING,
                    source=source,
                    requested_dispatch_class=declared,
                )
                db.add(task)
                db.commit()
                return task.id

        def stage(self, kind, task_id, *, reserved=False):
            self.counter += 1
            with sessions() as db:
                staged = transport.stage_task_command(
                    db,
                    task_id=task_id,
                    actor_user_id=user_id,
                    command_id=f"{kind.value}-{self.counter}",
                    kind=kind,
                    payload={},
                    reserved=reserved,
                )
                db.commit()
                return staged.staged_db_id

        def klass(self, db_id):
            with sessions() as db:
                return db.execute(
                    sa.select(TaskExecutionCommand.dispatch_class).where(
                        TaskExecutionCommand.id == db_id
                    )
                ).scalar_one()

    try:
        yield Host()
    finally:
        engine.dispose()


def _settings(monkeypatch, *, accept, fair):
    monkeypatch.setenv(ACCEPT, str(accept))
    monkeypatch.setenv(FAIR, str(fair))


def test_each_command_judges_the_gates_when_it_is_enqueued(host, monkeypatch):
    task_id = host.task(I)
    queued: list[tuple[int, str]] = []

    def step(kind, expected, *, reserved=False):
        queued.append((host.stage(kind, task_id, reserved=reserved), expected))
        for db_id, earlier in queued:
            assert host.klass(db_id) == earlier

    step(KINDS.START, B)
    _settings(monkeypatch, accept=True, fair=True)
    step(KINDS.MESSAGE, I)
    _settings(monkeypatch, accept=False, fair=True)
    step(KINDS.RESUME, B)
    _settings(monkeypatch, accept=True, fair=True)
    step(KINDS.RESUME, I, reserved=True)
    _settings(monkeypatch, accept=True, fair=False)
    step(KINDS.MESSAGE, B)


def test_an_answer_is_interactive_whatever_the_declaration(host, monkeypatch):
    _settings(monkeypatch, accept=True, fair=True)
    task_id = host.task(B)
    assert host.klass(host.stage(KINDS.RESUME_INPUT, task_id)) == I


def test_an_accepted_batch_declaration_lowers_an_interactive_source(host, monkeypatch):
    _settings(monkeypatch, accept=True, fair=True)
    task_id = host.task(B, source="internal")
    assert host.klass(host.stage(KINDS.MESSAGE, task_id)) == B


@pytest.mark.parametrize("kind", [KINDS.START, KINDS.MESSAGE])
def test_a_task_without_a_declaration_is_classified_as_before(host, monkeypatch, kind):
    _settings(monkeypatch, accept=True, fair=True)
    sdk = host.task(None, source="sdk")
    internal = host.task(None, source="internal")
    assert host.klass(host.stage(kind, sdk)) == B
    assert host.klass(host.stage(kind, internal)) == I


def test_the_classifier_sees_the_declaration_as_stored_with_the_gates_closed(host):
    seen = []

    def classify(db, command, view):
        seen.append(view.requested_dispatch_class)
        return admission.CommandClassification()

    admission.set_command_classifier(classify, admission_enabled=False)
    host.stage(KINDS.START, host.task(I))
    host.stage(KINDS.START, host.task("bogus"))
    host.stage(KINDS.START, host.task(None))
    assert seen == [I, "bogus", None]


def test_no_declaration_reads_neither_setting(host, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("a setting was read")

    monkeypatch.setattr(admission, "get_dispatch_accept_requested_class_enabled", boom)
    monkeypatch.setattr(transport, "_fair_share_enabled", boom)
    task_id = host.task(None)
    assert host.klass(host.stage(KINDS.START, task_id)) == B
    assert host.klass(host.stage(KINDS.MESSAGE, task_id)) == B


def test_an_unknown_stored_value_is_ignored_and_logged_once(host, monkeypatch, caplog):
    _settings(monkeypatch, accept=True, fair=True)
    task_id = host.task("bogus", source="internal")
    with caplog.at_level(logging.WARNING, logger=admission.logger.name):
        db_id = host.stage(KINDS.START, task_id)
    assert host.klass(db_id) == I
    messages = [
        r.getMessage()
        for r in caplog.records
        if "requested dispatch class" in r.message
    ]
    assert len(messages) == 1
    assert "bogus" in messages[0]


def test_staging_reads_the_declaration_in_the_one_task_snapshot(host):
    statements = []

    def record(_conn, _cursor, statement, *_args):
        if "FROM tasks" in statement and "tasks.channel_name" in statement:
            statements.append(statement)

    task_id = host.task(I)
    sa.event.listen(host.engine_, "before_cursor_execute", record)
    try:
        host.stage(KINDS.START, task_id)
    finally:
        sa.event.remove(host.engine_, "before_cursor_execute", record)
    assert len(statements) == 1
    assert "tasks.requested_dispatch_class" in statements[0]
