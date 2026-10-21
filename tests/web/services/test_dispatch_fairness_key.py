"""The tenant key written to each task command row and used to account for dispatch slots."""

import itertools

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
from xagent.web.services.task_dispatch_cap import DISPATCH_CLASS_INTERACTIVE

KINDS = transport.TaskCommandKind


@pytest.fixture
def host(tmp_path, monkeypatch):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'fairness.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(database_module, "_SessionLocal", sessions)
    with sessions() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.commit()
        user_id = user.id

    class Host:
        def __init__(self):
            self.sessions = sessions
            self.engine = engine
            self.user = user_id
            self.counter = itertools.count(1)

        def task(self):
            with sessions() as db:
                task = Task(user_id=user_id, title="Work", status=TaskStatus.PENDING)
                db.add(task)
                db.commit()
                return task.id

        def stage(self, kind, *, task_id, command_id=None):
            command_id = command_id or f"{kind.value}-{next(self.counter)}"
            with sessions() as db:
                staged = transport.stage_task_command(
                    db,
                    task_id=task_id,
                    actor_user_id=user_id,
                    command_id=command_id,
                    kind=kind,
                    payload={},
                )
                db.commit()
                return staged

        def key(self, db_id):
            with sessions() as db:
                return db.execute(
                    sa.select(TaskExecutionCommand.fairness_key).where(
                        TaskExecutionCommand.id == db_id
                    )
                ).scalar_one()

        def seed(self, task_id, kind, fairness_key):
            with sessions() as db:
                db.add(
                    TaskExecutionCommand(
                        task_id=task_id,
                        command_id=f"seed-{next(self.counter)}",
                        kind=kind,
                        payload={},
                        status="completed",
                        fairness_key=fairness_key,
                    )
                )
                db.commit()

        def command_updates(self):
            updates: list[str] = []

            def record(_conn, _cursor, statement, *_args):
                if statement.lstrip().startswith("UPDATE task_execution_commands"):
                    updates.append(statement)

            sa.event.listen(engine, "before_cursor_execute", record)
            self._listener = record
            return updates

        def stop_recording(self):
            sa.event.remove(engine, "before_cursor_execute", self._listener)

    try:
        yield Host()
    finally:
        engine.dispose()


def classifier_returning(**fields):
    def classify(db, command, view):
        return admission.CommandClassification(**fields)

    return classify


def test_a_new_execution_command_defaults_to_the_owner_key(host):
    task_id = host.task()
    staged = host.stage(KINDS.START, task_id=task_id)
    assert host.key(staged.staged_db_id) == f"user:{host.user}"


def test_a_classifier_key_is_written_with_the_class_in_one_flush(host):
    admission.set_command_classifier(
        classifier_returning(
            dispatch_class=DISPATCH_CLASS_INTERACTIVE, fairness_key="team:7"
        ),
        admission_enabled=False,
    )
    task_id = host.task()
    updates = host.command_updates()
    try:
        staged = host.stage(KINDS.START, task_id=task_id)
    finally:
        host.stop_recording()
    assert host.key(staged.staged_db_id) == "team:7"
    assert len(updates) == 1
    assert "fairness_key" in updates[0] and "dispatch_class" in updates[0]


@pytest.mark.parametrize("bad", ["", "x" * 65, 5])
def test_an_invalid_key_is_refused_and_the_enqueue_leaves_no_row(host, bad):
    with pytest.raises(ValueError, match="Invalid fairness key"):
        admission.CommandClassification(fairness_key=bad)

    def classify(db, command, view):
        return admission.CommandClassification(fairness_key=bad)

    admission.set_command_classifier(classify, admission_enabled=False)
    task_id = host.task()
    with host.sessions() as db:
        with pytest.raises(ValueError, match="Invalid fairness key"):
            transport.stage_task_command(
                db,
                task_id=task_id,
                actor_user_id=host.user,
                command_id="start-1",
                kind=KINDS.START,
                payload={},
            )
        db.rollback()
        assert db.query(TaskExecutionCommand).count() == 0


def test_the_longest_valid_key_is_accepted():
    assert admission.CommandClassification(fairness_key="x" * 64).fairness_key == (
        "x" * 64
    )
    assert admission.CommandClassification(fairness_key="x").fairness_key == "x"


def test_a_control_command_takes_the_key_of_the_latest_command_of_its_task(host):
    admission.set_command_classifier(
        classifier_returning(fairness_key="team:7"), admission_enabled=False
    )
    task_id = host.task()
    host.stage(KINDS.START, task_id=task_id)
    cancel = host.stage(KINDS.CANCEL, task_id=task_id)
    assert host.key(cancel.staged_db_id) == "team:7"


def test_a_control_command_after_pre_migration_rows_takes_the_empty_key(host):
    task_id = host.task()
    host.seed(task_id, "start", "")
    cancel = host.stage(KINDS.CANCEL, task_id=task_id)
    assert host.key(cancel.staged_db_id) == ""


def test_a_control_command_of_a_task_without_commands_takes_the_owner_key(host):
    task_id = host.task()
    pause = host.stage(KINDS.PAUSE, task_id=task_id)
    assert host.key(pause.staged_db_id) == f"user:{host.user}"


def test_a_repeated_command_id_changes_nothing(host):
    task_id = host.task()
    first = host.stage(KINDS.START, task_id=task_id, command_id="same")
    updates = host.command_updates()
    try:
        replay = host.stage(KINDS.START, task_id=task_id, command_id="same")
    finally:
        host.stop_recording()
    assert replay.created is False
    assert replay.staged_db_id == first.staged_db_id
    assert host.key(first.staged_db_id) == f"user:{host.user}"
    assert updates == []
