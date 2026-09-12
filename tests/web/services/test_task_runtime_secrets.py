"""Single-turn runtime inputs remain encrypted, scoped and transactional."""

import pytest

from xagent.core.tools.adapters.vibe.connector_runtime import (
    ConnectorRef,
    ConnectorRuntimeError,
)
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_runtime_secret import TaskRuntimeSecret
from xagent.web.models.user import User
from xagent.web.services.task_runtime_secrets import (
    bind_runtime_values_to_run,
    clean_finished_runtime_values,
    load_runtime_values,
    stage_runtime_values,
)

VALUES = {
    ConnectorRef("mcp", 1): {
        "secrets": {"token": "synthetic-secret"},
        "auth_selector": {"account": "synthetic-account"},
    }
}


@pytest.fixture
def task_id(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'secrets.db'}")
    with get_session_local()() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        task = Task(
            user_id=user.id,
            title="Inputs",
            status=TaskStatus.RUNNING,
            run_id="run-1",
            state_version=1,
            control_state="running",
        )
        db.add(task)
        db.commit()
        yield task.id
    Base.metadata.drop_all(bind=get_engine())


def stage(db, task_id):
    stage_runtime_values(db, task_id=task_id, turn_id="turn-1", values_by_ref=VALUES)
    assert bind_runtime_values_to_run(
        db, task_id=task_id, turn_id="turn-1", run_id="run-1"
    )


def test_ciphertext_round_trip_in_independent_session(task_id):
    with get_session_local()() as db:
        stage(db, task_id)
        row = db.query(TaskRuntimeSecret).one()
        assert "synthetic" not in row.ciphertext
        db.commit()
    with get_session_local()() as db:
        task = db.get(Task, task_id)
        assert load_runtime_values(db, task=task, turn_id="turn-1", required=True) == {
            ref.storage_key: values for ref, values in VALUES.items()
        }
        assert load_runtime_values(db, task=task, turn_id="turn-1", required=True)


def test_acceptance_rollback_removes_runtime_inputs(task_id):
    with get_session_local()() as db:
        stage(db, task_id)
        db.rollback()
    with get_session_local()() as db:
        assert db.query(TaskRuntimeSecret).count() == 0


@pytest.mark.parametrize("change", ["owner", "run", "ciphertext", "turn"])
def test_missing_or_mismatched_values_fail_closed(task_id, change):
    with get_session_local()() as db:
        stage(db, task_id)
        row = db.query(TaskRuntimeSecret).one()
        if change == "owner":
            db.get(
                User, db.get(Task, task_id).user_id
            ).actor_subject = "replacement-owner"
        elif change == "run":
            db.get(Task, task_id).run_id = "run-2"
        elif change == "ciphertext":
            row.ciphertext = "synthetic-plaintext"
        db.commit()
    with get_session_local()() as db:
        with pytest.raises(ConnectorRuntimeError) as error:
            load_runtime_values(
                db,
                task=db.get(Task, task_id),
                turn_id="other-turn" if change == "turn" else "turn-1",
                required=True,
            )
        assert "synthetic" not in str(error.value)


def test_compensation_preserves_queued_and_active_inputs(task_id):
    with get_session_local()() as db:
        stage(db, task_id)
        db.commit()
    clean_finished_runtime_values()
    with get_session_local()() as db:
        assert db.query(TaskRuntimeSecret).count() == 1
        db.get(Task, task_id).status = TaskStatus.PAUSED
        db.commit()
    clean_finished_runtime_values()
    with get_session_local()() as db:
        assert db.query(TaskRuntimeSecret).count() == 0


def test_compensation_waits_for_waiting_execution_to_release_lease(task_id):
    with get_session_local()() as db:
        stage(db, task_id)
        task = db.get(Task, task_id)
        task.status = TaskStatus.WAITING_FOR_USER
        task.runner_id = "worker"
        db.commit()
    clean_finished_runtime_values()
    with get_session_local()() as db:
        assert db.query(TaskRuntimeSecret).count() == 1
        db.get(Task, task_id).runner_id = None
        db.commit()
    clean_finished_runtime_values()
    with get_session_local()() as db:
        assert db.query(TaskRuntimeSecret).count() == 0


def test_cleanup_cannot_observe_inputs_before_transaction_binds_run(task_id):
    with get_session_local()() as acceptance:
        stage_runtime_values(
            acceptance, task_id=task_id, turn_id="turn-1", values_by_ref=VALUES
        )
        assert acceptance.query(TaskRuntimeSecret).one().run_id is None
        with get_session_local()() as observer:
            assert observer.query(TaskRuntimeSecret).count() == 0
        # Cleanup runs through its own session while acceptance is uncommitted.
        clean_finished_runtime_values()
        assert bind_runtime_values_to_run(
            acceptance, task_id=task_id, turn_id="turn-1", run_id="run-1"
        )
        acceptance.commit()
    clean_finished_runtime_values()
    with get_session_local()() as observer:
        row = observer.query(TaskRuntimeSecret).one()
        assert row.run_id == "run-1"
        assert load_runtime_values(
            observer, task=observer.get(Task, task_id), turn_id="turn-1", required=True
        )
