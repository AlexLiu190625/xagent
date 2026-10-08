"""Deleting a task removes its auto-recovery state and event rows."""

from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.api.admin_users import _purge_user_task_rows
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.user import User
from xagent.web.services.task_deletion import purge_task_rows

engine = engine_fixture


def _seed(db: Session, *, username: str) -> tuple[int, int, int]:
    user = User(username=username, password_hash="unused")
    db.add(user)
    db.flush()
    tasks = [Task(user_id=user.id, title=f"t{i}", description="d") for i in range(2)]
    db.add_all(tasks)
    db.flush()
    now = datetime.now(timezone.utc)
    for task in tasks:
        db.add(
            TaskAutoRecovery(
                task_id=task.id,
                run_id="run",
                reason="lease_expired",
                state="scheduled",
                paused_state_version=1,
                interrupted_at=now,
                episode_started_at=now,
                next_attempt_at=now,
            )
        )
        db.add(TaskRecoveryEvent(task_id=task.id, run_id="run", event="interrupted"))
        db.add(TaskRecoveryEvent(task_id=task.id, run_id="run", event="scheduled"))
    db.commit()
    return int(user.id), int(tasks[0].id), int(tasks[1].id)


def _counts(db: Session, task_id: int) -> tuple[int, int]:
    return (
        db.query(TaskAutoRecovery).filter_by(task_id=task_id).count(),
        db.query(TaskRecoveryEvent).filter_by(task_id=task_id).count(),
    )


@pytest.mark.parametrize("fk_enforced", [True, False])
def test_purge_task_rows_removes_recovery_rows(engine, fk_enforced):
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        if engine.dialect.name == "sqlite" and not fk_enforced:
            # Without the pragma, ON DELETE CASCADE cannot do the work, so
            # this case proves the explicit deletes do.
            db.execute(sa.text("PRAGMA foreign_keys=OFF"))
        _user_id, first, second = _seed(db, username=f"purge-owner-{fk_enforced}")
        assert purge_task_rows(db, task_id=first, detached_reason="task_deleted")
        db.commit()
        assert _counts(db, first) == (0, 0)
        assert _counts(db, second) == (1, 2)


@pytest.mark.parametrize("fk_enforced", [True, False])
def test_user_purge_removes_recovery_rows(engine, fk_enforced):
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        if engine.dialect.name == "sqlite" and not fk_enforced:
            db.execute(sa.text("PRAGMA foreign_keys=OFF"))
        user_id, first, second = _seed(db, username=f"user-purge-{fk_enforced}")
        _purge_user_task_rows(db, user_id=user_id)
        db.commit()
        assert _counts(db, first) == (0, 0)
        assert _counts(db, second) == (0, 0)
