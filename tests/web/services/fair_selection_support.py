"""Queued task commands for the fair-share selection tests, on any engine."""

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import sessionmaker

from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User


class World:
    """Commands queued for tenants, each of its own task, in a SQLite file."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.sessions = sessionmaker(engine, expire_on_commit=False)
        with self.sessions() as db:
            user = User(username="owner", password_hash="unused")
            db.add(user)
            db.commit()
            self.user = user.id
        self.count = 0

    def add(
        self,
        tenant,
        dispatch_class="batch",
        kind="start",
        status="pending",
        *,
        pinned=False,
        unavailable=False,
    ) -> int:
        """Queue one command of a new task; ``pinned`` puts the task's lease elsewhere."""
        self.count += 1
        later = datetime.now(timezone.utc) + timedelta(hours=1)
        with self.sessions() as db:
            task = Task(
                user_id=self.user,
                title="t",
                status=TaskStatus.PENDING,
                runner_id="other" if pinned else None,
                lease_expires_at=later if pinned else None,
            )
            db.add(task)
            db.flush()
            command = TaskExecutionCommand(
                task_id=task.id,
                command_id=f"c{self.count}",
                kind=kind,
                payload={},
                status=status,
                dispatch_class=dispatch_class,
                fairness_key=tenant,
                retry_available_at=later if unavailable else None,
            )
            db.add(command)
            db.commit()
            return command.id

    def task_of(self, command_id: int) -> int:
        with self.sessions() as db:
            return db.get(TaskExecutionCommand, command_id).task_id
