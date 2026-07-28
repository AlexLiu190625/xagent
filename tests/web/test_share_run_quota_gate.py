"""Per-share run quota at the execute_task chokepoint (#973, PR2).

The owner run gate bounds the owner's team quota, but every anonymous share
run bills the owner, so a per-link + per-guest rolling ceiling is enforced on
top at run start. This verifies the share-quota block short-circuits a share
task's run when the quota is exhausted, and is skipped for non-share tasks.
"""

from __future__ import annotations

import pytest

from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.share_rate_limit import reset_share_rate_limiter


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'share_run_quota.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture(autouse=True)
def _reset_limiter() -> None:
    reset_share_rate_limiter()
    yield
    reset_share_rate_limiter()


class _FakeAgentService:
    async def execute_task(self, **_kwargs):
        return {"success": True}

    def set_interrupt_checker(self, _checker):
        pass


def _make_task(db_session, *, agent_config: dict) -> Task:
    user = User(username="share-quota-user", password_hash="h", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="share run",
        description="test",
        status=TaskStatus.PENDING,
        execution_mode="auto",
        agent_config=agent_config,
    )
    db_session.add(task)
    db_session.commit()
    return task


@pytest.mark.asyncio
async def test_share_run_quota_blocks_share_task(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "0/day" leaves no room, so the very first share run is refused.
    monkeypatch.setenv("XAGENT_SHARE_RUN_QUOTA", "0/day")
    reset_share_rate_limiter()

    task = _make_task(
        db_session,
        agent_config={
            "auth_mode": "share",
            "guest_id": "guest-abc",
            "share_agent_id": 4242,
        },
    )

    result = await AgentServiceManager().execute_task(
        agent_service=_FakeAgentService(),
        task="hello",
        tracking_task_id=str(task.id),
        db_session=db_session,
        manage_task_lease=False,
    )

    assert result["success"] is False
    assert result["status"] == "quota_exceeded"
    assert result["error_code"] == "share_run_quota_exceeded"
    assert "usage limit" in result["output"]


@pytest.mark.asyncio
async def test_share_run_quota_skips_non_share_task(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-share task must never hit the share quota, even at 0/day."""
    monkeypatch.setenv("XAGENT_SHARE_RUN_QUOTA", "0/day")
    reset_share_rate_limiter()

    # No auth_mode == "share" marker: the share-quota branch is skipped, so the
    # run is not refused with the share error code (it proceeds to execution).
    task = _make_task(db_session, agent_config={})

    result = await AgentServiceManager().execute_task(
        agent_service=_FakeAgentService(),
        task="hello",
        tracking_task_id=str(task.id),
        db_session=db_session,
        manage_task_lease=False,
    )

    assert result.get("error_code") != "share_run_quota_exceeded"
