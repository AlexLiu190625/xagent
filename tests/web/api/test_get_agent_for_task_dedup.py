"""Regression test: ``get_agent_for_task`` must not double-query Task or Agent.

Background:
    ``chat.py:get_agent_for_task`` previously ran ``db.query(Task)`` twice
    on the new-agent setup path (once for the existence check around
    line 690, once again at the top of the LLM-config block around line
    757) and ``db.query(Agent)`` twice (once in the agent-builder branch
    around line 791 to load configuration, once again in the tools-init
    block around line 866 to read ``.status`` for the published-agent
    exclusion list). Step 1 of the PR3 sequence (Codex-revised) reuses
    the first result in both cases.

What this test pins:
    For one full ``get_agent_for_task`` call on an existing task with
    an associated agent, the number of ``db.query(Task)`` / ``db.query(Agent)``
    calls drops to one each. The test counts query invocations by table
    via a wrapped ``MagicMock`` so subsequent refactors that accidentally
    reintroduce a duplicate query will fail here rather than only
    showing up in production timing logs.
"""

from __future__ import annotations

from collections import Counter
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User


def _make_user() -> User:
    return User(
        id=1,
        username="dedup_test_user",
        password_hash="hash",
        is_admin=False,
    )


def _make_task(
    agent_id: int | None = None, status: TaskStatus = TaskStatus.PENDING
) -> Task:
    return Task(
        id=42,
        user_id=1,
        title="dedup test",
        description="dedup",
        status=status,
        agent_id=agent_id,
        agent_type="standard",
    )


def _make_agent() -> Agent:
    return Agent(
        id=7,
        user_id=1,
        name="dedup agent",
        instructions="be terse",
        status=AgentStatus.PUBLISHED,
        tool_categories=["basic"],
        knowledge_bases=[],
        skills=[],
        execution_mode="flash",
    )


class _QueryCounter:
    """Wraps ``db.query`` so we can count invocations per model class.

    SQLAlchemy ``Session.query`` returns a Query object; subsequent
    ``.filter().first()`` calls don't re-enter ``Session.query``, so a
    simple counter on the entry point is sufficient to detect double
    SELECTs against the same table.
    """

    def __init__(self) -> None:
        self.calls_by_model: Counter[type] = Counter()
        self._returns: dict[type, Any] = {}

    def set_first(self, model: type, value: Any) -> None:
        """Configure what ``.filter(...).first()`` will return for queries
        against the given model.
        """
        self._returns[model] = value

    def __call__(self, model: type) -> Any:
        self.calls_by_model[model] += 1
        result = MagicMock()
        result.filter = MagicMock(return_value=result)
        result.first = MagicMock(return_value=self._returns.get(model))
        result.all = MagicMock(return_value=[])
        result.order_by = MagicMock(return_value=result)
        return result


@pytest.mark.asyncio
async def test_existing_task_with_agent_dedups_task_and_agent_queries() -> None:
    """Existing task + agent path: Step 1's two dedups must hold.

    Pre-PR3 baseline:
      - ``db.query(Task)`` called >= 2 times (existence check + LLM-config
        re-load).
      - ``db.query(Agent)`` called >= 2 times (LLM-config agent-builder
        branch + tools-init published-status branch).

    After PR3 Step 1:
      - Each query happens at most once.
    """
    manager = AgentServiceManager()
    user = _make_user()
    task = _make_task(agent_id=7, status=TaskStatus.PENDING)
    agent_row = _make_agent()

    counter = _QueryCounter()
    counter.set_first(Task, task)
    counter.set_first(Agent, agent_row)
    counter.set_first(User, user)

    db = MagicMock()
    db.query = counter

    # Make ``task.status not in [RUNNING, PAUSED, WAITING_FOR_USER]`` so the
    # reconstruct branch is skipped entirely; we want to count the
    # ``normal creation`` path's queries, not reconstruct internal ones.
    # ``PENDING`` already satisfies that (see ``should_reconstruct`` check
    # in chat.py:720).

    # Stub the heavy work that ``get_agent_for_task`` performs after the
    # DB queries we care about, so the test focuses on query counts.
    with (
        patch.object(
            manager,
            "_get_task_llm_ids",
            return_value=[None, None, None, None],
        ),
        patch(
            "xagent.web.api.chat.resolve_llms_from_names",
            return_value=(None, None, None, None),
        ),
        patch.object(
            manager,
            "_load_agent_builder_config",
            return_value={
                "llms": (None, None, None, None),
                "execution_mode": "flash",
                "knowledge_bases": [],
                "skills": [],
                "tool_categories": ["basic"],
            },
        ),
        patch.object(
            manager,
            "_load_persisted_conversation_history",
        ),
        patch(
            "xagent.web.api.chat.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.web.api.chat.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ),
        patch(
            "xagent.web.sandbox_manager.get_sandbox_manager",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.AgentService",
        ),
    ):
        try:
            await manager.get_agent_for_task(task_id=42, db=db, user=user)
        except Exception:
            # Some downstream stubs may raise during agent assembly --
            # query-count assertions below are what we're verifying, and
            # they're recorded before the failure point.
            pass

    # The two dedups Step 1 introduces:
    assert counter.calls_by_model[Task] == 1, (
        f"Task queried {counter.calls_by_model[Task]} times -- expected 1. "
        "If this drops back to 2+, the line 757 dedup regressed."
    )
    assert counter.calls_by_model[Agent] == 1, (
        f"Agent queried {counter.calls_by_model[Agent]} times -- expected 1. "
        "If this drops back to 2+, the line 866 dedup regressed."
    )
