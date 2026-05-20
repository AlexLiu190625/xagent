"""Regression test: ``get_agent_for_task`` must not double-query Task or Agent
on the request session.

Background:
    Originally, ``chat.py:get_agent_for_task`` ran ``db.query(Task)``
    twice on the new-agent setup path (once for the existence check,
    once at the top of the LLM-config block) and ``db.query(Agent)``
    twice (once in the agent-builder branch to load configuration,
    once again in the tools-init block to read ``.status`` for the
    published-agent exclusion list). Step 1 of the PR3 sequence
    (Codex-revised) reused the first result in both cases, dropping
    each count to one.

    Step 3 of the same sequence then moved the entire LLM-config +
    agent-builder DB chain (Task re-read, Agent lookup, DBModel
    lookups, LLM access checks) into ``task_setup_snapshot``, which
    runs off-loop via ``asyncio.to_thread`` and uses its own
    ``SessionLocal``. The request session ``db`` is no longer
    responsible for any Agent query in the setup path.

What this test pins (post-Step-3 invariant):
    * ``db.query(Task)`` fires at most once on the request session
      (the existence check) -- the post-existence re-read is gone.
    * ``db.query(Agent)`` fires zero times on the request session --
      the snapshot owns that lookup now.

Snapshot-internal query counts are not the responsibility of this
file; those are covered by ``test_task_setup_snapshot.py``.
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
from xagent.web.services.task_setup_snapshot import (
    TaskSetupSnapshot,
    _AgentFields,
    _TaskFields,
)


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
    """Existing task + agent path: post-Step-3 the request session
    only handles the existence check.

    Pre-PR3 baseline:
      - ``db.query(Task)`` called >= 2 times (existence check + LLM-config
        re-load).
      - ``db.query(Agent)`` called >= 2 times (LLM-config agent-builder
        branch + tools-init published-status branch).

    After PR3 Step 1:
      - Each query happened at most once.

    After PR3 Step 3 (current invariant):
      - ``db.query(Task)`` happens at most once (existence check).
        The LLM-config block calls ``task_setup_snapshot`` which uses
        its own ``SessionLocal`` and is mocked out here -- so it
        contributes nothing to the counter on ``db``.
      - ``db.query(Agent)`` happens zero times. The snapshot loader
        owns the Agent lookup now; the duplicated published-agent
        status read is gone.
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

    # The snapshot loader uses its own SessionLocal (not the request
    # ``db``), so we replace it with a stubbed return that mirrors a
    # successful real load. The point of this test is the request
    # session's query counts; the snapshot's internal counts are
    # tested separately in ``test_task_setup_snapshot.py``.
    snapshot_stub = TaskSetupSnapshot(
        task=_TaskFields(
            id=42,
            user_id=1,
            status=TaskStatus.PENDING,
            agent_id=7,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
        ),
        task_pattern="single_call",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=_AgentFields(
            id=7,
            name="dedup agent",
            status=AgentStatus.PUBLISHED,
            instructions="be terse",
        ),
        agent_config={
            "llms": (None, None, None, None),
            "execution_mode": "flash",
            "instructions": "be terse",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": ["basic"],
        },
        excluded_agent_id=7,
    )

    # Make ``task.status not in [RUNNING, PAUSED, WAITING_FOR_USER]`` so the
    # reconstruct branch is skipped entirely; we want to count the
    # ``normal creation`` path's queries, not reconstruct internal ones.
    # ``PENDING`` already satisfies that (see ``should_reconstruct`` check
    # in chat.py:720).

    # Stub the heavy work that ``get_agent_for_task`` performs after the
    # DB queries we care about, so the test focuses on query counts.
    # ``load_task_setup_snapshot_sync`` is patched at the import site
    # inside chat.py: get_agent_for_task imports it at module load, so
    # patching the chat module's symbol intercepts the call. Returning
    # the stub above lets the rest of the function consume snapshot
    # fields without ever opening a real SessionLocal.
    with (
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=snapshot_stub,
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

    assert counter.calls_by_model[Task] == 1, (
        f"Task queried {counter.calls_by_model[Task]} times on the request "
        "session -- expected 1 (the existence check). If this jumps to 2+, "
        "either the dedup regressed or a new code path is re-reading Task "
        "via ``db`` instead of going through the snapshot."
    )
    assert counter.calls_by_model[Agent] == 0, (
        f"Agent queried {counter.calls_by_model[Agent]} times on the "
        "request session -- expected 0. Step 3 moved the agent lookup "
        "into ``task_setup_snapshot`` (its own SessionLocal). If this "
        "becomes non-zero, something is bypassing the snapshot and "
        "re-doing the agent lookup on the loop thread."
    )
