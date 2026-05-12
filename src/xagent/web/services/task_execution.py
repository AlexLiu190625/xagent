"""Reusable background task execution helper.

Encapsulates the "kick off a task in the background and register it
with the global ``background_task_manager``" pattern that previously
lived only inline inside the WebSocket message handler
(``web/api/websocket.py``). This module is callable from any entry
point that wants to run an agent task asynchronously -- the SDK's
``POST /v1/chat/tasks`` is the first such caller.

Design note:
    The existing WebSocket handler is deliberately NOT refactored to
    call this helper. It continues to use its own inline copy of the
    same logic so any subtle behavior the WS path depends on (error
    handling order, log lines, broadcast timing) stays frozen. We
    accept ~30 lines of duplication here in exchange for risk
    isolation -- a bug in the SDK kick-off path cannot regress the
    WS-based web UI flow. A future PR may unify the two once SDK
    traffic is stable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from ..models.task import Task
from ..models.user import User

logger = logging.getLogger(__name__)


async def start_task_in_background(
    *,
    task: Task,
    user_message: str,
    user: User,
    db: Session,
    force_fresh_execution: bool = False,
    context: Optional[Dict[str, Any]] = None,
) -> asyncio.Task[None]:
    """Schedule an agent task to run in the background.

    Internally:

      1. Creates an ``asyncio.Task`` wrapping
         ``execute_task_background`` (the same coroutine the WebSocket
         handler uses).
      2. Registers the task with the global ``background_task_manager``
         so subsequent calls on the same ``task.id`` wait for the
         prior turn to finish (single-flight per task).
      3. Returns the ``asyncio.Task`` handle. Callers normally fire and
         forget, but the handle is returned for tests that want to
         await completion or inspect status.

    Args:
        task: The persisted :class:`Task` row to run. Must already
            exist in the DB (caller is responsible for ``db.add`` +
            ``db.commit`` before invoking this helper).
        user_message: The latest user input that this background run
            should consume. Mirrors the WS handler argument shape so
            ``execute_task_background`` doesn't have to special-case
            entry points.
        user: The :class:`User` who owns the task. Used by
            ``execute_task_background`` to set ``UserContext`` for
            tenant-scoped lookups.
        db: SQLAlchemy session. ``execute_task_background`` will keep
            using it internally; the caller is expected to hand off
            ownership of the session for the duration of the
            background run.
        force_fresh_execution: When ``True``, the background coroutine
            ignores any prior reconstructible state for ``task.id`` and
            starts a brand new agent execution. Used by the WS handler
            when restarting a COMPLETED / FAILED task; SDK callers
            normally leave this False.
        context: Optional extra context dict passed to
            ``execute_task_background``; merged in with whatever
            execution_mode / process_description / examples the task
            row carries.

    Returns:
        The ``asyncio.Task`` that wraps the background coroutine.

    Notes:
        - Imports of ``execute_task_background`` and
          ``background_task_manager`` are done locally to avoid an
          import cycle (``web.services`` -> ``web.api.websocket`` ->
          back into anything that already imported services).
        - This helper does NOT modify ``task.status``. The background
          coroutine flips it to RUNNING when it actually picks up.
          Callers who need an immediately-visible status change should
          do so themselves before calling this function.
    """
    # Local import: keeping ``execute_task_background`` and
    # ``background_task_manager`` references out of module scope means
    # web/services modules don't pull in the entire websocket router
    # at import time, which would create a cycle through
    # ``web/app.py`` and confuse downstream importers.
    from ..api.websocket import background_task_manager, execute_task_background

    bg_task = asyncio.create_task(
        execute_task_background(
            task_id=int(task.id),
            user_message=user_message,
            context=context or {},
            agent_manager=_get_agent_manager(),
            user=user,
            task=task,
            db=db,
            force_fresh_execution=force_fresh_execution,
        )
    )
    background_task_manager.register_task(int(task.id), bg_task)
    logger.info(
        f"Task {task.id} kicked off in background "
        f"(source={getattr(task, 'source', None)})"
    )
    return bg_task


def _get_agent_manager() -> Any:
    """Resolve the global ``AgentServiceManager`` singleton.

    Wraps the same lookup the WS handler uses but factored out so the
    helper has one obvious dependency surface. The local import keeps
    services <-> api boundary one-way at module load.
    """
    from ..api.chat import get_agent_manager

    return get_agent_manager()
