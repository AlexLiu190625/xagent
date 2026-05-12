"""Reusable background task execution helper.

Encapsulates the "kick off a task in the background and register it
with the global ``background_task_manager``" pattern that previously
lived only inline inside the WebSocket message handler
(``web/api/websocket.py``). This module is callable from any entry
point that wants to run an agent task asynchronously -- the SDK's
``POST /v1/chat/tasks`` is the first such caller.

Design notes:

  - The existing WebSocket handler is deliberately NOT refactored to
    call this helper. It continues to use its own inline copy of the
    same logic so any subtle behavior the WS path depends on (error
    handling order, log lines, broadcast timing) stays frozen. We
    accept ~30 lines of duplication here in exchange for risk
    isolation -- a bug in the SDK kick-off path cannot regress the
    WS-based web UI flow. A future PR may unify the two once SDK
    traffic is stable.

  - Session lifecycle: the helper opens its own SQLAlchemy session
    for the background coroutine and does NOT borrow the caller's
    request-scoped session. FastAPI's ``Depends(get_db)`` closes the
    request session as soon as the response is returned, which would
    race against the background coroutine still using it. By opening
    a fresh session here that the bg coroutine owns end-to-end, we
    eliminate that race entirely.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from ..models.task import Task

logger = logging.getLogger(__name__)


async def start_task_in_background(
    *,
    task: Task,
    user_message: str,
    force_fresh_execution: bool = False,
    context: Optional[Dict[str, Any]] = None,
) -> "asyncio.Task[None]":
    """Schedule an agent task to run in the background.

    Internally:

      1. Opens a fresh SQLAlchemy session that the background
         coroutine owns end-to-end (caller's request-scoped session
         is NOT borrowed -- it would be closed by FastAPI before the
         bg coroutine finishes using it).
      2. Re-queries :class:`Task` and :class:`User` rows on the new
         session so they're attached to it (the caller's instances
         belong to the caller's already-detached session by the time
         this coroutine runs).
      3. Creates an ``asyncio.Task`` wrapping
         ``execute_task_background`` (the same coroutine the
         WebSocket handler uses).
      4. Registers the task with the global ``background_task_manager``
         so subsequent calls on the same ``task.id`` wait for the
         prior turn to finish (single-flight per task).
      5. Closes the bg session when the coroutine completes (success,
         error, or cancellation).

    Args:
        task: The persisted :class:`Task` row to run. Must already
            exist in the DB (caller is responsible for ``db.add`` +
            ``db.commit`` before invoking this helper). Only its
            ``id`` is used to re-query on the bg session.
        user_message: The latest user input that this background run
            should consume. Mirrors the WS handler argument shape so
            ``execute_task_background`` doesn't have to special-case
            entry points.
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
        Callers normally fire and forget, but the handle is returned
        for tests that want to await completion or inspect status.

    Notes:
        - Imports of ``execute_task_background`` and
          ``background_task_manager`` are done locally to avoid an
          import cycle (``web.services`` -> ``web.api.websocket`` ->
          back into anything that already imported services).
        - This helper does NOT modify ``task.status``. The background
          coroutine flips it to RUNNING when it actually picks up.
          Callers who need an immediately-visible status change
          should do so themselves on their own session before
          calling this function.
    """
    # Local import: keeping ``execute_task_background`` and
    # ``background_task_manager`` references out of module scope means
    # web/services modules don't pull in the entire websocket router
    # at import time, which would create a cycle through
    # ``web/app.py`` and confuse downstream importers.
    from ..api.websocket import background_task_manager, execute_task_background

    task_id = int(task.id)
    task_source = getattr(task, "source", None)
    # ``user_id`` is read on the caller's session before the closure
    # captures it; pure Python int has no session affinity.
    user_id = int(task.user_id)

    async def _runner() -> None:
        """Open a session, rebind ORM instances, run, then close."""
        from ..models.database import get_session_local
        from ..models.user import User

        SessionLocal = get_session_local()
        bg_db = SessionLocal()
        try:
            bg_task = bg_db.query(Task).filter(Task.id == task_id).first()
            bg_user = bg_db.query(User).filter(User.id == user_id).first()
            if bg_task is None or bg_user is None:
                # Task or owner disappeared between caller commit and
                # background pickup. Log and exit cleanly; the bg
                # registration was already done so the manager will
                # cleanup_task() in the protocol.
                logger.warning(
                    "Background task %s aborted: task or user vanished "
                    "(task=%s, user=%s)",
                    task_id,
                    bg_task,
                    bg_user,
                )
                return
            await execute_task_background(
                task_id=task_id,
                user_message=user_message,
                context=context or {},
                agent_manager=_get_agent_manager(),
                user=bg_user,
                task=bg_task,
                db=bg_db,
                force_fresh_execution=force_fresh_execution,
            )
        finally:
            bg_db.close()

    bg_task = asyncio.create_task(_runner())
    background_task_manager.register_task(task_id, bg_task)
    logger.info(f"Task {task_id} kicked off in background (source={task_source})")
    return bg_task


def _get_agent_manager() -> Any:
    """Resolve the global ``AgentServiceManager`` singleton.

    Wraps the same lookup the WS handler uses but factored out so the
    helper has one obvious dependency surface. The local import keeps
    services <-> api boundary one-way at module load.
    """
    from ..api.chat import get_agent_manager

    return get_agent_manager()
