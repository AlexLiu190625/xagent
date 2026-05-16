"""Single source of truth for task turn lifecycle.

Both the WebSocket UI path (``websocket.py:handle_chat_message``) and the
``/v1`` SDK endpoints (``v1/tasks.py``) route through this module. It owns
the parts of the lifecycle that *must* behave identically across both
transports so the same race / state-machine bugs don't grow back on
either side:

  - atomic state transitions (claim a task as RUNNING)
  - user message persistence (``task_chat_messages``)
  - background execution scheduling with a single-flight guard
  - assistant ``task.output`` / ``error_message`` sync after the bg
    coroutine returns

Things this module deliberately does **not** own (each transport keeps
its own adapter):

  - response shapes / error envelopes
    (``{"detail": ...}`` for ``/api/*`` vs ``{"error": {"code", "message"}}``
    for ``/v1/*``)
  - live broadcast events (WS sends ``task_started`` / ``task_completed``;
    SDK doesn't)

Background context — why we replaced the older ``task_execution.py``
helper with this orchestrator:

  - The atomic claim in ``v1/tasks.py`` previously filtered on
    ``status != RUNNING``, which let a brand-new PENDING task be
    claimed by an immediate follow-up ``POST /messages`` before the bg
    coroutine ever ran. Two bg coroutines could end up racing the same
    transcript and task.output.
  - ``background_task_manager.register_task`` overwrites the previous
    handle for a given ``task_id``. Combined with
    ``wait_for_previous``'s ``is current_task`` short-circuit, two
    concurrent kickoffs would each register themselves as "previous"
    and skip waiting. The orchestrator's ``_refuse_if_bg_inflight``
    closes this from the caller side.

These were both real races identified by code review; the fix is to
funnel both transports through this single chokepoint.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from ..models.task import Task, TaskStatus
from ..models.user import User

logger = logging.getLogger(__name__)


# Terminal statuses for the "is the previous turn finished?" check. A
# task in any of these is eligible for ``append_turn``. PENDING and
# RUNNING both signal "previous turn not done yet" → 409 busy. PAUSED
# is intentionally excluded for now (semantics unclear; revisit when
# pause/resume becomes a first-class SDK feature).
_TERMINAL_STATUSES = (TaskStatus.COMPLETED, TaskStatus.FAILED)


class TaskTurnError(Exception):
    """Raised when a turn cannot be started because the task is busy.

    Each transport adapter catches this and maps it to its own error
    shape:

      - ``/v1`` SDK endpoints → ``V1ApiError(TASK_BUSY, 409)``
      - WebSocket handler → broadcast an ``agent_error`` event
    """

    def __init__(self, reason: str = "busy"):
        super().__init__(reason)
        self.reason = reason


class TaskTurnOrchestrator:
    """Drive one task-turn lifecycle.

    All methods are static; the class is a namespace, not stateful.
    State lives in the database and in the global
    ``background_task_manager``.
    """

    @staticmethod
    async def start_new_turn(
        *,
        task: Task,
        user_message: str,
        user: User,
        db: Any,
        force_fresh_execution: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> "asyncio.Task[None]":
        """Start the first turn of a brand-new task, OR restart a
        terminal task with ``force_fresh_execution=True``.

        Callers (SDK ``POST /v1/chat/tasks``, WS first-message-on-new-task,
        WS restart-of-completed-task) must have already committed the
        ``Task`` row to the DB. This method:

          1. Refuses if a background coroutine is already in flight for
             this ``task_id`` (defense against ``register_task``
             overwrite race).
          2. Persists the user message to ``task_chat_messages`` so the
             bg run + GET /steps see it.
          3. Schedules ``execute_task_background`` on its own session
             with the SDK column sync hook attached.

        Args:
            task: The newly-committed Task row. ``status`` should be
                PENDING for fresh tasks; COMPLETED / FAILED is the
                restart case (use with ``force_fresh_execution=True``).
            user_message: First turn's user input (or replacement input
                if restarting). Already validated non-empty by the
                caller's Pydantic / WS message shape.
            user: The User who owns the task. Passed through for the
                bg coroutine's ``UserContext``.
            db: Caller's request-scoped session. Only used here to do
                the user-message INSERT + commit; the bg coroutine
                opens its own independent session.
            force_fresh_execution: When True, the bg coroutine ignores
                any prior reconstructible state for ``task.id`` and
                starts a fresh agent run. WS uses this for "user typed
                another message in a chat where the task was already
                COMPLETED/FAILED"; SDK doesn't currently use this path.
            context: Optional context dict (execution_mode,
                process_description, examples) merged into the bg run.

        Returns:
            The ``asyncio.Task`` wrapping the bg coroutine. Callers
            usually fire-and-forget; the handle is returned for tests.

        Raises:
            TaskTurnError("bg_inflight"): a previous bg coroutine for
                this task is still running. Caller should treat as
                busy.
        """
        _refuse_if_bg_inflight(int(task.id))

        # Persist the user message on the caller's session so the bg run
        # (which reads task_chat_messages) sees it. Caller still owns
        # commit/rollback of their session.
        from .chat_history_service import persist_user_message

        persist_user_message(
            db=db,
            task_id=int(task.id),
            user_id=int(user.id),
            content=user_message,
        )
        db.commit()

        return await _schedule_bg(
            task=task,
            user=user,
            user_message=user_message,
            force_fresh_execution=force_fresh_execution,
            context=context,
        )

    @staticmethod
    async def schedule_bg(
        *,
        task: Task,
        user_message: str,
        user: User,
        force_fresh_execution: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> "asyncio.Task[None]":
        """Schedule a bg coroutine for a task that's already been set up.

        Lower-level entry point than ``start_new_turn`` / ``append_turn``:
        it does **not** persist the user message and does **not** run an
        atomic claim. Caller is responsible for both. Used by the WS
        handler, which already persists the message and flips status to
        RUNNING inline with other UI-specific broadcast logic, but still
        needs the single-flight guard + the SDK column sync hook around
        the bg coroutine.

        Raises ``TaskTurnError('bg_inflight')`` if a previous bg
        coroutine for this task is still running.
        """
        _refuse_if_bg_inflight(int(task.id))
        return await _schedule_bg(
            task=task,
            user=user,
            user_message=user_message,
            force_fresh_execution=force_fresh_execution,
            context=context,
        )

    @staticmethod
    async def append_turn(
        *,
        task: Task,
        user_message: str,
        user: User,
        db: Any,
    ) -> "asyncio.Task[None]":
        """Append a follow-up turn to an existing task.

        Requires the previous turn to be in a terminal state
        (COMPLETED or FAILED). Atomically flips the row to RUNNING in
        the same UPDATE that records the new input — so two concurrent
        ``append_turn`` calls cannot both pass the check.

        PENDING is **rejected** as busy: a PENDING row means a previous
        turn was just scheduled but hasn't run yet, and starting another
        bg coroutine for the same task would race the first.

        Args:
            task: The task to append to (existing row). Caller already
                ran ownership / agent-id checks.
            user_message: The new user input.
            user: The User who owns the task.
            db: Caller's request-scoped session.

        Returns:
            The ``asyncio.Task`` wrapping the bg coroutine.

        Raises:
            TaskTurnError("busy"): previous turn isn't terminal (status
                is PENDING / RUNNING / PAUSED), or a bg coroutine is
                already in flight for this task_id.
        """
        # Check the bg manager BEFORE the durable UPDATE so a rejected
        # append never corrupts task.status / task.input. The bg manager's
        # running_tasks dict and the DB row are two separate sources of
        # truth — refusing here keeps the DB unchanged on rejection so
        # the previous turn's COMPLETED/FAILED state survives.
        #
        # A late check (post-UPDATE) corrupts the row in this scenario:
        # previous bg coroutine finished its run loop and flipped status
        # to COMPLETED but is still doing tail cleanup (_sync_sdk_columns
        # hasn't returned yet). A new append passes the atomic claim,
        # flips status back to RUNNING, then this guard refuses. The
        # endpoint returns 409, but the tail cleanup then sees RUNNING
        # and treats it as a stuck task, flipping status to FAILED with
        # a placeholder error_message. Net effect: a successful past
        # turn shows up as FAILED.
        #
        # Note: between this check and the UPDATE below there's a tiny
        # window for a fresh bg registration. _schedule_bg does its own
        # _refuse_if_bg_inflight before register_task, which closes that
        # window — the caller sees 409 at scheduling time instead of
        # post-UPDATE, but the DB row is still correct either way.
        _refuse_if_bg_inflight(int(task.id))

        # Atomic single-statement claim. The filter is "status is
        # terminal"; matches at most when the previous turn is fully
        # done. If two concurrent callers both run this UPDATE, only
        # one's rowcount is 1 (the other is 0 because the row's status
        # is already RUNNING by then).
        claimed = (
            db.query(Task)
            .filter(
                Task.id == task.id,
                Task.status.in_(_TERMINAL_STATUSES),
            )
            .update(
                {
                    Task.status: TaskStatus.RUNNING,
                    Task.input: user_message,
                },
                synchronize_session=False,
            )
        )
        db.commit()
        if claimed == 0:
            raise TaskTurnError("busy")

        from .chat_history_service import persist_user_message

        persist_user_message(
            db=db,
            task_id=int(task.id),
            user_id=int(user.id),
            content=user_message,
        )
        db.commit()
        db.refresh(task)

        return await _schedule_bg(
            task=task,
            user=user,
            user_message=user_message,
            force_fresh_execution=False,
            context=None,
        )


# ===== internal helpers =====


def _refuse_if_bg_inflight(task_id: int) -> None:
    """Raise ``TaskTurnError`` if the manager already has a non-done
    bg coroutine registered for this task_id.

    Why this exists: ``background_task_manager.register_task`` is a plain
    dict assignment that overwrites any previous handle. Without this
    guard, two scheduling calls in quick succession both register
    themselves; the second one's bg coroutine then calls
    ``wait_for_previous(task_id)``, which sees its own handle in the
    map and returns immediately (the ``is current_task`` short-circuit
    treats "I'm the only one registered" as "I'm previous, no wait"),
    so both bg coroutines race.

    Checking from the orchestrator side before register_task closes the
    window without touching the manager's semantics (the manager still
    works fine for the legitimate "previous task naturally completed"
    case).
    """
    from ..api.websocket import background_task_manager

    existing = background_task_manager.running_tasks.get(task_id)
    if existing is not None and not existing.done():
        raise TaskTurnError("bg_inflight")


async def _schedule_bg(
    *,
    task: Task,
    user: User,
    user_message: str,
    force_fresh_execution: bool,
    context: Optional[Dict[str, Any]],
) -> "asyncio.Task[None]":
    """Schedule the bg coroutine and register it with the manager.

    Owns the session lifecycle for the bg run: opens a fresh session
    inside ``_runner``, binds the task/user instances onto it (the
    caller's session may close before the bg coroutine finishes), runs
    ``execute_task_background``, calls ``_sync_sdk_columns``, then
    closes the session in ``finally``.
    """
    # Local imports keep this module's import surface light and avoid
    # the services -> api -> services cycle that would otherwise
    # happen at top-level import time.
    from ..api.websocket import background_task_manager, execute_task_background

    task_id = int(task.id)
    task_source = getattr(task, "source", None)
    user_id = int(user.id)  # pure int, no session affinity

    async def _runner() -> None:
        from ..models.database import get_session_local

        SessionLocal = get_session_local()
        bg_db = SessionLocal()
        try:
            bg_task = bg_db.query(Task).filter(Task.id == task_id).first()
            bg_user = bg_db.query(User).filter(User.id == user_id).first()
            if bg_task is None or bg_user is None:
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
            # SDK column sync: fill task.output / task.error_message
            # after execute_task_background returns. Wrap in try/except
            # because this coroutine runs inside asyncio.create_task --
            # an unhandled exception would be silently stored in the
            # Task's exception slot with no log.
            try:
                _sync_sdk_columns(bg_db, task_id)
            except Exception as sync_err:
                logger.error(
                    f"SDK column sync failed for task {task_id}: {sync_err}",
                    exc_info=True,
                )
        finally:
            bg_db.close()

    bg_task = asyncio.create_task(_runner())
    background_task_manager.register_task(task_id, bg_task)
    logger.info(
        f"Task {task_id} scheduled in background (source={task_source}, "
        f"force_fresh={force_fresh_execution})"
    )
    return bg_task


def _get_agent_manager() -> Any:
    """Resolve the global ``AgentServiceManager`` singleton.

    Local import keeps the services -> api boundary one-way at module
    load time.
    """
    from ..api.chat import get_agent_manager

    return get_agent_manager()


def _sync_sdk_columns(bg_db: Any, task_id: int) -> None:
    """Populate ``task.output`` / ``task.error_message`` after the bg
    coroutine returns.

    These columns are read by the SDK GET endpoint but never written by
    ``execute_task_background`` (which only updates ``task.status`` +
    inserts a row into ``task_chat_messages``). The orchestrator fills
    them so SDK clients see the latest assistant content / failure
    reason without having to query ``task_chat_messages`` themselves.

    Behavior matrix (after ``execute_task_background`` returns):

      - status == COMPLETED: read the latest assistant row from
        ``task_chat_messages`` and write to ``task.output``. Clear any
        stale ``error_message``.
      - status == FAILED: ``task.output`` stays whatever it was;
        ``error_message`` gets a generic placeholder if not already
        set. Detailed exception text is a Phase 2 follow-up (requires
        ``execute_task_background``'s except block to propagate the
        error).
      - status == RUNNING: the bg coroutine returned without updating
        status, which means its outer except block swallowed an
        exception (websocket.py ~L628). Flip the row to FAILED with a
        placeholder so SDK pollers exit the running state instead of
        polling forever.
      - status == PAUSED / other: leave alone (semantics unclear).

    Session note: ``execute_task_background`` commits via its own
    independent session (websocket.py ~L565). Our ``bg_db`` had the
    row cached in its identity map from the pre-run state, so without
    ``expire_all()`` a naive ``query(Task)`` would return the stale
    snapshot instead of the just-committed COMPLETED state.
    """
    from ..models.chat_message import TaskChatMessage

    bg_db.expire_all()

    fresh_task = bg_db.query(Task).filter(Task.id == task_id).first()
    if fresh_task is None:
        logger.warning(
            "SDK column sync skipped: task %s vanished after bg run", task_id
        )
        return

    status = fresh_task.status

    if status == TaskStatus.COMPLETED:
        latest_assistant = (
            bg_db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
            )
            .order_by(TaskChatMessage.id.desc())
            .first()
        )
        if latest_assistant is not None:
            fresh_task.output = latest_assistant.content
            fresh_task.error_message = None
            bg_db.commit()
            logger.info(
                f"SDK column sync: task {task_id} output written "
                f"({len(latest_assistant.content)} chars)"
            )
        else:
            logger.warning(
                f"SDK column sync: task {task_id} completed but "
                "no assistant message found in task_chat_messages"
            )
    elif status == TaskStatus.FAILED:
        if not fresh_task.error_message:
            fresh_task.error_message = "Task execution failed (see /steps for details)"
            bg_db.commit()
            logger.info(
                f"SDK column sync: task {task_id} marked failed "
                "with placeholder error_message"
            )
    elif status == TaskStatus.RUNNING:
        fresh_task.status = TaskStatus.FAILED
        fresh_task.error_message = (
            "Task execution failed without status update; see /steps."
        )
        bg_db.commit()
        logger.warning(
            f"SDK column sync: task {task_id} bg coroutine returned "
            "with status=RUNNING; flipping to FAILED"
        )
