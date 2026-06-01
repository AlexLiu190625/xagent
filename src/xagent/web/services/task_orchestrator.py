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

Both races are prevented by funneling the WebSocket and /v1 transports
through this single turn-lifecycle chokepoint -- the atomic claim
filter and ``_refuse_if_bg_inflight`` guard close them at the
orchestrator boundary.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..models.task import Task, TaskStatus
from ..models.user import User
from .hot_path_cache import invalidate_task_cache
from .task_lease_service import (
    acquire_task_lease_isolated,
    get_runner_id,
    run_task_lease_heartbeat,
)
from .task_setup_snapshot import load_task_setup_snapshot_sync
from .workforce_runtime import release_current_runner_task_lease_with_workforce_sync

logger = logging.getLogger(__name__)


# Statuses for the "can a user message start the next turn?" check. A
# task in any of these is eligible for ``TurnKind.APPEND``. PENDING is
# claimed by ``CREATE``; RUNNING is still busy; WAITING_FOR_USER is an
# answer to an explicit pending agent question and resumes that execution.
_APPENDABLE_STATUSES = (
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.PAUSED,
)


@dataclass(frozen=True)
class TaskTurnPayload:
    """Both message representations a single turn carries.

    A turn has two distinct message channels, and collapsing them into
    a single string loses the WS file-context input on its way to the
    LLM:

    - ``transcript_message`` — what gets persisted to
      ``task_chat_messages`` and shown back to the user / GET endpoint
    - ``execution_message`` — what the agent / LLM actually consumes;
      may be file-enriched, system-prefix-augmented, etc.

    When ``execution_message`` is ``None``, ``for_agent`` falls back to
    ``transcript_message`` (typical for SDK callers which only have one
    representation). WS callers pass both because the file-context
    append for the LLM input is intentionally not shown verbatim in the
    transcript.
    """

    transcript_message: str
    execution_message: Optional[str] = None
    # Per-turn uploaded-file metadata persisted alongside the transcript
    # row so historical replay can render the same clickable chips the
    # user saw live. Each entry is the minimal chip shape (file_id,
    # name, size, type) — already path-stripped by the websocket layer
    # before reaching here.
    attachments: Optional[List[Dict[str, Any]]] = None
    # Stable identity shared by the transcript row and the user_message trace
    # event for this user turn. Historical replay uses it to merge persisted
    # transcript rows with trace rows without collapsing repeated text.
    turn_id: str = field(default_factory=lambda: str(uuid4()))

    @property
    def for_agent(self) -> str:
        return self.execution_message or self.transcript_message


class TurnKind(str, enum.Enum):
    """Which transition the turn represents.

    ``kind`` answers "which status filter does the atomic claim use".
    Orthogonal to ``force_fresh`` (passed alongside to ``begin_turn``),
    which answers "does the agent reconstruct prior execution state or
    start fresh". The two cover four logical combinations; only three
    are reachable in practice (CREATE + force_fresh has no meaning
    because a brand-new task has no prior state to discard — see the
    assert in ``begin_turn``).

    Continuation paths (PAUSED / WAITING_FOR_USER resumed onto the same
    turn) are deliberately not modeled here: they go through
    ``dag_pattern.request_continuation`` instead, because continuation
    is the *same* turn picking up where it paused — terminal-field reset
    would be wrong.
    """

    CREATE = "create"  # PENDING → RUNNING; new task's first turn
    APPEND = "append"  # APPENDABLE → RUNNING; new turn on an existing task


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
    async def begin_turn(
        *,
        task: Task,
        payload: TaskTurnPayload,
        user: User,
        db: Any,
        kind: TurnKind,
        force_fresh: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> "asyncio.Task[None]":
        """Single entry for any new-turn transition (CREATE / APPEND).

        The single transactional primitive that owns the full turn-start
        contract:

          1. Refuse if a bg coroutine is still in flight for this task
             (``TaskTurnError("bg_inflight")``). Checked before any DB
             write so a rejected turn never mutates the row.
          2. Atomic UPDATE in one statement on the caller's session
             — the latest-turn snapshot invariant:

             - ``status = RUNNING``
             - ``input = payload.transcript_message``
             - ``output = NULL``          (clear prior-turn terminal field)
             - ``error_message = NULL``   (clear prior-turn terminal field)

             with filter:

             - ``kind == CREATE`` → ``status == PENDING``
             - ``kind == APPEND`` → ``status IN APPENDABLE_STATUSES``

             rowcount 0 → ``TaskTurnError("busy")``.
          3. ``persist_user_message(payload.transcript_message)`` in the
             same session (no commit yet).
          4. ``db.commit()`` ONCE for steps 2 + 3 together — the
             single-transaction turn-start contract. If any step above
             raises, the session is rolled back and neither the status
             flip nor the message persists, so a rejected turn never
             leaves an orphan user message in the transcript.
          5. Schedule the bg coroutine via ``_schedule_bg``, passing
             the full payload so the execution side receives the
             execution-only message channel, not just the transcript.

        Preconditions (caller contract — assert on entry):

          - ``db`` is a clean session: no uncommitted ``new`` /
            ``dirty`` / ``deleted`` instances. ``begin_turn`` commits
            the atomic UPDATE + message persist together; a dirty
            caller session would have its pending changes committed
            alongside, which is rarely the caller's intent. All current
            callers (SDK ``create_chat_task``, SDK
            ``append_message_to_task``, WS path) satisfy this; the
            assert is defensive against future callers.
          - ``kind == CREATE and force_fresh`` is invalid — a new task
            has no prior execution state to discard. Raises
            ``ValueError``.

        Args:
            task: The committed Task row. ``status`` should be PENDING
                for ``kind=CREATE`` or terminal for ``kind=APPEND``.
            payload: Two-channel message (transcript + execution); see
                :class:`TaskTurnPayload`.
            user: Task owner; passed through to the bg coroutine's
                ``UserContext``.
            db: Caller's request-scoped session. Used for steps 2-4
                (atomic claim + persist + commit). The bg coroutine
                opens its own independent session inside
                ``_schedule_bg``.
            kind: Which status filter the atomic claim uses; see
                :class:`TurnKind`.
            force_fresh: When True, the bg coroutine ignores any
                reconstructible prior execution state and starts a
                fresh agent run. WS terminal-task re-engage passes
                True; SDK callers pass False.
            context: Optional execution-context dict
                (execution_mode / process_description / examples)
                merged into the bg run.

        Returns:
            The ``asyncio.Task`` wrapping the bg coroutine. Callers
            usually fire-and-forget; the handle is returned for tests.

        Raises:
            ValueError: invalid ``kind`` / ``force_fresh`` combination,
                or caller session not clean.
            TaskTurnError("bg_inflight"): a previous bg coroutine for
                this task is still running.
            TaskTurnError("busy"): atomic claim filter mismatched the
                current row status (e.g. ``kind=APPEND`` against a
                non-terminal row).
        """
        if kind == TurnKind.CREATE and force_fresh:
            raise ValueError(
                "force_fresh has no meaning for kind=CREATE — a new task "
                "has no prior execution state to discard"
            )

        # Session-clean precondition. Catching this up-front prevents the
        # commit at step 4 from accidentally persisting unrelated
        # caller-staged objects.
        if db.new or db.dirty or db.deleted:
            raise ValueError(
                "begin_turn requires a clean db session "
                f"(new={len(db.new)}, dirty={len(db.dirty)}, "
                f"deleted={len(db.deleted)}); caller must commit or "
                "rollback its pending changes before calling begin_turn"
            )

        task_id = int(task.id)

        # Step 1: bg-inflight guard before any DB write.
        _refuse_if_bg_inflight(task_id)

        # Step 2 + 3 + 4: atomic claim + persist + single commit. We
        # don't commit between 2 and 3, so a failure at step 3 rolls
        # back step 2 cleanly — the rejected-turn-leaves-no-side-effect
        # contract.
        try:
            if kind == TurnKind.CREATE:
                status_filter = Task.status == TaskStatus.PENDING
            else:  # APPEND
                status_filter = Task.status.in_(_APPENDABLE_STATUSES)

            claimed = (
                db.query(Task)
                .filter(Task.id == task_id, status_filter)
                .update(
                    {
                        Task.status: TaskStatus.RUNNING,
                        Task.input: payload.transcript_message,
                        Task.output: None,
                        Task.error_message: None,
                    },
                    synchronize_session=False,
                )
            )
            if claimed == 0:
                db.rollback()
                raise TaskTurnError("busy")

            from .chat_history_service import persist_user_message_no_commit

            persisted_message = persist_user_message_no_commit(
                db=db,
                task_id=task_id,
                user_id=int(user.id),
                content=payload.transcript_message,
                attachments=payload.attachments,
                turn_id=payload.turn_id,
            )
            if persisted_message is not None:
                db.flush()
                before_message_id = int(persisted_message.id)
            else:
                before_message_id = None
            db.commit()
            invalidate_task_cache(task_id)
        except TaskTurnError:
            raise
        except Exception:
            db.rollback()
            raise

        db.refresh(task)

        # Step 5: hand off to lease-aware scheduler.
        return await _schedule_bg(
            task=task,
            user=user,
            payload=payload,
            force_fresh=force_fresh,
            context=context,
            before_message_id=before_message_id,
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


def _get_agent_manager() -> Any:
    """Resolve the global ``AgentServiceManager`` singleton.

    Local import keeps the services -> api boundary one-way at module
    load time.
    """
    from ..api.chat import get_agent_manager

    return get_agent_manager()


def _mark_task_failed_if_running(task_id: int, error_message: str) -> None:
    """Setup/run-error sentinel for ``_schedule_bg._runner``.

    ``acquire_task_lease_isolated`` sets ``task.status = RUNNING`` as
    part of taking the lease. If a later step in ``_runner`` raises
    (snapshot load, ``execute_task_background``) and no downstream
    handler moves the task to a terminal status, the release block
    would see ``status=RUNNING`` and write it back -- leaving the row
    visible as running but with no active worker (zombie state). This
    helper closes that window: ``_runner`` calls it from an outer
    ``except`` so the task is forced to ``FAILED`` before release.

    Guarded by ``status == RUNNING`` -- never overwrites a terminal /
    control status (``PAUSED`` / ``WAITING_FOR_USER`` / ``FAILED`` /
    ``COMPLETED``) that ``execute_task_background`` may have set
    inside its own inner ``try/except``. Opens / commits / closes
    its own session so the caller doesn't have to thread a session
    through the exception path.
    """
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    try:
        with SessionLocal() as db:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task is None or task.status != TaskStatus.RUNNING:
                return
            task.status = TaskStatus.FAILED
            task.error_message = error_message  # type: ignore[assignment]
            db.commit()
    except Exception as e:
        # Defensive: do not let this helper raise out of the ``except``
        # path that's already handling an error. Log loudly so the
        # zombie state, if it survives, is traceable.
        logger.error(
            "Failed to mark task %s as FAILED during setup/run error: %s",
            task_id,
            e,
            exc_info=True,
        )


# ===== finish_turn / _schedule_bg (new lifecycle API) =====


def finish_turn(bg_db: Any, task_id: int) -> None:
    """Symmetric terminal-field writer with lease ownership guard.

    Called from ``_schedule_bg._runner`` after ``execute_task_background``
    returns. Two key properties:

      - latest-turn snapshot invariant: COMPLETED, FAILED, and the
        RUNNING-fallback branch all leave the row in a state where the
        terminal field that *doesn't* apply to the current turn is
        cleared (COMPLETED clears ``error_message``; FAILED clears
        stale ``output``). SDK consumers reading ``/v1/chat/tasks/{id}``
        therefore never see a contradictory snapshot like
        ``status='failed' + output='prior successful answer'``.
      - lease ownership guard: the RUNNING-fallback branch refuses to
        flip the row to FAILED while another worker still holds a live
        lease, so a slow scheduler in this process can't overwrite the
        in-flight execution result of a different process.

    Uses :func:`get_runner_id` internally rather than accepting
    runner_id as a parameter so the comparison always reads the
    canonical process runner id and a separately-captured
    ``lease.runner_id`` can't drift from it.

    Branches:

      - ``status == COMPLETED``: set ``output`` from latest assistant
        message, clear ``error_message``
      - ``status == FAILED``: set ``error_message`` placeholder if
        absent, clear stale ``output``
      - ``status == RUNNING`` + other worker holds live lease: skip
        entirely (ownership guard)
      - ``status == RUNNING`` + we own lease or it's expired: flip to
        FAILED, set placeholder ``error_message``, clear stale
        ``output``
      - other statuses (PAUSED / WAITING_FOR_USER): leave alone
    """
    from ..models.chat_message import TaskChatMessage
    from .workforce_runtime import sync_workforce_run_status

    bg_db.expire_all()

    fresh = bg_db.query(Task).filter(Task.id == task_id).first()
    if fresh is None:
        logger.warning("finish_turn: task %s vanished after bg run", task_id)
        return

    status = fresh.status

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
            fresh.output = latest_assistant.content
            fresh.error_message = None
            sync_workforce_run_status(bg_db, fresh, TaskStatus.COMPLETED)
            bg_db.commit()
            invalidate_task_cache(task_id)
            logger.info(
                "finish_turn: task %s output written (%d chars)",
                task_id,
                len(latest_assistant.content),
            )
        else:
            logger.warning(
                "finish_turn: task %s completed but no assistant message found",
                task_id,
            )
            if sync_workforce_run_status(bg_db, fresh, TaskStatus.COMPLETED):
                bg_db.commit()
                invalidate_task_cache(task_id)
        return

    if status == TaskStatus.FAILED:
        changed = False
        if not fresh.error_message:
            fresh.error_message = "Task execution failed (see /steps for details)"
            changed = True
        if fresh.output is not None:
            # Latest-turn snapshot invariant: a failed turn must not
            # carry forward prior
            # successful output. SDK consumers reading the row otherwise
            # see a contradiction (status=failed + output populated).
            fresh.output = None
            changed = True
        run_changed = sync_workforce_run_status(bg_db, fresh, TaskStatus.FAILED)
        if changed or run_changed:
            bg_db.commit()
            invalidate_task_cache(task_id)
            logger.info(
                "finish_turn: task %s marked failed (cleared stale output)",
                task_id,
            )
        return

    if status == TaskStatus.RUNNING:
        # Lease ownership guard: a live lease held by another worker
        # means that worker is actively executing this task; we must
        # not overwrite its in-flight result with a FAILED snapshot.
        # ``lease_expires_at`` comes back tz-naive from SQLite (the column is
        # DateTime(timezone=True) but SQLite stores only the naked timestamp);
        # normalize to UTC so the comparison stays dialect-agnostic.
        expires_at = fresh.lease_expires_at
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        live_other_owner = (
            fresh.runner_id is not None
            and fresh.runner_id != get_runner_id()
            and expires_at is not None
            and expires_at > datetime.now(timezone.utc)
        )
        if live_other_owner:
            logger.info(
                "finish_turn: task %s owned by runner %s, lease alive "
                "until %s; skipping RUNNING fallback",
                task_id,
                fresh.runner_id,
                fresh.lease_expires_at,
            )
            return
        # Genuinely stuck: our bg coroutine returned, no live lease elsewhere.
        fresh.status = TaskStatus.FAILED
        fresh.error_message = "Task execution failed without status update; see /steps."
        fresh.output = None  # latest-turn snapshot invariant
        sync_workforce_run_status(bg_db, fresh, TaskStatus.FAILED)
        bg_db.commit()
        invalidate_task_cache(task_id)
        logger.warning(
            "finish_turn: task %s bg coroutine returned with status=RUNNING; "
            "flipping to FAILED",
            task_id,
        )
        return

    # PAUSED / WAITING_FOR_USER / other: leave alone.


async def _schedule_bg(
    *,
    task: Task,
    user: User,
    payload: TaskTurnPayload,
    force_fresh: bool,
    context: Optional[Dict[str, Any]],
    before_message_id: Optional[int] = None,
) -> "asyncio.Task[None]":
    """Lease-aware bg scheduler.

    Owns the full lease lifecycle for the bg run:

      - acquire at ``_runner`` entry. If another worker already holds
        the lease the scheduler returns immediately without invoking
        ``execute_task_background`` or ``finish_turn`` — the
        running-elsewhere short-circuit. ``finish_turn``'s ownership
        guard would catch the same situation a level deeper, but
        skipping at the entry means we never even attempt local work
        on a task another worker is executing.
      - heartbeat alongside the run.
      - release in ``finally`` as the single owner of the release
        call, regardless of whether ``execute_task_background``
        returned normally or raised. ``execute_task_background`` only
        writes ``task.status`` and never touches the lease columns;
        the scheduler is responsible for the whole lease lifecycle.
    """
    from ..api.websocket import background_task_manager, execute_task_background

    task_id = int(task.id)
    task_source = getattr(task, "source", None)
    user_id = int(user.id)

    async def _runner() -> None:
        # ``bg_db`` is opened lazily inside the post-run finalize block
        # only. We no longer keep a SessionLocal open across the entire
        # agent run -- that previously held an idle connection-pool
        # slot for tens of seconds to minutes (long-running agents)
        # without doing any work. The lease acquire / heartbeat /
        # snapshot load all open their own short-lived sessions, and
        # ``finish_turn`` + release run inside a single ``with`` block
        # below.
        from ..models.database import get_session_local

        lease = None
        try:
            # Running-elsewhere short-circuit: acquire lease before
            # doing anything else. If another worker owns it, skip
            # execution entirely so finish_turn never touches the row.
            #
            # The acquire is a conditional UPDATE + commit that
            # measured 3.75s of synchronous DB write on the main
            # event loop (issue #427). ``acquire_task_lease_isolated``
            # wraps the existing helper with its own SessionLocal so
            # the work runs on a worker thread.
            lease = await asyncio.to_thread(acquire_task_lease_isolated, task_id)
            if lease is None:
                logger.info(
                    "task %s acquired by another worker; skipping "
                    "execution and finish_turn",
                    task_id,
                )
                return

            # INVARIANT: ``asyncio.create_task(run_task_lease_heartbeat(...))``
            # MUST be scheduled before any ``await`` that yields the
            # loop (snapshot to_thread, agent setup, execute_task_background).
            # The lease has a bounded TTL; nothing downstream of acquire
            # may ride bare past this point. If a future refactor moves
            # the heartbeat creation below the snapshot load, a
            # contended worker could drop the lease while snapshot is
            # in-flight, hand the task to another runner mid-setup, and
            # double-run the same turn. Do not reorder.
            stop_event = asyncio.Event()
            hb_task = asyncio.create_task(run_task_lease_heartbeat(lease, stop_event))
            try:
                # Outer ``try/except`` is the lease-acquire-to-terminal
                # safety net: ``acquire_task_lease_isolated`` already
                # set ``status=RUNNING`` for this row, so any unhandled
                # exception from snapshot load / execute_task_background
                # would leave the task in a zombie state (visible as
                # running, no active worker) once the release block
                # below clears ``runner_id``. ``_mark_task_failed_if_running``
                # closes the window. We swallow the exception so
                # ``finish_turn`` + lease release still run cleanly with
                # the now-terminal status.
                try:
                    # Load the synchronous DB block on a worker thread
                    # so the main loop stays responsive. The loader
                    # opens / closes its own SessionLocal (no ORM
                    # leak), and the snapshot is passed straight
                    # through to execute_task_background →
                    # get_agent_for_task. That turns the previous
                    # chain of three redundant Task queries into a
                    # single off-loop read.
                    snapshot = await asyncio.to_thread(
                        load_task_setup_snapshot_sync, task_id, user_id
                    )
                    if snapshot is None:
                        logger.warning(
                            "bg task %s aborted: task vanished before snapshot load",
                            task_id,
                        )
                        _mark_task_failed_if_running(
                            task_id, "task vanished before snapshot load"
                        )
                        return

                    await execute_task_background(
                        task_id=task_id,
                        user_message=payload.transcript_message,
                        context=_execution_context_with_turn_id(
                            context, payload.turn_id
                        ),
                        agent_manager=_get_agent_manager(),
                        user_id=user_id,
                        before_message_id=before_message_id,
                        llm_user_message=payload.execution_message,
                        task_setup_snapshot=snapshot,
                    )
                except Exception as setup_or_run_err:
                    logger.error(
                        "bg task %s setup/run failed: %s",
                        task_id,
                        setup_or_run_err,
                        exc_info=True,
                    )
                    _mark_task_failed_if_running(
                        task_id,
                        f"setup/run error: "
                        f"{type(setup_or_run_err).__name__}: {setup_or_run_err}",
                    )
                    # Do not re-raise: ``finish_turn`` + release below
                    # must run so the lease is freed and the row is
                    # not stuck mid-lifecycle.

                # Short-lived finalize session. ``finish_turn`` only
                # reads / updates the task row once; opening here keeps
                # the pool slot freed for the entire agent run above.
                SessionLocal = get_session_local()
                with SessionLocal() as finalize_db:
                    try:
                        finish_turn(finalize_db, task_id)
                    except Exception as e:
                        logger.error(
                            "finish_turn failed for task %s: %s",
                            task_id,
                            e,
                            exc_info=True,
                        )
            finally:
                stop_event.set()
                try:
                    await hb_task
                except Exception:
                    pass
        finally:
            if lease is not None:
                # Single owner of release. Open a fresh short-lived
                # session for both the status read and the release UPDATE
                # so we don't hold a connection across the agent run.
                # Defensive: if the read raises (DB connectivity issue),
                # default to FAILED so the lease still gets released
                # instead of stuck-until-TTL.
                SessionLocal = get_session_local()
                with SessionLocal() as release_db:
                    final_status: TaskStatus = TaskStatus.FAILED
                    try:
                        fresh = (
                            release_db.query(Task).filter(Task.id == task_id).first()
                        )
                        if fresh is not None:
                            final_status = fresh.status
                    except Exception as query_err:
                        logger.warning(
                            "task %s status read failed during lease release "
                            "(%s); rolling session back and defaulting to FAILED",
                            task_id,
                            query_err,
                        )
                        try:
                            release_db.rollback()
                        except Exception:
                            pass
                    # Use the workforce-aware release helper: it wraps
                    # ``release_current_runner_task_lease`` (signature
                    # unchanged) and additionally syncs the workforce
                    # run status when the released task belongs to one.
                    # Both PR #461 (short-open/short-close release_db
                    # pattern) and PR #528 (workforce sync) compose
                    # cleanly here -- decorator-style, no perf regression.
                    try:
                        release_current_runner_task_lease_with_workforce_sync(
                            release_db, task_id, status=final_status
                        )
                    except Exception as e:
                        logger.warning(
                            "lease release failed for task %s: %s; "
                            "TTL expiry will reclaim it",
                            task_id,
                            e,
                        )

    bg_task = asyncio.create_task(_runner())
    background_task_manager.register_task(task_id, bg_task)
    logger.info(
        "task %s scheduled in background v2 (source=%s, force_fresh=%s)",
        task_id,
        task_source,
        force_fresh,
    )
    return bg_task


def _execution_context_with_turn_id(
    context: Optional[Dict[str, Any]], turn_id: str
) -> Dict[str, Any]:
    execution_context = dict(context or {})
    if turn_id:
        execution_context["turn_id"] = turn_id
    return execution_context
