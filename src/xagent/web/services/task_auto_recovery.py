"""Interruption bookkeeping for automatic task recovery.

When a run is interrupted rather than finished, the settling transaction
records why in ``task_auto_recovery`` (one row per task) and appends a
``task_recovery_events`` row. Those rows are metadata only: nothing here
changes a task's status, control state, error message or lifecycle
projections, which stay the settling writer's own decision.

Phase 1 has no executor, so a recorded interruption is never ``scheduled``.
Its state says only that nothing automatic will act on the run: ``manual``
(only the user can act -- resume a PAUSED task; a FAILED one is terminal and
the row just records why), ``ineligible`` (a task kind automatic recovery will
never touch) or ``disabled`` (the settling path is gated by
``XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED`` and that switch is off).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from ...config import (
    get_shared_task_execution_enabled,
    get_task_infra_failure_pause_enabled,
)
from ...core.agent.checkpoint import checkpoint_progress_marker
from ...core.agent.interruption import InterruptionReason
from ..models.task import Task, TaskStatus
from ..models.task_auto_recovery import (
    TaskAutoRecovery,
    TaskAutoRecoveryState,
    TaskRecoveryEvent,
    TaskRecoveryEventType,
)
from ..models.trigger import TriggerType
from ..models.workforce import WorkforceRun
from .task_execution_controller import TaskControlState
from .task_lease_service import (
    CheckpointRecoveryResolution,
    CheckpointRecoveryVerdict,
    TaskLeaseRecoveryCandidate,
)
from .workforce_runtime import extract_workforce_run_id

logger = logging.getLogger(__name__)

# Callers that own their own protocol for a stopped run: SDK and A2A clients,
# external cancellation, and anonymous visitors of a widget or shared link.
_INELIGIBLE_SOURCES = frozenset({"sdk", "a2a", "external", "widget", "shared_link"})
_TRIGGER_TYPES = frozenset(member.value for member in TriggerType)


@dataclass(frozen=True)
class AutoRecoveryEligibility:
    """Whether a task may ever be resumed without its user.

    ``kind`` names the eligible task family (later policy picks windows by
    it); ``detail`` is the ``ineligible:<why>`` label stored on the row.
    """

    eligible: bool
    kind: str | None = None
    detail: str | None = None


def _ineligible(why: str) -> AutoRecoveryEligibility:
    return AutoRecoveryEligibility(eligible=False, detail=f"ineligible:{why}")


def auto_recovery_eligibility(db: Session, task: Task) -> AutoRecoveryEligibility:
    """Classify ``task`` for automatic recovery; the single source of truth.

    Rules apply in order and the first match wins:

    1. SDK, A2A, external, widget and shared-link tasks are ineligible.
    2. Workforce tasks are eligible unless their run is a preview (or the
       run row is gone, so it cannot be shown not to be one).
    3. Trigger tasks are eligible.
    4. Channel tasks are eligible only under shared execution; a combined
       host runs them inline in the bot, which has no durable way to deliver
       an answer produced by a later resume.
    5. Hidden internal tasks are previews (agent builder, preview chat).
    6. Everything else is an eligible normal task.
    """

    source = str(task.source or "internal")
    if source in _INELIGIBLE_SOURCES:
        return _ineligible(f"source_{source}")

    workforce_run_id = extract_workforce_run_id(task)
    if workforce_run_id is not None:
        run = db.get(WorkforceRun, workforce_run_id)
        if run is None:
            return _ineligible("workforce_run_missing")
        if run.is_preview:
            return _ineligible("preview")
        return AutoRecoveryEligibility(eligible=True, kind="workforce")

    if source == "trigger":
        agent_config = task.agent_config
        trigger_type = (
            agent_config.get("trigger_type") if isinstance(agent_config, dict) else None
        )
        return AutoRecoveryEligibility(
            eligible=True,
            kind=(
                f"trigger_{trigger_type}"
                if isinstance(trigger_type, str) and trigger_type in _TRIGGER_TYPES
                else "trigger"
            ),
        )

    if task.channel_id is not None:
        if not get_shared_task_execution_enabled():
            return _ineligible("channel_inline")
        return AutoRecoveryEligibility(eligible=True, kind="channel")

    if source == "internal" and task.is_visible is False:
        return _ineligible("preview")

    return AutoRecoveryEligibility(eligible=True, kind="normal")


def record_interruption_no_commit(
    db: Session,
    *,
    task: Task,
    reason: InterruptionReason,
    task_status: TaskStatus,
    interrupted_at: datetime,
    progress_marker: str | None,
    gated_by_infra_pause_switch: bool,
) -> TaskAutoRecovery | None:
    """Upsert ``task``'s recovery row for one interruption and log the event.

    ``task`` must be the row as the settling write left it, in the same
    transaction: its ``run_id`` and ``state_version`` become the row's run
    and fence. A run without an id is not recorded -- it cannot be fenced.

    ``gated_by_infra_pause_switch`` says whether the caller's PAUSED outcome
    depends on ``XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED``. Settlement paths
    that pause instead of failing are; lease-expiry recovery is not (it paused
    before this switch existed), so it never records ``disabled``. An
    ineligible task records ``ineligible`` whatever the switch says.

    ``paused_state_version`` is the task's ``state_version`` after the
    settling write, PAUSED or FAILED; only a PAUSED row's fence is ever read.

    Episodes: a recorded interruption of the same run at the same progress
    marker continues the current no-progress episode (its start and
    ``no_progress_resumes`` are kept). Any progress starts a new episode, and
    a different run also restarts ``total_resumes``. A missing marker (an
    undecodable or absent checkpoint) is unknown, not progress: it continues
    the episode and keeps the last known marker.
    """

    run_id = task.run_id
    if run_id is None:
        return None

    eligibility = auto_recovery_eligibility(db, task)
    # Ineligibility is permanent and the switch is not, so it wins: a
    # ``disabled`` row always means "turning the switch on would help".
    if not eligibility.eligible:
        state, state_detail = TaskAutoRecoveryState.INELIGIBLE, eligibility.detail
    elif gated_by_infra_pause_switch and not get_task_infra_failure_pause_enabled():
        state, state_detail = TaskAutoRecoveryState.DISABLED, None
    else:
        state, state_detail = TaskAutoRecoveryState.MANUAL, None

    row = db.get(TaskAutoRecovery, task.id)
    same_run = row is not None and row.run_id == run_id
    if same_run and row is not None and progress_marker is None:
        progress_marker = row.progress_marker
    same_episode = (
        same_run
        and row is not None
        and (row.progress_marker is None or row.progress_marker == progress_marker)
    )
    if row is None:
        row = TaskAutoRecovery(task_id=task.id)
        db.add(row)
    if not same_episode:
        row.episode_started_at = interrupted_at
        row.no_progress_resumes = 0
    if not same_run:
        row.total_resumes = 0
        row.last_command_id = None
    row.run_id = run_id
    row.reason = reason.value
    row.state = state.value
    row.state_detail = state_detail
    row.paused_state_version = int(task.state_version or 0)
    row.interrupted_at = interrupted_at
    row.progress_marker = progress_marker
    row.next_attempt_at = None
    row.last_error = None

    detail: dict[str, Any] = {"task_status": task_status.value, "state": state.value}
    if state_detail is not None:
        detail["state_detail"] = state_detail
    if eligibility.kind is not None:
        detail["kind"] = eligibility.kind
    db.add(
        TaskRecoveryEvent(
            task_id=task.id,
            run_id=run_id,
            event=(
                TaskRecoveryEventType.INELIGIBLE
                if state is TaskAutoRecoveryState.INELIGIBLE
                else TaskRecoveryEventType.INTERRUPTED
            ).value,
            reason=reason.value,
            detail=detail,
        )
    )
    db.flush()
    logger.info(
        "Recorded task interruption: task_id=%s run_id=%s reason=%s state=%s "
        "component=auto-recovery",
        task.id,
        run_id,
        reason.value,
        state.value,
    )
    return row


def lease_expiry_interruption_reason(
    candidate: TaskLeaseRecoveryCandidate,
    verdict: CheckpointRecoveryVerdict,
) -> InterruptionReason | None:
    """Why lease recovery stopped ``candidate``'s run, by its verdict.

    A recoverable run becomes PAUSED: a user pause when the run died while
    PAUSE_REQUESTED (the user already decided), else ``lease_expired``. The
    FAILED verdicts keep their own terminal reasons. ``INDETERMINATE`` writes
    nothing, so it has no reason.
    """

    if verdict is CheckpointRecoveryVerdict.RECOVERABLE:
        if candidate.control_state == TaskControlState.PAUSE_REQUESTED.value:
            return InterruptionReason.USER_PAUSE
        return InterruptionReason.LEASE_EXPIRED
    if verdict is CheckpointRecoveryVerdict.UNKNOWN_TOOL_EFFECT:
        return InterruptionReason.UNKNOWN_TOOL_EFFECT
    if verdict is CheckpointRecoveryVerdict.NOT_RECOVERABLE:
        return InterruptionReason.NOT_RECOVERABLE
    return None


def resolution_progress_marker(
    db: Session, task_id: int, resolution: CheckpointRecoveryResolution
) -> str | None:
    """Progress fingerprint of a recoverable verdict's checkpoint, if any.

    A legacy payload is decoded first, because its message list and tool
    ledger may be stored as refs; only counts are needed, so blob hashes are
    not verified. Refs that fail to decode (``CheckpointMessageDecodeError``)
    still record the interruption, just without a marker; any other lookup
    error propagates to the caller's SAVEPOINT and skips the row.
    """

    from .trace_message_storage import (
        CheckpointMessageDecodeError,
        decode_trace_event_data,
    )

    if (
        resolution.verdict is not CheckpointRecoveryVerdict.RECOVERABLE
        or resolution.checkpoint is None
    ):
        return None
    data: Any = resolution.checkpoint
    if resolution.encoded:
        try:
            data = decode_trace_event_data(
                db,
                task_id=task_id,
                data=data,
                strict=True,
                verify_blob_hashes=False,
            )
        except CheckpointMessageDecodeError:
            logger.warning(
                "Task %s checkpoint refs are undecodable; recording its "
                "interruption without a progress marker",
                task_id,
            )
            return None
    snapshot = data.get("snapshot") if isinstance(data, dict) else None
    return checkpoint_progress_marker(snapshot if isinstance(snapshot, dict) else None)


def record_lease_expiry_interruption_no_commit(
    db: Session,
    *,
    task: Task,
    candidate: TaskLeaseRecoveryCandidate,
    resolution: CheckpointRecoveryResolution,
    task_status: TaskStatus,
    recovered_at: datetime,
) -> None:
    """Record a lease recovery's interruption inside its open transaction.

    Runs in a SAVEPOINT and does not raise past it for a failure inside the
    recording itself:
    the fenced status write and its projections must commit exactly as they
    would without this metadata. Rolling the whole recovery back instead
    would retry it next tick, but a failure that repeats deterministically
    (a schema the migration has not reached, a value the row cannot hold)
    would then leave the task RUNNING with an expired lease forever. A
    skipped row only means the task takes no part in automatic recovery,
    which is the safe direction. If the SAVEPOINT itself cannot be rolled
    back, or opening or releasing it fails, the transaction itself is broken
    and the commit would fail anyway; that error propagates and the whole
    recovery retries next tick.

    Preconditions, both met by the lease-recovery transaction today:

    - The fenced status write has already run in this transaction: on
      SQLite that UPDATE is what opens it, so the SAVEPOINT is nested rather
      than autocommitted.
    - Nothing staged earlier in the transaction is still pending in the
      session. ``flush`` has no scope, so pending objects would be emitted
      inside the SAVEPOINT and discarded with it on failure. Sessions are
      built with ``autoflush=False`` and every earlier staging step (the
      orphaned-delivery reconciliation) flushes itself; a new step added
      before this call must do the same.
    """

    reason = lease_expiry_interruption_reason(candidate, resolution.verdict)
    if reason is None:
        return
    savepoint = db.begin_nested()
    try:
        record_interruption_no_commit(
            db,
            task=task,
            reason=reason,
            task_status=task_status,
            interrupted_at=recovered_at,
            progress_marker=resolution_progress_marker(db, int(task.id), resolution),
            gated_by_infra_pause_switch=False,
        )
    except Exception:
        savepoint.rollback()
        logger.exception(
            "Recording the lease-expiry interruption of task %s failed; "
            "recovering it without auto-recovery metadata",
            task.id,
        )
        return
    savepoint.commit()
