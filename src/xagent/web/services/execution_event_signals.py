"""Operator signals for execution-event writes and their unknown outcomes.

Two problem classes are covered. ``report_commit_failure`` reports a V2
execution-event commit that failed after retries. ``report_unknown_outcome``
reports a case where the system does not know whether something durable
happened (a tool's external effect, a turn or command acceptance, a finalize
whose acknowledgement was lost, a settlement deferred to lease recovery).
One of its kinds, ``finalize_ack_reconciled``, is a confirmed success (a lost
acknowledgement the read-back proved applied), counted for visibility only:
alerts on the unknown-outcome counter must exclude it.

Each helper emits two kinds of signal for one occurrence: a single structured
log line that locates the task (and the tool attempt where relevant), and a
metric with bounded attributes for rates. Neither carries payload content or
exception text; exception messages can embed user data, so only type names,
identifiers, SQLSTATEs and numbers are reported. Task ids appear in logs only,
never as metric attributes.
"""

from __future__ import annotations

import logging

from ...core.agent.checkpoint import ExecutionEventPersistenceError
from ...core.runtime_performance import increment_counter
from .db_write_retry import cause_links, database_sqlstate

logger = logging.getLogger(__name__)

COMMIT_FAILURE_COUNTER = "xagent.execution_event.commit.failures"
UNKNOWN_OUTCOME_COUNTER = "xagent.execution_event.unknown_outcomes"


def _failure_type(error: BaseException) -> str:
    """Class name of the first cause that is not the persistence wrapper."""
    for link in cause_links(error):
        if not isinstance(link, ExecutionEventPersistenceError):
            return type(link).__name__
    return type(error).__name__


def report_commit_failure(
    site: str,
    error: BaseException,
    *,
    task_id: int,
    run_id: str | None,
    retried: bool,
    tool_attempt_id: str | None = None,
) -> None:
    """Report one failed V2 execution-event commit, after retries are over.

    Call it once per failed write: after retries are exhausted or for a
    failure that is not retryable, never per attempt. Never raises, so a
    failure to report cannot mask the original error.
    """
    try:
        error_type = _failure_type(error)
        increment_counter(
            COMMIT_FAILURE_COUNTER,
            attributes={"site": site, "error.type": error_type, "storage": "v2"},
        )
        message = (
            "execution_event_commit_failed task_id=%s run_id=%s storage=v2 "
            "component=execution-events site=%s error_type=%s sqlstate=%s "
            "retried=%s"
        )
        args: list[object] = [
            task_id,
            run_id or "-",
            site,
            error_type,
            database_sqlstate(error) or "-",
            retried,
        ]
        if tool_attempt_id:
            message += " tool_attempt_id=%s"
            args.append(tool_attempt_id)
        logger.error(message, *args)
    except Exception:
        # Observability must never change the behavior of the measured path.
        return


def report_unknown_outcome(
    kind: str,
    *,
    task_id: int,
    run_id: str | None = None,
    ref: str | None = None,
    storage: str = "unknown",
    level: int = logging.WARNING,
) -> None:
    """Report one case where it is unknown whether something durable happened.

    ``kind`` names the case and is the counter's ``operation`` attribute, so
    it must stay a small fixed set: ``tool_effect``, ``turn_commit``,
    ``command_commit``, ``finalize_ack_unresolved``,
    ``resume_finalize_unreconciled``, ``settlement_deferred`` -- and
    ``finalize_ack_reconciled``, which is not unknown but a confirmed success
    (the read-back proved a lost acknowledgement's COMMIT applied), reported
    at INFO and counted for visibility only: alerts on this counter must
    exclude it. ``storage`` is ``v1``, ``v2`` or ``unknown``; ``turn_commit``,
    ``command_commit`` and ``settlement_deferred`` report ``unknown`` because
    the storage version is not in scope where they occur. ``ref`` is an
    identifier only (a turn id, a command id, a tool attempt id), never
    text. Call it once per occurrence. Never raises, so a failure to report
    cannot change the path being measured.
    """
    # Observability must never change the behavior of the measured path.
    # Two independent guards: a failing metric must not suppress the log line.
    try:
        increment_counter(
            UNKNOWN_OUTCOME_COUNTER,
            attributes={"operation": kind, "storage": storage},
        )
    except Exception:
        pass
    try:
        logger.log(
            level,
            "execution_event_outcome_unknown task_id=%s run_id=%s storage=%s "
            "component=execution-events kind=%s ref=%s",
            task_id,
            run_id or "-",
            storage,
            kind,
            ref or "-",
        )
    except Exception:
        pass
