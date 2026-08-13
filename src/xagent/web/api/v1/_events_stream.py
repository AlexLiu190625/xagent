"""SSE transport layer for ``GET /v1/chat/tasks/{task_id}/events``.

This module owns everything about *moving bytes to the client and
deciding when to stop* -- registration into the shared connection
``manager``, the outbound frame queue, the 30-second watchdog, the
1-hour absolute duration cap, and per-task / per-principal concurrency
limits. It emits four lifecycle events -- ``task.status``,
``task.completed``, ``task.input_required``, and ``stream.error`` --
and nothing else.

Explicitly out of scope here:
  - Projecting ``step.*`` / ``message.*`` content from trace events.
  - Buffering live frames during attach warm-up. Without that buffer,
    a live status update emitted between sink registration and the
    first ``task.status`` frame can arrive out of order or duplicate
    the first frame; this is accepted. Worst case the client sees one
    extra, stale ``task.status``. That's harmless because the
    authoritative terminal/close determination always comes from the
    watchdog reading the task row (or the attach-time snapshot read),
    never from frame ordering.
  - Populating ``task.input_required``'s ``prompt`` field from the
    agent's question text. Only the watchdog closes the stream on a
    task stuck waiting for user input (within one watchdog cycle), and
    it always sends ``prompt: null`` -- question-text sniffing from
    live frames belongs to the content-projection layer, not this
    transport layer.

Sink instances duck-type the ``websocket.ConnectionManager`` connection
contract (an object with an async ``send_text(str)`` method) so they
register into the *same* shared ``manager`` real WebSocket connections
use, and ride the same ``broadcast_to_task`` fan-out.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from time import monotonic
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, cast

from fastapi import WebSocket
from fastapi.responses import StreamingResponse

from ...models.agent_api_key import AgentApiKey
from ...models.database import get_session_local
from ...models.task import TaskStatus
from ...services.db_runtime import run_db_io_cancellation_safe
from ..websocket import _is_versioned_task_event, manager
from .deps import ApiKeyPrincipal, active_runtime_key_filters
from .errors import V1ApiError, V1ErrorCode

if TYPE_CHECKING:
    # Only for the type checker -- importing ``tasks`` at module scope
    # would cycle back here (``tasks.py`` imports this module to wire the
    # endpoint). Callers pass a snapshot-reading callable at call time
    # instead (see ``TaskSnapshotReader`` below).
    from .tasks import _TaskInfoSnapshot

logger = logging.getLogger(__name__)

# -- Tunables (all injectable per-call for tests; production always uses
# the defaults below via ``tasks.py``). ---------------------------------

HEARTBEAT_INTERVAL_SECONDS = 15.0
WATCHDOG_INTERVAL_SECONDS = 30.0
# Same shape/value as A2A's cap (``web/api/a2a.py:105``); a separate v1
# constant because the two streams' generators are structurally different
# (A2A polls every <=0.5s and re-checks the deadline for free; v1 is
# queue-consumption-based, so each wait on the outbound queue must be
# capped at the heartbeat interval so the deadline gets re-checked often
# enough).
STREAM_MAX_DURATION_SECONDS = 60.0 * 60.0
OUTBOUND_QUEUE_MAX_SIZE = 256
PER_TASK_STREAM_CAP = 2
PER_PRINCIPAL_STREAM_CAP = 32

_TERMINAL_STATUSES = (TaskStatus.COMPLETED, TaskStatus.FAILED)

# A sync callable: (task_id, principal) -> ``_TaskInfoSnapshot``-shaped
# object (duck-typed: ``.status`` / ``.control_state`` / ``.output`` /
# ``.error``), raising ``V1ApiError(TASK_NOT_FOUND, 404)`` when the task
# is missing or not owned. Always run through ``run_db_io_cancellation_safe``
# by this module -- never called directly.
TaskSnapshotReader = Callable[[int, "ApiKeyPrincipal"], Any]


# -- Wire format --------------------------------------------------------


def _sse_frame(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def status_frame(status: str) -> str:
    return _sse_frame("task.status", {"status": status})


def completed_frame(*, status: str, output: str | None, error: str | None) -> str:
    return _sse_frame(
        "task.completed", {"status": status, "output": output, "error": error}
    )


def input_required_frame(task_id: int) -> str:
    # ``prompt`` is always null: populating it from the agent's question
    # text is the content-projection layer's job, not this transport.
    return _sse_frame("task.input_required", {"task_id": task_id, "prompt": None})


_ERROR_MESSAGES = {
    "resync_required": (
        "The output queue overflowed; call steps() to resync, then re-attach."
    ),
    "unauthorized": "The API key used to open this stream is no longer valid.",
    "task_deleted": "The task no longer exists.",
    "stream_expired": "This stream reached its maximum allowed duration.",
}


def error_frame(code: str) -> str:
    return _sse_frame("stream.error", {"code": code, "message": _ERROR_MESSAGES[code]})


# -- Sink -----------------------------------------------------------------


class V1EventStreamSink:
    """One SSE consumer's broadcast-frame filter and outbound queue.

    Bound to exactly one ``task_id`` and one ``principal`` for the life
    of the connection. The principal is used only for quota accounting
    and the watchdog's key-validity check -- it is **not** an
    authorization gate for who may act on the task; that check already
    happened once, at attach time, via ``get_principal_from_api_key`` +
    ``_resolve_task_or_404``.
    """

    def __init__(
        self, *, task_id: int, principal_key_prefix: str, initial_status: str
    ) -> None:
        self.task_id = task_id
        self.principal_key_prefix = principal_key_prefix
        self.dropped_frame_count = 0
        self.completion_hint = asyncio.Event()
        # Each element is ``(frame_text, is_close)``. ``is_close`` travels
        # with the frame itself rather than being inferred from
        # ``self._closing`` at dequeue time -- the generator can be
        # suspended at ``yield`` on an *earlier*, non-close frame while a
        # concurrent ``enqueue_close`` flips ``_closing`` and appends the
        # close frame behind it; reading ``_closing`` after that yield
        # would wrongly treat the earlier frame as the close and return
        # without ever delivering the close frame still sitting in the
        # queue.
        self._queue: "asyncio.Queue[tuple[str, bool]]" = asyncio.Queue(
            maxsize=OUTBOUND_QUEUE_MAX_SIZE
        )
        self._closing = False
        self._last_status = initial_status
        # Recorded at construction time (always inside the endpoint's own
        # request coroutine) so later state-mutating calls -- however they
        # got here -- can be asserted to run on the same loop.
        self._owner_loop = asyncio.get_running_loop()

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def queue(self) -> "asyncio.Queue[tuple[str, bool]]":
        return self._queue

    def _assert_owner_loop(self) -> None:
        if asyncio.get_running_loop() is not self._owner_loop:
            raise RuntimeError("v1 SSE sink touched from a foreign event loop")

    def enqueue_status(self, status: str) -> None:
        """Enqueue a deduped ``task.status`` frame. Never closes the stream."""
        self._assert_owner_loop()
        if self._closing or status == self._last_status:
            return
        self._last_status = status
        self._put_or_overflow(status_frame(status))

    def enqueue_close(self, frame_text: str) -> bool:
        """Close exactly once; the first caller wins.

        Drains any queued backlog before inserting the close frame so
        the close frame always has room -- even when called *because*
        the queue just overflowed. Losing unread backlog on close is
        intentional: every close reason tells the client to resync
        (``steps()`` + re-attach) rather than trust the tail of the
        stream.
        """
        self._assert_owner_loop()
        if self._closing:
            return False
        self._closing = True
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait((frame_text, True))
        return True

    def _put_or_overflow(self, frame_text: str) -> None:
        # Defense in depth: every current caller already checks ``closing``
        # first, but a closed sink must never grow its queue again no
        # matter how it's reached, so the guard lives here too.
        if self._closing:
            return
        try:
            self._queue.put_nowait((frame_text, False))
        except asyncio.QueueFull:
            self.enqueue_close(error_frame("resync_required"))

    async def send_text(self, text: str) -> None:
        """Receive one broadcast frame. Duck-types ``WebSocket.send_text``.

        Must never raise. ``broadcast_to_task`` (``websocket.py:4504-4532``)
        catches network errors from a connection's ``send_text`` and drops
        the connection, but re-raises anything else (``:4525-4532``) --
        that re-raise happens on the *task's own* outbound event path, so
        an uncaught exception here would abort the broadcast for every
        other listener on the task, not just this stream. This is the
        one deliberate blanket ``except Exception`` in this module;
        dropped frames are counted, not silent.
        """
        try:
            self._assert_owner_loop()
            if self._closing:
                return
            message = json.loads(text)
            if not isinstance(message, dict):
                return
            if not self._binding_matches(message):
                return
            if message.get("type") == "task_completed":
                # Acceleration signal only: the authoritative completion
                # frame still comes from the watchdog reading the task
                # row, just woken up early instead of waiting out its
                # normal cadence. This keeps the sink itself from ever
                # touching the database -- no query happens here.
                self.completion_hint.set()
                return
            if _is_versioned_task_event(message):
                status = message.get("status")
                if isinstance(status, str):
                    self.enqueue_status(status)
        except Exception:
            self.dropped_frame_count += 1
            logger.exception(
                "v1 SSE sink dropped a broadcast frame for task %s", self.task_id
            )

    def _binding_matches(self, message: dict[str, Any]) -> bool:
        """Drop frames whose task_id doesn't match this stream's binding
        (defense in depth -- ``manager`` already scopes delivery to this
        task's connections, so this only matters if a connection is ever
        reassigned via ``move_connection``)."""
        candidate = message.get("task_id")
        if candidate is None and message.get("type") == "task_completed":
            task_obj = message.get("task")
            if isinstance(task_obj, dict):
                candidate = task_obj.get("id")
        if candidate is None:
            return True
        try:
            return int(candidate) == self.task_id
        except (TypeError, ValueError):
            return False


# -- Per-principal concurrency accounting --------------------------------

_principal_stream_counts: dict[str, int] = {}


def try_reserve_principal_slot(key_prefix: str) -> bool:
    current = _principal_stream_counts.get(key_prefix, 0)
    if current >= PER_PRINCIPAL_STREAM_CAP:
        return False
    _principal_stream_counts[key_prefix] = current + 1
    return True


def principal_slot_available(key_prefix: str) -> bool:
    """Read-only capacity check -- would ``try_reserve_principal_slot``
    currently succeed for this key. Used at response-construction time
    (``build_event_stream_response``) so a 429 is raised before any
    stream opens, without mutating the counter yet: the actual
    reservation happens once the generator starts running (see
    ``_generate``), so a response that's constructed but never iterated
    never touches this counter."""
    return _principal_stream_counts.get(key_prefix, 0) < PER_PRINCIPAL_STREAM_CAP


def release_principal_slot(key_prefix: str) -> None:
    current = _principal_stream_counts.get(key_prefix, 0)
    if current <= 1:
        _principal_stream_counts.pop(key_prefix, None)
    else:
        _principal_stream_counts[key_prefix] = current - 1


def reset_principal_stream_counts_for_testing() -> None:
    _principal_stream_counts.clear()


def count_task_sinks(task_id: int) -> int:
    """Count only v1 SSE sinks for a task -- WebSocket connections on the
    same task_id don't share this concurrency cap."""
    return sum(
        1
        for connection in manager.connections_for_task(task_id)
        if isinstance(connection, V1EventStreamSink)
    )


# -- Watchdog --------------------------------------------------------------


def _is_runtime_key_active(key_prefix: str) -> bool:
    """One indexed lookup, no bcrypt (the handshake already verified the
    secret; this only re-checks revoked/paused)."""
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        return (
            db.query(AgentApiKey.id)
            .filter(*active_runtime_key_filters(key_prefix))
            .first()
            is not None
        )


async def watchdog_check_once(
    sink: V1EventStreamSink,
    task_id: int,
    principal: ApiKeyPrincipal,
    *,
    read_task_snapshot: TaskSnapshotReader,
) -> bool:
    """One watchdog check pass. Returns True iff it closed the stream.

    Order: key validity first (the auth axis, fail closed), then the
    task row's own state. All reads go through
    ``run_db_io_cancellation_safe``; none of this runs on the sink's
    ``send_text`` hot path, so the sink itself still never touches the
    database.
    """
    key_active = await run_db_io_cancellation_safe(
        lambda: _is_runtime_key_active(sink.principal_key_prefix)
    )
    if not key_active:
        return sink.enqueue_close(error_frame("unauthorized"))

    try:
        snapshot = await run_db_io_cancellation_safe(
            lambda: read_task_snapshot(task_id, principal)
        )
    except V1ApiError as exc:
        if exc.code is V1ErrorCode.TASK_NOT_FOUND:
            return sink.enqueue_close(error_frame("task_deleted"))
        raise

    status = snapshot.status
    if status in _TERMINAL_STATUSES:
        return sink.enqueue_close(
            completed_frame(
                status=status.value,
                output=snapshot.output,
                error=snapshot.error,
            )
        )
    if (
        status is TaskStatus.WAITING_FOR_USER
        and snapshot.control_state != "resume_requested"
    ):
        return sink.enqueue_close(input_required_frame(task_id))
    if status is TaskStatus.PAUSED:
        # SDK wait() semantics: PAUSED keeps waiting, doesn't close.
        # A "was this paused by an orphaned process?" check was considered
        # and rejected: a normal pause and an orphaned one leave identical
        # values in every lease-tracking column (runner id, lease
        # expiry, last heartbeat, control state) -- both the normal-pause
        # and lease-recovery code paths stamp those columns with "now",
        # so they only measure how long ago the pause happened, not
        # whether anyone is still managing it. Closing on that signal
        # would kill legitimately paused streams too. The 1-hour absolute
        # cap is what actually bounds an orphaned paused stream's lifetime.
        sink.enqueue_status(status.value)
        return False
    return False  # pending/running: keep streaming


async def _watchdog_loop(
    sink: V1EventStreamSink,
    task_id: int,
    principal: ApiKeyPrincipal,
    *,
    read_task_snapshot: TaskSnapshotReader,
    interval_seconds: float,
) -> None:
    """Runs every ``interval_seconds`` and also wakes early on a
    ``task_completed`` broadcast hint: same check, just run sooner.

    A single failed check (e.g. a transient DB error) must not end
    watchdog coverage for the rest of the stream's lifetime -- that
    would leave an orphaned stream open until the 1-hour absolute cap
    with nobody watching it. So every per-cycle check is wrapped and
    logged; only cancellation (this loop's own teardown, not a check
    failure) ends the loop early.
    """
    while not sink.closing:
        try:
            await asyncio.wait_for(
                sink.completion_hint.wait(), timeout=interval_seconds
            )
        except asyncio.TimeoutError:
            pass
        else:
            sink.completion_hint.clear()
        if sink.closing:
            return
        try:
            if await watchdog_check_once(
                sink, task_id, principal, read_task_snapshot=read_task_snapshot
            ):
                return
        except Exception:
            logger.exception(
                "v1 SSE watchdog check failed for task %s; retrying next cycle",
                task_id,
            )


# -- Response assembly ----------------------------------------------------


async def _terminal_snapshot_stream(
    snapshot: "_TaskInfoSnapshot",
) -> AsyncIterator[str]:
    """Attach-time fast path for an already-terminal task: emit
    ``task.status`` + ``task.completed`` and end. No sink, no
    registration, no watchdog -- there's nothing left to watch."""
    yield status_frame(snapshot.status.value)
    yield completed_frame(
        status=snapshot.status.value, output=snapshot.output, error=snapshot.error
    )


async def _generate(
    task_id: int,
    principal: ApiKeyPrincipal,
    *,
    key_prefix: str,
    initial_status: str,
    read_task_snapshot: TaskSnapshotReader,
    watchdog_interval_seconds: float,
    stream_max_duration_seconds: float,
    heartbeat_interval_seconds: float,
) -> AsyncIterator[str]:
    """Build the sink, register it, run the stream, tear it all down.

    Sink construction, ``manager`` registration, and the per-principal
    slot reservation all happen *inside* this ``try`` -- deliberately
    not in ``build_event_stream_response`` -- because an async
    generator's body doesn't run at all until it's first iterated. If
    registration/reservation happened at response-construction time
    instead, a ``StreamingResponse`` that gets built but never iterated
    (e.g. the caller closes it before Starlette ever pulls a chunk)
    would leak both: this generator's ``finally`` -- the only code that
    unregisters and releases -- would simply never execute. Deferring
    both into here means whatever starts this generator (even just one
    ``aclose()`` with no frames read) is guaranteed to reach the
    ``finally`` and clean up exactly what it reserved.

    The 429 capacity *checks* still happen earlier, in
    ``build_event_stream_response`` (read-only, no mutation) -- so a
    rejected attach still fails before any stream bytes are sent; only
    the actual counter mutation and registration move here.
    """
    deadline = monotonic() + stream_max_duration_seconds
    sink = V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status=initial_status
    )
    principal_slot_reserved = False
    watchdog_task: "asyncio.Task[None] | None" = None
    try:
        # Soft cap: `principal_slot_available` in `build_event_stream_response`
        # already did the *check* before this generator started, and that
        # check is what carries the normal 429 -- an attach that loses the
        # race here has already been told "yes, come in", so this
        # reservation only accounts for capacity, it never rejects. If a
        # concurrent burst of attaches for the same principal all pass the
        # earlier read-only check and then land here before any of them
        # releases a slot, `try_reserve_principal_slot` returns False for
        # the late arrivals: the count is briefly over `PER_PRINCIPAL_STREAM_CAP`,
        # it's logged, and the stream is served anyway. The sentinel
        # (`principal_slot_reserved`) stays False for these streams, so the
        # `finally` below correctly skips `release_principal_slot` for them
        # -- nothing was reserved, so nothing is released. The count
        # self-heals as soon as any concurrently-open stream for this
        # principal finishes and releases its own slot.
        principal_slot_reserved = try_reserve_principal_slot(key_prefix)
        if not principal_slot_reserved:
            logger.warning(
                "v1 SSE per-principal cap best-effort exceeded for "
                "key_prefix=%s under concurrent attach burst; serving the "
                "stream anyway (soft cap, no reservation held)",
                key_prefix,
            )
        manager.register_connection(cast(WebSocket, sink), task_id)
        watchdog_task = asyncio.create_task(
            _watchdog_loop(
                sink,
                task_id,
                principal,
                read_task_snapshot=read_task_snapshot,
                interval_seconds=watchdog_interval_seconds,
            )
        )
        # The initial yield must be inside this ``try`` too: a generator
        # closed (``aclose()``, e.g. on client disconnect) while suspended
        # here still has to unregister the sink and cancel the watchdog.
        yield status_frame(initial_status)
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                # First-to-close-wins: a no-op if the watchdog already
                # closed the stream first in the same tick.
                sink.enqueue_close(error_frame("stream_expired"))
            wait_budget = heartbeat_interval_seconds if remaining > 0 else 1.0
            try:
                frame_text, is_close = await asyncio.wait_for(
                    sink.queue.get(), timeout=wait_budget
                )
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            yield frame_text
            # ``is_close`` is the flag recorded on *this* frame at enqueue
            # time, not ``sink.closing`` re-read now -- see the queue
            # element's docstring in ``V1EventStreamSink.__init__`` for why
            # that distinction is load-bearing under concurrent
            # ``enqueue_close`` calls.
            if is_close:
                return
    finally:
        # The watchdog wait-and-cancel is wrapped in its own try/finally so
        # that `manager.disconnect` and `release_principal_slot` below --
        # the two calls that actually undo what this generator reserved --
        # still run even if awaiting the cancelled watchdog task raises
        # something unexpected. Without this inner `finally`, a raise here
        # would skip straight past both cleanup calls, leaking the sink
        # registration and (if held) the per-principal slot.
        try:
            if watchdog_task is not None:
                watchdog_task.cancel()
                # Narrowed to ``CancelledError`` only (not a blanket
                # ``BaseException``): the watchdog loop itself now catches
                # and logs every per-cycle ``Exception`` internally (see
                # ``_watchdog_loop``) and retries rather than dying, so the
                # only exception this teardown should ever observe here is
                # the cancellation just requested above. If the watchdog
                # task raises anything else, that's a real bug and must
                # propagate instead of being silently swallowed.
                with contextlib.suppress(asyncio.CancelledError):
                    await watchdog_task
        finally:
            # ``manager`` is typed against a real ``WebSocket``; this sink only
            # duck-types its ``send_text`` contract (module docstring) -- the
            # cast documents that intentional narrowing instead of suppressing
            # the type error blanket-wide. A safe no-op if registration above
            # never ran or never completed.
            manager.disconnect(cast(WebSocket, sink))
            if principal_slot_reserved:
                release_principal_slot(key_prefix)


async def build_event_stream_response(
    *,
    task_id: int,
    principal: ApiKeyPrincipal,
    initial_snapshot: "_TaskInfoSnapshot",
    read_task_snapshot: TaskSnapshotReader,
    watchdog_interval_seconds: float = WATCHDOG_INTERVAL_SECONDS,
    stream_max_duration_seconds: float = STREAM_MAX_DURATION_SECONDS,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> StreamingResponse:
    """Assemble the SSE ``StreamingResponse`` for one attach.

    ``initial_snapshot`` must already be authorized (i.e. come from
    ``_resolve_task_or_404`` via ``read_task_snapshot``) -- this
    function does no auth of its own. Concurrency caps (429) are
    checked here, before the generator (and therefore the sink and the
    per-principal reservation) ever exists, so a rejected attach never
    touches ``manager`` or the per-principal counter.
    """
    if initial_snapshot.status in _TERMINAL_STATUSES:
        return StreamingResponse(
            _terminal_snapshot_stream(initial_snapshot),
            media_type="text/event-stream",
        )

    if count_task_sinks(task_id) >= PER_TASK_STREAM_CAP:
        raise V1ApiError(V1ErrorCode.RATE_LIMITED, 429)

    key_prefix = principal.key.key_prefix
    if not principal_slot_available(key_prefix):
        raise V1ApiError(V1ErrorCode.RATE_LIMITED, 429)

    return StreamingResponse(
        _generate(
            task_id,
            principal,
            key_prefix=key_prefix,
            initial_status=initial_snapshot.status.value,
            read_task_snapshot=read_task_snapshot,
            watchdog_interval_seconds=watchdog_interval_seconds,
            stream_max_duration_seconds=stream_max_duration_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        ),
        media_type="text/event-stream",
    )
