"""Tests for the v1 SSE transport layer (``GET /v1/chat/tasks/{id}/events``).

Covers the transport layer only: registration, the sink's frame filter
and exception discipline, the watchdog, the 1-hour cap, and concurrency
caps. Step/message content projection isn't part of this layer, so it
isn't tested here. Tests either drive ``_events_stream`` directly (fast,
deterministic -- used whenever a test needs injected short intervals or
precise control over the race between the watchdog and the deadline) or
go through the real HTTP endpoint (used for the 401/404/429/terminal-
attach shapes, where the JSON-vs-event-stream distinction actually
matters).
"""

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from xagent.web.api.v1 import _events_stream as es
from xagent.web.api.v1 import tasks as v1_tasks
from xagent.web.api.v1.deps import ApiKeyPrincipal, _resolve_principal_from_credentials
from xagent.web.api.v1.errors import V1ApiError, V1ErrorCode
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.task import Task, TaskStatus

from ..conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


# ===== local helpers (mirrors test_tasks.py's pattern) =====


@pytest.fixture(autouse=True)
def mock_start_task():
    """Every test here creates tasks through the real POST endpoint (so
    ownership/source="sdk" scoping is authentic) but never wants an
    actual agent execution to run."""
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


@pytest.fixture(autouse=True)
def reset_principal_stream_counts():
    es.reset_principal_stream_counts_for_testing()
    yield
    es.reset_principal_stream_counts_for_testing()


def _create_agent_with_key() -> tuple[int, str]:
    headers = _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "v1 events test agent",
            "description": "test",
            "instructions": "you are a test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]
    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    return agent_id, key_resp.json()["full_key"]


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def _create_task(full_key: str, agent_id: int, content: str = "hello") -> int:
    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": content}},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


def _principal_for(full_key: str) -> ApiKeyPrincipal:
    """Resolve a real principal through the production auth path (bcrypt
    included) rather than hand-rolling a snapshot -- keeps tests honest
    about what ``get_principal_from_api_key`` actually returns."""
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=full_key)
    return _resolve_principal_from_credentials(creds)


def _set_task_status(
    task_id: int,
    status: TaskStatus,
    *,
    output: str | None = None,
    error_message: str | None = None,
    control_state: str | None = None,
) -> None:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = status
        if output is not None:
            task.output = output
        if error_message is not None:
            task.error_message = error_message
        if control_state is not None:
            task.control_state = control_state
        db.commit()
    finally:
        db.close()


def _key_prefix_for_agent(agent_id: int) -> str:
    db = _direct_db_session()
    try:
        return str(
            db.query(AgentApiKey)
            .filter(AgentApiKey.agent_id == agent_id)
            .one()
            .key_prefix
        )
    finally:
        db.close()


def _make_sink(task_id: int = 1, status: str = "running") -> es.V1EventStreamSink:
    return es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix="pfx-test", initial_status=status
    )


# ===== sink.send_text never raises =====


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "42",
        "null",
        "[]",
        json.dumps({"type": "task_completed", "task": "not-a-dict"}),
        json.dumps(
            {"type": "task_started", "task_id": "not-an-int", "status": "running"}
        ),
    ],
)
async def test_sink_send_text_never_raises_on_malformed_input(raw):
    sink = _make_sink()
    await sink.send_text(raw)  # must not raise


async def test_sink_send_text_never_raises_when_classification_throws(monkeypatch):
    sink = _make_sink()

    def _boom(message):
        raise RuntimeError("boom")

    monkeypatch.setattr(es, "_is_versioned_task_event", _boom)
    before = sink.dropped_frame_count
    await sink.send_text(
        json.dumps(
            {"type": "task_started", "task_id": sink.task_id, "status": "running"}
        )
    )  # must not raise
    assert sink.dropped_frame_count == before + 1


async def test_sink_send_text_never_raises_from_foreign_loop():
    sink = _make_sink()
    sink._owner_loop = object()  # simulate a foreign event loop
    await sink.send_text(
        json.dumps(
            {"type": "task_started", "task_id": sink.task_id, "status": "running"}
        )
    )  # must not raise
    assert sink.dropped_frame_count == 1
    assert sink.queue.empty()


# ===== unauthorized / not-owned get plain JSON, no stream bytes =====


def test_events_unauthorized_401():
    resp = client.get("/v1/chat/tasks/1/events")
    assert resp.status_code == 401
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_events_not_owned_404():
    _, full_key = _create_agent_with_key()
    resp = client.get("/v1/chat/tasks/999999/events", headers=_bearer(full_key))
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["error"]["code"] == "task_not_found"


# ===== attach on an already-terminal task =====


def test_events_terminal_task_immediate_close():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _set_task_status(task_id, TaskStatus.COMPLETED, output="the answer")

    resp = client.get(f"/v1/chat/tasks/{task_id}/events", headers=_bearer(full_key))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert body.count("event: task.status") == 1
    assert body.count("event: task.completed") == 1
    blocks = [b for b in body.split("\n\n") if b.strip()]
    status_data = json.loads(blocks[0].split("data: ", 1)[1])
    completed_data = json.loads(blocks[1].split("data: ", 1)[1])
    assert status_data == {"status": "completed"}
    assert completed_data == {
        "status": "completed",
        "output": "the answer",
        "error": None,
    }
    # No sink is registered for the terminal fast path -- nothing to close.
    assert es.count_task_sinks(task_id) == 0


# ===== bounded outbound queue, overflow closes with resync_required =====


async def test_slow_consumer_queue_overflow_closes_with_resync_required():
    sink = _make_sink()
    for i in range(es.OUTBOUND_QUEUE_MAX_SIZE + 5):
        sink._put_or_overflow(f"frame-{i}")
    assert sink.closing is True
    # Memory doesn't grow past the cap even under sustained overflow --
    # exactly one frame (the close frame) survives.
    assert sink.queue.qsize() == 1
    assert sink.queue.get_nowait() == (es.error_frame("resync_required"), True)


# ===== key revoked/paused closes within one watchdog cycle =====


@pytest.mark.parametrize("field", ["revoked_at", "paused_at"])
async def test_watchdog_closes_within_one_cycle_on_key_invalidation(field):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)

    db = _direct_db_session()
    try:
        key_row = db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).one()
        setattr(key_row, field, datetime.now(UTC))
        db.commit()
    finally:
        db.close()

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is True
    assert sink.closing is True
    assert sink.queue.get_nowait() == (es.error_frame("unauthorized"), True)


async def test_watchdog_paused_does_not_close():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    _set_task_status(task_id, TaskStatus.PAUSED)

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is False
    assert sink.closing is False
    assert sink.queue.get_nowait() == (es.status_frame("paused"), False)


async def test_watchdog_waiting_for_user_closes_with_null_prompt_input_required():
    """The watchdog is the only trigger this transport layer implements
    for input_required, and it always sends a null prompt -- populating
    it from the agent's question text belongs to the content-projection
    layer."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    _set_task_status(task_id, TaskStatus.WAITING_FOR_USER, control_state="idle")

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is True
    assert sink.closing is True
    assert sink.queue.get_nowait() == (es.input_required_frame(task_id), True)
    assert (
        json.loads(es.input_required_frame(task_id).split("data: ", 1)[1])["prompt"]
        is None
    )


async def test_watchdog_waiting_for_user_with_resume_requested_does_not_close():
    """The ``resume_requested`` carve-out: a task about to resume isn't
    "stuck waiting" even though its status hasn't flipped yet."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    _set_task_status(
        task_id, TaskStatus.WAITING_FOR_USER, control_state="resume_requested"
    )

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is False
    assert sink.closing is False


async def test_watchdog_terminal_task_row_closes_with_completed():
    """The watchdog itself is the authoritative source for task.completed,
    independent of the attach-time fast path for an already-terminal
    task or the broadcast acceleration hint."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    _set_task_status(task_id, TaskStatus.FAILED, error_message="boom")

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is True
    assert sink.queue.get_nowait() == (
        es.completed_frame(status="failed", output=None, error="boom"),
        True,
    )


async def test_watchdog_missing_task_row_closes_with_task_deleted():
    """A task row that vanishes out from under an open stream
    (hard-deleted) surfaces as task_deleted, not a hang or a 500."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)

    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).delete()
        db.commit()
    finally:
        db.close()

    sink = es.V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status="running"
    )
    closed = await es.watchdog_check_once(
        sink, task_id, principal, read_task_snapshot=v1_tasks._load_task_info_snapshot
    )
    assert closed is True
    assert sink.queue.get_nowait() == (es.error_frame("task_deleted"), True)


async def test_watchdog_survives_transient_check_failure_and_still_closes():
    """A single failed watchdog cycle (e.g. a transient DB error reading
    the task row) must not silence the watchdog for the rest of the
    stream's life. Before this fix, an uncaught exception inside the
    watchdog loop killed the loop's background task permanently and
    silently -- the generator's teardown swallowed it via a blanket
    ``except BaseException``, so nothing was left watching the stream
    until the 1-hour absolute cap. This injects a ``read_task_snapshot``
    that raises on its first call and returns a real, terminal
    snapshot on its second, and asserts the stream still reaches
    ``task.completed`` -- i.e. the watchdog retried instead of dying."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    call_count = 0

    def flaky_reader(task_id_, principal_):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom - transient read failure")
        return v1_tasks._load_task_info_snapshot(task_id_, principal_)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=flaky_reader,
        watchdog_interval_seconds=0.01,
        stream_max_duration_seconds=1000,
        heartbeat_interval_seconds=1000,
    )
    first = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
    assert "event: task.status" in first

    _set_task_status(task_id, TaskStatus.COMPLETED, output="done")

    frames = []

    async def _drain():
        async for frame in resp.body_iterator:
            frames.append(frame)

    await asyncio.wait_for(_drain(), timeout=2)
    assert call_count >= 2  # the first (failing) cycle and the retry
    assert any("event: task.completed" in f for f in frames)
    assert es.count_task_sinks(task_id) == 0


# ===== endpoint signature carries no request-scoped DB dependency =====


def test_events_endpoint_signature_has_no_db_dependency():
    import inspect

    sig = inspect.signature(v1_tasks.stream_chat_task_events)
    assert set(sig.parameters) == {"task_id", "principal"}


# ===== generator teardown unregisters the sink =====


async def test_sink_unregistered_after_generator_teardown():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        watchdog_interval_seconds=1000,
        stream_max_duration_seconds=1000,
        heartbeat_interval_seconds=1000,
    )
    first = await resp.body_iterator.__anext__()
    assert "event: task.status" in first
    assert es.count_task_sinks(task_id) == 1

    # Simulate what Starlette does on client disconnect / response teardown.
    await resp.body_iterator.aclose()
    assert es.count_task_sinks(task_id) == 0


# ===== a generator that's built but never started leaks nothing =====


async def test_unstarted_generator_leaves_no_registration_or_reservation():
    """An async generator's body doesn't run until it's first iterated.
    Registering the sink and reserving the per-principal slot both
    happen *inside* ``_generate``'s own ``try`` (not at
    ``build_event_stream_response`` construction time) specifically so
    that a ``StreamingResponse`` that's constructed but never iterated
    -- ``__anext__`` never called even once -- leaves nothing behind:
    no sink in ``manager``, no held slot in the per-principal counter.
    Before that fix, registration/reservation ran at construction time,
    so this exact sequence (build, then ``aclose()`` with zero reads)
    leaked both forever, since the only code that ever released them
    lived inside the generator body that never got to run."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        watchdog_interval_seconds=1000,
        stream_max_duration_seconds=1000,
        heartbeat_interval_seconds=1000,
    )
    # No __anext__() call at all -- the generator body has never run.
    assert es.count_task_sinks(task_id) == 0
    assert es._principal_stream_counts.get(key_prefix, 0) == 0

    await resp.body_iterator.aclose()
    assert es.count_task_sinks(task_id) == 0
    assert es._principal_stream_counts.get(key_prefix, 0) == 0


# ===== per-principal stream cap =====


async def test_try_reserve_principal_slot_caps_at_limit():
    key_prefix = "pfx-cap-unit-test"
    for _ in range(es.PER_PRINCIPAL_STREAM_CAP):
        assert es.try_reserve_principal_slot(key_prefix) is True
    assert es.try_reserve_principal_slot(key_prefix) is False
    for _ in range(es.PER_PRINCIPAL_STREAM_CAP):
        es.release_principal_slot(key_prefix)
    assert es.try_reserve_principal_slot(key_prefix) is True
    es.release_principal_slot(key_prefix)


def test_events_principal_cap_returns_429_json():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    key_prefix = _key_prefix_for_agent(agent_id)
    for _ in range(es.PER_PRINCIPAL_STREAM_CAP):
        assert es.try_reserve_principal_slot(key_prefix)

    resp = client.get(f"/v1/chat/tasks/{task_id}/events", headers=_bearer(full_key))
    assert resp.status_code == 429
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["error"]["code"] == "rate_limited"
    assert es.count_task_sinks(task_id) == 0  # rejected before registration


async def test_principal_slot_released_after_real_stream_teardown():
    """A real attach through ``build_event_stream_response`` (not the
    raw counter functions in isolation) must actually give its slot
    back on teardown -- otherwise every stream a key ever opens
    permanently eats one of its 32 concurrent slots."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    key_prefix = _key_prefix_for_agent(agent_id)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        watchdog_interval_seconds=1000,
        stream_max_duration_seconds=1000,
        heartbeat_interval_seconds=1000,
    )
    await resp.body_iterator.__anext__()
    assert es._principal_stream_counts.get(key_prefix, 0) == 1

    await resp.body_iterator.aclose()
    assert es._principal_stream_counts.get(key_prefix, 0) == 0


# ===== absolute deadline emits stream_expired before closing =====


async def test_absolute_deadline_emits_expired_then_closes():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        watchdog_interval_seconds=1000,
        stream_max_duration_seconds=0.05,
        heartbeat_interval_seconds=0.01,
    )
    frames = [frame async for frame in resp.body_iterator]
    # Zero or more heartbeats may land before the deadline trips, but the
    # close frame must be exactly one, and it must be last (nothing is
    # yielded after it).
    assert "event: task.status" in frames[0]
    assert frames.count(": ping\n\n") == len(frames) - 2  # everything but status+close
    assert "event: stream.error" in frames[-1]
    assert "stream_expired" in frames[-1]
    assert es.count_task_sinks(task_id) == 0


# ===== watchdog vs. deadline race closes exactly once =====


async def test_enqueue_close_is_idempotent_first_writer_wins():
    sink = _make_sink()
    first = sink.enqueue_close(es.error_frame("stream_expired"))
    second = sink.enqueue_close(es.error_frame("task_deleted"))
    assert first is True
    assert second is False
    assert sink.queue.qsize() == 1
    assert sink.queue.get_nowait() == (es.error_frame("stream_expired"), True)


# ===== close frame isn't lost when it's enqueued while the generator is
# suspended mid-yield on an earlier, non-close frame =====


async def test_close_frame_delivered_when_enqueued_between_two_yields():
    """The generator can be suspended at ``yield frame_text`` (Starlette
    awaiting the socket write) when a concurrent close call lands. The
    close-ness of the *next* frame the generator delivers must come
    from that frame's own queue entry, not from re-reading
    ``sink.closing`` after the unrelated earlier yield resumes -- by
    then ``closing`` is already true even though the frame just sent
    wasn't the close frame. Driven by hand via ``asend`` to control
    exactly where the generator is suspended when the race hits."""

    async def _never_read(task_id, principal):  # pragma: no cover
        raise AssertionError("watchdog must not run in this test")

    task_id = 987_654_321  # sentinel, not a real DB row -- unused by this test
    gen = es._generate(
        task_id,
        None,
        key_prefix="pfx-close-frame-race-test",
        initial_status="running",
        read_task_snapshot=_never_read,
        watchdog_interval_seconds=10_000,
        stream_max_duration_seconds=10_000,
        heartbeat_interval_seconds=10_000,
    )
    try:
        first = await gen.asend(None)
        assert "event: task.status" in first

        # ``_generate`` builds and registers the sink internally now (fix
        # for the double-leak on an unstarted generator) -- fetch the
        # live instance via the same ``manager`` registry the generator
        # just registered it into.
        sink = next(
            c
            for c in es.manager.connections_for_task(task_id)
            if isinstance(c, es.V1EventStreamSink)
        )

        sink.enqueue_status("paused")
        second = await gen.asend(None)
        assert "paused" in second
        # The generator is now suspended right after yielding ``second``
        # -- exactly the window in which the real consumer would be
        # awaiting the socket write while other tasks run.

        won = sink.enqueue_close(
            es.completed_frame(status="completed", output="done", error=None)
        )
        assert won is True

        third = await gen.asend(None)
        assert "event: task.completed" in third
        with pytest.raises(StopAsyncIteration):
            await gen.asend(None)
    finally:
        # ``_generate``'s own ``finally`` already released the slot it
        # reserved once the close frame was delivered above; ``aclose()``
        # here only matters if an assertion failed earlier and the
        # generator is still suspended mid-stream.
        await gen.aclose()


async def test_close_exactly_once_under_watchdog_deadline_race():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)  # non-terminal at attach
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        watchdog_interval_seconds=0.001,
        stream_max_duration_seconds=0.001,
        heartbeat_interval_seconds=0.001,
    )
    # Flip terminal *after* attach so the watchdog's next tick and the
    # already-short deadline both want to close the stream.
    _set_task_status(task_id, TaskStatus.COMPLETED, output="done")

    closing_events = ("event: task.completed", "event: stream.error")
    frames = []
    async for frame in resp.body_iterator:
        frames.append(frame)
        if len(frames) > 10:
            break
    close_frames = [f for f in frames if any(name in f for name in closing_events)]
    assert len(close_frames) == 1
    assert close_frames[0] == frames[-1]
    assert es.count_task_sinks(task_id) == 0


# ===== frames bound to a different task_id are discarded =====


async def test_sink_discards_foreign_task_frames():
    sink = _make_sink(task_id=1)
    await sink.send_text(
        json.dumps({"type": "task_started", "task_id": 2, "status": "running"})
    )
    assert sink.queue.empty()
    assert sink.dropped_frame_count == 0  # a graceful discard, not an error


# ===== per-task concurrency cap =====


async def test_events_per_task_cap_third_stream_429():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    responses = []
    for _ in range(es.PER_TASK_STREAM_CAP):
        resp = await es.build_event_stream_response(
            task_id=task_id,
            principal=principal,
            initial_snapshot=snapshot,
            read_task_snapshot=v1_tasks._load_task_info_snapshot,
            watchdog_interval_seconds=1000,
            stream_max_duration_seconds=1000,
            heartbeat_interval_seconds=1000,
        )
        # Real delivery (Starlette) always pulls the first chunk before a
        # response can close; advance each generator once so its ``finally``
        # cleanup is reachable, same as it would be in production.
        await resp.body_iterator.__anext__()
        responses.append(resp)

    assert es.count_task_sinks(task_id) == es.PER_TASK_STREAM_CAP
    with pytest.raises(V1ApiError) as exc_info:
        await es.build_event_stream_response(
            task_id=task_id,
            principal=principal,
            initial_snapshot=snapshot,
            read_task_snapshot=v1_tasks._load_task_info_snapshot,
        )
    assert exc_info.value.code is V1ErrorCode.RATE_LIMITED
    assert exc_info.value.http_status == 429

    for resp in responses:
        await resp.body_iterator.aclose()
    assert es.count_task_sinks(task_id) == 0


# ===== the sink never touches the database on its per-frame path =====


async def test_sink_send_text_never_queries_db(monkeypatch):
    sink = _make_sink()
    calls: list[int] = []
    original_get_session_local = es.get_session_local

    def _tracking_get_session_local():
        calls.append(1)
        return original_get_session_local()

    monkeypatch.setattr(es, "get_session_local", _tracking_get_session_local)

    for i in range(20):
        await sink.send_text(
            json.dumps(
                {"type": "task_started", "task_id": sink.task_id, "status": f"s{i}"}
            )
        )
    await sink.send_text(
        json.dumps({"type": "task_completed", "task": {"id": sink.task_id}})
    )
    assert calls == []


# ===== the heartbeat comment line actually gets sent while idle =====


async def test_heartbeat_actually_sent_when_idle():
    """The 15s heartbeat comment must actually be emitted while the
    stream is otherwise idle -- this is a real, non-vacuous assertion
    that at least one ``: ping`` frame arrives, not a shape check that
    would pass whether zero or many heartbeats fired (that was the bug
    in the older deadline test, which only ever asserted a frame-count
    identity that holds either way). Injects a short heartbeat interval
    with a long watchdog interval and deadline so a heartbeat is the
    only thing that can produce a frame here."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        watchdog_interval_seconds=1000,
        stream_max_duration_seconds=1000,
        heartbeat_interval_seconds=0.01,
    )
    pings = 0
    try:

        async def _collect() -> None:
            nonlocal pings
            async for frame in resp.body_iterator:
                if frame == ": ping\n\n":
                    pings += 1
                    return

        await asyncio.wait_for(_collect(), timeout=2)
    finally:
        await resp.body_iterator.aclose()
    assert pings >= 1


# ===== a broadcast-shaped frame reaching the sink ends up as real
# SSE output text out of the generator, not just in the internal queue =====


async def test_broadcast_frame_reaches_generator_output_as_task_status():
    """Wiring test for the full path: ``sink.send_text`` (what
    ``broadcast_to_task`` calls) -> ``enqueue_status`` -> the outbound
    queue -> ``_generate``'s yield. Uses a status distinct from the
    initial attach status ("running") so the assertion can't pass
    merely because the *first* frame already says ``task.status`` --
    it must be *this* broadcast that produced the second frame."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)
    assert snapshot.status.value == "running"

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        watchdog_interval_seconds=1000,
        stream_max_duration_seconds=1000,
        heartbeat_interval_seconds=1000,
    )
    first = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
    assert json.loads(first.split("data: ", 1)[1]) == {"status": "running"}

    sink = next(
        c
        for c in es.manager.connections_for_task(task_id)
        if isinstance(c, es.V1EventStreamSink)
    )
    await sink.send_text(
        json.dumps({"type": "task_paused", "task_id": task_id, "status": "paused"})
    )

    second = await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)
    assert "event: task.status" in second
    assert json.loads(second.split("data: ", 1)[1]) == {"status": "paused"}

    await resp.body_iterator.aclose()


# ===== consecutive identical statuses only produce one frame =====


async def test_consecutive_same_status_broadcasts_are_deduped_to_one_frame():
    sink = _make_sink(task_id=1, status="running")
    for _ in range(3):
        await sink.send_text(
            json.dumps({"type": "task_paused", "task_id": 1, "status": "paused"})
        )
    assert sink.queue.qsize() == 1
    assert sink.queue.get_nowait() == (es.status_frame("paused"), False)


# ===== the completion_hint wakes the watchdog early, not on its next
# periodic tick =====


async def test_completion_hint_wakes_watchdog_before_its_next_periodic_tick():
    """The ``task_completed`` broadcast hint is supposed to be an
    acceleration path -- the watchdog wakes and checks immediately
    instead of waiting out its normal interval. This pins that it's a
    genuine early wake, not something that happens to work because the
    interval is already short: the watchdog interval here is 1000s, so
    the only way this test can finish inside its 2s bound is if the
    hint actually woke it early. If the hint stopped working, this
    would hang until the bound trips and the test would fail instead of
    silently passing."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    principal = _principal_for(full_key)
    snapshot = v1_tasks._load_task_info_snapshot(task_id, principal)

    resp = await es.build_event_stream_response(
        task_id=task_id,
        principal=principal,
        initial_snapshot=snapshot,
        read_task_snapshot=v1_tasks._load_task_info_snapshot,
        watchdog_interval_seconds=1000,
        stream_max_duration_seconds=1000,
        heartbeat_interval_seconds=1000,
    )
    await asyncio.wait_for(resp.body_iterator.__anext__(), timeout=2)

    _set_task_status(task_id, TaskStatus.COMPLETED, output="done")
    sink = next(
        c
        for c in es.manager.connections_for_task(task_id)
        if isinstance(c, es.V1EventStreamSink)
    )
    await sink.send_text(
        json.dumps({"type": "task_completed", "task": {"id": task_id}})
    )

    frames = []

    async def _drain() -> None:
        async for frame in resp.body_iterator:
            frames.append(frame)

    await asyncio.wait_for(_drain(), timeout=2)
    assert any("event: task.completed" in f for f in frames)
