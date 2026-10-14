"""Server heartbeat negotiation and lifecycle on the ``/ws/chat`` endpoint."""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import WebSocketDisconnect

from xagent.web.api import websocket as websocket_api

TASK_ID = 41
CAPABILITIES_FRAME = {"type": "connection_capabilities", "capabilities": ["heartbeat"]}
HEARTBEAT_FRAME = {"type": "heartbeat"}
STATUS_MARKER = {"type": "status_request_handled"}


class _FakeSocket:
    """Minimal accepted socket: records every text frame the server sends.

    ``confirmation_delay`` slows only the capabilities frame, so a heartbeat
    task that was started too early would overtake it on the wire.
    """

    def __init__(self, liveness: list[str], *, confirmation_delay: float = 0.0) -> None:
        self.query_params = SimpleNamespace(
            getlist=lambda key: list(liveness) if key == "liveness" else []
        )
        self.accept = AsyncMock()
        self.close = AsyncMock()
        self.frames: list[dict[str, Any]] = []
        self.heartbeat_seen = asyncio.Event()
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self._confirmation_delay = confirmation_delay

    async def send_text(self, text: str) -> None:
        frame = json.loads(text)
        if frame == CAPABILITIES_FRAME and self._confirmation_delay:
            await asyncio.sleep(self._confirmation_delay)
        self.frames.append(frame)
        if frame == HEARTBEAT_FRAME:
            self.heartbeat_seen.set()

    async def receive_text(self) -> str:
        item = await self._inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def client_sends(self, item: Any) -> None:
        self._inbox.put_nowait(item)

    def heartbeats(self) -> int:
        return sum(1 for frame in self.frames if frame == HEARTBEAT_FRAME)


async def _status_marker(websocket: _FakeSocket, *_: Any) -> None:
    websocket.frames.append(STATUS_MARKER)


@contextlib.contextmanager
def _endpoint_env(
    manager: websocket_api.ConnectionManager, **overrides: Any
) -> Iterator[None]:
    """Patch the endpoint's collaborators so no database is touched."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(websocket_api, "manager", manager))
        stack.enter_context(
            patch.object(
                websocket_api,
                "get_authenticated_user",
                AsyncMock(return_value=SimpleNamespace(id=7, is_admin=False)),
            )
        )
        stack.enter_context(
            patch.object(
                websocket_api,
                "_private_websocket_task_access_sync",
                lambda **_: "authorized",
            )
        )
        stack.enter_context(
            patch.object(websocket_api, "handle_status_request", _status_marker)
        )
        for name, value in overrides.items():
            stack.enter_context(patch.object(websocket_api, name, value))
        yield


def _run_endpoint(websocket: _FakeSocket) -> asyncio.Task[None]:
    return asyncio.ensure_future(
        websocket_api.websocket_chat_endpoint(websocket, TASK_ID, None)  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("liveness", "negotiated"),
    [
        ([], False),
        (["heartbeat"], True),
        (["ping"], False),
        (["heartbeat", "heartbeat"], False),
    ],
)
async def test_ws_chat_liveness_negotiation(
    liveness: list[str], negotiated: bool
) -> None:
    manager = websocket_api.ConnectionManager()
    websocket = _FakeSocket(liveness, confirmation_delay=0.1)

    with _endpoint_env(manager, HEARTBEAT_INTERVAL_SECONDS=0.005):
        endpoint = _run_endpoint(websocket)
        try:
            if negotiated:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(websocket.heartbeat_seen.wait(), timeout=2)
            else:
                # Many heartbeat intervals pass; nothing may be sent.
                await asyncio.sleep(0.3)
        finally:
            websocket.client_sends(WebSocketDisconnect())
            await asyncio.gather(endpoint, return_exceptions=True)

    wire_types = [frame["type"] for frame in websocket.frames]
    if negotiated:
        assert websocket.frames[0] == CAPABILITIES_FRAME
        assert websocket.frames.count(CAPABILITIES_FRAME) == 1
        # The acknowledgement precedes the opening status snapshot and every
        # heartbeat.
        assert websocket.frames.index(CAPABILITIES_FRAME) < websocket.frames.index(
            STATUS_MARKER
        )
        assert websocket.frames.index(CAPABILITIES_FRAME) < websocket.frames.index(
            HEARTBEAT_FRAME
        )
    else:
        assert "connection_capabilities" not in wire_types
        assert "heartbeat" not in wire_types
        assert wire_types == [STATUS_MARKER["type"]]


@pytest.mark.asyncio
async def test_ws_chat_heartbeat_survives_blocked_loop() -> None:
    manager = websocket_api.ConnectionManager()
    release_handler = asyncio.Event()

    async def blocked_execute_task(*_: Any) -> None:
        await release_handler.wait()

    # Phase 1: the read loop is stuck inside a long handler, and the heartbeat
    # keeps flowing; after a normal disconnect the task is gone.
    first = _FakeSocket(["heartbeat"])
    with _endpoint_env(
        manager,
        HEARTBEAT_INTERVAL_SECONDS=0.05,
        handle_execute_task=blocked_execute_task,
    ):
        endpoint = _run_endpoint(first)
        first.client_sends(json.dumps({"type": "execute_task"}))
        await asyncio.sleep(0.4)
        assert not endpoint.done()
        assert first.heartbeats() >= 2

        release_handler.set()
        first.client_sends(WebSocketDisconnect())
        await asyncio.gather(endpoint, return_exceptions=True)

        assert manager.connection_count() == 0
        beats_at_exit = first.heartbeats()
        await asyncio.sleep(0.2)
        assert first.heartbeats() == beats_at_exit

    # Phase 2: the heartbeat task dies with an exception outside the swallowed
    # send-failure group; the connection is still removed from the manager.
    second = _FakeSocket(["heartbeat"])
    real_send = manager.send_personal_message
    heartbeat_failure = {"armed": False}

    async def send_then_fail(message: dict, websocket: Any) -> None:
        if message == HEARTBEAT_FRAME and heartbeat_failure["armed"]:
            raise ValueError("heartbeat serialisation fault")
        await real_send(message, websocket)

    with (
        _endpoint_env(
            manager,
            HEARTBEAT_INTERVAL_SECONDS=0.05,
            handle_execute_task=blocked_execute_task,
        ),
        patch.object(manager, "send_personal_message", send_then_fail),
    ):
        endpoint = _run_endpoint(second)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(second.heartbeat_seen.wait(), timeout=2)
        assert second.heartbeats() >= 1
        assert manager.connection_count() == 1
        heartbeat_failure["armed"] = True
        await asyncio.sleep(0.2)
        second.client_sends(WebSocketDisconnect())
        results = await asyncio.gather(endpoint, return_exceptions=True)

        assert results == [None]
        assert manager.connection_count() == 0


def test_heartbeat_interval_stays_within_half_the_dead_threshold() -> None:
    assert (
        websocket_api.HEARTBEAT_INTERVAL_SECONDS * 2
        <= websocket_api.CLIENT_DEAD_AFTER_SECONDS
    )
