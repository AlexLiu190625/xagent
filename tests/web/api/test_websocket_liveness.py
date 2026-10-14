"""Server heartbeat negotiation and lifecycle on the ``/ws/chat`` endpoint."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import WebSocketDisconnect

from xagent.web.api import websocket as websocket_api

TASK_ID = 41
CAPABILITIES_FRAME = {"type": "connection_capabilities", "capabilities": ["heartbeat"]}
HEARTBEAT_FRAME = {"type": "heartbeat"}
STATUS_MARKER = {"type": "status_request_handled"}


class _FakeSocket:
    """Records sent frames; ``confirmation_delay`` slows only the capabilities
    frame so a heartbeat task started too early would overtake it."""

    def __init__(self, liveness: list[str], *, confirmation_delay: float = 0.0) -> None:
        self.query_params = SimpleNamespace(
            getlist=lambda key: list(liveness) if key == "liveness" else []
        )
        self.accept = AsyncMock()
        self.close = AsyncMock()
        self.frames: list[dict[str, Any]] = []
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self._confirmation_delay = confirmation_delay

    async def send_text(self, text: str) -> None:
        frame = json.loads(text)
        if frame == CAPABILITIES_FRAME and self._confirmation_delay:
            await asyncio.sleep(self._confirmation_delay)
        self.frames.append(frame)

    async def receive_text(self) -> str:
        item = await self._inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def client_sends(self, item: Any) -> None:
        self._inbox.put_nowait(item)


async def _status_marker(websocket: _FakeSocket, *_: Any) -> None:
    websocket.frames.append(STATUS_MARKER)


def _endpoint_env(manager: websocket_api.ConnectionManager, **overrides: Any) -> Any:
    """Patch the endpoint's collaborators so no database is touched."""
    return patch.multiple(
        websocket_api,
        manager=manager,
        get_authenticated_user=AsyncMock(
            return_value=SimpleNamespace(id=7, is_admin=False)
        ),
        _private_websocket_task_access_sync=lambda **_: "authorized",
        handle_status_request=_status_marker,
        **overrides,
    )


async def _finished(endpoint: asyncio.Task[None]) -> list[Any]:
    # Bounded, so an endpoint that never stops its heartbeat task fails the
    # test instead of hanging it.
    return await asyncio.wait_for(
        asyncio.gather(endpoint, return_exceptions=True), timeout=5
    )


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
        await asyncio.sleep(0.3)  # many heartbeat intervals
        websocket.client_sends(WebSocketDisconnect())
        await _finished(endpoint)

    if negotiated:
        # The acknowledgement precedes the status snapshot and every heartbeat.
        assert websocket.frames[0] == CAPABILITIES_FRAME
        assert STATUS_MARKER in websocket.frames
        assert HEARTBEAT_FRAME in websocket.frames
    else:
        assert websocket.frames == [STATUS_MARKER]


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
        assert first.frames.count(HEARTBEAT_FRAME) >= 2

        release_handler.set()
        first.client_sends(WebSocketDisconnect())
        await _finished(endpoint)

        assert manager.connection_count() == 0
        beats_at_exit = first.frames.count(HEARTBEAT_FRAME)
        await asyncio.sleep(0.2)
        assert first.frames.count(HEARTBEAT_FRAME) == beats_at_exit

    # Phase 2: the heartbeat task dies with an exception outside the swallowed
    # send-failure group; the connection is still removed from the manager.
    async def failing_heartbeat(_: Any) -> None:
        raise ValueError("heartbeat fault")

    second = _FakeSocket(["heartbeat"])
    with _endpoint_env(
        manager,
        _run_chat_heartbeat=failing_heartbeat,
    ):
        endpoint = _run_endpoint(second)
        await asyncio.sleep(0.1)
        assert manager.connection_count() == 1
        second.client_sends(WebSocketDisconnect())
        assert await _finished(endpoint) == [None]
        assert manager.connection_count() == 0


def test_heartbeat_interval_stays_within_half_the_dead_threshold() -> None:
    assert (
        websocket_api.HEARTBEAT_INTERVAL_SECONDS * 2
        <= websocket_api.CLIENT_DEAD_AFTER_SECONDS
    )
