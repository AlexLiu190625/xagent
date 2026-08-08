"""Tests for the shared authenticated WebSocket transport owner."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import is_dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocket

from xagent.web.api import progress_ws
from xagent.web.api import websocket as websocket_api
from xagent.web.api import websocket_auth


def _asgi_websocket(
    *, denial_extension: bool
) -> tuple[WebSocket, list[dict[str, object]]]:
    """Create a pre-handshake WebSocket and collect its outgoing ASGI messages."""

    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "websocket.connect"}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    extensions = {"websocket.http.response": {}} if denial_extension else {}
    websocket = WebSocket(
        {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "ws",
            "path": "/ws/chat/1",
            "raw_path": b"/ws/chat/1",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "subprotocols": [],
            "extensions": extensions,
        },
        receive,
        send,
    )
    return websocket, messages


@pytest.mark.asyncio
async def test_missing_token_returns_none_without_starting_database_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_db_io = MagicMock()
    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", run_db_io)

    assert await websocket_auth.get_authenticated_user(MagicMock(), None) is None
    run_db_io.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_token_returns_none_and_closes_worker_owned_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    class TrackingSession:
        def __enter__(self) -> "TrackingSession":
            return self

        def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
            closed.append(True)

    monkeypatch.setattr(websocket_auth, "get_session_local", lambda: TrackingSession)
    monkeypatch.setattr(
        websocket_auth, "get_user_from_websocket_token", lambda _token, _db: None
    )

    assert await websocket_auth.get_authenticated_user(MagicMock(), "invalid") is None
    assert closed == [True]


@pytest.mark.asyncio
async def test_valid_token_returns_frozen_detached_principal_from_worker_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = threading.get_ident()
    auth_threads: list[int] = []
    closed: list[bool] = []

    class TrackingSession:
        def __enter__(self) -> "TrackingSession":
            return self

        def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
            closed.append(True)

    def authenticate(_token: str, _db: object) -> SimpleNamespace:
        auth_threads.append(threading.get_ident())
        return SimpleNamespace(id=73, is_admin=True)

    monkeypatch.setattr(websocket_auth, "get_session_local", lambda: TrackingSession)
    monkeypatch.setattr(websocket_auth, "get_user_from_websocket_token", authenticate)

    principal = await websocket_auth.get_authenticated_user(MagicMock(), "signed")

    assert principal == websocket_auth.WebSocketPrincipal(id=73, is_admin=True)
    assert is_dataclass(principal)
    assert principal.__dataclass_params__.frozen is True
    assert auth_threads == [auth_threads[0]]
    assert auth_threads[0] != event_loop_thread
    assert closed == [True]


@pytest.mark.asyncio
async def test_cancellation_propagates_from_database_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancelled(_operation: object) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", cancelled)

    with pytest.raises(asyncio.CancelledError):
        await websocket_auth.get_authenticated_user(MagicMock(), "signed")


@pytest.mark.asyncio
async def test_operational_auth_failure_sends_sanitized_extension_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket, messages = _asgi_websocket(denial_extension=True)

    async def timeout(_operation: object) -> None:
        raise TimeoutError("database pool token=secret")

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", timeout)

    with pytest.raises(websocket_auth._WebSocketAuthenticationTerminated):
        await websocket_auth.get_authenticated_user(websocket, "signed-token")

    assert messages == [
        {
            "type": "websocket.http.response.start",
            "status": 503,
            "headers": [
                (b"content-length", b"44"),
                (b"content-type", b"application/json"),
            ],
        },
        {
            "type": "websocket.http.response.body",
            "body": b'{"detail":"Service temporarily unavailable"}',
        },
    ]
    serialized_messages = json.dumps(messages, default=lambda value: value.decode())
    assert "signed-token" not in serialized_messages
    assert "database pool" not in serialized_messages


@pytest.mark.asyncio
async def test_operational_auth_failure_without_extension_accepts_then_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket, messages = _asgi_websocket(denial_extension=False)

    async def timeout(_operation: object) -> None:
        raise TimeoutError("database pool token=secret")

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", timeout)

    with pytest.raises(websocket_auth._WebSocketAuthenticationTerminated):
        await websocket_auth.get_authenticated_user(websocket, "signed-token")

    assert messages == [
        {"type": "websocket.accept", "subprotocol": None, "headers": []},
        {
            "type": "websocket.close",
            "code": 1011,
            "reason": "Internal server error",
        },
    ]
    serialized_messages = json.dumps(messages, default=lambda value: value.decode())
    assert "signed-token" not in serialized_messages
    assert "database pool" not in serialized_messages


@pytest.mark.asyncio
async def test_denial_send_failure_is_terminal_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = MagicMock()
    websocket.scope = {"extensions": {"websocket.http.response": {}}}
    websocket.send_denial_response = AsyncMock(side_effect=ConnectionError("closed"))
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()

    async def timeout(_operation: object) -> None:
        raise TimeoutError("database pool")

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", timeout)

    with pytest.raises(websocket_auth._WebSocketAuthenticationTerminated):
        await websocket_auth.get_authenticated_user(websocket, "signed-token")

    websocket.send_denial_response.assert_awaited_once()
    websocket.accept.assert_not_awaited()
    websocket.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_extensionless_accept_failure_is_terminal_without_close_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = MagicMock()
    websocket.scope = {"extensions": {}}
    websocket.accept = AsyncMock(side_effect=ConnectionError("closed"))
    websocket.close = AsyncMock()
    websocket.send_denial_response = AsyncMock()

    async def timeout(_operation: object) -> None:
        raise TimeoutError("database pool")

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", timeout)

    with pytest.raises(websocket_auth._WebSocketAuthenticationTerminated):
        await websocket_auth.get_authenticated_user(websocket, "signed-token")

    websocket.accept.assert_awaited_once()
    websocket.close.assert_not_awaited()
    websocket.send_denial_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_extensionless_close_failure_is_terminal_without_second_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = MagicMock()
    websocket.scope = {"extensions": {}}
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock(side_effect=ConnectionError("closed"))
    websocket.send_denial_response = AsyncMock()

    async def timeout(_operation: object) -> None:
        raise TimeoutError("database pool")

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", timeout)

    with pytest.raises(websocket_auth._WebSocketAuthenticationTerminated):
        await websocket_auth.get_authenticated_user(websocket, "signed-token")

    websocket.accept.assert_awaited_once()
    websocket.close.assert_awaited_once_with(code=1011, reason="Internal server error")
    websocket.send_denial_response.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("control_signal", [KeyboardInterrupt, SystemExit])
async def test_process_control_signals_are_not_translated(
    monkeypatch: pytest.MonkeyPatch,
    control_signal: type[BaseException],
) -> None:
    websocket = MagicMock()
    websocket.send_denial_response = AsyncMock()

    async def interrupted(_operation: object) -> None:
        raise control_signal()

    monkeypatch.setattr(websocket_auth, "run_db_io_cancellation_safe", interrupted)

    with pytest.raises(control_signal):
        await websocket_auth.get_authenticated_user(websocket, "signed-token")

    websocket.send_denial_response.assert_not_awaited()


async def _authentication_terminated(*_args: object, **_kwargs: object) -> None:
    raise websocket_auth._WebSocketAuthenticationTerminated()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "endpoint_args"),
    [
        (websocket_api.websocket_chat_endpoint, (MagicMock(), 1, "signed-token")),
        (websocket_api.websocket_builder_chat_endpoint, (MagicMock(), "signed-token")),
        (websocket_api.websocket_build_preview_endpoint, (MagicMock(), "signed-token")),
    ],
)
async def test_main_endpoints_return_after_shared_terminal_authentication(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: object,
    endpoint_args: tuple[MagicMock, object, str],
) -> None:
    websocket = endpoint_args[0]
    websocket.close = AsyncMock()
    websocket.accept = AsyncMock()
    manager = MagicMock()
    manager.connect = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", manager)
    monkeypatch.setattr(
        websocket_api, "get_authenticated_user", _authentication_terminated
    )

    await endpoint(*endpoint_args)  # type: ignore[operator]

    websocket.close.assert_not_awaited()
    websocket.accept.assert_not_awaited()
    manager.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_progress_returns_after_shared_terminal_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = MagicMock()
    websocket.close = AsyncMock()
    websocket.accept = AsyncMock()
    broadcaster = MagicMock()
    broadcaster.connect = AsyncMock()
    authenticated_user = AsyncMock(side_effect=_authentication_terminated)
    monkeypatch.setattr(
        progress_ws, "get_authenticated_user", authenticated_user, raising=False
    )
    monkeypatch.setattr(progress_ws, "progress_broadcaster", broadcaster)

    await progress_ws.progress_websocket_endpoint(websocket, "task", "signed-token")

    authenticated_user.assert_awaited_once_with(websocket, "signed-token")
    websocket.close.assert_not_awaited()
    websocket.accept.assert_not_awaited()
    broadcaster.connect.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "endpoint_args"),
    [
        (websocket_api.websocket_chat_endpoint, (MagicMock(), 1, "invalid")),
        (websocket_api.websocket_builder_chat_endpoint, (MagicMock(), "invalid")),
        (websocket_api.websocket_build_preview_endpoint, (MagicMock(), "invalid")),
    ],
)
async def test_main_endpoints_retain_invalid_credential_close_codes(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: object,
    endpoint_args: tuple[MagicMock, object, str],
) -> None:
    websocket = endpoint_args[0]
    websocket.close = AsyncMock()
    monkeypatch.setattr(
        websocket_api, "get_authenticated_user", AsyncMock(return_value=None)
    )

    await endpoint(*endpoint_args)  # type: ignore[operator]

    websocket.close.assert_awaited_once_with(
        code=4001, reason="Authentication required"
    )


@pytest.mark.asyncio
async def test_progress_retains_invalid_credential_close_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = MagicMock()
    websocket.close = AsyncMock()
    authenticated_user = AsyncMock(return_value=None)
    monkeypatch.setattr(
        progress_ws, "get_authenticated_user", authenticated_user, raising=False
    )

    await progress_ws.progress_websocket_endpoint(websocket, "task", "invalid")

    authenticated_user.assert_awaited_once_with(websocket, "invalid")
    websocket.close.assert_awaited_once_with(code=1008)
