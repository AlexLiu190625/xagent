"""Rate-limit enforcement on the public share endpoints (#973, PR2) and the
widget websocket (#1056).

Each anonymous share surface returns 429 once its bucket is exhausted; the
websocket gates close pre-accept (connect) or reject the turn in-band (turn).
The autouse conftest fixture resets the share limiter before every test; each
test tightens the relevant limit to 1/minute via env and resets again so the
new limiter reads it.
"""

from __future__ import annotations

import contextlib
import io
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.user import User
from xagent.web.services.share_rate_limit import reset_share_rate_limiter

from .conftest import _admin_headers, _direct_db_session, _setup_admin, client

pytestmark = pytest.mark.usefixtures("_test_db")


def _user_id() -> int:
    _setup_admin()
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == "admin").one().id)
    finally:
        db.close()


def _published_share_agent(token: str) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=_user_id(),
            name="RL Agent",
            description="d",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            share_enabled=True,
            share_token=token,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _create_workforce(name: str) -> str:
    headers = _admin_headers()
    owner = _user_id()

    def _agent(agent_name: str, tok: str) -> int:
        db = _direct_db_session()
        try:
            a = Agent(
                user_id=owner,
                name=agent_name,
                description="d",
                instructions="i",
                execution_mode="balanced",
                status=AgentStatus.PUBLISHED,
                share_enabled=True,
                share_token=tok,
            )
            db.add(a)
            db.commit()
            db.refresh(a)
            return int(a.id)
        finally:
            db.close()

    resp = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": name,
            "description": "rl",
            "manager_agent_id": _agent(f"{name} Mgr", f"{name}-mgr"),
            "workers": [
                {
                    "source_type": "existing",
                    "agent_id": _agent(f"{name} Wrk", f"{name}-wrk"),
                    "alias": "w1",
                    "assignment_instructions": "go",
                    "enabled": True,
                    "sort_order": 1,
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    wf_id = int(resp.json()["id"])
    published = client.post(f"/api/workforces/{wf_id}/publish", headers=headers)
    assert published.status_code == 200, published.text
    share = client.post(f"/api/workforces/{wf_id}/share-link", headers=headers)
    assert share.status_code == 200, share.text
    return str(share.json()["share_token"])


def _guest_headers(token: str) -> dict[str, str]:
    resp = client.post("/api/share/auth", json={"share_token": token})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_share_auth_returns_429_over_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_SHARE_AUTH_RATE_LIMIT", "1/minute")
    reset_share_rate_limiter()
    _published_share_agent("rl-auth-tok")

    first = client.post("/api/share/auth", json={"share_token": "rl-auth-tok"})
    assert first.status_code == 200, first.text
    second = client.post("/api/share/auth", json={"share_token": "rl-auth-tok"})
    assert second.status_code == 429, second.text


def test_share_task_create_returns_429_over_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _published_share_agent("rl-create-tok")
    guest = _guest_headers("rl-create-tok")

    # Tighten only after the guest token is minted (auth has its own bucket).
    monkeypatch.setenv("XAGENT_SHARE_TASK_CREATE_RATE_LIMIT", "1/minute")
    reset_share_rate_limiter()

    body = {"title": "hi", "description": "hi"}
    first = client.post("/api/share/chat/task/create", headers=guest, json=body)
    assert first.status_code == 200, first.text
    second = client.post("/api/share/chat/task/create", headers=guest, json=body)
    assert second.status_code == 429, second.text


def test_share_upload_returns_429_over_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = _create_workforce("RL Upload WF")
    guest = _guest_headers(token)

    monkeypatch.setenv("XAGENT_SHARE_UPLOAD_RATE_LIMIT", "1/minute")
    reset_share_rate_limiter()

    def _upload():
        return client.post(
            "/api/share/files/upload",
            headers=guest,
            data={"task_type": "task"},
            files={"file": ("n.txt", io.BytesIO(b"x"), "text/plain")},
        )

    assert _upload().status_code == 200
    assert _upload().status_code == 429


class _FakeWebSocket:
    """Minimal websocket double: yields queued frames, then disconnects."""

    def __init__(self, frames: list[str]) -> None:
        self._frames = list(frames)
        self.accepted = False
        self.closed = False

    async def accept(self) -> None:
        # The endpoint accepts before auth so denial reasons survive the
        # handshake (#973); the double just records it.
        self.accepted = True

    async def receive_text(self) -> str:
        from fastapi import WebSocketDisconnect

        if not self._frames:
            raise WebSocketDisconnect(code=1000)
        return self._frames.pop(0)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        # Recorded (not raised) so tests can pin the "socket stays open"
        # contract: a real close is a terminal server action even though this
        # double would keep yielding frames after it.
        self.closed = True


@pytest.mark.asyncio
async def test_ws_turn_rate_limited_rejects_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rate-limited websocket turn is rejected (message_rejected) and never
    dispatched to handle_chat_message; the connection stays open."""
    from xagent.web.api import public_chat_access as pca

    monkeypatch.setenv("XAGENT_SHARE_WS_TURN_RATE_LIMIT", "1/minute")
    reset_share_rate_limiter()

    ctx = pca.ShareChatAccessContext(
        user=MagicMock(id=1),
        share_token="tok",
        guest_id="guest-ws",
        agent=MagicMock(),
    )
    monkeypatch.setattr(pca, "get_share_chat_user", lambda *a, **k: ctx)
    monkeypatch.setattr(pca, "get_task_for_share_context", lambda *a, **k: MagicMock())

    @contextlib.contextmanager
    def _fake_db():
        yield MagicMock()

    monkeypatch.setattr(pca, "db_session_context", _fake_db)
    monkeypatch.setattr(
        pca, "manager", MagicMock(connect=AsyncMock(), disconnect=MagicMock())
    )
    monkeypatch.setattr(pca, "handle_status_request", AsyncMock())
    dispatch = AsyncMock()
    monkeypatch.setattr(pca, "handle_chat_message", dispatch)
    delivery = AsyncMock()
    monkeypatch.setattr(pca, "send_message_delivery", delivery)

    # Two chat turns: the first is admitted (dispatched), the second trips the
    # 1/minute per-guest bucket and is rejected.
    frames = [
        json.dumps({"type": "chat", "client_message_id": "m1", "message": "hi"}),
        json.dumps({"type": "chat", "client_message_id": "m2", "message": "again"}),
    ]
    websocket = _FakeWebSocket(frames)
    await pca.share_chat_websocket_endpoint(websocket=websocket, task_id=1, token="jwt")

    assert dispatch.await_count == 1  # only the admitted turn dispatched
    assert delivery.await_count == 1  # the rejected turn got a delivery ack
    assert websocket.closed is False  # throttled turn must not close the socket
    _, kwargs = delivery.await_args
    assert kwargs["accepted"] is False
    assert kwargs["client_message_id"] == "m2"
    assert kwargs["rejection_outcome"] == "not_accepted"


@pytest.mark.asyncio
async def test_ws_turn_rate_limited_without_client_id_sends_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rate-limited turn that carries no client_message_id still surfaces the
    throttle: send_message_delivery no-ops without an id, so a generic error
    frame is sent instead (the client isn't left with a silently dropped turn)."""
    from xagent.web.api import public_chat_access as pca

    monkeypatch.setenv("XAGENT_SHARE_WS_TURN_RATE_LIMIT", "1/minute")
    reset_share_rate_limiter()

    ctx = pca.ShareChatAccessContext(
        user=MagicMock(id=1),
        share_token="tok",
        guest_id="guest-noid",
        agent=MagicMock(),
    )
    monkeypatch.setattr(pca, "get_share_chat_user", lambda *a, **k: ctx)
    monkeypatch.setattr(pca, "get_task_for_share_context", lambda *a, **k: MagicMock())

    @contextlib.contextmanager
    def _fake_db():
        yield MagicMock()

    monkeypatch.setattr(pca, "db_session_context", _fake_db)
    manager = MagicMock(
        connect=AsyncMock(),
        disconnect=MagicMock(),
        send_personal_message=AsyncMock(),
    )
    monkeypatch.setattr(pca, "manager", manager)
    monkeypatch.setattr(pca, "handle_status_request", AsyncMock())
    monkeypatch.setattr(pca, "handle_chat_message", AsyncMock())
    monkeypatch.setattr(pca, "send_message_delivery", AsyncMock())

    # Two untagged chat turns: the second trips the 1/minute bucket. Without a
    # client_message_id, the rejection must arrive as a generic error frame.
    frames = [
        json.dumps({"type": "chat", "message": "hi"}),
        json.dumps({"type": "chat", "message": "again"}),
    ]
    websocket = _FakeWebSocket(frames)
    await pca.share_chat_websocket_endpoint(websocket=websocket, task_id=1, token="jwt")

    error_frames = [
        call.args[0]
        for call in manager.send_personal_message.await_args_list
        if call.args and call.args[0].get("type") == "error"
    ]
    assert len(error_frames) == 1
    assert "too quickly" in error_frames[0]["message"]
    assert websocket.closed is False  # throttled turn must not close the socket


def test_ws_connect_over_limit_is_refused_pre_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over-limit share WS connection attempts are refused before the
    handshake is accepted (#993 F5): accept-before-auth means every admitted
    attempt completes a 101 upgrade even with a garbage token, so the attempts
    themselves carry a per-IP budget. TestClient surfaces a pre-accept close at
    context-manager entry (post-accept denials raise at receive instead —
    that asymmetry is what pins the ordering)."""
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setenv("XAGENT_SHARE_WS_CONNECT_IP_RATE_LIMIT", "1/minute")
    reset_share_rate_limiter()

    url = "/api/share/chat/ws/1?token=garbage"
    # First attempt is admitted: it upgrades, then auth rejects it post-accept.
    with client.websocket_connect(url) as ws:
        with pytest.raises(WebSocketDisconnect) as first:
            ws.receive_text()
    assert first.value.code == 4003

    # Second attempt from the same caller trips the budget pre-accept.
    with pytest.raises(WebSocketDisconnect) as denied:
        with client.websocket_connect(url):
            pass
    assert denied.value.code == 4008


@pytest.mark.asyncio
@pytest.mark.parametrize("frame_type", ["chat", "execute_task"])
async def test_widget_ws_turn_rate_limited_rejects_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    frame_type: str,
) -> None:
    """A rate-limited widget websocket turn is rejected (message_rejected) and
    never dispatched; the connection stays open (#1056). Keyed on entity + IP,
    not the client-supplied widget guest_id. Both run-starting frame types are
    gated, so both are driven through."""
    from xagent.web.api import public_chat_access as pca

    monkeypatch.setenv("XAGENT_WIDGET_WS_TURN_IP_RATE_LIMIT", "1/minute")
    reset_share_rate_limiter()

    ctx = pca.PublicChatAccessContext(
        user=MagicMock(id=1, is_admin=False),
        channel_id=None,
        guest_id="widget-guest",
        auth_mode="widget",
        widget_agent_id=7,
    )
    monkeypatch.setattr(pca, "get_public_chat_user", lambda *a, **k: ctx)
    monkeypatch.setattr(pca, "get_task_for_public_context", lambda *a, **k: MagicMock())
    monkeypatch.setattr(pca, "remote_ip_from_request", lambda ws: "9.9.9.9")
    monkeypatch.setattr(
        pca, "manager", MagicMock(connect=AsyncMock(), disconnect=MagicMock())
    )
    monkeypatch.setattr(pca, "handle_status_request", AsyncMock())
    # One mock behind both run-starting handlers: the admitted turn dispatches
    # to whichever handler matches frame_type, the throttled one to neither.
    dispatch = AsyncMock()
    monkeypatch.setattr(pca, "handle_chat_message", dispatch)
    monkeypatch.setattr(pca, "handle_execute_task", dispatch)
    delivery = AsyncMock()
    monkeypatch.setattr(pca, "send_message_delivery", delivery)

    # Two turns of the same type: the first is admitted (dispatched), the
    # second trips the 1/minute per-IP bucket and is rejected.
    frames = [
        json.dumps({"type": frame_type, "client_message_id": "m1", "message": "hi"}),
        json.dumps({"type": frame_type, "client_message_id": "m2", "message": "again"}),
    ]
    websocket = _FakeWebSocket(frames)
    await pca.public_chat_websocket_endpoint(
        websocket=websocket,
        task_id=1,
        token="jwt",
        expected_auth_mode="widget",
    )

    assert dispatch.await_count == 1  # only the admitted turn dispatched
    assert delivery.await_count == 1  # the rejected turn got a delivery ack
    assert websocket.closed is False  # throttled turn must not close the socket
    _, kwargs = delivery.await_args
    assert kwargs["accepted"] is False
    assert kwargs["client_message_id"] == "m2"
    assert kwargs["rejection_outcome"] == "not_accepted"


@pytest.mark.asyncio
async def test_widget_ws_turn_gate_does_not_gate_interventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interventions are control messages, not new runs: they pass even when
    the turn budget is exhausted (mirrors the share endpoint's contract)."""
    from xagent.web.api import public_chat_access as pca

    monkeypatch.setenv("XAGENT_WIDGET_WS_TURN_IP_RATE_LIMIT", "1/minute")
    reset_share_rate_limiter()

    ctx = pca.PublicChatAccessContext(
        user=MagicMock(id=1, is_admin=False),
        channel_id=None,
        guest_id="widget-guest",
        auth_mode="widget",
        widget_workforce_id=3,
    )
    monkeypatch.setattr(pca, "get_public_chat_user", lambda *a, **k: ctx)
    monkeypatch.setattr(pca, "get_task_for_public_context", lambda *a, **k: MagicMock())
    monkeypatch.setattr(pca, "remote_ip_from_request", lambda ws: "7.7.7.7")
    monkeypatch.setattr(
        pca, "manager", MagicMock(connect=AsyncMock(), disconnect=MagicMock())
    )
    monkeypatch.setattr(pca, "handle_status_request", AsyncMock())
    chat_dispatch = AsyncMock()
    monkeypatch.setattr(pca, "handle_chat_message", chat_dispatch)
    intervention_dispatch = AsyncMock()
    monkeypatch.setattr(pca, "handle_intervention", intervention_dispatch)
    monkeypatch.setattr(pca, "send_message_delivery", AsyncMock())

    # Exhaust the turn budget with the first chat, then send an intervention:
    # it must still dispatch.
    frames = [
        json.dumps({"type": "chat", "client_message_id": "m1", "message": "hi"}),
        json.dumps({"type": "intervention", "action": "noop"}),
    ]
    await pca.public_chat_websocket_endpoint(
        websocket=_FakeWebSocket(frames),
        task_id=1,
        token="jwt",
        expected_auth_mode="widget",
    )

    assert chat_dispatch.await_count == 1
    assert intervention_dispatch.await_count == 1


def test_widget_ws_connect_over_limit_is_refused_pre_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over-limit widget WS connection attempts are refused before the
    handshake is accepted (#1056), mirroring the share gate. Since #1057 the
    widget endpoint also accepts before auth, so a garbage token's denial is a
    post-accept 4003 (it surfaces at receive) while the connect-gate rejection
    stays pre-accept (it surfaces at context-manager entry) — that asymmetry is
    what pins the gate ordering, exactly as on the share path."""
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setenv("XAGENT_WIDGET_WS_CONNECT_IP_RATE_LIMIT", "1/minute")
    reset_share_rate_limiter()

    url = "/api/widget/chat/ws/1?token=garbage"
    # First attempt is admitted: it upgrades, then auth rejects it post-accept.
    with client.websocket_connect(url) as ws:
        with pytest.raises(WebSocketDisconnect) as first:
            ws.receive_text()
    assert first.value.code == 4003

    # Second attempt from the same caller trips the budget pre-accept.
    with pytest.raises(WebSocketDisconnect) as denied:
        with client.websocket_connect(url):
            pass
    assert denied.value.code == 4008


def test_ws_close_reason_clamps_to_123_utf8_bytes() -> None:
    """WS close reasons are capped at 123 UTF-8 bytes by the protocol;
    ``exc.detail`` is ``Any`` in FastAPI, so the close path coerces and clamps
    on a codepoint boundary instead of blowing up mid-close (#993 F6)."""
    from xagent.web.api.public_chat_access import _ws_close_reason

    assert _ws_close_reason("short reason") == "short reason"
    assert _ws_close_reason(None) == ""
    assert _ws_close_reason({"nested": "detail"}) == str({"nested": "detail"})

    clamped = _ws_close_reason("б" * 200)  # 2-byte codepoint
    encoded = clamped.encode("utf-8")
    assert len(encoded) <= 123
    # 123 is odd, so a naive byte slice would split the final 2-byte
    # codepoint; the clamp must land on a boundary.
    assert clamped == "б" * 61
