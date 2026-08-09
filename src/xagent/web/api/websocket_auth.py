"""Shared authentication ownership for authenticated WebSocket transports."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import WebSocket
from fastapi.responses import JSONResponse

from ..auth_dependencies import get_user_from_websocket_token
from ..models.database import get_session_local
from ..services.db_runtime import run_db_io_cancellation_safe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSocketPrincipal:
    """The complete authenticated identity needed by WebSocket transports."""

    id: int
    is_admin: bool
    guest_id: str | None = None
    widget_entity_key: str | None = None


class _WebSocketAuthenticationTerminated(Exception):
    """Authentication already sent a terminal transport response."""


def _load_websocket_principal_sync(token: str) -> WebSocketPrincipal | None:
    """Authenticate one token inside a worker-owned short Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        user = get_user_from_websocket_token(token, db)
        if user is None or user.id is None:
            return None
        return WebSocketPrincipal(id=int(user.id), is_admin=bool(user.is_admin))


async def get_authenticated_user(
    websocket: WebSocket, token: str | None = None
) -> WebSocketPrincipal | None:
    """Load a detached principal or return ``None`` for rejected credentials.

    Operational authentication failures are raised after this owner sends its
    terminal transport response. Cancellation and other process-control signals
    propagate unchanged.
    """

    if not token:
        return None
    try:
        return await run_db_io_cancellation_safe(
            lambda: _load_websocket_principal_sync(token)
        )
    except Exception as exc:
        route_template = getattr(websocket.scope.get("route"), "path", "<unresolved>")
        logger.exception(
            "WebSocket authentication infrastructure failure "
            "transport=websocket route=%s",
            route_template,
        )
        try:
            if "websocket.http.response" in websocket.scope.get("extensions", {}):
                await websocket.send_denial_response(
                    JSONResponse(
                        status_code=503,
                        content={"detail": "Service temporarily unavailable"},
                    )
                )
            else:
                await websocket.accept()
                await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            logger.exception(
                "WebSocket authentication terminal response failure "
                "transport=websocket route=%s",
                route_template,
            )
        raise _WebSocketAuthenticationTerminated() from exc
