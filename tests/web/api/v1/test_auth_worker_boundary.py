"""Concurrency and ownership invariants for /v1 API-key authentication."""

import threading
from dataclasses import FrozenInstanceError

import httpx
import pytest
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import event

from xagent.core.utils.api_key import verify_api_key as real_verify_api_key
from xagent.web.api.v1 import deps
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import get_engine
from xagent.web.services import api_keys as api_key_service

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    app_for_tests,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


def _create_agent_key(*, tool_categories: list[str] | None = None) -> tuple[str, int]:
    headers = _admin_headers()
    response = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "worker boundary agent",
            "description": "auth worker test",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    )
    assert response.status_code == 200, response.text
    agent_id = int(response.json()["id"])

    db = _direct_db_session()
    try:
        db.query(Agent).filter(Agent.id == agent_id).update(
            {
                "status": AgentStatus.PUBLISHED,
                "tool_categories": tool_categories,
            }
        )
        db.commit()
    finally:
        db.close()

    key_response = client.post(
        f"/api/agents/{agent_id}/api-key",
        headers=headers,
    )
    assert key_response.status_code == 200, key_response.text
    return str(key_response.json()["full_key"]), agent_id


def _create_personal_key() -> str:
    response = client.post("/api/me/personal-keys", headers=_admin_headers())
    assert response.status_code == 200, response.text
    return str(response.json()["full_key"])


@pytest.mark.asyncio
async def test_personal_auth_checks_in_connection_before_bcrypt_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_key = _create_personal_key()
    engine = get_engine()
    loop_thread_id = threading.get_ident()
    events: list[tuple[str, int]] = []

    def on_checkout(*_args: object) -> None:
        events.append(("checkout", threading.get_ident()))

    def on_checkin(*_args: object) -> None:
        events.append(("checkin", threading.get_ident()))  # codespell:ignore checkin

    def observed_verify(raw: str, stored_hash: str) -> bool:
        events.append(("bcrypt", threading.get_ident()))
        return real_verify_api_key(raw, stored_hash)

    event.listen(engine, "checkout", on_checkout)
    event.listen(engine, "checkin", on_checkin)  # codespell:ignore checkin
    monkeypatch.setattr(
        api_key_service,
        "verify_api_key",
        observed_verify,
    )
    try:
        transport = httpx.ASGITransport(
            app=app_for_tests,
            raise_app_exceptions=False,
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as async_client:
            response = await async_client.get(
                "/v1/me",
                headers={"Authorization": f"Bearer {full_key}"},
            )
    finally:
        event.remove(engine, "checkout", on_checkout)
        event.remove(engine, "checkin", on_checkin)  # codespell:ignore checkin

    assert response.status_code == 200, response.text
    names = [name for name, _thread_id in events]
    checkin_index = names.index("checkin")  # codespell:ignore checkin
    assert names.index("checkout") < checkin_index < names.index("bcrypt")
    bcrypt_thread_id = next(thread_id for name, thread_id in events if name == "bcrypt")
    assert bcrypt_thread_id != loop_thread_id


def test_agent_auth_service_returns_frozen_detached_identity() -> None:
    full_key, agent_id = _create_agent_key(tool_categories=["mcp:ShiftCare"])

    identity = api_key_service.authenticate_agent_api_key(full_key)

    assert identity.agent_id == agent_id
    assert identity.user_id > 0
    assert identity.execution_mode == "balanced"
    assert identity.tool_categories == ("mcp:ShiftCare",)
    assert identity.status == AgentStatus.PUBLISHED.value
    assert identity.key_prefix
    assert not hasattr(identity, "_sa_instance_state")
    with pytest.raises(FrozenInstanceError):
        identity.agent_id = -1  # type: ignore[misc]


def test_personal_auth_service_returns_frozen_detached_identity() -> None:
    full_key = _create_personal_key()

    identity = api_key_service.authenticate_personal_api_key(full_key)

    assert identity.user_id > 0
    assert identity.is_admin is True
    assert identity.username == "admin"
    assert identity.email == "admin@example.com"
    assert identity.key_prefix
    assert not hasattr(identity, "_sa_instance_state")
    with pytest.raises(FrozenInstanceError):
        identity.user_id = -1  # type: ignore[misc]


@pytest.mark.asyncio
async def test_auth_dependency_returns_agent_identity_without_request_session() -> None:
    full_key, agent_id = _create_agent_key(tool_categories=[])

    identity = await deps.get_agent_from_api_key(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=full_key)
    )

    assert identity.agent_id == agent_id
    assert identity.tool_categories == ()


def test_usage_tracking_swallows_update_rollback_and_close_failures(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingUsageSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            raise RuntimeError("close failed")

        def query(self, _model):
            raise RuntimeError("update failed")

        def rollback(self) -> None:
            raise RuntimeError("rollback failed")

    monkeypatch.setattr(
        api_key_service,
        "get_session_local",
        lambda: FailingUsageSession,
    )

    with caplog.at_level("WARNING", logger="xagent.web.services.api_keys"):
        api_key_service.record_agent_api_key_usage("best-effort-prefix")

    assert "Failed to record API key usage" in caplog.text
