from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from xagent.core.utils.encryption import decrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import (
    MCPOAuthConnectRequest,
    MCPOAuthDiscoverRequest,
    MCPOAuthStatusResponse,
    connect_mcp_oauth,
    delete_mcp_oauth_grant,
    discover_mcp_oauth,
    get_mcp_oauth_status,
    mcp_oauth_callback,
)
from xagent.web.models import MCPOAuthClient, MCPOAuthFlowState, MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "mcp-oauth.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    user = User(username="alice", password_hash="x", is_admin=False)
    other_user = User(username="bob", password_hash="x", is_admin=False)
    db.add_all([user, other_user])
    db.commit()
    db.refresh(user)
    db.refresh(other_user)

    yield db, user, other_user
    db.close()
    engine.dispose()


def _request(path: str, headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    parsed = urlparse(path)
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": parsed.path,
            "query_string": parsed.query.encode(),
            "headers": headers or [],
        }
    )


def _discovery() -> SimpleNamespace:
    return SimpleNamespace(
        resource="https://mcp.example.com/mcp",
        scopes=("records.read",),
        protected_resource=SimpleNamespace(
            authorization_servers=("https://auth.example.com",),
        ),
        authorization_server=SimpleNamespace(
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            client_id_metadata_document_supported=True,
            raw={"issuer": "https://auth.example.com"},
        ),
    )


def _add_mcp_oauth_server(db, user: User) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": "records",
            "managed": "external",
            "transport": "streamable_http",
            "url": "https://mcp.example.com/mcp",
            "auth": {
                "type": "mcp_oauth",
                "resource": "https://mcp.example.com/mcp",
                "issuer": "https://auth.example.com",
                "scope": "records.read",
                "client_id": "client-123",
                "client_secret": "client-secret",
                "redirect_uri": "https://xagent.example.com/api/mcp/oauth/callback",
                "token_endpoint_auth_method": "client_secret_post",
            },
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()
    return server


def _add_callback_client_and_state(
    db,
    user: User,
    *,
    state: str,
    metadata_json: dict | None = None,
) -> tuple[MCPServer, MCPOAuthClient, MCPOAuthFlowState]:
    server = _add_mcp_oauth_server(db, user)
    client = MCPOAuthClient(
        mcp_server_id=server.id,
        issuer="https://auth.example.com",
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        client_id="client-123",
        token_endpoint_auth_method="none",
        redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        metadata_json=metadata_json,
    )
    flow_state = MCPOAuthFlowState(
        state=state,
        mcp_server_id=server.id,
        user_id=user.id,
        resource_owner_key="resource-owner-a",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        code_verifier=mcp_api.encrypt_value("verifier-123"),
        redirect_after="/mcp",
        expires_at=mcp_api._utc_now() + timedelta(minutes=10),
    )
    db.add_all([client, flow_state])
    db.commit()
    return server, client, flow_state


def _set_user_mcp_active(db, user: User, server: MCPServer, is_active: bool) -> None:
    user_mcp = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .one()
    )
    user_mcp.is_active = is_active
    db.commit()


@pytest.mark.asyncio
async def test_connect_creates_pkce_state_and_redirects(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    server.headers = {"Authorization": "Bearer static-token"}
    db.commit()

    async def fake_discover(*args, **kwargs):
        assert kwargs["headers"] is None
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    response = await connect_mcp_oauth(
        server.id,
        MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
        user,
        db,
    )

    assert response.status_code == 303
    location = response.headers["location"]
    parsed = urlparse(location)
    query = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://auth.example.com/authorize"
    )
    assert query["response_type"] == ["code"]
    assert query["client_id"] == ["client-123"]
    assert query["redirect_uri"] == [
        "https://xagent.example.com/api/mcp/oauth/callback"
    ]
    assert query["resource"] == ["https://mcp.example.com/mcp"]
    assert query["scope"] == ["records.read"]
    assert query["code_challenge_method"] == ["S256"]
    assert "code_challenge" in query
    assert "code_verifier" not in query

    flow_state = db.query(MCPOAuthFlowState).one()
    assert flow_state.state == query["state"][0]
    assert flow_state.resource_owner_key == f"xagent:user:{user.id}"
    assert flow_state.redirect_after == "/settings/mcp"
    assert decrypt_value(flow_state.code_verifier) != flow_state.code_verifier

    client = db.query(MCPOAuthClient).one()
    assert client.client_id == "client-123"
    assert client.client_secret != "client-secret"
    assert decrypt_value(client.client_secret) == "client-secret"


def test_connect_request_rejects_public_resource_owner_key():
    with pytest.raises(ValueError):
        MCPOAuthConnectRequest.model_validate(
            {
                "redirect_after": "/settings/mcp",
                "resource_owner_key": "external:public-request",
            }
        )


@pytest.mark.asyncio
async def test_connect_can_return_authorization_url_json(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)

    async def fake_discover(*args, **kwargs):
        return _discovery()

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fake_discover)

    response = await connect_mcp_oauth(
        server.id,
        MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
        user,
        db,
        accept="application/json",
    )

    assert isinstance(response, dict)
    authorization_url = response["authorization_url"]
    parsed = urlparse(authorization_url)
    query = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://auth.example.com/authorize"
    )
    assert query["client_id"] == ["client-123"]
    assert query["resource"] == ["https://mcp.example.com/mcp"]
    assert db.query(MCPOAuthFlowState).count() == 1


@pytest.mark.asyncio
async def test_oauth_routes_reject_inactive_user_mcp_server(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    _set_user_mcp_active(db, user, server, False)

    async def fail_discover(*args, **kwargs):
        pytest.fail("inactive MCP server must not run OAuth discovery")

    monkeypatch.setattr(mcp_api, "discover_mcp_oauth_metadata", fail_discover)

    with pytest.raises(mcp_api.HTTPException) as exc:
        await discover_mcp_oauth(
            server.id,
            MCPOAuthDiscoverRequest(),
            user,
            db,
        )
    assert exc.value.status_code == 404

    with pytest.raises(mcp_api.HTTPException) as exc:
        await connect_mcp_oauth(
            server.id,
            MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
            user,
            db,
        )
    assert exc.value.status_code == 404

    with pytest.raises(mcp_api.HTTPException) as exc:
        await get_mcp_oauth_status(server.id, user, db)
    assert exc.value.status_code == 404

    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        resource_owner_key=f"xagent:user:{user.id}",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token=mcp_api.encrypt_value("own-access-token"),
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)

    with pytest.raises(mcp_api.HTTPException) as exc:
        await delete_mcp_oauth_grant(server.id, grant.id, user, db)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_callback_exchanges_code_and_stores_encrypted_grant(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    client = MCPOAuthClient(
        mcp_server_id=server.id,
        issuer="https://auth.example.com",
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        client_id="client-123",
        client_secret=mcp_api.encrypt_value("client-secret"),
        token_endpoint_auth_method="client_secret_post",
        redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
    )
    flow_state = MCPOAuthFlowState(
        state="state-123",
        mcp_server_id=server.id,
        user_id=user.id,
        resource_owner_key="resource-owner-a",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        code_verifier=mcp_api.encrypt_value("verifier-123"),
        redirect_after="/mcp",
        expires_at=mcp_api._utc_now() + timedelta(minutes=10),
    )
    db.add_all([client, flow_state])
    db.commit()

    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        form = dict(item.split("=") for item in request.content.decode().split("&"))
        assert form["grant_type"] == "authorization_code"
        assert form["code"] == "auth-code"
        assert form["code_verifier"] == "verifier-123"
        assert form["resource"] == "https%3A%2F%2Fmcp.example.com%2Fmcp"
        assert form["client_secret"] == "client-secret"
        return httpx.Response(
            200,
            json={
                "access_token": "plain-access-token",
                "refresh_token": "plain-refresh-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "records.read",
            },
        )

    def async_client_factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(mcp_api.httpx, "AsyncClient", async_client_factory)

    response = await mcp_oauth_callback(
        _request("/api/mcp/oauth/callback?code=auth-code&state=state-123"),
        db,
    )

    assert response.status_code == 307
    assert response.headers["location"] == "/mcp"
    grant = db.query(MCPOAuthGrant).one()
    assert grant.resource_owner_key == "resource-owner-a"
    assert grant.access_token != "plain-access-token"
    assert decrypt_value(grant.access_token) == "plain-access-token"
    assert decrypt_value(grant.refresh_token) == "plain-refresh-token"
    assert db.query(MCPOAuthFlowState).one().consumed_at is not None


@pytest.mark.asyncio
async def test_callback_accepts_matching_issuer_when_supported(db_session, monkeypatch):
    db, user, _ = db_session
    _add_callback_client_and_state(
        db,
        user,
        state="issuer-match-state",
        metadata_json={"authorization_response_iss_parameter_supported": True},
    )

    async def fake_exchange(**kwargs):
        return {
            "access_token": "plain-access-token",
            "token_type": "Bearer",
            "scope": "records.read",
        }

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)

    response = await mcp_oauth_callback(
        _request(
            "/api/mcp/oauth/callback?code=auth-code&state=issuer-match-state"
            "&iss=https%3A%2F%2Fauth.example.com"
        ),
        db,
    )

    assert response.status_code == 307
    assert db.query(MCPOAuthGrant).count() == 1


@pytest.mark.asyncio
async def test_callback_rejects_missing_required_issuer_before_token_exchange(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_callback_client_and_state(
        db,
        user,
        state="issuer-required-state",
        metadata_json={"authorization_response_iss_parameter_supported": True},
    )

    async def fail_exchange(**kwargs):
        pytest.fail("token exchange must not run when callback issuer is required")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fail_exchange)

    with pytest.raises(mcp_api.HTTPException) as exc:
        await mcp_oauth_callback(
            _request(
                "/api/mcp/oauth/callback?code=auth-code&state=issuer-required-state"
            ),
            db,
        )

    assert exc.value.detail["code"] == "issuer_mismatch"
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_rejects_mismatched_issuer_before_token_exchange(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_callback_client_and_state(
        db,
        user,
        state="issuer-mismatch-state",
        metadata_json={"authorization_response_iss_parameter_supported": False},
    )

    async def fail_exchange(**kwargs):
        pytest.fail("token exchange must not run when callback issuer mismatches")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fail_exchange)

    with pytest.raises(mcp_api.HTTPException) as exc:
        await mcp_oauth_callback(
            _request(
                "/api/mcp/oauth/callback?code=auth-code&state=issuer-mismatch-state"
                "&iss=https%3A%2F%2Fevil.example.com"
            ),
            db,
        )

    assert exc.value.detail["code"] == "issuer_mismatch"
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_rejects_error_response_mismatched_issuer(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_callback_client_and_state(
        db,
        user,
        state="issuer-error-state",
        metadata_json={"authorization_response_iss_parameter_supported": True},
    )

    async def fail_exchange(**kwargs):
        pytest.fail("token exchange must not run for authorization error callbacks")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fail_exchange)

    with pytest.raises(mcp_api.HTTPException) as exc:
        await mcp_oauth_callback(
            _request(
                "/api/mcp/oauth/callback?error=access_denied&state=issuer-error-state"
                "&iss=https%3A%2F%2Fevil.example.com"
            ),
            db,
        )

    assert exc.value.detail["code"] == "issuer_mismatch"
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_accepts_absent_issuer_when_not_supported(
    db_session, monkeypatch
):
    db, user, _ = db_session
    _add_callback_client_and_state(
        db,
        user,
        state="issuer-unsupported-state",
        metadata_json={"authorization_response_iss_parameter_supported": False},
    )

    async def fake_exchange(**kwargs):
        return {
            "access_token": "plain-access-token",
            "token_type": "Bearer",
            "scope": "records.read",
        }

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)

    response = await mcp_oauth_callback(
        _request(
            "/api/mcp/oauth/callback?code=auth-code&state=issuer-unsupported-state"
        ),
        db,
    )

    assert response.status_code == 307
    assert db.query(MCPOAuthGrant).count() == 1


@pytest.mark.asyncio
async def test_callback_rejects_state_replay(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    db.add(
        MCPOAuthFlowState(
            state="used-state",
            mcp_server_id=server.id,
            user_id=user.id,
            resource_owner_key="resource-owner-a",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="records.read",
            code_verifier=mcp_api.encrypt_value("verifier-123"),
            redirect_after="/mcp",
            expires_at=mcp_api._utc_now() + timedelta(minutes=10),
            consumed_at=mcp_api._utc_now(),
        )
    )
    db.commit()

    with pytest.raises(mcp_api.HTTPException) as exc:
        await mcp_oauth_callback(
            _request("/api/mcp/oauth/callback?code=auth-code&state=used-state"),
            db,
        )

    assert exc.value.detail["code"] == "state_already_consumed"


@pytest.mark.asyncio
async def test_callback_rejects_expired_state(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    db.add(
        MCPOAuthFlowState(
            state="expired-state",
            mcp_server_id=server.id,
            user_id=user.id,
            resource_owner_key="resource-owner-a",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="records.read",
            code_verifier=mcp_api.encrypt_value("verifier-123"),
            redirect_after="/mcp",
            expires_at=mcp_api._utc_now() - timedelta(minutes=1),
        )
    )
    db.commit()

    with pytest.raises(mcp_api.HTTPException) as exc:
        await mcp_oauth_callback(
            _request("/api/mcp/oauth/callback?code=auth-code&state=expired-state"),
            db,
        )

    assert exc.value.detail["code"] == "expired_state"


@pytest.mark.asyncio
async def test_callback_rejects_state_after_user_loses_mcp_access(db_session):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    db.add(
        MCPOAuthFlowState(
            state="orphaned-access-state",
            mcp_server_id=server.id,
            user_id=user.id,
            resource_owner_key="resource-owner-a",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="records.read",
            code_verifier=mcp_api.encrypt_value("verifier-123"),
            redirect_after="/mcp",
            expires_at=mcp_api._utc_now() + timedelta(minutes=10),
        )
    )
    db.query(UserMCPServer).filter(
        UserMCPServer.user_id == user.id,
        UserMCPServer.mcpserver_id == server.id,
    ).delete()
    db.commit()

    with pytest.raises(mcp_api.HTTPException) as exc:
        await mcp_oauth_callback(
            _request(
                "/api/mcp/oauth/callback?code=auth-code&state=orphaned-access-state"
            ),
            db,
        )

    assert exc.value.detail["code"] == "invalid_state"
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_rejects_state_after_user_mcp_server_is_deactivated(
    db_session, monkeypatch
):
    db, user, _ = db_session
    server, _, _ = _add_callback_client_and_state(
        db,
        user,
        state="inactive-access-state",
    )
    _set_user_mcp_active(db, user, server, False)

    async def fail_exchange(**kwargs):
        pytest.fail("token exchange must not run for inactive MCP server access")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fail_exchange)

    with pytest.raises(mcp_api.HTTPException) as exc:
        await mcp_oauth_callback(
            _request(
                "/api/mcp/oauth/callback?code=auth-code&state=inactive-access-state"
            ),
            db,
        )

    assert exc.value.detail["code"] == "invalid_state"
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_callback_reports_token_exchange_failure(db_session, monkeypatch):
    db, user, _ = db_session
    server = _add_mcp_oauth_server(db, user)
    db.add(
        MCPOAuthClient(
            mcp_server_id=server.id,
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            client_id="client-123",
            token_endpoint_auth_method="none",
            redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        )
    )
    db.add(
        MCPOAuthFlowState(
            state="bad-token-state",
            mcp_server_id=server.id,
            user_id=user.id,
            resource_owner_key="resource-owner-a",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="records.read",
            code_verifier=mcp_api.encrypt_value("verifier-123"),
            redirect_after="/mcp",
            expires_at=mcp_api._utc_now() + timedelta(minutes=10),
        )
    )
    db.commit()

    real_async_client = httpx.AsyncClient

    def async_client_factory(*args, **kwargs):
        return real_async_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(400, json={"error": "invalid_grant"})
            )
        )

    monkeypatch.setattr(mcp_api.httpx, "AsyncClient", async_client_factory)

    with pytest.raises(mcp_api.HTTPException) as exc:
        await mcp_oauth_callback(
            _request("/api/mcp/oauth/callback?code=auth-code&state=bad-token-state"),
            db,
        )

    assert exc.value.detail["code"] == "token_exchange_failed"
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_status_and_delete_are_scoped_to_current_user(db_session):
    db, user, other_user = db_session
    server = _add_mcp_oauth_server(db, user)
    own_grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        resource_owner_key="resource-owner-a",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token=mcp_api.encrypt_value("own-access-token"),
    )
    other_grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=other_user.id,
        resource_owner_key="resource-owner-b",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token=mcp_api.encrypt_value("other-access-token"),
    )
    db.add_all([own_grant, other_grant])
    db.commit()
    db.refresh(own_grant)
    db.refresh(other_grant)

    status_response = await get_mcp_oauth_status(server.id, user, db)

    assert isinstance(status_response, MCPOAuthStatusResponse)
    assert [grant.id for grant in status_response.grants] == [own_grant.id]

    with pytest.raises(mcp_api.HTTPException) as exc:
        await delete_mcp_oauth_grant(server.id, other_grant.id, user, db)
    assert exc.value.status_code == 404

    await delete_mcp_oauth_grant(server.id, own_grant.id, user, db)
    db.refresh(own_grant)
    assert own_grant.status == "revoked"
    assert own_grant.revoked_at is not None

    status_response = await get_mcp_oauth_status(server.id, user, db)
    assert status_response.grants == []
