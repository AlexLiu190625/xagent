"""Integration tests for the agent API key admin endpoints.

Covers the three endpoints at ``/api/agents/{agent_id}/api-key``:

  - POST: generate or rotate the active key (happy + reset + 401 + 404)
  - GET:  read active key metadata (happy + no-key 404 + cross-user 404)
  - DELETE: idempotent revoke (active -> revoked / no-active -> no-op /
            double-call true idempotency)

Each test uses a fresh SQLite database via the ``_test_db`` fixture
(matching ``test_agents_kb_tool_validation.py``); admin user is set up
via ``setup-admin`` and an optional second user is registered via the
public ``/register`` endpoint to exercise cross-user 404 paths.
"""

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from xagent.web.api.agents import router as agents_router
from xagent.web.api.auth import auth_router
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.database import Base, get_db, get_engine

# ===== App + DB plumbing =====


def _override_get_db():
    """Yield the default DB session; required so dependency_overrides has a callable."""
    db = None
    try:
        db = next(get_db())
        yield db
    finally:
        if db is not None:
            db.close()


app_for_tests = FastAPI()
app_for_tests.include_router(auth_router)
app_for_tests.include_router(agents_router)
app_for_tests.dependency_overrides[get_db] = _override_get_db
client = TestClient(app_for_tests)


@pytest.fixture(autouse=True)
def _test_db():
    """Per-test SQLite DB so tables come up empty.

    Matches the pattern in ``test_agents_kb_tool_validation.py``: tmp
    sqlite file, ``init_db`` lays out the schema, drop all on teardown.
    """
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    db_url = f"sqlite:///{temp_db_path}"
    init_db(db_url=db_url)

    yield

    Base.metadata.drop_all(bind=get_engine())
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass


# ===== Auth helpers =====


def _setup_admin() -> None:
    """Idempotent admin bootstrap; mirrors ``test_agents_kb_tool_validation``."""
    status = client.get("/api/auth/setup-status")
    assert status.status_code == 200
    if status.json().get("needs_setup", True):
        resp = client.post(
            "/api/auth/setup-admin",
            json={"username": "admin", "password": "admin123"},
        )
        assert resp.status_code == 200


def _login(username: str = "admin", password: str = "admin123") -> dict[str, str]:
    """Log in and return the bearer header dict ready to splat into a request."""
    resp = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _headers() -> dict[str, str]:
    """Setup admin if needed and return the admin's auth header."""
    _setup_admin()
    return _login()


def _register_second_user(
    username: str = "bob", password: str = "bobpass1"
) -> dict[str, str]:
    """Register a second user via the public endpoint and return their auth header."""
    resp = client.post(
        "/api/auth/register",
        json={"username": username, "password": password},
    )
    assert resp.status_code == 200, resp.text
    return _login(username, password)


# ===== Domain helpers =====


def _create_agent(headers: dict[str, str], name: str = "Test Agent") -> int:
    """Create a minimal agent under the given user; return its id."""
    resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "test",
            "instructions": "You are a test agent.",
            "execution_mode": "balanced",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def _direct_db_session() -> Session:
    """Open a session against the same test DB FastAPI is using.

    Used in a couple of tests that need to inspect rows directly (e.g.
    confirm an active key was inserted) rather than only via the HTTP
    surface.
    """
    return next(get_db())


# ===== POST /{agent_id}/api-key =====


class TestPostGenerateApiKey:
    """POST /api/agents/{agent_id}/api-key — generate or rotate."""

    def test_happy_path_returns_full_key_and_creates_row(self):
        headers = _headers()
        agent_id = _create_agent(headers)

        resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 200, resp.text

        body = resp.json()
        # full_key format: xag_<6 alnum>_<32 alnum>
        full_key = body["full_key"]
        assert full_key.startswith("xag_")
        parts = full_key.split("_")
        assert len(parts) == 3
        assert parts[0] == "xag"
        assert len(parts[1]) == 6
        assert len(parts[2]) == 32
        assert body["key_prefix"] == parts[1]
        assert "created_at" in body

        # DB row exists, active, prefix matches
        db = _direct_db_session()
        try:
            rows = db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).all()
            assert len(rows) == 1
            assert rows[0].key_prefix == body["key_prefix"]
            assert rows[0].revoked_at is None
            # Hash is bcrypt, NOT the plaintext full_key
            assert rows[0].key_hash != full_key
            assert rows[0].key_hash.startswith("$2b$12$")
        finally:
            db.close()

    def test_second_post_rotates_and_revokes_old(self):
        """Second POST revokes the old active row and creates a new one."""
        headers = _headers()
        agent_id = _create_agent(headers)

        first = client.post(f"/api/agents/{agent_id}/api-key", headers=headers).json()
        second = client.post(f"/api/agents/{agent_id}/api-key", headers=headers).json()
        assert first["full_key"] != second["full_key"]
        assert first["key_prefix"] != second["key_prefix"]

        db = _direct_db_session()
        try:
            rows = (
                db.query(AgentApiKey)
                .filter(AgentApiKey.agent_id == agent_id)
                .order_by(AgentApiKey.id)
                .all()
            )
            assert len(rows) == 2
            # First row is revoked, second is active.
            assert rows[0].revoked_at is not None
            assert rows[0].key_prefix == first["key_prefix"]
            assert rows[1].revoked_at is None
            assert rows[1].key_prefix == second["key_prefix"]
        finally:
            db.close()

    def test_unauthorized_returns_401(self):
        """No Authorization header -> 401."""
        # We still need an agent to target, but the auth gate fires before
        # ownership; create the agent under admin, then call without header.
        headers = _headers()
        agent_id = _create_agent(headers)
        resp = client.post(f"/api/agents/{agent_id}/api-key")
        # python-jose / HTTPBearer raises 401 with "Not authenticated"
        # when the header is missing; 403 is the FastAPI default for
        # HTTPBearer missing credentials. Accept either.
        assert resp.status_code in (401, 403)

    def test_other_users_agent_returns_404(self):
        """Calling POST on someone else's agent returns 404 (not 403)."""
        admin_headers = _headers()
        admin_agent_id = _create_agent(admin_headers, name="admin agent")

        bob_headers = _register_second_user()
        # Bob tries to generate a key for the admin's agent
        resp = client.post(f"/api/agents/{admin_agent_id}/api-key", headers=bob_headers)
        assert resp.status_code == 404
        # The detail must NOT indicate "permission denied" -- it must
        # look identical to "this agent does not exist".
        assert "Agent not found" in resp.json()["detail"]

    def test_nonexistent_agent_returns_404(self):
        headers = _headers()
        resp = client.post("/api/agents/9999999/api-key", headers=headers)
        assert resp.status_code == 404

    def test_integrity_error_returns_409_rotation_conflict(self):
        """Concurrent rotate race -- partial unique constraint fires at commit.

        We can't easily orchestrate two real concurrent connections in
        SQLite tests, so we monkey-patch ``Session.commit`` to raise
        ``IntegrityError`` once. That exercises the exact branch the
        production race would hit (commit fails because partial unique
        index rejects the second active row), and asserts the endpoint
        translates it into HTTP 409 with the stable ``rotation_conflict``
        code rather than leaking a 500 + raw SQL message.

        ``_create_agent`` and ``_headers`` run BEFORE the patch context,
        so the setup commits succeed; only the commit inside POST
        /api-key sees the simulated race.
        """
        headers = _headers()
        agent_id = _create_agent(headers)

        # Patch the SQLAlchemy ``Session.commit`` only inside the POST
        # call. The handler catches IntegrityError -> 409, rolls back,
        # and never calls commit again on this session, so the patch is
        # exercised exactly once.
        fake_error = IntegrityError(
            "UNIQUE constraint failed: agent_api_keys.agent_id",
            params=None,
            orig=Exception("simulated race"),
        )
        with patch.object(Session, "commit", side_effect=fake_error):
            resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)

        assert resp.status_code == 409
        assert resp.json()["detail"] == "rotation_conflict"
        # Crucially: the raw SQL error string must NOT appear in the
        # client-visible response.
        assert "UNIQUE constraint failed" not in resp.text
        assert "agent_api_keys" not in resp.text

    def test_internal_error_response_does_not_leak_str_e(self):
        """Non-IntegrityError 500 path must not echo str(e) to the client."""
        headers = _headers()
        agent_id = _create_agent(headers)

        # Patch commit to raise an unrelated RuntimeError -- this should
        # hit the generic ``except Exception`` branch and surface as a
        # sanitized 500 ("Internal server error"), not the raw message.
        secret_message = "secret-internal-detail-do-not-leak"
        with patch.object(Session, "commit", side_effect=RuntimeError(secret_message)):
            resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)

        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error"
        assert secret_message not in resp.text


# ===== GET /{agent_id}/api-key =====


class TestGetActiveApiKey:
    """GET /api/agents/{agent_id}/api-key — read active key metadata."""

    def test_happy_path_returns_masked(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        post_resp = client.post(
            f"/api/agents/{agent_id}/api-key", headers=headers
        ).json()
        prefix = post_resp["key_prefix"]

        resp = client.get(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["key_prefix"] == prefix
        assert body["masked_key"] == f"xag_{prefix}_••••••••"
        assert "created_at" in body
        # full_key MUST NOT be in the GET response
        assert "full_key" not in body

    def test_no_active_key_returns_404(self):
        """Agent owned but never had a key -> 404 no_active_key."""
        headers = _headers()
        agent_id = _create_agent(headers)
        resp = client.get(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 404
        assert resp.json()["detail"] == "no_active_key"

    def test_revoked_key_returns_404(self):
        """After DELETE, GET returns 404 no_active_key."""
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
        client.delete(f"/api/agents/{agent_id}/api-key", headers=headers)

        resp = client.get(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 404
        assert resp.json()["detail"] == "no_active_key"

    def test_other_users_agent_returns_404(self):
        admin_headers = _headers()
        admin_agent_id = _create_agent(admin_headers)
        client.post(f"/api/agents/{admin_agent_id}/api-key", headers=admin_headers)

        bob_headers = _register_second_user()
        resp = client.get(f"/api/agents/{admin_agent_id}/api-key", headers=bob_headers)
        assert resp.status_code == 404
        # Same "agent not found" detail -- never reveals the existence
        # of admin's key, only that the agent itself isn't bob's.
        assert resp.json()["detail"] == "Agent not found"


# ===== DELETE /{agent_id}/api-key =====


class TestDeleteApiKey:
    """DELETE /api/agents/{agent_id}/api-key — idempotent revoke."""

    def test_revoke_active_returns_true(self):
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post(f"/api/agents/{agent_id}/api-key", headers=headers)

        resp = client.delete(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["revoked"] is True
        assert body["revoked_at"] is not None

        # DB confirms the row is now revoked
        db = _direct_db_session()
        try:
            row = db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).first()
            assert row is not None
            assert row.revoked_at is not None
        finally:
            db.close()

    def test_revoke_with_no_active_returns_false_idempotent(self):
        """DELETE on an agent with no active key is a 200 no-op."""
        headers = _headers()
        agent_id = _create_agent(headers)
        resp = client.delete(f"/api/agents/{agent_id}/api-key", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["revoked"] is False
        assert body["revoked_at"] is None

    def test_double_revoke_is_idempotent(self):
        """Two consecutive DELETEs: first revokes, second is a no-op."""
        headers = _headers()
        agent_id = _create_agent(headers)
        client.post(f"/api/agents/{agent_id}/api-key", headers=headers)

        first = client.delete(f"/api/agents/{agent_id}/api-key", headers=headers).json()
        assert first["revoked"] is True

        second = client.delete(
            f"/api/agents/{agent_id}/api-key", headers=headers
        ).json()
        assert second["revoked"] is False
        assert second["revoked_at"] is None

    def test_other_users_agent_returns_404(self):
        admin_headers = _headers()
        admin_agent_id = _create_agent(admin_headers)
        client.post(f"/api/agents/{admin_agent_id}/api-key", headers=admin_headers)

        bob_headers = _register_second_user()
        resp = client.delete(
            f"/api/agents/{admin_agent_id}/api-key", headers=bob_headers
        )
        assert resp.status_code == 404
