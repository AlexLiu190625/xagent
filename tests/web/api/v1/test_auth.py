"""Integration tests for the /v1/* personal management auth dependency.

Drives /v1/me to verify each personal-key failure path returns the
stable ``{"error": {"code": "invalid_api_key", ...}}`` envelope.

Test plumbing (client, _test_db fixture, auth helpers) is shared via
``tests/web/api/conftest.py``.
"""

import time
from unittest.mock import patch

import bcrypt
import pytest

from xagent.core.utils.api_key import BCRYPT_COST

from ..conftest import _admin_headers, client

# Opt this file into the shared conftest ``_test_db`` fixture. See the
# note in test_agent_api_keys.py for why we use ``usefixtures`` with a
# string name rather than importing the fixture directly.
pytestmark = pytest.mark.usefixtures("_test_db")


def _create_agent_and_key() -> tuple[int, str, str]:
    """Helper: create an agent + generate its first API key.

    Returns: (agent_id, full_key, key_prefix)
    """
    headers = _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "v1 auth test agent",
            "description": "for /v1/* auth tests",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]

    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    body = key_resp.json()
    return agent_id, body["full_key"], body["key_prefix"]


def _create_personal_key() -> tuple[str, str]:
    """Helper: create a personal management key for the admin user."""
    headers = _admin_headers()
    key_resp = client.post("/api/me/personal-keys", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    body = key_resp.json()
    return body["full_key"], body["key_prefix"]


# ===== happy path =====


def test_valid_personal_key_returns_me_response():
    """A freshly generated personal key authenticates /v1/me."""
    full_key, prefix = _create_personal_key()

    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {full_key}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["principal_type"] == "user"
    assert body["user_id"] > 0
    assert body["email"] == "admin"
    assert body["name"] == "admin"
    assert body["key_prefix"] == prefix


def test_agent_runtime_key_cannot_authenticate_me():
    """Runtime keys are not accepted by management identity endpoints."""
    _agent_id, full_key, _prefix = _create_agent_and_key()
    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {full_key}"})
    _assert_invalid_api_key(resp)


# ===== failure paths -- all must return the same envelope =====


def _assert_invalid_api_key(resp) -> None:
    """Every auth failure should respond with the same shape."""
    assert resp.status_code == 401, resp.text
    body = resp.json()
    assert body == {
        "error": {
            "code": "invalid_api_key",
            "message": body["error"]["message"],  # message is free text
        }
    }
    # Ensure no internal SQL message or raw exception slipped into message
    msg = body["error"]["message"]
    assert "bcrypt" not in msg.lower()
    assert "sqlalchemy" not in msg.lower()


def test_missing_authorization_header_returns_401():
    resp = client.get("/v1/me")
    _assert_invalid_api_key(resp)


def test_malformed_authorization_header_returns_401():
    resp = client.get("/v1/me", headers={"Authorization": "Bearer not_a_key"})
    _assert_invalid_api_key(resp)


def test_wrong_brand_prefix_returns_401():
    resp = client.get(
        "/v1/me", headers={"Authorization": "Bearer sk_ABCDEF_" + "x" * 32}
    )
    _assert_invalid_api_key(resp)


def test_unknown_prefix_returns_401():
    """A well-formed key with a prefix that's never been issued."""
    fake_key = "xag_personal_ZZZZZZ_" + "x" * 32
    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {fake_key}"})
    _assert_invalid_api_key(resp)


def test_known_prefix_wrong_secret_returns_401():
    """Prefix is real but the secret doesn't bcrypt-match."""
    full_key, _prefix = _create_personal_key()
    # Replace just the secret half with a different (but well-formed) value
    parts = full_key.split("_")
    parts[3] = "y" * 32
    wrong_key = "_".join(parts)
    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {wrong_key}"})
    _assert_invalid_api_key(resp)


def test_revoked_key_returns_401():
    """Once DELETE rotates / revokes, the old key must stop working."""
    full_key, prefix = _create_personal_key()
    admin = _admin_headers()
    keys = client.get("/api/me/personal-keys", headers=admin)
    assert keys.status_code == 200
    key_id = next(row["id"] for row in keys.json() if row["key_prefix"] == prefix)
    revoke = client.delete(f"/api/me/personal-keys/{key_id}", headers=admin)
    assert revoke.status_code == 200
    assert revoke.json()["revoked"] is True

    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {full_key}"})
    _assert_invalid_api_key(resp)


# ===== timing oracle defense =====


def test_unknown_prefix_takes_similar_time_to_wrong_secret():
    """Prefix-miss must burn bcrypt time like a real verify would.

    Both paths should be ~100ms on commodity hardware. We use generous
    bounds (each within 2x of the other) so CI runners' jitter doesn't
    flake the test. The defense is to keep the order of magnitude the
    same, not to clock to the millisecond.
    """
    full_key, _prefix = _create_personal_key()
    parts = full_key.split("_")
    parts[3] = "z" * 32
    wrong_secret_key = "_".join(parts)

    # Warm the bcrypt module a bit so first-call overhead doesn't skew
    bcrypt.checkpw(b"warm", bcrypt.hashpw(b"warm", bcrypt.gensalt(rounds=BCRYPT_COST)))

    # Wrong secret (prefix hits index, then bcrypt runs)
    t0 = time.perf_counter()
    resp1 = client.get(
        "/v1/me", headers={"Authorization": f"Bearer {wrong_secret_key}"}
    )
    real_t = time.perf_counter() - t0
    assert resp1.status_code == 401

    # Unknown prefix (index miss, then verify_dummy runs)
    fake_key = "xag_personal_ZZZZZZ_" + "x" * 32
    t0 = time.perf_counter()
    resp2 = client.get("/v1/me", headers={"Authorization": f"Bearer {fake_key}"})
    dummy_t = time.perf_counter() - t0
    assert resp2.status_code == 401

    # They should be within an order of magnitude. Asserting roughly:
    # the slower one is at most 3x the faster one. Wide bounds keep CI
    # flakes down; the real safeguard is the verify_dummy call itself.
    ratio = max(real_t, dummy_t) / max(min(real_t, dummy_t), 1e-6)
    assert ratio < 3.0, (
        f"timing asymmetry too large: real={real_t * 1000:.1f}ms, "
        f"dummy={dummy_t * 1000:.1f}ms, ratio={ratio:.2f}"
    )


# ===== /v1/* internal_error envelope (catch-all) =====


def test_internal_exception_returns_v1_envelope_not_fastapi_detail():
    """Non-V1ApiError exceptions on /v1/* must still match SDK contract.

    If an upstream layer (db.query, bcrypt, dependency) raises an
    unexpected exception, the response MUST be the stable
    ``{"error": {"code": "internal_error", "message": ...}}`` shape --
    not FastAPI's default ``{"detail": "Internal Server Error"}``,
    which would break SDK clients that key off ``body.error.code``.

    We force the failure by patching ``parse_api_key`` (called inside
    the auth dep) to raise a RuntimeError. That gets the request past
    the FastAPI routing layer but blows up inside our handler chain
    BEFORE V1ApiError is raised, exercising the generic Exception
    branch of the global handler.
    """
    secret_internal_msg = "secret-internal-detail-do-not-leak"
    with patch(
        "xagent.web.api.v1.deps.parse_api_key",
        side_effect=RuntimeError(secret_internal_msg),
    ):
        resp = client.get(
            "/v1/me",
            headers={"Authorization": "Bearer xag_personal_ABCDEF_" + "x" * 32},
        )

    # Must be 500 in the V1 envelope, not 500 with FastAPI's detail key.
    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body == {
        "error": {
            "code": "internal_error",
            "message": "Internal server error.",
        }
    }
    # Sanity: no internal exception message leaks into the response
    assert secret_internal_msg not in resp.text
    # Sanity: NOT the default FastAPI {"detail": ...} shape
    assert "detail" not in body
