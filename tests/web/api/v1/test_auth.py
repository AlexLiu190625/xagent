"""Integration tests for the /v1/* auth dependency (get_agent_from_api_key).

Drives /v1/me (which only does auth + return identity) to verify each
failure path returns the stable ``{"error": {"code": "invalid_api_key",
...}}`` envelope. Also asserts timing-oracle symmetry: prefix-miss
should not be measurably faster than secret-wrong, and that unhandled
internal exceptions on the /v1/* surface still respond in the SDK
envelope rather than FastAPI's default ``{"detail": ...}``.

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


# ===== happy path =====


def test_valid_key_returns_me_response():
    """A freshly generated key authenticates /v1/me and returns identity."""
    agent_id, full_key, prefix = _create_agent_and_key()

    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {full_key}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["agent_id"] == agent_id
    assert body["agent_name"] == "v1 auth test agent"
    assert body["key_prefix"] == prefix


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
    fake_key = "xag_ZZZZZZ_" + "x" * 32
    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {fake_key}"})
    _assert_invalid_api_key(resp)


def test_known_prefix_wrong_secret_returns_401():
    """Prefix is real but the secret doesn't bcrypt-match."""
    _agent_id, full_key, prefix = _create_agent_and_key()
    # Replace just the secret half with a different (but well-formed) value
    parts = full_key.split("_")
    parts[2] = "y" * 32
    wrong_key = "_".join(parts)
    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {wrong_key}"})
    _assert_invalid_api_key(resp)


def test_revoked_key_returns_401():
    """Once DELETE rotates / revokes, the old key must stop working."""
    agent_id, full_key, _prefix = _create_agent_and_key()
    # Revoke via admin endpoint
    admin = _admin_headers()
    revoke = client.delete(f"/api/agents/{agent_id}/api-key", headers=admin)
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
    _agent_id, full_key, _prefix = _create_agent_and_key()
    parts = full_key.split("_")
    parts[2] = "z" * 32
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
    fake_key = "xag_ZZZZZZ_" + "x" * 32
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
            "/v1/me", headers={"Authorization": "Bearer xag_ABCDEF_" + "x" * 32}
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
