"""Integration tests for scoped personal API-key management."""

import pytest

from xagent.web.models.user import User
from xagent.web.services.personal_key_scope import (
    PersonalKeyAccessScope,
    set_personal_key_scope_hook,
)

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)


pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _clear_personal_key_scope_hook():
    set_personal_key_scope_hook(None)
    yield
    set_personal_key_scope_hook(None)


def _create_personal_key(headers: dict[str, str]) -> dict:
    response = client.post("/api/me/personal-keys", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User.id).filter(User.username == username).one()[0])
    finally:
        db.close()


def test_list_returns_only_self_keys_without_disclosing_one_time_secrets():
    headers = _admin_headers()
    created = _create_personal_key(headers)

    response = client.get("/api/personal-api-keys", headers=headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["can_manage_others"] is False
    item = next(item for item in body["items"] if item["id"] == created["id"])
    assert item["key_prefix"] == created["key_prefix"]
    assert item["owner"] == {
        "id": _user_id("admin"),
        "username": "admin",
        "email": "admin@example.com",
    }
    assert created["full_key"] not in response.text
    assert "full_key" not in item


def test_default_scope_returns_404_for_another_owners_key():
    admin_headers = _admin_headers()
    bob_headers = _register_second_user()
    bobs_key = _create_personal_key(bob_headers)

    response = client.delete(
        f"/api/personal-api-keys/{bobs_key['id']}", headers=admin_headers
    )

    assert response.status_code == 404


def test_authorized_scope_lists_and_revokes_other_owner_keys_idempotently():
    admin_headers = _admin_headers()
    bob_headers = _register_second_user()
    bobs_key = _create_personal_key(bob_headers)
    bob_id = _user_id("bob")

    set_personal_key_scope_hook(
        lambda _db, actor: PersonalKeyAccessScope(
            owner_user_ids=(int(actor.id), bob_id), can_manage_others=True
        )
    )

    listed = client.get("/api/personal-api-keys", headers=admin_headers)

    assert listed.status_code == 200, listed.text
    assert listed.json()["can_manage_others"] is True
    item = next(
        item for item in listed.json()["items"] if item["id"] == bobs_key["id"]
    )
    assert item["owner"]["id"] == bob_id
    assert item["owner"]["username"] == "bob"

    first_revoke = client.delete(
        f"/api/personal-api-keys/{bobs_key['id']}", headers=admin_headers
    )
    second_revoke = client.delete(
        f"/api/personal-api-keys/{bobs_key['id']}", headers=admin_headers
    )

    assert first_revoke.status_code == 200, first_revoke.text
    assert first_revoke.json()["revoked"] is True
    assert second_revoke.status_code == 200, second_revoke.text
    assert second_revoke.json() == {
        "revoked": False,
        "revoked_at": first_revoke.json()["revoked_at"],
    }


def test_legacy_personal_key_creation_contract_is_unchanged():
    headers = _admin_headers()

    response = client.post("/api/me/personal-keys", headers=headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"id", "full_key", "key_prefix", "created_at", "expires_at"}
    assert body["full_key"].startswith("xag_personal_")
