"""Integration tests for /v1 management endpoints."""

from pathlib import Path

import pytest

from xagent.templates.manager import TemplateManager
from xagent.web.models.model import Model as DBModel
from xagent.web.models.user import User, UserModel

from ..conftest import _admin_headers, _direct_db_session, app_for_tests, client

pytestmark = pytest.mark.usefixtures("_test_db")


def _personal_key() -> str:
    headers = _admin_headers()
    resp = client.post("/api/me/personal-keys", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["full_key"]


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def _write_template(root: Path) -> None:
    (root / "qa.yaml").write_text(
        """
id: qa
name: Q&A Assistant
category: General
descriptions:
  en: Answers questions from provided context.
features:
  en:
    - Ask questions
connections: []
agent_config:
  instructions: Answer clearly.
  skills:
    - retrieval
  tool_categories:
    - web_search
  knowledge_bases:
    - template-kb
  suggested_prompts:
    - Ask anything
  execution_mode: balanced
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture
def template_manager(tmp_path):
    _write_template(tmp_path)
    manager = TemplateManager(templates_root=tmp_path)
    app_for_tests.state.template_manager = manager
    return manager


def test_personal_key_management_round_trip():
    headers = _admin_headers()
    create = client.post("/api/me/personal-keys", headers=headers)
    assert create.status_code == 200, create.text
    body = create.json()
    assert body["full_key"].startswith("xag_personal_")

    list_resp = client.get("/api/me/personal-keys", headers=headers)
    assert list_resp.status_code == 200
    assert any(row["key_prefix"] == body["key_prefix"] for row in list_resp.json())

    revoke = client.delete(f"/api/me/personal-keys/{body['id']}", headers=headers)
    assert revoke.status_code == 200
    assert revoke.json()["revoked"] is True


def test_v1_me_uses_personal_key():
    key = _personal_key()
    resp = client.get("/v1/me", headers=_bearer(key))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["principal_type"] == "user"
    assert body["email"] == "admin"


def test_v1_create_agent_defaults_to_runtime_key():
    key = _personal_key()
    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "SDK-created agent",
            "description": "created from management SDK",
            "instructions": "Be useful.",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["agent"]["name"] == "SDK-created agent"
    assert body["api_key"]["full_key"].startswith("xag_")


def test_v1_create_agent_can_skip_runtime_key():
    key = _personal_key()
    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "No key agent",
            "instructions": "Be useful.",
            "generate_runtime_key": False,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["api_key"] is None


def test_v1_create_agent_rejects_string_model_ids():
    key = _personal_key()
    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "Bad model shape",
            "instructions": "Be useful.",
            "models": {"general": "deepseek-v4-flash"},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


def test_v1_create_agent_rejects_unknown_model_ids():
    key = _personal_key()
    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "Unknown model",
            "instructions": "Be useful.",
            "models": {"general": 999999},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


def test_v1_create_agent_rejects_inaccessible_model_ids():
    key = _personal_key()
    db = _direct_db_session()
    try:
        other = User(username="model-owner", password_hash="hash")
        db.add(other)
        db.flush()
        model = DBModel(
            model_id="other-private-model",
            category="llm",
            model_provider="openai",
            model_name="gpt-4",
            api_key="test-api-key",
            base_url="https://api.openai.com/v1",
            is_active=True,
        )
        db.add(model)
        db.flush()
        db.add(
            UserModel(
                user_id=other.id,
                model_id=model.id,
                is_owner=True,
                is_shared=False,
            )
        )
        db.commit()
        model_id = model.id
    finally:
        db.close()

    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "Inaccessible model",
            "instructions": "Be useful.",
            "models": {"general": model_id},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


def test_runtime_key_cannot_call_management_endpoint():
    personal = _personal_key()
    create = client.post(
        "/v1/agents",
        headers=_bearer(personal),
        json={"name": "Runtime only", "instructions": "hi"},
    )
    assert create.status_code == 200, create.text
    runtime_key = create.json()["api_key"]["full_key"]

    resp = client.get("/v1/agents", headers=_bearer(runtime_key))
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_personal_key_cannot_call_runtime_endpoint():
    personal = _personal_key()
    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(personal),
        json={
            "agent_id": 1,
            "message": {"role": "user", "content": "hello"},
        },
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_from_template_creates_agent(template_manager):
    key = _personal_key()
    resp = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={"template_id": "qa", "name": "Template agent"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["agent"]["name"] == "Template agent"
    assert body["agent"]["instructions"] == "Answer clearly."
    assert body["agent"]["skills"] == ["retrieval"]
    assert body["agent"]["tool_categories"] == ["web_search"]
    assert body["agent"]["knowledge_bases"] == ["template-kb"]
    assert body["agent"]["suggested_prompts"] == ["Ask anything"]
    assert body["api_key"]["full_key"].startswith("xag_")


def test_from_template_allows_empty_list_overrides(template_manager):
    key = _personal_key()
    resp = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={
            "template_id": "qa",
            "name": "Template agent without defaults",
            "knowledge_bases": [],
            "skills": [],
            "tool_categories": [],
            "suggested_prompts": [],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["agent"]["knowledge_bases"] == []
    assert body["agent"]["skills"] == []
    assert body["agent"]["tool_categories"] == []
    assert body["agent"]["suggested_prompts"] == []


def test_unknown_template_returns_stable_error(template_manager):
    key = _personal_key()
    resp = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={"template_id": "missing"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "template_not_found"
