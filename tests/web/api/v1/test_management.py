"""Integration tests for /v1 management endpoints."""

import asyncio
import inspect
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.templates.manager import TemplateManager
from xagent.web.api.v1 import agents as v1_agents
from xagent.web.models import database as database_module
from xagent.web.models.agent import Agent
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.model import Model as DBModel
from xagent.web.models.user import User, UserModel
from xagent.web.services import agent_management

from ..conftest import _admin_headers, _direct_db_session, app_for_tests, client

pytestmark = pytest.mark.usefixtures("_test_db")


def _personal_key() -> str:
    headers = _admin_headers()
    resp = client.post("/api/me/personal-keys", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["full_key"]


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def _post_commit_pool_exhausting_session_factory():
    """Return a real one-slot QueuePool occupied immediately after commit."""
    engine = create_engine(
        database_module.get_engine().url,
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
    )
    held_connections = []

    class PostCommitPoolExhaustingSession(Session):
        def commit(self) -> None:
            super().commit()
            if not held_connections:
                # ``super().commit()`` returned the transaction's connection.
                # Occupy that sole slot so any post-commit ORM refresh or
                # expired-attribute load exercises a real QueuePool timeout.
                held_connections.append(engine.connect())

    factory = sessionmaker(
        bind=engine,
        class_=PostCommitPoolExhaustingSession,
        autocommit=False,
        autoflush=False,
    )
    return factory, held_connections, engine


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
  suggested_prompts:
    - Ask anything
  execution_mode: balanced
""".strip(),
        encoding="utf-8",
    )
    # Pre-sets a knowledge base without the knowledge tool category, so
    # from-template must reject it -- exercises that template-sourced KB
    # fields go through the same validation as user input.
    (root / "kb-no-tool.yaml").write_text(
        """
id: kb-no-tool
name: KB without tool
category: General
descriptions:
  en: Pre-sets a knowledge base but not the knowledge tool category.
connections: []
agent_config:
  instructions: Answer.
  tool_categories:
    - web_search
  knowledge_bases:
    - template-kb
  execution_mode: balanced
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "handler",
    [
        v1_agents.list_agents,
        v1_agents.create_agent,
        v1_agents.create_agent_from_template,
        v1_agents.rotate_agent_runtime_key,
    ],
)
def test_v1_agent_routes_do_not_inject_request_db_session(handler):
    """A /v1 agent request must not own a Session across async work."""
    assert "db" not in inspect.signature(handler).parameters


@pytest.mark.asyncio
async def test_agent_list_worker_owns_session_and_does_not_block_event_loop(
    monkeypatch,
):
    _admin_headers()
    db = _direct_db_session()
    try:
        user_id = db.query(User.id).filter(User.username == "admin").scalar()
    finally:
        db.close()
    assert user_id is not None

    operation = getattr(agent_management, "list_agents_for_user_worker_owned", None)
    assert callable(operation), "missing worker-owned agent-list entry point"

    loop_thread = threading.get_ident()
    service_threads: list[int] = []
    original = agent_management.AgentManagementService.list_agents_for_user

    def slow_list(self, requested_user_id):
        service_threads.append(threading.get_ident())
        time.sleep(0.05)
        return original(self, requested_user_id)

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "list_agents_for_user",
        slow_list,
    )

    ticks = 0
    stop = False

    async def ticker() -> None:
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.005)

    ticker_task = asyncio.create_task(ticker())
    try:
        result = await operation(user_id=user_id)
    finally:
        stop = True
        await ticker_task

    assert isinstance(result, list)
    assert service_threads and service_threads[0] != loop_thread
    assert ticks >= 3


@pytest.mark.asyncio
async def test_agent_create_opens_worker_session_after_kb_await(monkeypatch):
    _admin_headers()
    db = _direct_db_session()
    try:
        user_id = db.query(User.id).filter(User.username == "admin").scalar()
    finally:
        db.close()
    assert user_id is not None

    operation = getattr(agent_management, "create_agent_worker_owned", None)
    assert callable(operation), "missing worker-owned agent-create entry point"

    real_factory = database_module.get_session_local()
    session_threads: list[int] = []

    class TrackingSessionFactory:
        def __call__(self):
            session_threads.append(threading.get_ident())
            return real_factory()

    monkeypatch.setattr(
        agent_management,
        "get_session_local",
        lambda: TrackingSessionFactory(),
    )

    async def visible_kbs(names, *, user_id, is_admin):
        assert names == ["visible-kb"]
        assert session_threads == []
        await asyncio.sleep(0)
        assert session_threads == []
        return []

    monkeypatch.setattr(agent_management, "find_missing_knowledge_bases", visible_kbs)

    loop_thread = threading.get_ident()
    result = await operation(
        user_id=user_id,
        is_admin=True,
        name="worker-owned create",
        description=None,
        instructions="Be useful.",
        knowledge_bases=["visible-kb"],
        tool_categories=["knowledge"],
        generate_runtime_key=False,
    )

    assert isinstance(result.agent, dict)
    assert result.agent["name"] == "worker-owned create"
    assert session_threads and all(tid != loop_thread for tid in session_threads)

    db = _direct_db_session()
    try:
        assert db.query(Agent).filter(Agent.name == "worker-owned create").count() == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_template_resolution_finishes_before_worker_session_opens(monkeypatch):
    _admin_headers()
    db = _direct_db_session()
    try:
        user_id = db.query(User.id).filter(User.username == "admin").scalar()
    finally:
        db.close()
    assert user_id is not None

    operation = getattr(
        agent_management, "create_agent_from_template_worker_owned", None
    )
    assert callable(operation), "missing worker-owned template-create entry point"

    real_factory = database_module.get_session_local()
    session_threads: list[int] = []

    class TrackingSessionFactory:
        def __call__(self):
            session_threads.append(threading.get_ident())
            return real_factory()

    monkeypatch.setattr(
        agent_management,
        "get_session_local",
        lambda: TrackingSessionFactory(),
    )

    class FakeTemplateManager:
        async def get_template(self, template_id):
            assert template_id == "worker-template"
            assert session_threads == []
            await asyncio.sleep(0)
            assert session_threads == []
            return {
                "name": "Worker template",
                "descriptions": {"en": "Template description"},
                "agent_config": {"instructions": "Template instructions"},
            }

    result = await operation(
        template_manager=FakeTemplateManager(),
        user_id=user_id,
        is_admin=True,
        template_id="worker-template",
        generate_runtime_key=False,
    )

    assert isinstance(result.agent, dict)
    assert result.agent["name"] == "Worker template"
    assert session_threads


@pytest.mark.asyncio
async def test_runtime_key_rotation_runs_in_worker_owned_session(monkeypatch):
    _admin_headers()
    db = _direct_db_session()
    try:
        user_id = db.query(User.id).filter(User.username == "admin").scalar()
        assert user_id is not None
        agent = Agent(
            user_id=user_id,
            name="worker rotation",
            instructions="Be useful.",
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
    finally:
        db.close()

    operation = getattr(agent_management, "rotate_agent_runtime_key_worker_owned", None)
    assert callable(operation), "missing worker-owned key-rotation entry point"

    loop_thread = threading.get_ident()
    service_threads: list[int] = []
    original = agent_management.AgentManagementService.generate_agent_runtime_key

    def tracked_rotation(self, *, user_id, agent_id):
        service_threads.append(threading.get_ident())
        return original(self, user_id=user_id, agent_id=agent_id)

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "generate_agent_runtime_key",
        tracked_rotation,
    )

    result = await operation(user_id=user_id, agent_id=agent_id)

    assert result is not None
    assert result.full_key.startswith("xag_")
    assert service_threads and service_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_agent_create_does_not_checkout_again_after_secret_commit(monkeypatch):
    """A committed agent/key must not be stranded by post-commit pool pressure."""
    _admin_headers()
    db = _direct_db_session()
    try:
        user_id = db.query(User.id).filter(User.username == "admin").scalar()
    finally:
        db.close()
    assert user_id is not None

    factory, held_connections, engine = _post_commit_pool_exhausting_session_factory()
    monkeypatch.setattr(agent_management, "get_session_local", lambda: factory)
    name = "create without post-commit checkout"
    try:
        result = await agent_management.create_agent_worker_owned(
            user_id=user_id,
            is_admin=True,
            name=name,
            description=None,
            instructions="Be useful.",
        )
    finally:
        for connection in held_connections:
            connection.close()
        engine.dispose()

    assert result.api_key is not None
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.name == name).one()
        active_prefixes = {
            row.key_prefix
            for row in db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent.id,
                AgentApiKey.revoked_at.is_(None),
            )
            .all()
        }
        assert active_prefixes == {result.api_key.key_prefix}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_runtime_key_rotation_does_not_checkout_again_after_commit(monkeypatch):
    """A rotation succeeds even when no pool slot exists after its commit."""
    _admin_headers()
    db = _direct_db_session()
    try:
        user_id = db.query(User.id).filter(User.username == "admin").scalar()
        assert user_id is not None
        agent = Agent(
            user_id=user_id,
            name="rotation without post-commit checkout",
            instructions="Be useful.",
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
    finally:
        db.close()

    factory, held_connections, engine = _post_commit_pool_exhausting_session_factory()
    monkeypatch.setattr(agent_management, "get_session_local", lambda: factory)
    try:
        result = await agent_management.rotate_agent_runtime_key_worker_owned(
            user_id=user_id,
            agent_id=agent_id,
        )
    finally:
        for connection in held_connections:
            connection.close()
        engine.dispose()

    assert result is not None
    db = _direct_db_session()
    try:
        assert db.query(Agent).filter(Agent.id == agent_id).count() == 1
        active_prefixes = {
            row.key_prefix
            for row in db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.revoked_at.is_(None),
            )
            .all()
        }
        assert active_prefixes == {result.key_prefix}
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("from_template", [False, True])
async def test_cancelled_agent_create_revokes_only_committed_undelivered_runtime_key(
    monkeypatch,
    from_template: bool,
):
    """Cancellation keeps the durable agent but revokes its undelivered key."""
    _admin_headers()
    db = _direct_db_session()
    try:
        user_id = db.query(User.id).filter(User.username == "admin").scalar()
    finally:
        db.close()
    assert user_id is not None

    entered_after_commit = threading.Event()
    release_worker = threading.Event()
    initial_key_prefix: list[str] = []
    original_create = (
        agent_management.AgentManagementService.create_agent_with_optional_key
    )

    def create_then_block(self, **kwargs):
        result = original_create(self, **kwargs)
        assert result[1] is not None
        initial_key_prefix.append(result[1].key_prefix)
        entered_after_commit.set()
        assert release_worker.wait(timeout=5)
        return result

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "create_agent_with_optional_key",
        create_then_block,
    )

    name = f"cancelled-{'template' if from_template else 'plain'}-create"
    if from_template:

        class TemplateManager:
            async def get_template(self, template_id):
                assert template_id == "cancelled-template"
                return {
                    "name": name,
                    "descriptions": {"en": "test"},
                    "agent_config": {"instructions": "Be useful."},
                }

        caller = asyncio.create_task(
            agent_management.create_agent_from_template_worker_owned(
                template_manager=TemplateManager(),
                user_id=user_id,
                is_admin=True,
                template_id="cancelled-template",
            )
        )
    else:
        caller = asyncio.create_task(
            agent_management.create_agent_worker_owned(
                user_id=user_id,
                is_admin=True,
                name=name,
                description=None,
                instructions="Be useful.",
            )
        )

    assert await asyncio.to_thread(entered_after_commit.wait, 5)
    caller.cancel()
    release_worker.set()
    with pytest.raises(asyncio.CancelledError):
        await caller

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.name == name).one()
        assert initial_key_prefix
        assert (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent.id,
                AgentApiKey.key_prefix == initial_key_prefix[0],
                AgentApiKey.revoked_at.is_(None),
            )
            .count()
            == 0
        )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_cancelled_agent_create_preserves_concurrent_rotation(monkeypatch):
    """Create compensation cannot delete a concurrently updated durable agent."""
    _admin_headers()
    db = _direct_db_session()
    try:
        user_id = db.query(User.id).filter(User.username == "admin").scalar()
    finally:
        db.close()
    assert user_id is not None

    entered_after_commit = threading.Event()
    release_worker = threading.Event()
    initial_key_prefix: list[str] = []
    original_create = (
        agent_management.AgentManagementService.create_agent_with_optional_key
    )

    def create_then_block(self, **kwargs):
        result = original_create(self, **kwargs)
        assert result[1] is not None
        initial_key_prefix.append(result[1].key_prefix)
        entered_after_commit.set()
        assert release_worker.wait(timeout=5)
        return result

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "create_agent_with_optional_key",
        create_then_block,
    )

    name = "cancelled create with concurrent rotation"
    caller = asyncio.create_task(
        agent_management.create_agent_worker_owned(
            user_id=user_id,
            is_admin=True,
            name=name,
            description=None,
            instructions="Be useful.",
        )
    )
    assert await asyncio.to_thread(entered_after_commit.wait, 5)

    db = _direct_db_session()
    try:
        agent_id = int(db.query(Agent.id).filter(Agent.name == name).scalar())
    finally:
        db.close()
    concurrent_key = await agent_management.rotate_agent_runtime_key_worker_owned(
        user_id=user_id,
        agent_id=agent_id,
    )
    assert concurrent_key is not None

    caller.cancel()
    release_worker.set()
    with pytest.raises(asyncio.CancelledError):
        await caller

    db = _direct_db_session()
    try:
        assert db.query(Agent).filter(Agent.id == agent_id).count() == 1
        initial_key = (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.key_prefix == initial_key_prefix[0],
            )
            .one()
        )
        assert initial_key.revoked_at is not None
        concurrent_row = (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.key_prefix == concurrent_key.key_prefix,
            )
            .one()
        )
        assert concurrent_row.revoked_at is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_cancelled_runtime_key_rotation_revokes_the_undelivered_key(monkeypatch):
    """A cancelled rotation must not leave its newly minted key active."""
    _admin_headers()
    db = _direct_db_session()
    try:
        user_id = db.query(User.id).filter(User.username == "admin").scalar()
        assert user_id is not None
        agent = Agent(
            user_id=user_id,
            name="cancelled runtime rotation",
            instructions="Be useful.",
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
    finally:
        db.close()

    first_key = await agent_management.rotate_agent_runtime_key_worker_owned(
        user_id=user_id, agent_id=agent_id
    )
    assert first_key is not None

    entered_after_commit = threading.Event()
    release_worker = threading.Event()
    created_prefix: list[str] = []
    original_rotate = agent_management.AgentManagementService.generate_agent_runtime_key

    def rotate_then_block(self, *, user_id, agent_id):
        result = original_rotate(self, user_id=user_id, agent_id=agent_id)
        assert result is not None
        created_prefix.append(result.key_prefix)
        entered_after_commit.set()
        assert release_worker.wait(timeout=5)
        return result

    monkeypatch.setattr(
        agent_management.AgentManagementService,
        "generate_agent_runtime_key",
        rotate_then_block,
    )

    caller = asyncio.create_task(
        agent_management.rotate_agent_runtime_key_worker_owned(
            user_id=user_id,
            agent_id=agent_id,
        )
    )
    assert await asyncio.to_thread(entered_after_commit.wait, 5)
    caller.cancel()
    release_worker.set()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert created_prefix
    db = _direct_db_session()
    try:
        assert (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.key_prefix == created_prefix[0],
                AgentApiKey.revoked_at.is_(None),
            )
            .count()
            == 0
        )
    finally:
        db.close()


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
    assert body["username"] == "admin"
    assert body["email"] == "admin@example.com"


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


def test_v1_create_agent_rolls_back_agent_when_key_step_fails():
    """Agent + first runtime key commit atomically: if the key step
    fails, the agent row must not persist, so a client retry with the
    same name succeeds instead of hitting a stale duplicate-name."""
    key = _personal_key()
    name = "atomic-create agent"

    with patch(
        "xagent.web.services.agent_management.AgentApiKeyService.stage_rotated_key",
        side_effect=RuntimeError("staged key write blew up"),
    ):
        resp = client.post(
            "/v1/agents",
            headers=_bearer(key),
            json={"name": name, "instructions": "Be useful."},
        )
    assert resp.status_code == 500, resp.text

    # The aborted create must leave no row behind.
    db = _direct_db_session()
    try:
        leftover = db.query(Agent).filter(Agent.name == name).count()
    finally:
        db.close()
    assert leftover == 0

    # A clean retry with the same name now goes through.
    retry = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={"name": name, "instructions": "Be useful."},
    )
    assert retry.status_code == 200, retry.text
    assert retry.json()["agent"]["name"] == name


def test_v1_create_agent_maps_commit_conflict_to_409():
    """A unique-constraint IntegrityError at commit is translated to a
    409 rotation conflict, not a 500."""
    key = _personal_key()
    with patch(
        "sqlalchemy.orm.Session.commit",
        side_effect=IntegrityError(
            "INSERT", {}, Exception("uq_agent_api_keys_agent_active")
        ),
    ):
        resp = client.post(
            "/v1/agents",
            headers=_bearer(key),
            json={"name": "commit conflict agent", "instructions": "x"},
        )
    assert resp.status_code == 409, resp.text


def test_v1_create_agent_rejects_kb_without_knowledge_category():
    """KB selected but the knowledge tool category is not enabled -> 400,
    same invariant /api/agents enforces."""
    key = _personal_key()
    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "kb no tool",
            "instructions": "x",
            "knowledge_bases": ["some-kb"],
            "tool_categories": ["web_search"],
        },
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"


def test_v1_create_agent_rejects_invisible_kb():
    """KB not visible to the user -> 400."""
    key = _personal_key()
    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "kb invisible",
            "instructions": "x",
            "knowledge_bases": ["nonexistent-kb"],
            "tool_categories": ["knowledge"],
        },
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"


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
    assert body["agent"]["knowledge_bases"] == []
    assert body["agent"]["suggested_prompts"] == ["Ask anything"]
    assert body["api_key"]["full_key"].startswith("xag_")


def test_from_template_strips_agent_tool_category(template_manager):
    """A tool_categories override containing ``agent`` is silently
    stripped, same as the plain create path (issue #802)."""
    key = _personal_key()
    resp = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={
            "template_id": "qa",
            "name": "Template agent stripped",
            "tool_categories": ["web_search", "agent"],
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"]["tool_categories"] == ["web_search"]


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


def test_from_template_rolls_back_agent_when_key_step_fails(template_manager):
    """The from-template path shares the plain-create atomic boundary: a
    key-step failure must not leave the template-derived agent behind."""
    key = _personal_key()
    name = "atomic-template agent"

    with patch(
        "xagent.web.services.agent_management.AgentApiKeyService.stage_rotated_key",
        side_effect=RuntimeError("staged key write blew up"),
    ):
        resp = client.post(
            "/v1/agents/from-template",
            headers=_bearer(key),
            json={"template_id": "qa", "name": name},
        )
    assert resp.status_code == 500, resp.text

    db = _direct_db_session()
    try:
        leftover = db.query(Agent).filter(Agent.name == name).count()
    finally:
        db.close()
    assert leftover == 0

    retry = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={"template_id": "qa", "name": name},
    )
    assert retry.status_code == 200, retry.text
    assert retry.json()["agent"]["name"] == name


def test_from_template_rejects_template_kb_without_knowledge_category(
    template_manager,
):
    """Template-sourced KB fields go through the same validation as user
    input: a template that pre-sets a KB without the knowledge category
    is rejected, not silently persisted."""
    key = _personal_key()
    resp = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={"template_id": "kb-no-tool", "name": "from bad template"},
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"


def test_from_template_rejects_invisible_kb_override(template_manager):
    """A KB override that is not visible to the user -> 400 on the
    from-template path too."""
    key = _personal_key()
    resp = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={
            "template_id": "qa",
            "name": "from template invisible kb",
            "knowledge_bases": ["nonexistent-kb"],
            "tool_categories": ["knowledge"],
        },
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"
