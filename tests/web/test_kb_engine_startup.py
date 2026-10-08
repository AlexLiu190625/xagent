"""The backend, the agent worker and the Celery worker refuse a mismatched KB engine."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from cryptography.fernet import Fernet

from xagent.core.tools.core.RAG_tools.core.exceptions import ConfigurationError
from xagent.core.tools.core.RAG_tools.storage.vector_backend import KB_ENGINE_RECORD


@pytest.fixture(autouse=True)
def milvus_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path))
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "lancedb")
    (tmp_path / KB_ENGINE_RECORD).write_text("milvus\n")


@pytest.mark.asyncio
async def test_backend_startup_refuses_before_touching_the_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web import app

    initialize = AsyncMock()
    monkeypatch.setattr(app, "_initialize_database_and_admit_runtime", initialize)

    with pytest.raises(ConfigurationError, match="KB engine is milvus"):
        await app.startup_event()
    initialize.assert_not_called()


@pytest.mark.asyncio
async def test_agent_worker_refuses_before_accepting_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web import worker

    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "worker")
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    for name in (
        "configure_db",
        "validate_worker_schema",
        "validate_interaction_rollout_at_startup",
        "register_local_browser_runtime",
    ):
        monkeypatch.setattr(worker, name, Mock())

    with pytest.raises(ConfigurationError, match="KB engine is milvus"):
        await worker.run_worker(stop=asyncio.Event())
    worker.register_local_browser_runtime.assert_not_called()


def test_celery_worker_exits_at_worker_init() -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    with pytest.raises(SystemExit, match="KB engine is milvus"):
        worker_init.send(sender=None)


def test_celery_worker_exits_on_any_engine_check_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from celery.signals import worker_init

    from xagent.core.tools.core.RAG_tools.storage import vector_backend
    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    def unwritable() -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(vector_backend, "lock_deployment_kb_engine", unwritable)

    with pytest.raises(SystemExit, match="disk unavailable"):
        worker_init.send(sender=None)
