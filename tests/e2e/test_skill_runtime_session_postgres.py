from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg2
import pytest
from docker.errors import APIError

from tests.e2e.app_harness import (
    create_e2e_user,
    disable_external_app_services,
    init_e2e_db,
    run_e2e_app_client,
)
from tests.e2e.minio_harness import (
    _docker_available,
    _docker_client,
    _free_port,
)
from xagent.web.models.database import get_engine
from xagent.web.models.skill import UserSkill, UserSkillFile

pytestmark = [pytest.mark.e2e, pytest.mark.docker]

POSTGRES_PASSWORD = "xagent_test"
POSTGRES_DATABASE = "xagent_test"


@pytest.fixture
def postgres_url() -> Iterator[str]:
    if not _docker_available():
        pytest.skip("Requires reachable Docker daemon")

    client = _docker_client()
    container = None
    host_port = 0
    for _ in range(5):
        host_port = _free_port()
        container_name = f"xagent-postgres-e2e-{uuid4().hex[:12]}"
        try:
            container = client.containers.run(
                "postgres:16-bookworm",
                detach=True,
                name=container_name,
                ports={"5432/tcp": host_port},
                tmpfs={"/var/lib/postgresql/data": "rw,size=256m"},
                environment={
                    "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
                    "POSTGRES_DB": POSTGRES_DATABASE,
                },
            )
            break
        except APIError as exc:
            if "address already in use" not in str(exc):
                raise
            try:
                stale = client.containers.get(container_name)
                stale.remove(force=True)
            except Exception:
                pass
    if container is None:
        pytest.skip("Could not allocate a free host port for PostgreSQL")

    database_url = (
        "postgresql+psycopg2://postgres:"
        f"{POSTGRES_PASSWORD}@127.0.0.1:{host_port}/{POSTGRES_DATABASE}"
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                connection = psycopg2.connect(
                    host="127.0.0.1",
                    port=host_port,
                    user="postgres",
                    password=POSTGRES_PASSWORD,
                    dbname=POSTGRES_DATABASE,
                    connect_timeout=1,
                )
            except psycopg2.OperationalError:
                time.sleep(0.25)
            else:
                connection.close()
                break
        else:
            raise RuntimeError("PostgreSQL did not become ready")

        yield database_url
    finally:
        container.remove(force=True)


def _configure_postgres_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    postgres_url: str,
) -> None:
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    monkeypatch.setenv("XAGENT_DB_POOL_SIZE", "1")
    monkeypatch.setenv("XAGENT_DB_MAX_OVERFLOW", "0")
    monkeypatch.setenv("XAGENT_DB_POOL_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("XAGENT_FILE_STORAGE_STARTUP_SYNC_ENABLED", "false")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized"))
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "lancedb"))
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "lancedb-path"))
    monkeypatch.setenv("LANCEDB_AUTO_MIGRATE", "false")


def _seed_personal_skill(db: Any, *, user_id: int) -> str:
    name = "session-safe"
    content = b"""---
name: session-safe
description: Pool handoff regression fixture
when_to_use: Test authenticated Skill database reads
---

# Session Safe
"""
    skill = UserSkill(
        user_id=user_id,
        name=name,
        origin="custom",
        created_by_user_id=user_id,
        updated_by_user_id=user_id,
    )
    skill.files.append(
        UserSkillFile(
            path="SKILL.md",
            content=content,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            media_type="text/markdown",
        )
    )
    db.add(skill)
    db.commit()
    return name


def test_authenticated_skill_routes_handoff_one_slot_postgres_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    postgres_url: str,
) -> None:
    _configure_postgres_app(
        monkeypatch,
        tmp_path=tmp_path,
        postgres_url=postgres_url,
    )
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    with SessionLocal() as db:
        user = create_e2e_user(db, username="skill-runtime-user")
        skill_name = _seed_personal_skill(db, user_id=user.id)

    with run_e2e_app_client(
        monkeypatch,
        username=user.username,
        user_id=user.id,
    ) as app:
        skills_response = app.client.get("/api/skills/", headers=app.headers)
        assert skills_response.status_code == 200, skills_response.text
        assert skill_name in {item["name"] for item in skills_response.json()}
        assert get_engine().pool.checkedout() == 0

        installed_response = app.client.get(
            "/api/skill-hub/installed",
            headers=app.headers,
        )
        assert installed_response.status_code == 200, installed_response.text
        assert skill_name in {item["name"] for item in installed_response.json()}
        assert get_engine().pool.checkedout() == 0
