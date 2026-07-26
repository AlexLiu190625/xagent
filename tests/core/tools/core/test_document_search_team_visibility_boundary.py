"""Regression coverage for the document-search team visibility boundary."""

from __future__ import annotations

import asyncio
import threading
from contextlib import suppress

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool

from xagent.core.tools.core import document_search
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    ListCollectionsResult,
)
from xagent.web.services.knowledge_base_team_scope import (
    KnowledgeBaseAccess,
    set_knowledge_base_team_hooks,
)


@pytest.fixture(autouse=True)
def _clear_team_knowledge_base_hooks():
    """Keep process-global application hooks isolated between tests."""
    set_knowledge_base_team_hooks()
    yield
    set_knowledge_base_team_hooks()


def _collections_result(*collections: CollectionInfo) -> ListCollectionsResult:
    return ListCollectionsResult(
        status="success",
        collections=list(collections),
        total_count=len(collections),
        message="ok",
    )


def _install_collection_listing(monkeypatch: pytest.MonkeyPatch) -> CollectionInfo:
    personal = CollectionInfo(name="personal")
    shared = CollectionInfo(name="shared")

    async def list_for_user(
        user_id: int | None = None, is_admin: bool = False
    ) -> ListCollectionsResult:
        del is_admin
        if user_id == 2:
            return _collections_result(shared)
        return _collections_result(personal)

    monkeypatch.setattr(document_search, "list_collections", list_for_user)
    return personal


def _one_slot_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.25,
    )


async def _wait_for(event: threading.Event) -> None:
    await asyncio.wait_for(asyncio.to_thread(event.wait), timeout=5)


async def _await_without_leaking(task: asyncio.Task[object]) -> None:
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


@pytest.mark.asyncio
async def test_team_visibility_hook_waits_off_loop_and_releases_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked hook checkout must leave the loop responsive until release."""
    _install_collection_listing(monkeypatch)
    engine = _one_slot_engine()
    held_connection = engine.connect()
    hook_started = threading.Event()
    hook_closed = threading.Event()
    ticker_advanced = threading.Event()
    observed: dict[str, int] = {}
    loop_thread_id = threading.get_ident()

    def visibility_hook(db: Session | None, user_id: int) -> list[KnowledgeBaseAccess]:
        assert db is None
        assert user_id == 1
        observed["hook_thread_id"] = threading.get_ident()
        hook_started.set()
        try:
            with Session(engine) as session:
                observed["sql_thread_id"] = threading.get_ident()
                session.execute(text("SELECT 1"))
        finally:
            hook_closed.set()
        return [KnowledgeBaseAccess(name="shared", storage_user_id=2)]

    async def tick_after_hook_starts() -> None:
        await _wait_for(hook_started)
        await asyncio.sleep(0)
        ticker_advanced.set()

    set_knowledge_base_team_hooks(visibility=visibility_hook)
    ticker = asyncio.create_task(tick_after_hook_starts())
    listing = asyncio.create_task(
        document_search._list_visible_collections(user_id=1, is_admin=False)
    )
    try:
        await _wait_for(hook_started)
        await _wait_for(ticker_advanced)
        assert not listing.done()
        assert engine.pool.checkedout() == 1

        held_connection.close()
        result = await listing
        await _wait_for(hook_closed)

        assert observed["hook_thread_id"] != loop_thread_id
        assert observed["sql_thread_id"] != loop_thread_id
        assert [collection.name for collection in result.collections] == [
            "personal",
            "shared",
        ]
        shared = next(
            collection
            for collection in result.collections
            if collection.name == "shared"
        )
        assert shared.ownership == "team"
        assert shared.storage_user_id == 2
        assert engine.pool.checkedout() == 0
    finally:
        if not held_connection.closed:
            held_connection.close()
        await _await_without_leaking(listing)
        await _await_without_leaking(ticker)
        engine.dispose()


@pytest.mark.asyncio
async def test_cancellation_drains_team_visibility_session_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation must wait for the hook-owned Session to close."""
    _install_collection_listing(monkeypatch)
    engine = _one_slot_engine()
    held_connection = engine.connect()
    hook_started = threading.Event()
    hook_closed = threading.Event()

    def visibility_hook(db: Session | None, user_id: int) -> list[KnowledgeBaseAccess]:
        assert db is None
        assert user_id == 1
        hook_started.set()
        try:
            with Session(engine) as session:
                session.execute(text("SELECT 1"))
        finally:
            hook_closed.set()
        return []

    set_knowledge_base_team_hooks(visibility=visibility_hook)
    listing = asyncio.create_task(
        document_search._list_visible_collections(user_id=1, is_admin=False)
    )
    try:
        await _wait_for(hook_started)
        listing.cancel()
        await asyncio.sleep(0)
        assert not listing.done()
        assert not hook_closed.is_set()
        assert engine.pool.checkedout() == 1

        held_connection.close()
        await _wait_for(hook_closed)
        with pytest.raises(asyncio.CancelledError):
            await listing
        assert engine.pool.checkedout() == 0
    finally:
        if not held_connection.closed:
            held_connection.close()
        await _await_without_leaking(listing)
        engine.dispose()


@pytest.mark.asyncio
async def test_team_visibility_hook_error_preserves_identity_and_closes_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hook error is not wrapped and its worker-owned Session is released."""
    _install_collection_listing(monkeypatch)
    engine = _one_slot_engine()
    hook_error = RuntimeError("team visibility failed")
    hook_closed = threading.Event()

    def visibility_hook(db: Session | None, user_id: int) -> list[KnowledgeBaseAccess]:
        assert db is None
        assert user_id == 1
        try:
            with Session(engine) as session:
                session.execute(text("SELECT 1"))
                raise hook_error
        finally:
            hook_closed.set()

    set_knowledge_base_team_hooks(visibility=visibility_hook)
    try:
        with pytest.raises(RuntimeError) as raised:
            await document_search._list_visible_collections(user_id=1, is_admin=False)
        assert raised.value is hook_error
        await _wait_for(hook_closed)
        assert engine.pool.checkedout() == 0
    finally:
        engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id", "is_admin", "install_hook"),
    [
        (None, False, True),
        (1, True, True),
        (1, False, False),
    ],
    ids=["anonymous", "admin", "no-hook"],
)
async def test_visibility_bypasses_keep_personal_collections_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    user_id: int | None,
    is_admin: bool,
    install_hook: bool,
) -> None:
    """Anonymous, admin, and unconfigured hook paths keep the personal result."""
    personal = _install_collection_listing(monkeypatch)
    hook_calls: list[tuple[Session | None, int]] = []

    def visibility_hook(
        db: Session | None, hooked_user_id: int
    ) -> list[KnowledgeBaseAccess]:
        hook_calls.append((db, hooked_user_id))
        return [KnowledgeBaseAccess(name="shared", storage_user_id=2)]

    if install_hook:
        set_knowledge_base_team_hooks(visibility=visibility_hook)

    result = await document_search._list_visible_collections(
        user_id=user_id, is_admin=is_admin
    )

    assert result.collections == [personal]
    assert result.total_count == 1
    assert hook_calls == []
