"""Async-boundary tests for team knowledge-base visibility overlays."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from xagent.core.tools.core import document_search
from xagent.core.tools.core.RAG_tools.core.schemas import ListCollectionsResult
from xagent.web.services.knowledge_base_team_scope import (
    set_knowledge_base_team_hooks,
)


@pytest.mark.asyncio
async def test_team_visibility_hook_runs_off_the_event_loop(monkeypatch):
    async def empty_collections(*, user_id, is_admin):
        return ListCollectionsResult(
            status="success",
            collections=[],
            total_count=0,
            message="ok",
        )

    monkeypatch.setattr(document_search, "list_collections", empty_collections)

    loop_thread = threading.get_ident()
    hook_threads: list[int] = []

    def slow_visibility_hook(db, user_id):
        assert db is None
        assert user_id == 7
        hook_threads.append(threading.get_ident())
        time.sleep(0.05)
        return []

    set_knowledge_base_team_hooks(visibility=slow_visibility_hook)
    ticks = 0
    stop = False

    async def ticker() -> None:
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.005)

    ticker_task = asyncio.create_task(ticker())
    try:
        result = await document_search._list_visible_collections(7, False)
    finally:
        stop = True
        await ticker_task
        set_knowledge_base_team_hooks()

    assert result.total_count == 0
    assert hook_threads and hook_threads[0] != loop_thread
    assert ticks >= 3
