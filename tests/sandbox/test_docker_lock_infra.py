"""
Unit tests for DockerSandboxService's per-name lifecycle lock (_named_lock)
and control-object construction (_get_live_control). Pure asyncio tests
against a minimal fake Docker client; no Docker daemon required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import xagent.sandbox.docker_sandbox as docker_sandbox_module
from xagent.sandbox.docker_sandbox import (
    DockerSandboxService,
    MemDockerStore,
    _SandboxControl,
)


class _FakeContainerCollection:
    """Minimal Docker container collection stub: always reports no containers."""

    def list(self, *args, **kwargs):
        return []


class _FakeDockerClient:
    """Minimal Docker client stub sufficient for service construction and delete()."""

    def __init__(self) -> None:
        self.containers = _FakeContainerCollection()

    def ping(self):
        return True


def _make_service() -> DockerSandboxService:
    return DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())


class TestNamedLockIdentityAndMutualExclusion:
    """Pin the split-brain fix: a waiter queued behind a holder must land on
    the same lock entry the holder used, not a freshly-constructed one."""

    @pytest.mark.asyncio
    async def test_named_lock_stays_mutually_exclusive_across_release_and_requeue(
        self,
    ):
        service = _make_service()
        concurrent_holders = 0
        max_concurrent = 0

        entered_b = asyncio.Event()
        release_b = asyncio.Event()
        entered_a = asyncio.Event()
        release_a = asyncio.Event()
        entered_c = asyncio.Event()
        release_c = asyncio.Event()

        async def holder(entered: asyncio.Event, release: asyncio.Event) -> None:
            nonlocal concurrent_holders, max_concurrent
            async with service._named_lock("shared-name"):
                concurrent_holders += 1
                max_concurrent = max(max_concurrent, concurrent_holders)
                entered.set()
                await release.wait()
                concurrent_holders -= 1

        # B takes the lock first.
        task_b = asyncio.create_task(holder(entered_b, release_b))
        await entered_b.wait()

        # A queues behind B while B still holds it.
        task_a = asyncio.create_task(holder(entered_a, release_a))
        await asyncio.sleep(0)
        entry_while_b_holds = service._locks["shared-name"]
        assert entry_while_b_holds.waiters == 2

        # B finishes; the entry must survive (A is still waiting on it) so
        # that A ends up acquiring the SAME entry rather than racing a
        # newly-constructed one against a delete()/other holder.
        release_b.set()
        await task_b
        await entered_a.wait()
        assert service._locks["shared-name"] is entry_while_b_holds

        # A now holds it; C attempts to acquire concurrently and must queue
        # behind A rather than proceeding on an independent lock instance.
        task_c = asyncio.create_task(holder(entered_c, release_c))
        await asyncio.sleep(0)
        assert concurrent_holders == 1
        assert not entered_c.is_set()

        release_a.set()
        await task_a
        await entered_c.wait()

        release_c.set()
        await task_c

        assert max_concurrent == 1
        assert "shared-name" not in service._locks


class TestNamedLockWaiterRecycling:
    """Pin the entry-recycling contract: only evict when unused."""

    @pytest.mark.asyncio
    async def test_entry_recycled_when_no_waiters_remain(self):
        service = _make_service()
        async with service._named_lock("solo"):
            assert "solo" in service._locks
        assert "solo" not in service._locks

    @pytest.mark.asyncio
    async def test_entry_retained_while_a_waiter_is_pending(self):
        service = _make_service()
        entered_holder = asyncio.Event()
        release_holder = asyncio.Event()
        entered_waiter = asyncio.Event()
        release_waiter = asyncio.Event()

        async def holder(entered: asyncio.Event, release: asyncio.Event) -> None:
            async with service._named_lock("busy"):
                entered.set()
                await release.wait()

        task_holder = asyncio.create_task(holder(entered_holder, release_holder))
        await entered_holder.wait()

        task_waiter = asyncio.create_task(holder(entered_waiter, release_waiter))
        await asyncio.sleep(0)
        assert service._locks["busy"].waiters == 2

        # Releasing the holder must not drop the entry: the waiter is still
        # queued on it.
        release_holder.set()
        await task_holder
        assert "busy" in service._locks

        release_waiter.set()
        await task_waiter
        assert "busy" not in service._locks


class TestNamedLockCancellationSafety:
    """Pin that a cancelled waiter rolls back its waiter count and leaks nothing."""

    @pytest.mark.asyncio
    async def test_cancel_while_waiting_rolls_back_waiter_count_and_entry(self):
        service = _make_service()
        entered_holder = asyncio.Event()
        release_holder = asyncio.Event()

        async def holder() -> None:
            async with service._named_lock("cancel-me"):
                entered_holder.set()
                await release_holder.wait()

        task_holder = asyncio.create_task(holder())
        await entered_holder.wait()

        async def waiter() -> None:
            async with service._named_lock("cancel-me"):
                raise AssertionError("cancelled waiter must never enter the body")

        task_waiter = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        assert service._locks["cancel-me"].waiters == 2

        task_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_waiter

        # Cancellation rolled the waiter count back; only the holder remains.
        assert service._locks["cancel-me"].waiters == 1

        release_holder.set()
        await task_holder

        # Once the holder also finishes, the entry must not have leaked.
        assert "cancel-me" not in service._locks

    @pytest.mark.asyncio
    async def test_double_cancel_while_waiting_rolls_back_cleanly(self):
        # Waiter bookkeeping in _named_lock is fully synchronous: the
        # rollback path in the `except BaseException` branch has no `await`
        # in it, so a second cancel() delivered on top of the first, before
        # the waiter task gets a chance to run its own cancellation
        # handling, must still leave the waiter count and entry correctly
        # recovered.
        service = _make_service()
        entered_holder = asyncio.Event()
        release_holder = asyncio.Event()

        async def holder() -> None:
            async with service._named_lock("double-cancel"):
                entered_holder.set()
                await release_holder.wait()

        task_holder = asyncio.create_task(holder())
        await entered_holder.wait()

        async def waiter() -> None:
            async with service._named_lock("double-cancel"):
                raise AssertionError("cancelled waiter must never enter the body")

        task_waiter = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        assert service._locks["double-cancel"].waiters == 2

        task_waiter.cancel()
        task_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_waiter

        assert service._locks["double-cancel"].waiters == 1

        release_holder.set()
        await task_holder

        assert "double-cancel" not in service._locks


class TestGetLiveControl:
    """Pin _get_live_control as the deleted-aware construction point.

    _get_live_control() asserts that _named_lock(name) is held, so every
    call below happens inside that context, matching its real call sites.
    """

    @pytest.mark.asyncio
    async def test_reuses_the_same_live_control_instance(self):
        service = _make_service()
        async with service._named_lock("box"):
            first = service._get_live_control("box")
            second = service._get_live_control("box")
        assert second is first

    @pytest.mark.asyncio
    async def test_replaces_a_deleted_control_with_a_fresh_instance(self):
        service = _make_service()
        async with service._named_lock("box"):
            first = service._get_live_control("box")
            first.deleted = True

            replaced = service._get_live_control("box")

        assert replaced is not first
        assert replaced.deleted is False
        assert service._controls["box"] is replaced

    def test_asserts_when_called_without_holding_the_named_lock(self):
        service = _make_service()
        with pytest.raises(AssertionError):
            service._get_live_control("unlocked")


class TestDeleteIdentityCheckedPop:
    """Pin delete()'s identity-checked pop: a replaced control must survive."""

    @pytest.mark.asyncio
    async def test_delete_does_not_evict_a_control_installed_after_its_lookup(
        self, monkeypatch
    ):
        service = _make_service()
        old_control = service._get_control("box")
        new_control = _SandboxControl(name="box")

        real_find_container = service._find_container

        async def find_container_and_swap(name: str):
            # Simulate another in-flight path installing a fresh control
            # object for this name between delete()'s control lookup and its
            # cleanup pop.
            service._controls[name] = new_control
            return await real_find_container(name)

        monkeypatch.setattr(service, "_find_container", find_container_and_swap)

        await service.delete("box")

        assert old_control is not new_control
        assert service._controls.get("box") is new_control


class TestSandboxControlSingleConstructionPoint:
    """Source-level pin: only _get_control and _get_live_control may
    construct a _SandboxControl. A new inline construction elsewhere is a
    regression of the single-construction-point contract.

    Two sanctioned construction points: _get_control (legacy paths,
    deleted-preserving) and _get_live_control (lock-held paths,
    deleted-replacing); no inline construction elsewhere.
    """

    def test_sandbox_control_constructed_in_exactly_two_places(self):
        source = Path(docker_sandbox_module.__file__).read_text()
        assert source.count("_SandboxControl(") == 2
