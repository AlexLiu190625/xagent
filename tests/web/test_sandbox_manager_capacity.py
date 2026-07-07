"""Tests for the sandbox container cap and LRU idle eviction.

Uses a stateful fake ``SandboxService`` so the container count reflects
creations and deletions — no Docker-only assumptions.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import xagent.web.sandbox_manager as sandbox_manager_module
from xagent.web.sandbox_manager import SandboxCapacityError, SandboxManager


class _FakeService:
    """In-memory sandbox service tracking existing container names."""

    def __init__(self, initial: tuple[str, ...] = ()) -> None:
        self.containers: set[str] = set(initial)
        self.peak = len(self.containers)
        self.deleted: list[str] = []

    async def list_sandboxes(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                name=name,
                state="stopped",
                template=SimpleNamespace(type="image", image="img:v1"),
                config=SimpleNamespace(),
            )
            for name in sorted(self.containers)
        ]

    async def get_or_create(self, name, template=None, config=None):
        self.containers.add(name)
        self.peak = max(self.peak, len(self.containers))
        sandbox = MagicMock()
        sandbox.name = name
        return sandbox

    async def delete(self, name: str) -> None:
        self.containers.discard(name)
        self.deleted.append(name)


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(sandbox_manager_module, "time", fake)
    return fake


@pytest.fixture
def _env(monkeypatch, tmp_path):
    """Isolate sandbox env config and neutralize code mounts."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/repo/src", "/app/src", "ro")],
        ),
    ):
        yield


def _make_manager(initial: tuple[str, ...] = ()) -> tuple[SandboxManager, _FakeService]:
    service = _FakeService(initial)
    return SandboxManager(service), service


@pytest.mark.asyncio
async def test_cap_never_exceeded_under_concurrent_creates(
    _env, clock, monkeypatch
) -> None:
    monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONTAINERS", "2")
    manager, service = _make_manager()

    await asyncio.gather(
        *(manager.get_or_create_sandbox("task", str(i)) for i in range(5))
    )

    assert service.peak <= 2
    assert len(service.containers) <= 2


@pytest.mark.asyncio
async def test_lru_idle_sandbox_is_evicted_with_workers(
    _env, clock, monkeypatch
) -> None:
    monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONTAINERS", "3")
    manager, service = _make_manager(("task::old", "task::old::worker::0", "task::new"))
    # task::new was recently active; task::old has no recorded activity and
    # counts as idle since startup (LRU-oldest).
    clock.advance(50)
    async with manager._activity_guard:
        manager._touch_locked("task::new")

    await manager.get_or_create_sandbox("task", "incoming")

    assert "task::old" in service.deleted
    assert "task::old::worker::0" in service.deleted
    assert service.containers == {"task::new", "task::incoming"}
    assert service.peak <= 3


@pytest.mark.asyncio
async def test_nothing_evictable_raises_capacity_error(
    _env, clock, monkeypatch
) -> None:
    monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONTAINERS", "2")
    manager, service = _make_manager(("task::1", "task::2"))
    manager._lease_providers["task::1"] = MagicMock()
    manager._lease_providers["task::2"] = MagicMock()
    assert await manager.attach("task", "1")
    assert await manager.attach("task", "2")

    with pytest.raises(SandboxCapacityError, match="capacity limit reached"):
        await manager.get_or_create_sandbox("task", "3")

    assert service.containers == {"task::1", "task::2"}
    assert service.deleted == []


@pytest.mark.asyncio
async def test_active_sandboxes_are_never_evicted(_env, clock, monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONTAINERS", "2")
    manager, service = _make_manager(("task::busy", "task::idle"))
    manager._lease_providers["task::busy"] = MagicMock()
    assert await manager.attach("task", "busy")

    await manager.get_or_create_sandbox("task", "incoming")

    assert "task::busy" in service.containers
    assert "task::idle" in service.deleted


@pytest.mark.asyncio
async def test_existing_sandbox_reuse_does_not_evict(_env, clock, monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONTAINERS", "1")
    manager, service = _make_manager(("task::1",))

    sandbox = await manager.get_or_create_sandbox("task", "1")

    assert sandbox.name == "task::1"
    assert service.deleted == []


@pytest.mark.asyncio
async def test_worker_creation_never_evicts_own_primary(
    _env, clock, monkeypatch
) -> None:
    monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONTAINERS", "1")
    manager, service = _make_manager(("task::1",))

    with pytest.raises(SandboxCapacityError):
        await manager.get_or_create_sandbox("task", "1::worker::0")

    assert "task::1" in service.containers


@pytest.mark.asyncio
async def test_cap_unset_means_no_gate(_env, clock, monkeypatch) -> None:
    monkeypatch.delenv("XAGENT_SANDBOX_MAX_CONTAINERS", raising=False)
    manager, service = _make_manager(("task::1", "task::2", "task::3"))

    for i in range(4, 8):
        await manager.get_or_create_sandbox("task", str(i))

    assert service.deleted == []
    assert len(service.containers) == 7


@pytest.mark.asyncio
async def test_evicted_sandbox_is_recreated_on_later_use(
    _env, clock, monkeypatch
) -> None:
    monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONTAINERS", "1")
    manager, service = _make_manager()

    first = await manager.get_or_create_sandbox("task", "1")
    clock.advance(10)
    await manager.get_or_create_sandbox("task", "2")  # evicts task::1
    clock.advance(10)
    recreated = await manager.get_or_create_sandbox("task", "1")  # evicts task::2

    assert "task::1" in service.deleted
    assert "task::2" in service.deleted
    assert service.containers == {"task::1"}
    assert first is not recreated
    assert service.peak <= 1


@pytest.mark.asyncio
async def test_in_flight_lifecycle_keys_are_not_eviction_victims(
    _env, clock, monkeypatch
) -> None:
    """A key whose lifecycle lock is held (creation/cleanup in flight) is
    skipped by victim selection instead of deadlocking or double-deleting."""
    monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONTAINERS", "1")
    manager, service = _make_manager(("task::busykey",))

    async with manager._lifecycle_locked("task::busykey"):
        with pytest.raises(SandboxCapacityError):
            await manager.get_or_create_sandbox("task", "other")

    assert "task::busykey" in service.containers


@pytest.mark.asyncio
async def test_concurrent_recreate_during_eviction_gets_fresh_container(
    _env, clock, monkeypatch
) -> None:
    """A same-key get_or_create_lease_provider racing an in-flight capacity
    eviction must not be handed a provider around the cached (doomed)
    sandbox: the eviction claim purges the instance cache, so the recreate
    cache-misses, queues behind the capacity gate, and builds a fresh
    container after the deletion finished."""
    monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONTAINERS", "1")
    manager, service = _make_manager(("task::victim",))
    old_sandbox = MagicMock()
    manager._cache["task::victim"] = old_sandbox
    manager._config_cache["task::victim"] = MagicMock()

    delete_started = asyncio.Event()
    delete_release = asyncio.Event()
    original_delete = service.delete

    async def slow_delete(name: str) -> None:
        delete_started.set()
        await delete_release.wait()
        await original_delete(name)

    service.delete = slow_delete  # type: ignore[method-assign]

    evictor = asyncio.create_task(manager.get_or_create_sandbox("task", "new"))
    await delete_started.wait()

    racer = asyncio.create_task(manager.get_or_create_lease_provider("task", "victim"))
    for _ in range(10):
        await asyncio.sleep(0)
    # With the old code the racer would cache-hit the doomed sandbox and
    # complete here; now it must be queued behind the capacity gate.
    assert not racer.done()

    delete_release.set()
    await evictor
    provider = await racer

    assert provider.primary_sandbox is not old_sandbox
    assert "task::victim" in service.deleted
    assert service.containers == {"task::victim"}


@pytest.mark.asyncio
async def test_worker_lease_at_cap_degrades_to_primary(
    _env, clock, monkeypatch
) -> None:
    """A concurrency-safe lease whose worker cannot fit under the cap must
    degrade to the primary sandbox (same sharing semantics as unsafe
    leases) instead of failing the tool mid-task, and must not cache the
    degraded result so worker creation is retried later."""
    monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONTAINERS", "1")
    manager, service = _make_manager()

    provider = await manager.get_or_create_lease_provider("task", "1")
    assert service.containers == {"task::1"}

    async with provider.lease(concurrency_safe=True) as sandbox:
        assert sandbox is provider.primary_sandbox

    assert provider._workers == {}
    assert service.containers == {"task::1"}

    # With the cap lifted, the same lease creates a real worker again.
    monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONTAINERS", "5")
    async with provider.lease(concurrency_safe=True) as sandbox:
        assert sandbox is not provider.primary_sandbox
        assert sandbox.name.startswith("task::1::worker::")


@pytest.mark.asyncio
async def test_victim_turning_active_between_pick_and_claim_is_spared(
    _env, clock, monkeypatch
) -> None:
    """When the picked victim becomes active between selection and claim,
    the claim returns False and the eviction loop retries with the next
    victim instead of deleting the now-active sandbox."""
    monkeypatch.setenv("XAGENT_SANDBOX_MAX_CONTAINERS", "2")
    manager, service = _make_manager(("task::a", "task::b"))
    # task::a is LRU-oldest (no recorded activity); task::b is fresher.
    clock.advance(50)
    async with manager._activity_guard:
        manager._touch_locked("task::b")

    original_claim = manager._claim_idle_sandbox
    raced: list[str] = []

    async def claim_with_race(base_name: str) -> bool:
        if base_name == "task::a" and not raced:
            raced.append(base_name)
            manager._lease_providers["task::a"] = MagicMock()
            assert await manager.attach("task", "a")
        return await original_claim(base_name)

    manager._claim_idle_sandbox = claim_with_race  # type: ignore[method-assign]

    await manager.get_or_create_sandbox("task", "c")

    assert raced == ["task::a"]
    assert "task::a" in service.containers  # spared: became active
    assert "task::b" in service.deleted  # next victim evicted instead
    assert "task::c" in service.containers
