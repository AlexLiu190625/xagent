"""
Sandbox management in application layer.
"""

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import (
    get_boxlite_home_dir,
    get_sandbox_cpus,
    get_sandbox_env,
    get_sandbox_host_storage_root,
    get_sandbox_idle_ttl,
    get_sandbox_image,
    get_sandbox_max_concurrency,
    get_sandbox_max_containers,
    get_sandbox_memory,
    get_sandbox_sweep_interval,
    get_sandbox_volumes,
    get_storage_root,
    get_uploads_dir,
)
from ..core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    build_code_mount_volumes,
)
from ..sandbox import SandboxService
from ..sandbox.base import Sandbox, SandboxConfig, SandboxTemplate

logger = logging.getLogger(__name__)

_WORKER_LIFECYCLE_MARKER = "::worker::"


class SandboxCapacityError(RuntimeError):
    """The sandbox container cap is reached and no idle sandbox is evictable.

    Distinct from sandbox-service unavailability: by default the web layer
    rejects the task with this error instead of falling back to local
    execution (see XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY).
    """

    def __init__(self, *, cap: int, in_use: int) -> None:
        super().__init__(
            f"Sandbox capacity limit reached ({in_use} containers, cap {cap}) "
            "and all sandboxes are busy. Please retry when a running task "
            "finishes, or raise XAGENT_SANDBOX_MAX_CONTAINERS."
        )
        self.cap = cap
        self.in_use = in_use


@dataclass
class _SandboxActivity:
    """Per-lifecycle activity state used for reclamation decisions."""

    ref_count: int = 0
    last_activity: float = 0.0


@dataclass
class _LifecycleLockEntry:
    """Per-lifecycle lock with holder/waiter tracking for safe eviction."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    waiters: int = 0


class SandboxLease:
    """Async context manager for one leased sandbox execution slot."""

    def __init__(
        self,
        provider: "SandboxLeaseProvider",
        *,
        concurrency_safe: bool,
    ) -> None:
        self._provider = provider
        self._concurrency_safe = concurrency_safe
        self._slot: int | None = None
        self._sandbox: Sandbox | None = None

    async def __aenter__(self) -> Sandbox:
        if not self._concurrency_safe:
            self._sandbox = self._provider.primary_sandbox
            return self._sandbox

        self._slot = await self._provider.acquire_worker_slot()
        try:
            self._sandbox = await self._provider.get_worker_sandbox(self._slot)
            return self._sandbox
        except Exception:
            await self._provider.release_worker_slot(self._slot)
            self._slot = None
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._slot is not None:
            await self._provider.release_worker_slot(self._slot)
            self._slot = None
        self._sandbox = None


class SandboxLeaseProvider:
    """Lease primary or worker sandboxes for sandboxed tool execution."""

    def __init__(
        self,
        *,
        manager: "SandboxManager",
        lifecycle_type: str,
        lifecycle_id: str,
        primary_sandbox: Sandbox,
        workspace_config: Mapping[str, Any] | None,
        max_concurrency: int,
    ) -> None:
        self._manager = manager
        self._lifecycle_type = lifecycle_type
        self._lifecycle_id = lifecycle_id
        self._workspace_config = workspace_config
        self._available_slots: asyncio.Queue[int] = asyncio.Queue()
        self._worker_locks: dict[int, asyncio.Lock] = {}
        self._workers: dict[int, Sandbox] = {}
        self.primary_sandbox = primary_sandbox
        for slot in range(max(1, max_concurrency)):
            self._available_slots.put_nowait(slot)

    def lease(self, *, concurrency_safe: bool) -> SandboxLease:
        """Return an async context manager for the requested execution mode."""
        return SandboxLease(self, concurrency_safe=concurrency_safe)

    async def acquire_worker_slot(self) -> int:
        """Reserve one worker slot, waiting when all workers are busy."""
        return await self._available_slots.get()

    async def release_worker_slot(self, slot: int) -> None:
        """Return a worker slot to the provider."""
        self._available_slots.put_nowait(slot)

    async def get_worker_sandbox(self, slot: int) -> Sandbox:
        """Get or lazily create a worker sandbox for a slot.

        When the container cap leaves no room for a worker, the lease
        degrades to the primary sandbox instead of failing the tool
        mid-task — the same sharing semantics non-concurrency-safe leases
        already have, trading isolation for availability. The degraded
        result is not cached, so a later lease retries worker creation
        once capacity frees up.
        """
        if slot in self._workers:
            return self._workers[slot]

        if slot not in self._worker_locks:
            self._worker_locks[slot] = asyncio.Lock()

        async with self._worker_locks[slot]:
            if slot in self._workers:
                return self._workers[slot]
            try:
                worker = await self._manager.get_or_create_sandbox(
                    self._lifecycle_type,
                    f"{self._lifecycle_id}::worker::{slot}",
                    workspace_config=self._workspace_config,
                )
            except SandboxCapacityError as exc:
                logger.warning(
                    "No capacity for worker sandbox %s::%s::worker::%d; "
                    "degrading to the primary sandbox: %s",
                    self._lifecycle_type,
                    self._lifecycle_id,
                    slot,
                    exc,
                )
                return self.primary_sandbox
            self._workers[slot] = worker
            return worker

    async def cleanup_worker_sandboxes(self) -> None:
        """Delete worker sandboxes while keeping the primary sandbox cached."""
        await self._manager.delete_worker_sandboxes(
            self._lifecycle_type,
            self._lifecycle_id,
        )
        self._workers.clear()


class SandboxPathMapper:
    """Translate backend-visible workspace paths into sandbox volume tuples."""

    def __init__(
        self,
        *,
        backend_storage_root: Path,
        host_storage_root: Path | None,
        sandbox_storage_root: Path | None = None,
    ) -> None:
        self.backend_storage_root = self._as_backend_path(backend_storage_root)
        self.host_storage_root = host_storage_root
        self.sandbox_storage_root = self._as_backend_path(
            sandbox_storage_root or self.backend_storage_root
        )

    @classmethod
    def from_env(cls) -> "SandboxPathMapper":
        return cls(
            backend_storage_root=get_storage_root(),
            host_storage_root=get_sandbox_host_storage_root(),
        )

    @property
    def uses_host_storage_root(self) -> bool:
        return self.host_storage_root is not None

    @staticmethod
    def _as_backend_path(path: str | Path) -> Path:
        backend_path = Path(os.path.expandvars(str(path))).expanduser()
        if not backend_path.is_absolute():
            backend_path = Path.cwd() / backend_path
        return backend_path

    def _relative_to_backend_storage(self, backend_path: Path) -> Path | None:
        try:
            return backend_path.relative_to(self.backend_storage_root)
        except ValueError:
            return None

    def to_host_bind_source(self, backend_path: str | Path) -> Path:
        path = self._as_backend_path(backend_path)
        if self.host_storage_root is None:
            return path

        relative_path = self._relative_to_backend_storage(path)
        if relative_path is None:
            return path
        return self.host_storage_root / relative_path

    def to_sandbox_target(self, backend_path: str | Path) -> Path:
        path = self._as_backend_path(backend_path)
        if self.host_storage_root is None:
            return path

        relative_path = self._relative_to_backend_storage(path)
        if relative_path is None:
            return path
        return self.sandbox_storage_root / relative_path

    def volume_for_backend_path(
        self, backend_path: str | Path, mode: str = "rw"
    ) -> tuple[str, str, str]:
        return (
            str(self.to_host_bind_source(backend_path)),
            str(self.to_sandbox_target(backend_path)),
            mode,
        )


class SandboxManager:
    """Manages sandbox instances, their activity state, and reclamation.

    Concurrency model — the invariants every change must preserve:

    Synchronization primitives, from innermost to outermost:

    - ``_activity_guard`` (one asyncio.Lock): makes compound check-then-act
      decisions on activity state atomic — attach's provider-existence
      check + ref-count increment, release's decrement + provider pop, and
      the eviction claim (ref-count re-check + provider pop + instance
      cache purge). Critical sections must stay fully synchronous: never
      ``await`` while holding it, and never acquire it inside a ``finally``
      (a cancellation delivered at that await point would skip the cleanup).
      Independent single dict operations do NOT need it.
    - ``_lifecycle_locks`` (per lifecycle key, waiter-counted): serialize
      lease provider creation with release-to-zero worker cleanup and with
      the idle sweep, per key. Entries are garbage-collected when the last
      holder/waiter leaves.
    - ``_locks`` + ``_locks_guard`` (per sandbox name): serialize container
      creation per name inside ``get_or_create_sandbox``.
    - ``_capacity_gate`` (global): serializes the cap check + eviction +
      container creation so concurrent creations for different names cannot
      all pass the count check.

    Ordering rules:

    - lifecycle lock -> per-name lock -> capacity gate is the only nesting
      direction; never acquire a lifecycle lock while holding the gate
      (a same-key creator holds its lifecycle lock while waiting for the
      gate, so gate -> lifecycle closes a deadlock cycle). Capacity
      eviction therefore does NOT lock the victim's lifecycle: it relies on
      the gate plus the atomic claim purging the instance cache, which
      forces any concurrent same-key re-creation to cache-miss and queue
      behind the gate until the deletion finished.

    Safety contract:

    - A lifecycle with a non-zero ref-count is never deleted, stopped, or
      evicted — by the sweep, by capacity eviction, or by any race between
      them. Both reclamation paths go through ``_evict_idle_sandbox``,
      whose claim re-validates the ref-count under ``_activity_guard``.
    """

    def __init__(self, service: SandboxService):
        """
        Initialize sandbox manager.

        Args:
            service: SandboxService instance for creating sandboxes
        """
        self._service: SandboxService = service
        self._cache: dict[str, Sandbox] = {}
        self._config_cache: dict[str, SandboxConfig] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        # Activity tracking: lease providers, active-task ref-counts, and
        # last-activity timestamps keyed by the primary sandbox name. This is
        # the single source of truth reclamation decisions are made from.
        self._lease_providers: dict[str, SandboxLeaseProvider] = {}
        self._activity: dict[str, _SandboxActivity] = {}
        self._activity_guard = asyncio.Lock()
        self._lifecycle_locks: dict[str, _LifecycleLockEntry] = {}
        self._startup_monotonic = time.monotonic()
        # Global gate serializing the capacity check with container creation:
        # per-name locks cannot stop two concurrent creations for different
        # names from both passing the count check.
        self._capacity_gate = asyncio.Lock()

    @staticmethod
    def make_sandbox_name(lifecycle_type: str, lifecycle_id: str) -> str:
        """Build a sandbox name from lifecycle type and id."""
        return f"{lifecycle_type}::{lifecycle_id}"

    @staticmethod
    def parse_sandbox_name(name: str) -> tuple[str, str]:
        """Parse a sandbox name into (lifecycle_type, lifecycle_id).

        Raises:
            ValueError: Invalid sandbox name format.
        """
        parts = name.split("::", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid sandbox name format: {name!r}")
        return parts[0], parts[1]

    @staticmethod
    def _base_lifecycle_id(lifecycle_id: str) -> str:
        """Return the owner lifecycle id for primary and worker sandboxes."""
        return lifecycle_id.split(_WORKER_LIFECYCLE_MARKER, 1)[0]

    @classmethod
    def _worker_sandbox_prefix(cls, lifecycle_type: str, lifecycle_id: str) -> str:
        return (
            cls.make_sandbox_name(lifecycle_type, lifecycle_id)
            + _WORKER_LIFECYCLE_MARKER
        )

    @classmethod
    def _base_sandbox_name(cls, lifecycle_type: str, lifecycle_id: str) -> str:
        """Primary sandbox name owning activity state for a lifecycle key."""
        return cls.make_sandbox_name(
            lifecycle_type, cls._base_lifecycle_id(lifecycle_id)
        )

    @asynccontextmanager
    async def _lifecycle_locked(self, base_name: str) -> AsyncIterator[None]:
        """Serialize provider creation and release-to-zero cleanup per key.

        Entries are dropped once no holder or waiter remains, so the dict
        does not grow with every lifecycle key ever seen.

        The waiter bookkeeping is deliberately not guarded by
        ``_activity_guard``: each step is a single synchronous operation
        with no compound invariant, and awaiting a lock inside ``finally``
        could leak the waiter count if the task were cancelled at that
        await point.
        """
        entry = self._lifecycle_locks.get(base_name)
        if entry is None:
            entry = _LifecycleLockEntry()
            self._lifecycle_locks[base_name] = entry
        entry.waiters += 1

        try:
            await entry.lock.acquire()
        except BaseException:
            entry.waiters -= 1
            self._drop_lifecycle_lock_if_unused(base_name, entry)
            raise

        try:
            yield
        finally:
            entry.lock.release()
            entry.waiters -= 1
            self._drop_lifecycle_lock_if_unused(base_name, entry)

    def _drop_lifecycle_lock_if_unused(
        self, base_name: str, entry: _LifecycleLockEntry
    ) -> None:
        if entry.waiters > 0:
            return
        if self._lifecycle_locks.get(base_name) is entry:
            self._lifecycle_locks.pop(base_name, None)

    def _touch_locked(self, base_name: str) -> _SandboxActivity:
        """Bump last-activity for a key; caller must hold ``_activity_guard``."""
        activity = self._activity.get(base_name)
        if activity is None:
            activity = _SandboxActivity()
            self._activity[base_name] = activity
        activity.last_activity = time.monotonic()
        return activity

    async def attach(self, lifecycle_type: str, lifecycle_id: str) -> bool:
        """Mark one task as actively using the lifecycle's lease provider.

        Returns False when no lease provider is cached for the key — nothing
        is attached and the caller must not release. A sandbox with a
        non-zero ref-count is never reclaimed.
        """
        base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
        async with self._activity_guard:
            if base_name not in self._lease_providers:
                return False
            self._touch_locked(base_name).ref_count += 1
        return True

    async def release(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        on_last_release: Optional[Callable[[], None]] = None,
    ) -> bool:
        """Release one active task for a lifecycle key.

        When the last task releases, the cached lease provider is dropped,
        ``on_last_release`` is invoked (still under the per-key lifecycle
        lock, before worker deletion, so callers can evict their own caches
        exactly once), and the lifecycle's worker sandboxes are deleted.

        A release without a matching attach (ref-count already zero) is
        ignored with a warning: running the cleanup path anyway could tear
        down a freshly created, not-yet-attached provider.

        Returns True when this call released the last active task.
        """
        base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
        async with self._lifecycle_locked(base_name):
            async with self._activity_guard:
                activity = self._touch_locked(base_name)
                if activity.ref_count == 0:
                    logger.warning(
                        "Ignoring sandbox release without matching attach for %s",
                        base_name,
                    )
                    return False
                if activity.ref_count > 1:
                    activity.ref_count -= 1
                    return False
                activity.ref_count = 0
                self._lease_providers.pop(base_name, None)

            if on_last_release is not None:
                on_last_release()

            await self.delete_worker_sandboxes(
                lifecycle_type, self._base_lifecycle_id(lifecycle_id)
            )
        return True

    def ref_count(self, lifecycle_type: str, lifecycle_id: str) -> int:
        """Number of active tasks attached to a lifecycle key."""
        activity = self._activity.get(
            self._base_sandbox_name(lifecycle_type, lifecycle_id)
        )
        return activity.ref_count if activity is not None else 0

    def last_activity_at(self, lifecycle_type: str, lifecycle_id: str) -> float:
        """Monotonic timestamp of the last recorded activity for a key.

        Keys with no recorded activity (e.g. containers discovered after a
        backend restart) report idle since manager startup.
        """
        activity = self._activity.get(
            self._base_sandbox_name(lifecycle_type, lifecycle_id)
        )
        if activity is None:
            return self._startup_monotonic
        return activity.last_activity

    def _get_sandbox_image_and_config(self) -> tuple[str, SandboxConfig]:
        """Get sandbox image and configuration from centralized config module."""
        image = get_sandbox_image()
        config = SandboxConfig()
        path_mapper = SandboxPathMapper.from_env()

        # CPU
        cpus = get_sandbox_cpus()
        if cpus is not None:
            config.cpus = cpus

        # MEM
        memory = get_sandbox_memory()
        if memory is not None:
            config.memory = memory

        # ENV
        env = get_sandbox_env()
        if env:
            config.env = env

        # VOL
        volumes = get_sandbox_volumes(
            host_side_sources=path_mapper.uses_host_storage_root
        )
        if volumes:
            config.volumes = volumes

        return image, config

    @staticmethod
    def _append_unique_volume(
        volumes: list[tuple[str, str, str]], volume: tuple[str, str, str]
    ) -> None:
        if volume not in volumes:
            volumes.append(volume)

    @staticmethod
    def _workspace_mount_paths(
        lifecycle_type: str,
        lifecycle_id: str,
        workspace_config: Mapping[str, Any] | None,
    ) -> list[tuple[Path, bool]]:
        paths: list[tuple[Path, bool]] = []

        if workspace_config:
            base_dir = workspace_config.get("base_dir")
            if base_dir:
                paths.append((Path(str(base_dir)), True))

            for raw_dir in workspace_config.get("allowed_external_dirs") or []:
                paths.append((Path(str(raw_dir)), False))
        elif lifecycle_type == "user":
            owner_lifecycle_id = SandboxManager._base_lifecycle_id(lifecycle_id)
            paths.append((get_uploads_dir() / f"user_{owner_lifecycle_id}", True))

        return paths

    @staticmethod
    def _config_equivalent(left: SandboxConfig, right: SandboxConfig) -> bool:
        return (
            left.cpus == right.cpus
            and left.memory == right.memory
            and (left.env or {}) == (right.env or {})
            and set(left.volumes or []) == set(right.volumes or [])
        )

    @staticmethod
    def _ensure_config_equivalent(
        sandbox_name: str,
        cached_config: SandboxConfig | None,
        desired_config: SandboxConfig,
    ) -> None:
        if cached_config is None:
            return
        if SandboxManager._config_equivalent(cached_config, desired_config):
            return
        raise RuntimeError(
            f"Sandbox {sandbox_name!r} already exists with different runtime "
            "configuration. Use a distinct lifecycle id for different workspace "
            "mounts."
        )

    def _build_sandbox_config(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        ensure_dir: bool,
        workspace_config: Mapping[str, Any] | None = None,
    ) -> tuple[str, SandboxConfig]:
        image, config = self._get_sandbox_image_and_config()
        config_volumes = list(config.volumes) if config.volumes else []
        default_volumes = self._make_default_volumes(
            lifecycle_type,
            lifecycle_id,
            ensure_dir=ensure_dir,
            workspace_config=workspace_config,
        )
        config.volumes = config_volumes + default_volumes
        return image, config

    def _make_default_volumes(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        ensure_dir: bool,
        workspace_config: Mapping[str, Any] | None = None,
    ) -> list[tuple[str, str, str]]:
        """
        Build default volume mounts.

        Code directories are always mounted read-only.
        User workspace is additionally mounted read-write for user lifecycle type.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id
            ensure_dir: When True, create the host directory
            workspace_config: Actual tool workspace configuration, when known
        """
        # Code mounts are always present (at least src/)
        volumes: list[tuple[str, str, str]] = list(build_code_mount_volumes())
        path_mapper = SandboxPathMapper.from_env()

        # Mount actual workspace roots as read-write.
        for backend_path, should_create in self._workspace_mount_paths(
            lifecycle_type,
            lifecycle_id,
            workspace_config,
        ):
            if ensure_dir:
                try:
                    if should_create or backend_path.exists():
                        os.makedirs(backend_path, exist_ok=True)
                except OSError as exc:
                    logger.warning(
                        "Failed to prepare sandbox workspace mount %s: %s",
                        backend_path,
                        exc,
                    )

            self._append_unique_volume(
                volumes, path_mapper.volume_for_backend_path(backend_path, "rw")
            )

        return volumes

    async def get_or_create_sandbox(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        workspace_config: Mapping[str, Any] | None = None,
    ) -> Sandbox:
        """
        Get or create a sandbox.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id
            workspace_config: Actual tool workspace configuration to mount

        Returns:
            Sandbox instance
        """
        sandbox_name = self.make_sandbox_name(lifecycle_type, lifecycle_id)
        async with self._activity_guard:
            self._touch_locked(self._base_sandbox_name(lifecycle_type, lifecycle_id))

        image, desired_config = self._build_sandbox_config(
            lifecycle_type,
            lifecycle_id,
            ensure_dir=False,
            workspace_config=workspace_config,
        )

        cached_config = self._config_cache.get(sandbox_name)
        if sandbox_name in self._cache:
            self._ensure_config_equivalent(sandbox_name, cached_config, desired_config)
            return self._cache[sandbox_name]

        # Acquire per-name lock to prevent concurrent creation
        async with self._locks_guard:
            if sandbox_name not in self._locks:
                self._locks[sandbox_name] = asyncio.Lock()
            lock = self._locks[sandbox_name]

        async with lock:
            # Double-check after acquiring lock
            cached_config = self._config_cache.get(sandbox_name)
            if sandbox_name in self._cache:
                self._ensure_config_equivalent(
                    sandbox_name, cached_config, desired_config
                )
                return self._cache[sandbox_name]

            # Get base image and config from environment variables
            image, config = self._build_sandbox_config(
                lifecycle_type,
                lifecycle_id,
                ensure_dir=True,
                workspace_config=workspace_config,
            )
            logger.info(
                "Getting/creating sandbox: image=%r, cpus=%r, memory=%r, volumes=%r, env_count=%r",
                image,
                config.cpus,
                config.memory,
                config.volumes,
                len(config.env or {}),
            )

            template = SandboxTemplate(type="image", image=image)

            logger.debug(f"Getting or creating sandbox for: {sandbox_name}")
            cap = get_sandbox_max_containers()
            if cap is None:
                sandbox = await self._service.get_or_create(
                    sandbox_name,
                    template=template,
                    config=config,
                )
            else:
                async with self._capacity_gate:
                    await self._ensure_capacity_for(sandbox_name, cap)
                    sandbox = await self._service.get_or_create(
                        sandbox_name,
                        template=template,
                        config=config,
                    )

            self._cache[sandbox_name] = sandbox
            self._config_cache[sandbox_name] = config
            return sandbox

    async def create_lease_provider(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        workspace_config: Mapping[str, Any] | None = None,
    ) -> SandboxLeaseProvider:
        """Create a lease provider for primary and worker sandboxes."""
        primary = await self.get_or_create_sandbox(
            lifecycle_type,
            lifecycle_id,
            workspace_config=workspace_config,
        )
        return SandboxLeaseProvider(
            manager=self,
            lifecycle_type=lifecycle_type,
            lifecycle_id=lifecycle_id,
            primary_sandbox=primary,
            workspace_config=workspace_config,
            max_concurrency=get_sandbox_max_concurrency(),
        )

    async def _list_managed_sandbox_names(self) -> set[str]:
        """Names of existing managed containers (warmup/unparsable excluded)."""
        names: set[str] = set()
        listed_sandboxes = await self._service.list_sandboxes()
        for sb in listed_sandboxes or []:
            if not isinstance(sb.name, str):
                continue
            try:
                self.parse_sandbox_name(sb.name)
            except ValueError:
                continue
            names.add(sb.name)
        return names

    async def _pick_eviction_victim(
        self, existing: set[str], protected_base: str, skip: set[str]
    ) -> Optional[str]:
        """Pick the LRU idle primary from ``existing`` (no claim).

        Skips primaries with active tasks, the protected key, and keys whose
        lifecycle lock is currently held or awaited (in-flight creation or
        release-to-zero cleanup).
        """
        async with self._activity_guard:
            candidates: list[tuple[float, str]] = []
            for name in existing:
                try:
                    lifecycle_type, lifecycle_id = self.parse_sandbox_name(name)
                except ValueError:
                    continue
                base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
                if name != base_name:
                    # Workers are deleted with their primary.
                    continue
                if base_name == protected_base or base_name in skip:
                    continue
                lock_entry = self._lifecycle_locks.get(base_name)
                if lock_entry is not None and (
                    lock_entry.lock.locked() or lock_entry.waiters > 0
                ):
                    continue
                activity = self._activity.get(base_name)
                if activity is not None and activity.ref_count > 0:
                    continue
                last_activity = (
                    activity.last_activity
                    if activity is not None
                    else self._startup_monotonic
                )
                candidates.append((last_activity, base_name))

            if not candidates:
                return None
            return min(candidates)[1]

    async def _claim_idle_sandbox(self, base_name: str) -> bool:
        """Atomically claim an idle lifecycle for deletion.

        Under the activity guard: re-validates that no task is attached,
        then drops the lease provider (new attaches fail) and the cached
        sandbox/config instances for the primary and its workers. Purging
        the instance cache is what makes eviction safe against a concurrent
        same-key re-creation: with the cache empty, ``get_or_create_sandbox``
        cannot short-circuit and hand out the doomed container — it falls
        through to the capacity gate and recreates only after the deletion
        has finished.

        Returns False when the lifecycle became active since selection.
        """
        worker_prefix = base_name + _WORKER_LIFECYCLE_MARKER
        async with self._activity_guard:
            activity = self._activity.get(base_name)
            if activity is not None and activity.ref_count > 0:
                return False
            self._lease_providers.pop(base_name, None)
            for name in [
                n for n in self._cache if n == base_name or n.startswith(worker_prefix)
            ]:
                self._cache.pop(name, None)
                self._config_cache.pop(name, None)
            return True

    async def _evict_idle_sandbox(self, base_name: str, *, reason: str) -> bool:
        """Claim and delete one idle primary together with its workers.

        Shared primitive for the idle sweep and capacity eviction. The
        caller must hold the context that excludes a concurrent same-key
        re-creation from completing against the old container: the sweep
        holds the victim's per-key lifecycle lock; capacity eviction holds
        the global capacity gate (which every post-claim re-creation must
        pass through, because the claim purged the instance cache).

        Returns False when the lifecycle became active and must be spared.
        """
        if not await self._claim_idle_sandbox(base_name):
            return False

        logger.info("Reclaiming idle sandbox %s (%s)", base_name, reason)
        lifecycle_type, lifecycle_id = self.parse_sandbox_name(base_name)
        await self.delete_sandbox(lifecycle_type, lifecycle_id)
        return True

    async def _ensure_capacity_for(self, sandbox_name: str, cap: int) -> None:
        """Make room under the container cap for one new sandbox.

        Caller must hold ``_capacity_gate``. Evicts LRU idle primaries (with
        their workers) until the new container fits; raises
        ``SandboxCapacityError`` when nothing is evictable. If listing the
        service fails, enforcement is skipped for this creation (fail-open:
        the daemon being unreachable will fail the creation itself anyway).
        """
        try:
            existing = await self._list_managed_sandbox_names()
        except Exception as exc:
            logger.warning(
                "Failed to list sandboxes for capacity check; "
                "skipping enforcement for %s: %s",
                sandbox_name,
                exc,
            )
            return

        if sandbox_name in existing:
            return

        lifecycle_type, lifecycle_id = self.parse_sandbox_name(sandbox_name)
        protected_base = self._base_sandbox_name(lifecycle_type, lifecycle_id)

        # Victims whose deletion was already attempted this pass: a failed
        # delete leaves the container listed and would otherwise be re-picked
        # forever.
        tried_victims: set[str] = set()
        while len(existing) >= cap:
            victim = await self._pick_eviction_victim(
                existing, protected_base, tried_victims
            )
            if victim is None:
                raise SandboxCapacityError(cap=cap, in_use=len(existing))

            if not await self._evict_idle_sandbox(
                victim, reason=f"LRU eviction under container cap {cap}"
            ):
                # Became active between selection and claim; the picker's
                # ref-count check will exclude it on the next round.
                continue
            tried_victims.add(victim)

            try:
                existing = await self._list_managed_sandbox_names()
            except Exception as exc:
                logger.warning(
                    "Failed to re-list sandboxes after eviction; "
                    "skipping further enforcement for %s: %s",
                    sandbox_name,
                    exc,
                )
                return

    async def get_or_create_lease_provider(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        workspace_config: Mapping[str, Any] | None = None,
    ) -> SandboxLeaseProvider:
        """Get the cached lease provider for a lifecycle key or create one.

        Creation is serialized per key with release-to-zero cleanup, so a new
        provider can never create worker sandboxes while an old provider's
        workers are still being deleted.
        """
        base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
        async with self._lifecycle_locked(base_name):
            async with self._activity_guard:
                provider = self._lease_providers.get(base_name)
                if provider is not None:
                    self._touch_locked(base_name)
                    return provider

            provider = await self.create_lease_provider(
                lifecycle_type,
                lifecycle_id,
                workspace_config=workspace_config,
            )

            async with self._activity_guard:
                self._lease_providers[base_name] = provider
                self._touch_locked(base_name)
            return provider

    async def delete_sandbox(self, lifecycle_type: str, lifecycle_id: str) -> None:
        """
        Delete sandbox.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id
        """
        sandbox_names = await self._find_lifecycle_sandbox_names(
            lifecycle_type,
            lifecycle_id,
            include_primary=True,
            include_workers=True,
        )
        await self._delete_sandbox_names(sandbox_names)

    async def delete_worker_sandboxes(
        self, lifecycle_type: str, lifecycle_id: str
    ) -> None:
        """Delete worker sandboxes for a lifecycle while preserving the primary."""
        sandbox_names = await self._find_lifecycle_sandbox_names(
            lifecycle_type,
            lifecycle_id,
            include_primary=False,
            include_workers=True,
        )
        await self._delete_sandbox_names(sandbox_names)

    async def _find_lifecycle_sandbox_names(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        include_primary: bool,
        include_workers: bool,
    ) -> set[str]:
        sandbox_name = self.make_sandbox_name(lifecycle_type, lifecycle_id)
        worker_prefix = self._worker_sandbox_prefix(lifecycle_type, lifecycle_id)
        sandbox_names = {
            name
            for name in self._cache
            if (include_primary and name == sandbox_name)
            or (include_workers and name.startswith(worker_prefix))
        }
        if include_primary:
            sandbox_names.add(sandbox_name)

        try:
            listed_sandboxes = await self._service.list_sandboxes()
        except Exception as exc:
            logger.warning("Failed to list sandboxes for cleanup: %s", exc)
            return sandbox_names

        for sb in listed_sandboxes or []:
            name = sb.name
            if not isinstance(name, str):
                continue
            if include_primary and name == sandbox_name:
                sandbox_names.add(name)
            elif include_workers and name.startswith(worker_prefix):
                sandbox_names.add(name)

        return sandbox_names

    async def _delete_sandbox_names(self, sandbox_names: set[str]) -> None:
        for name in sorted(sandbox_names):
            try:
                await self._service.delete(name)
                logger.debug(f"Sandbox deleted: {name}")
            except Exception as e:
                logger.error(f"Failed to delete sandbox {name}: {e}")
            finally:
                # Always evict from cache — even on failure the instance
                # may be in an unknown state and should be recreated.
                self._cache.pop(name, None)
                self._config_cache.pop(name, None)
                self._locks.pop(name, None)
                # Only primary names appear in these maps; worker names no-op.
                # Plain pops on purpose: each is a single synchronous
                # operation with no compound invariant, and awaiting
                # ``_activity_guard`` inside ``finally`` would risk skipping
                # the eviction entirely if the task were cancelled at that
                # await point.
                self._lease_providers.pop(name, None)
                self._activity.pop(name, None)

    async def sweep_idle_sandboxes(self, idle_ttl: float) -> list[str]:
        """Delete sandboxes with no attached tasks that are idle past the TTL.

        Candidates come from both the in-memory activity map and the sandbox
        service listing, so containers surviving a backend restart are also
        reclaimed: with no recorded activity they report idle since manager
        startup and get one TTL grace period.

        Each deletion re-checks ref-count and idle time under the per-key
        lifecycle lock, and the eviction decision plus provider removal are
        atomic under the activity guard, so a sweep can never delete a
        sandbox a task is concurrently attaching or recreating. Workspace data lives on bind
        mounts and survives; the next use recreates the sandbox.

        Args:
            idle_ttl: Idle threshold in seconds (> 0).

        Returns:
            Primary sandbox names that were reclaimed.
        """
        try:
            listed_sandboxes = await self._service.list_sandboxes()
        except Exception as exc:
            logger.warning("Failed to list sandboxes for idle sweep: %s", exc)
            listed_sandboxes = []

        # Only keys with an existing container (or cached instance) are
        # candidates; activity entries alone have nothing left to reclaim.
        candidates: set[str] = set()
        listed_names = [
            sb.name for sb in listed_sandboxes or [] if isinstance(sb.name, str)
        ]
        for name in [*listed_names, *self._cache]:
            try:
                lifecycle_type, lifecycle_id = self.parse_sandbox_name(name)
            except ValueError:
                continue
            candidates.add(self._base_sandbox_name(lifecycle_type, lifecycle_id))

        reclaimed: list[str] = []
        for base_name in sorted(candidates):
            try:
                lifecycle_type, lifecycle_id = self.parse_sandbox_name(base_name)
            except ValueError:
                continue

            async with self._lifecycle_locked(base_name):
                idle_for = time.monotonic() - self.last_activity_at(
                    lifecycle_type, lifecycle_id
                )
                if idle_for <= idle_ttl:
                    continue

                # _evict_idle_sandbox re-validates the ref-count and drops
                # the provider in one atomic step under the activity guard:
                # an attach can never land between the check and the pop
                # that makes attaches fail.
                if await self._evict_idle_sandbox(
                    base_name,
                    reason=f"idle for {idle_for:.0f}s, TTL {idle_ttl:.0f}s",
                ):
                    reclaimed.append(base_name)

        return reclaimed

    async def run_idle_sweep_loop(self) -> None:
        """Periodically reclaim idle sandboxes until cancelled.

        Reads XAGENT_SANDBOX_IDLE_TTL / XAGENT_SANDBOX_SWEEP_INTERVAL; when
        no TTL is configured the loop exits immediately and behavior is
        identical to deployments without idle reclamation.
        """
        idle_ttl = get_sandbox_idle_ttl()
        if idle_ttl is None:
            logger.debug("Sandbox idle reclamation disabled (no TTL configured)")
            return

        sweep_interval = get_sandbox_sweep_interval()
        logger.info(
            "Sandbox idle reclamation enabled: TTL %.0fs, sweep interval %.0fs",
            idle_ttl,
            sweep_interval,
        )
        while True:
            await asyncio.sleep(sweep_interval)
            try:
                reclaimed = await self.sweep_idle_sandboxes(idle_ttl)
                if reclaimed:
                    logger.info(
                        "Idle sweep reclaimed %d sandbox(es): %s",
                        len(reclaimed),
                        ", ".join(reclaimed),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Idle sandbox sweep failed: %s", exc)

    async def warmup(self) -> None:
        """
        Warmup default image.
        Uses empty config for warmup to avoid unnecessary volume mounts.
        """
        image = get_sandbox_image()
        warmup_name = "__warmup__"
        try:
            template = SandboxTemplate(type="image", image=image)
            # Use empty config for warmup - no need for volumes/env
            warmup_config = SandboxConfig()
            async with await self._service.get_or_create(
                warmup_name, template=template, config=warmup_config
            ):
                pass
            await self._service.delete(warmup_name)
            logger.info(f"Sandbox image warmup completed: {image}")
        except Exception as e:
            logger.error(f"Failed to warmup sandbox image: {e}")

    async def cleanup(self) -> None:
        """Stop all running sandboxes.

        Delete sandboxes whose config (image, cpus, memory, volumes)
        differs from the current environment so they get recreated
        with the correct settings next time.

        Note:
            If ``get_uploads_dir()`` (via ``XAGENT_UPLOADS_DIR`` env var) changes
            between deployments, all user sandboxes will be detected as
            having stale volume mounts and will be deleted for recreation.
        """
        try:
            sandboxes = await self._service.list_sandboxes()
            if not sandboxes:
                logger.info("No sandboxes to clean up")
                return

            image, config = self._get_sandbox_image_and_config()

            for sb in sandboxes:
                try:
                    lifecycle_type, lifecycle_id = None, None
                    try:
                        lifecycle_type, lifecycle_id = self.parse_sandbox_name(sb.name)
                    except ValueError:
                        # Not a normal managed sandbox name, stop
                        if sb.state == "running":
                            box = await self._service.get_or_create(
                                sb.name, template=sb.template, config=sb.config
                            )
                            await box.stop()
                            logger.debug(f"Stopped sandbox: {sb.name}")
                        continue

                    # Delete sandbox if config changed (force recreate on next start)
                    image_changed = sb.template.image != image
                    cpus_changed = sb.config.cpus != config.cpus
                    memory_changed = sb.config.memory != config.memory

                    # volumes comparison: None and empty list are treated as equal, ignore order
                    old_volumes = sb.config.volumes or []

                    default_volumes = self._make_default_volumes(
                        lifecycle_type, lifecycle_id, ensure_dir=False
                    )
                    config_volumes = list(config.volumes) if config.volumes else []
                    # Merge volumes
                    new_volumes = config_volumes + default_volumes

                    volumes_changed = set(old_volumes) != set(new_volumes)

                    # env comparison: None and empty dict are treated as equal
                    old_env = sb.config.env or {}
                    new_env = config.env or {}
                    env_changed = old_env != new_env

                    if (
                        image_changed
                        or cpus_changed
                        or memory_changed
                        or volumes_changed
                        or env_changed
                    ):
                        changes = []
                        if image_changed:
                            changes.append(f"image: {sb.template.image} -> {image}")
                        if cpus_changed:
                            changes.append(f"cpus: {sb.config.cpus} -> {config.cpus}")
                        if memory_changed:
                            changes.append(
                                f"memory: {sb.config.memory} -> {config.memory}"
                            )
                        if env_changed:
                            old_env_str = (
                                ";".join([f"{k}={v}" for k, v in old_env.items()])
                                if old_env
                                else "none"
                            )
                            new_env_str = (
                                ";".join([f"{k}={v}" for k, v in new_env.items()])
                                if new_env
                                else "none"
                            )
                            changes.append(f"env: {old_env_str} -> {new_env_str}")
                        if volumes_changed:
                            old_vol_str = (
                                ";".join([f"{h}:{g}:{m}" for h, g, m in old_volumes])
                                if old_volumes
                                else "none"
                            )
                            new_vol_str = (
                                ";".join([f"{h}:{g}:{m}" for h, g, m in new_volumes])
                                if new_volumes
                                else "none"
                            )
                            changes.append(f"volumes: {old_vol_str} -> {new_vol_str}")
                        logger.info(
                            f"Config changed for sandbox [{sb.name}]: "
                            f"{', '.join(changes)}, deleting"
                        )
                        await self._service.delete(sb.name)
                        continue

                    # Stop running sandboxes with matching image
                    if sb.state == "running":
                        box = await self._service.get_or_create(
                            sb.name, template=sb.template, config=sb.config
                        )
                        await box.stop()
                        logger.debug(f"Stopped sandbox: {sb.name}")
                except Exception as e:
                    logger.error(f"Failed to handle sandbox {sb.name}: {e}")

            self._cache.clear()
            self._config_cache.clear()
            self._locks.clear()
            self._lease_providers.clear()
            self._activity.clear()
            logger.info("Sandbox cleanup completed")
        except Exception as e:
            logger.error(f"Failed to cleanup sandboxes: {e}")


# Global sandbox manager instance
_sandbox_manager: Optional[SandboxManager] = None
_sandbox_manager_lock = threading.Lock()
_sandbox_manager_initialized = False


def _create_sandbox_service() -> Optional[SandboxService]:
    """
    Create sandbox service based on environment configuration.

    Environment variables:
    - SANDBOX_ENABLED: Enable/disable sandbox (default: true)
    - SANDBOX_IMPLEMENTATION: Implementation type (default: docker)
      - docker: Use Docker sandbox
      - boxlite: Use Boxlite sandbox
    - BOXLITE_HOME_DIR: Boxlite home directory (optional)

    Returns:
        SandboxService instance or None if disabled
    """
    # Check if sandbox is enabled
    sandbox_enabled = os.getenv("SANDBOX_ENABLED", "false").lower() == "true"
    if not sandbox_enabled:
        logger.info("Sandbox is disabled via SANDBOX_ENABLED environment variable")
        return None

    # Get implementation type
    implementation = os.getenv("SANDBOX_IMPLEMENTATION", "docker")

    if implementation == "boxlite":
        return _create_boxlite_service()
    elif implementation == "docker":
        return _create_docker_service()
    else:
        logger.warning(
            f"Unknown sandbox implementation: {implementation}, falling back to docker"
        )
        return _create_docker_service()


def _create_boxlite_service() -> Optional[SandboxService]:
    """Create Boxlite sandbox service."""
    try:
        from ..sandbox import BoxliteSandboxService
    except ImportError:
        logger.error("boxlite is not installed.")
        return None

    from .sandbox_store import DBBoxliteStore

    store = DBBoxliteStore()
    # Get home directory
    home_dir = get_boxlite_home_dir()

    service = None
    try:
        service = BoxliteSandboxService(
            store=store, home_dir=None if home_dir is None else str(home_dir)
        )
        logger.info(
            f"Created Boxlite sandbox service (home_dir={home_dir or 'default'})"
        )
    except Exception as e:
        logger.error(f"Failed to create Boxlite sandbox service: {e}")

    return service


def _create_docker_service() -> Optional[SandboxService]:
    """Create Docker sandbox service."""
    try:
        from ..sandbox import DockerSandboxService
    except ImportError:
        logger.error("docker sandbox dependencies are not installed.")
        return None

    from .sandbox_store import DBDockerStore

    store = DBDockerStore()

    service = None
    try:
        service = DockerSandboxService(store=store)
        logger.info("Created Docker sandbox service")
    except Exception as e:
        logger.error(f"Failed to create Docker sandbox service: {e}")

    return service


def get_sandbox_manager() -> Optional[SandboxManager]:
    """
    Get or create global sandbox manager instance.

    Thread-safe singleton pattern with double-checked locking.

    Returns:
        SandboxManager instance or None if sandbox is disabled
    """
    global _sandbox_manager, _sandbox_manager_initialized

    # Fast path: already initialized (either successfully or service was None)
    if _sandbox_manager_initialized:
        return _sandbox_manager

    # Slow path: need to initialize
    with _sandbox_manager_lock:
        # Double-check after acquiring lock
        if _sandbox_manager_initialized:
            return _sandbox_manager

        # Get sandbox service
        service = _create_sandbox_service()
        if service is None:
            _sandbox_manager_initialized = True
            return None

        # Create sandbox manager
        _sandbox_manager = SandboxManager(service)
        _sandbox_manager_initialized = True
        logger.info("Created global sandbox manager")

        return _sandbox_manager
