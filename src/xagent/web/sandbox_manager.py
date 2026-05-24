"""
Sandbox management in application layer.
"""

import asyncio
import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from ..config import (
    get_boxlite_home_dir,
    get_sandbox_cpus,
    get_sandbox_env,
    get_sandbox_host_storage_root,
    get_sandbox_image,
    get_sandbox_memory,
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
    """
    Manages sandbox instances.
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
            paths.append((get_uploads_dir() / f"user_{lifecycle_id}", True))

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
            sandbox = await self._service.get_or_create(
                sandbox_name,
                template=template,
                config=config,
            )

            self._cache[sandbox_name] = sandbox
            self._config_cache[sandbox_name] = config
            return sandbox

    async def delete_sandbox(self, lifecycle_type: str, lifecycle_id: str) -> None:
        """
        Delete sandbox.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id
        """
        sandbox_name = self.make_sandbox_name(lifecycle_type, lifecycle_id)
        try:
            await self._service.delete(sandbox_name)
            logger.debug(f"Sandbox deleted: {sandbox_name}")
        except Exception as e:
            logger.error(f"Failed to delete sandbox {sandbox_name}: {e}")
        finally:
            # Always evict from cache — even on failure the instance
            # may be in an unknown state and should be recreated.
            self._cache.pop(sandbox_name, None)
            self._config_cache.pop(sandbox_name, None)
            self._locks.pop(sandbox_name, None)

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
