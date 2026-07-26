"""
Docker sandbox implementation.
"""

from __future__ import annotations

import abc
import asyncio
import io
import logging
import os
import posixpath
import re
import shutil
import tarfile
import tempfile
import textwrap
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from hashlib import sha1
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, cast

from docker.errors import APIError, ImageNotFound, NotFound

import docker

from ..config import get_sandbox_image
from .base import (
    SPEC_CONTRACT_VERSION,
    CodeType,
    ExecResult,
    ObservedRuntimeFacts,
    ResolvedSandboxRuntimeSpec,
    Sandbox,
    SandboxAlreadyExistsError,
    SandboxConfig,
    SandboxInfo,
    SandboxInspection,
    SandboxNotFoundError,
    SandboxRuntimeConflictError,
    SandboxService,
    SandboxSnapshot,
    SandboxTemplate,
)

if TYPE_CHECKING:
    from docker.models.containers import Container

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = get_sandbox_image()

LABEL_MANAGED = "xagent.managed"
LABEL_SANDBOX_NAME = "xagent.sandbox.name"
LABEL_TEMPLATE_TYPE = "xagent.sandbox.template.type"
LABEL_SNAPSHOT_ID = "xagent.sandbox.snapshot_id"
# Written only by create() (the new explicit lifecycle API), never by the
# legacy get_or_create() path: their presence is the attestation that
# spec_matches_inspection() keys off of. Immutable once written.
LABEL_SPEC_FINGERPRINT = "xagent.sandbox.spec.fingerprint"
LABEL_SPEC_VERSION = "xagent.sandbox.spec.version"
CONTAINER_NAME_PREFIX = "xagent_sandbox_"
SNAPSHOT_REPOSITORY = "xagent-sandbox-snapshot"
_CPU_NANOS = 1_000_000_000


class DockerStore(abc.ABC):
    """Store for persisting Docker sandbox metadata."""

    @abc.abstractmethod
    def get_info(self, name: str) -> Optional[SandboxInfo]:
        """Get sandbox info."""

    @abc.abstractmethod
    def add_info(self, name: str, info: SandboxInfo) -> None:
        """Add sandbox info."""

    @abc.abstractmethod
    def update_info_state(self, name: str, state: str) -> None:
        """Update sandbox state."""

    @abc.abstractmethod
    def delete_info(self, name: str) -> None:
        """Delete sandbox info."""

    @abc.abstractmethod
    def get_snapshot(self, snapshot_id: str) -> Optional[SandboxSnapshot]:
        """Get snapshot info."""

    @abc.abstractmethod
    def add_snapshot(self, snapshot: SandboxSnapshot) -> None:
        """Add snapshot info."""

    @abc.abstractmethod
    def list_snapshots(self) -> list[SandboxSnapshot]:
        """List snapshot info."""

    @abc.abstractmethod
    def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete snapshot info."""


class MemDockerStore(DockerStore):
    """In-memory implementation of DockerStore."""

    def __init__(self) -> None:
        self._metadata: dict[str, SandboxInfo] = {}
        self._snapshots: dict[str, SandboxSnapshot] = {}

    def get_info(self, name: str) -> Optional[SandboxInfo]:
        return self._metadata.get(name)

    def add_info(self, name: str, info: SandboxInfo) -> None:
        self._metadata[name] = info

    def update_info_state(self, name: str, state: str) -> None:
        if name in self._metadata:
            self._metadata[name].state = state

    def delete_info(self, name: str) -> None:
        self._metadata.pop(name, None)

    def get_snapshot(self, snapshot_id: str) -> Optional[SandboxSnapshot]:
        return self._snapshots.get(snapshot_id)

    def add_snapshot(self, snapshot: SandboxSnapshot) -> None:
        self._snapshots[snapshot.snapshot_id] = snapshot

    def list_snapshots(self) -> list[SandboxSnapshot]:
        return list(self._snapshots.values())

    def delete_snapshot(self, snapshot_id: str) -> None:
        self._snapshots.pop(snapshot_id, None)


def _create_docker_client() -> Any:
    """Create a Docker SDK client using the standard Docker environment config.

    The Docker SDK can also talk to Docker-compatible runtimes such as Podman
    when ``DOCKER_HOST`` points at a compatible socket/service.
    """
    return cast(Any, docker.from_env())


def is_docker_available() -> bool:
    """Return whether Docker is reachable."""
    try:
        client = _create_docker_client()
        client.ping()
    except Exception as e:
        logger.exception(
            "No Docker-compatible runtime API is reachable. "
            "For Podman or other non-default runtimes, start the service/socket and "
            "set DOCKER_HOST to the compatible endpoint. error=%s",
            e,
        )
        return False
    return True


def _make_safe_name(name: str) -> str:
    """Convert an arbitrary sandbox identifier into a Docker-safe name."""
    # Convert to safe name
    base = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-.") or "sandbox"
    # Add a sha1 suffix to prevent duplicate names
    digest = sha1(name.encode("utf-8")).hexdigest()[:10]
    return f"{base.lower()}-{digest}"


def _container_name(name: str) -> str:
    """Build the managed Docker container name for a sandbox."""
    return f"{CONTAINER_NAME_PREFIX}{_make_safe_name(name)}"


def _snapshot_tag(snapshot_id: str) -> str:
    """Build the managed Docker image tag for a snapshot."""
    safe = _make_safe_name(snapshot_id)
    return f"{SNAPSHOT_REPOSITORY}:{safe}"


def _get_state(status: str | None) -> str:
    """Map Docker container status to the sandbox state model."""
    if not status:
        return "unknown"
    lowered = status.lower()
    if lowered == "running":
        return "running"
    if lowered in {"created", "exited", "paused", "dead", "restarting"}:
        return "stopped"
    return "unknown"


def _parse_container_config(container: Container) -> SandboxInfo:
    """Reconstruct SandboxInfo from Docker inspect data."""
    attrs = cast(dict[str, Any], container.attrs)
    config_data = cast(dict[str, Any], attrs.get("Config") or {})
    host_config = cast(dict[str, Any], attrs.get("HostConfig") or {})
    state = cast(dict[str, Any], attrs.get("State") or {})

    env_map: dict[str, str] = {}
    # Docker stores env vars as ["KEY=value", ...]
    for item in cast(list[str], config_data.get("Env") or []):
        if "=" in item:
            key, value = item.split("=", 1)
            env_map[key] = value

    volumes: list[tuple[str, str, str]] = []
    # Only bind mounts
    for mount in cast(list[dict[str, Any]], attrs.get("Mounts") or []):
        if mount.get("Type") != "bind":
            continue
        source = str(mount.get("Source") or "")
        target = str(mount.get("Destination") or "")
        mode = "ro" if bool(mount.get("RW")) is False else "rw"
        if source and target:
            volumes.append((source, target, mode))

    ports: list[tuple[int, int]] = []
    port_bindings = cast(
        dict[str, list[dict[str, str]]], host_config.get("PortBindings") or {}
    )
    for guest_port, host_bindings in port_bindings.items():
        container_port = int(str(guest_port).split("/", 1)[0])
        for binding in host_bindings or []:
            host_port = binding.get("HostPort")
            if host_port:
                ports.append((int(host_port), container_port))

    nano_cpus = int(host_config.get("NanoCpus") or 0)
    cpus = nano_cpus // _CPU_NANOS if nano_cpus else 1
    memory_bytes = int(host_config.get("Memory") or 0)
    memory = memory_bytes // (1024 * 1024) if memory_bytes else 512

    labels = container.labels
    template_type = labels.get(LABEL_TEMPLATE_TYPE, "image")
    if template_type == "snapshot" and labels.get(LABEL_SNAPSHOT_ID):
        template = SandboxTemplate(
            type="snapshot", snapshot_id=labels[LABEL_SNAPSHOT_ID]
        )
    else:
        template = SandboxTemplate(
            type="image", image=str(config_data.get("Image") or "")
        )
    config = SandboxConfig(
        working_dir=str(config_data.get("WorkingDir") or "/home"),
        cpus=max(1, cpus),
        memory=max(128, memory),
        env=env_map or None,
        volumes=volumes or None,
        network_isolated=bool(
            attrs.get("NetworkSettings", {}).get("Networks") == {}
            or host_config.get("NetworkMode") == "none"
        ),
        ports=ports or None,
    )
    return SandboxInfo(
        name=str(labels.get(LABEL_SANDBOX_NAME, container.name)),
        state=_get_state(str(state.get("Status"))),
        template=template,
        config=config,
        created_at=str(attrs.get("Created") or ""),
    )


def _merge_info(
    runtime_info: SandboxInfo, stored_info: Optional[SandboxInfo]
) -> SandboxInfo:
    """Merge runtime info and stored info."""
    if stored_info is None:
        return runtime_info
    return SandboxInfo(
        name=stored_info.name,
        state=runtime_info.state,
        template=stored_info.template,
        config=stored_info.config,
        created_at=runtime_info.created_at,
    )


def _build_inspection(container: Container) -> SandboxInspection:
    """Build a point-in-time SandboxInspection directly from Docker inspect data.

    Unlike ``_parse_container_config``, this keeps raw backend units
    (``HostConfig.NanoCpus`` / ``HostConfig.Memory``) rather than the
    divided-and-clamped ``SandboxConfig`` values, so a live edit such as
    ``docker update --cpus 0.5`` remains observable in the returned facts.
    Shared by ``inspect()`` (no side effects, no lock held) and ``create()``'s
    publish-before-verify step; the caller is responsible for reloading the
    container beforehand so ``container.attrs`` reflects current state.
    """
    attrs = cast(dict[str, Any], container.attrs)
    config_data = cast(dict[str, Any], attrs.get("Config") or {})
    host_config = cast(dict[str, Any], attrs.get("HostConfig") or {})
    state = cast(dict[str, Any], attrs.get("State") or {})

    env_map: dict[str, str] = {}
    for item in cast(list[str], config_data.get("Env") or []):
        if "=" in item:
            key, value = item.split("=", 1)
            env_map[key] = value

    volumes: list[tuple[str, str, str]] = []
    for mount in cast(list[dict[str, Any]], attrs.get("Mounts") or []):
        if mount.get("Type") != "bind":
            continue
        source = str(mount.get("Source") or "")
        target = str(mount.get("Destination") or "")
        mode = "ro" if bool(mount.get("RW")) is False else "rw"
        if source and target:
            volumes.append((source, target, mode))

    ports: list[tuple[int, int]] = []
    port_bindings = cast(
        dict[str, list[dict[str, str]]], host_config.get("PortBindings") or {}
    )
    for guest_port, host_bindings in port_bindings.items():
        container_port = int(str(guest_port).split("/", 1)[0])
        for binding in host_bindings or []:
            host_port = binding.get("HostPort")
            if host_port:
                ports.append((int(host_port), container_port))

    labels = dict(container.labels)
    raw_status = str(state.get("Status") or "")
    network_settings = cast(dict[str, Any], attrs.get("NetworkSettings") or {})
    runtime_networks = tuple(
        cast(dict[str, Any], network_settings.get("Networks") or {})
    )

    facts = ObservedRuntimeFacts(
        raw_status=raw_status,
        image_ref=cast(Optional[str], config_data.get("Image")),
        image_digest=cast(Optional[str], attrs.get("Image")),
        raw_nano_cpus=cast(Optional[int], host_config.get("NanoCpus")),
        raw_memory_bytes=cast(Optional[int], host_config.get("Memory")),
        env=env_map,
        volumes=tuple(volumes),
        ports=tuple(ports),
        network_isolated=bool(config_data.get("NetworkDisabled")),
        runtime_networks=runtime_networks,
        labels=labels,
        created_at=cast(Optional[str], attrs.get("Created")),
        working_dir=cast(Optional[str], config_data.get("WorkingDir")),
    )
    return SandboxInspection(
        state="running" if _get_state(raw_status) == "running" else "stopped",
        facts=facts,
        fingerprint_label=labels.get(LABEL_SPEC_FINGERPRINT),
        version_label=labels.get(LABEL_SPEC_VERSION),
    )


def _check_no_conflicting_volumes(
    volumes: Optional[list[tuple[str, str, str]]],
) -> None:
    """Reject desired volumes that share a host path but disagree downstream.

    ``_create_container`` builds its Docker ``volumes`` dict keyed by host
    path, so two entries with the same host path but a different guest path
    or mode would silently drop one of them. Exactly identical triples
    (duplicates) are accepted and simply collapse; this only rejects a real
    disagreement, normalizing paths first so equivalent-but-differently-
    spelled host paths are treated as the same key.
    """
    if not volumes:
        return
    seen: dict[str, tuple[str, str]] = {}
    for host_path, guest_path, mode in volumes:
        key = (posixpath.normpath(guest_path), mode)
        normalized_host = posixpath.normpath(host_path)
        prior = seen.get(normalized_host)
        if prior is not None and prior != key:
            raise SandboxRuntimeConflictError(
                f"Conflicting desired volume mounts for host path "
                f"{normalized_host!r}: {prior} vs {key}"
            )
        seen[normalized_host] = key


def _check_no_conflicting_ports(
    ports: Optional[list[tuple[int, int]]],
) -> None:
    """Reject desired ports that share a guest port but disagree on host port.

    ``_create_container`` builds its Docker ``ports`` dict keyed by guest
    port, so two entries with the same guest port but a different host port
    would silently drop one of them. Exactly identical pairs (duplicates)
    are accepted and simply collapse; this only rejects a real disagreement.
    """
    if not ports:
        return
    seen: dict[int, int] = {}
    for host_port, guest_port in ports:
        prior = seen.get(guest_port)
        if prior is not None and prior != host_port:
            raise SandboxRuntimeConflictError(
                f"Conflicting desired port mappings for guest port "
                f"{guest_port}: host {prior} vs {host_port}"
            )
        seen[guest_port] = host_port


def _find_publish_mismatches(
    desired: ResolvedSandboxRuntimeSpec,
    resolved_image: str,
    inspection: SandboxInspection,
) -> list[str]:
    """Return the field names whose observed value disagrees with ``desired``.

    Used only by create()'s publish-before-verify step. Returns field names
    only (never the actual values) since this feeds directly into a raised
    error message and desired/observed values may carry sensitive paths.

    The image check applies identically to both template types: for a
    snapshot-based create, ``resolved_image`` is the snapshot's own image
    tag, and ``facts.image_ref`` (Docker's ``Config.Image``) equals that tag
    once the container has actually been created from it, so there is no
    need for a separate label-based check on the snapshot leg.

    cpus/memory are re-checked here (immediately after start, in raw backend
    units) in addition to the live re-check ``spec_matches_inspection`` does
    later: this is the first opportunity to catch e.g. Docker silently
    clamping an out-of-range request, before the container is ever
    published.
    """
    mismatches: list[str] = []
    facts = inspection.facts

    if facts.image_ref != resolved_image:
        mismatches.append("image")
    if set(facts.volumes) != set(desired.volumes):
        mismatches.append("volumes")
    if set(facts.ports) != set(desired.ports):
        mismatches.append("ports")
    if facts.working_dir != desired.working_dir:
        mismatches.append("working_dir")
    if facts.network_isolated != desired.network_isolated:
        mismatches.append("network_isolated")
    if (facts.raw_nano_cpus or 0) != int(desired.cpus * _CPU_NANOS):
        mismatches.append("cpus")
    if (facts.raw_memory_bytes or 0) != int(desired.memory * 1024 * 1024):
        mismatches.append("memory")
    return mismatches


def _write_tar_from_local_path(
    local_path: str, arcname: str, file_obj: io.BufferedRandom
) -> None:
    """Pack a local file into a tar stream for Docker put_archive."""
    with tarfile.open(fileobj=file_obj, mode="w") as tar:
        tar.add(local_path, arcname=arcname)
    file_obj.seek(0)


def _write_tar_from_content(
    content: str, arcname: str, file_obj: io.BufferedRandom
) -> None:
    """Pack in-memory text content into a tar stream for Docker put_archive."""
    data = content.encode("utf-8")
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    with tarfile.open(fileobj=file_obj, mode="w") as tar:
        tar.addfile(info, io.BytesIO(data))
    file_obj.seek(0)


def _exec_capped(
    container: Any, cmd: list[str], env: Optional[dict[str, str]], cap: int
) -> ExecResult:
    """Exec ``cmd`` streaming stdout/stderr and keeping at most ``cap`` bytes of
    each, so the host docker client never buffers an unbounded remote flood
    (#3). ``container.exec_run(demux=True)`` reads the whole stream into host
    memory before returning; the low-level exec API streams instead, letting us
    drop everything past the cap while still draining to EOF (the caller's
    ``timeout`` bounds how long that can take). Blocking — run in a thread."""
    api = container.client.api
    exec_id = api.exec_create(
        container.id, cmd, environment=env, stdout=True, stderr=True
    )["Id"]
    out = bytearray()
    err = bytearray()
    out_truncated = False
    err_truncated = False
    for stdout_chunk, stderr_chunk in api.exec_start(exec_id, stream=True, demux=True):
        if stdout_chunk:
            room = cap - len(out)
            if room > 0:
                out.extend(stdout_chunk[:room])
            if len(stdout_chunk) > max(room, 0):
                out_truncated = True
        if stderr_chunk:
            room = cap - len(err)
            if room > 0:
                err.extend(stderr_chunk[:room])
            if len(stderr_chunk) > max(room, 0):
                err_truncated = True
    exit_code = api.exec_inspect(exec_id).get("ExitCode")
    return ExecResult(
        exit_code=exit_code if exit_code is not None else -1,
        stdout=bytes(out).decode("utf-8", errors="replace"),
        stderr=bytes(err).decode("utf-8", errors="replace"),
        truncated=out_truncated or err_truncated,
        error_message=None,
    )


def _write_stream_to_file(
    stream: Any, file_obj: io.BufferedRandom | io.BufferedWriter
) -> None:
    """Copy a streamed Docker archive into a local file object."""
    for chunk in stream:
        file_obj.write(chunk)
    file_obj.flush()
    file_obj.seek(0)


def _extract_single_file_from_tar(
    tar_file_obj: io.BufferedRandom | io.BufferedReader,
    output_file_obj: io.BufferedWriter | io.BytesIO,
) -> None:
    """Extract the first regular file from a Docker get_archive tar stream."""
    with tarfile.open(fileobj=tar_file_obj, mode="r:*") as tar:
        member = next((item for item in tar if item.isfile()), None)
        if member is None:
            raise FileNotFoundError("No file found in archive")
        fileobj = tar.extractfile(member)
        if fileobj is None:
            raise FileNotFoundError(f"Could not read file from archive: {member.name}")
        shutil.copyfileobj(fileobj, output_file_obj)
        output_file_obj.flush()


def _archive_path_exists(container: Container, remote_path: str) -> bool:
    """Check file existence."""
    try:
        container.get_archive(remote_path)
        return True
    except NotFound:
        return False


@dataclass
class _NamedLockEntry:
    """Per-sandbox-name lifecycle lock with holder/waiter tracking for safe eviction."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    waiters: int = 0


@dataclass
class _SandboxControl:
    """Shared concurrency guard for operations targeting the same sandbox."""

    name: str
    active_ops: int = 0
    new_operations_paused: bool = False
    deleted: bool = False
    file_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    exec_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def acquire_operation(self) -> None:
        """Register a new sandbox operation, blocking while new operations are paused."""
        async with self.cond:
            while self.new_operations_paused and not self.deleted:
                await self.cond.wait()
            if self.deleted:
                raise RuntimeError(f"Sandbox {self.name!r} has been deleted")
            self.active_ops += 1

    async def release_operation(self) -> None:
        """Mark a sandbox operation as finished."""
        async with self.cond:
            self.active_ops -= 1
            if self.active_ops == 0:
                self.cond.notify_all()

    @asynccontextmanager
    async def operation(self) -> AsyncIterator[None]:
        """Track a sandbox operation and always release it on cancellation."""
        await self.acquire_operation()
        try:
            yield
        finally:
            await asyncio.shield(self.release_operation())

    async def pause_new_operations(self, mark_deleted: bool) -> None:
        """Block new operations and wait for in-flight work to finish."""
        async with self.cond:
            self.new_operations_paused = True
            while self.active_ops > 0:
                await self.cond.wait()
            if mark_deleted:
                self.deleted = True

    async def resume_new_operations(self) -> None:
        """Allow new operations again after a non-destructive pause."""
        async with self.cond:
            if not self.deleted:
                self.new_operations_paused = False
            self.cond.notify_all()

    @asynccontextmanager
    async def exclusive_access(self, *, mark_deleted: bool) -> AsyncIterator[None]:
        """Block new operations and wait for exclusive lifecycle access."""
        await self.pause_new_operations(mark_deleted=mark_deleted)
        try:
            yield
        finally:
            await asyncio.shield(self.resume_new_operations())


class DockerSandbox(Sandbox):
    """Runtime sandbox implementation backed by a managed Docker container."""

    def __init__(
        self,
        sandbox_name: str,
        container: Container,
        info: SandboxInfo,
        store: DockerStore,
        control: _SandboxControl,
    ) -> None:
        self._container = container
        self._name = sandbox_name
        self._info = info
        self._store = store
        self._control = control

    @property
    def name(self) -> str:
        """Sandbox name (unique identifier)."""
        return self._name

    async def _require_container(self) -> Container:
        """Return the managed container or raise if it has been deleted."""
        try:
            await asyncio.to_thread(self._container.reload)
        except NotFound as e:
            raise SandboxNotFoundError(
                f"Sandbox container not found: {self._name}"
            ) from e
        return self._container

    async def _exec_in_container(
        self,
        command: str,
        *args: str,
        env: Optional[dict[str, str]] = None,
        max_output_bytes: Optional[int] = None,
    ) -> ExecResult:
        """Execute a command directly against the current container instance."""
        container = await self._require_container()
        cmd: list[str] = [command, *args]
        try:
            if max_output_bytes is not None:
                return await asyncio.to_thread(
                    _exec_capped, container, cmd, env, max_output_bytes
                )
            result = await asyncio.to_thread(
                container.exec_run,
                cmd,
                environment=env,
                demux=True,
                stdout=True,
                stderr=True,
            )
        except Exception as exc:
            return ExecResult(
                exit_code=1,
                stdout="",
                stderr="",
                error_message=str(exc),
            )

        output = cast(tuple[bytes | None, bytes | None] | None, result.output)
        stdout_bytes, stderr_bytes = output if output is not None else (b"", b"")
        return ExecResult(
            exit_code=cast(int, result.exit_code),
            stdout=(stdout_bytes or b"").decode("utf-8", errors="replace"),
            stderr=(stderr_bytes or b"").decode("utf-8", errors="replace"),
            error_message=None,
        )

    async def stop(self) -> None:
        """Stop the sandbox container while preserving filesystem state."""
        async with self._control.exclusive_access(mark_deleted=False):
            container = await self._require_container()
            await asyncio.to_thread(container.stop)
            self._store.update_info_state(self._name, "stopped")

    async def info(self) -> SandboxInfo:
        """Return current sandbox metadata derived from Docker inspect."""
        container = await self._require_container()

        runtime_info = _parse_container_config(container)
        self._info.state = runtime_info.state

        return self._info

    async def exec(
        self,
        command: str,
        *args: str,
        env: Optional[dict[str, str]] = None,
        max_output_bytes: Optional[int] = None,
    ) -> ExecResult:
        """Execute a shell command inside the sandbox.

        Exec calls are serialized per sandbox to avoid Docker SDK stream
        corruption when concurrent execs read from the same container socket.
        """
        async with self._control.operation():
            async with self._control.exec_lock:
                return await self._exec_in_container(
                    command, *args, env=env, max_output_bytes=max_output_bytes
                )

    async def run_code(
        self,
        code: str,
        code_type: CodeType = "python",
        env: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Execute code snippet."""
        code = textwrap.dedent(code)
        if code_type == "python":
            return await self.exec("python", "-c", code, env=env)
        elif code_type == "javascript":
            return await self.exec("node", "-e", code, env=env)
        raise ValueError(f"Unsupported code type: {code_type}")

    async def upload_file(
        self, local_path: str, remote_path: str, overwrite: bool = False
    ) -> None:
        """Upload a local file into the sandbox filesystem."""
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        async with self._control.operation():
            async with self._control.file_lock:
                # Serialize tar-based file transfers so concurrent writes do not produce partially-overwritten archives at the destination path.
                if not overwrite:
                    container = await self._require_container()
                    exists = await asyncio.to_thread(
                        _archive_path_exists, container, remote_path
                    )
                    if exists:
                        raise FileExistsError(
                            f"Remote file already exists: {remote_path}"
                        )

                remote_dir = posixpath.dirname(remote_path) or "/"
                mkdir = await self._exec_in_container("mkdir", "-p", remote_dir)
                if mkdir.exit_code != 0:
                    raise RuntimeError(f"Failed to create remote dir: {mkdir.stderr}")

                container = await self._require_container()
                with tempfile.TemporaryFile() as archive_file:
                    _write_tar_from_local_path(
                        local_path, posixpath.basename(remote_path), archive_file
                    )
                    ok = await asyncio.to_thread(
                        container.put_archive, remote_dir, archive_file
                    )
                    if not ok:
                        raise RuntimeError(f"Failed to upload file to {remote_path}")

    async def download_file(
        self, remote_path: str, local_path: str, overwrite: bool = False
    ) -> None:
        """Download a file from the sandbox to the local filesystem."""
        if not overwrite and os.path.exists(local_path):
            raise FileExistsError(f"Local file already exists: {local_path}")

        async with self._control.operation():
            async with self._control.file_lock:
                container = await self._require_container()
                try:
                    stream, _ = await asyncio.to_thread(
                        container.get_archive, remote_path
                    )
                except NotFound as e:
                    raise FileNotFoundError(
                        f"Remote file not found: {remote_path}"
                    ) from e

                local_dir = os.path.dirname(local_path)
                if local_dir:
                    os.makedirs(local_dir, exist_ok=True)
                with tempfile.TemporaryFile() as archive_file:
                    await asyncio.to_thread(_write_stream_to_file, stream, archive_file)
                    with open(local_path, "wb") as file_obj:
                        _extract_single_file_from_tar(archive_file, file_obj)

    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None:
        """Write text content directly to a file inside the sandbox."""
        async with self._control.operation():
            async with self._control.file_lock:
                if not overwrite:
                    container = await self._require_container()
                    exists = await asyncio.to_thread(
                        _archive_path_exists, container, remote_path
                    )
                    if exists:
                        raise FileExistsError(
                            f"Remote file already exists: {remote_path}"
                        )

                remote_dir = posixpath.dirname(remote_path) or "/"
                mkdir = await self._exec_in_container("mkdir", "-p", remote_dir)
                if mkdir.exit_code != 0:
                    raise RuntimeError(f"Failed to create remote dir: {mkdir.stderr}")

                container = await self._require_container()
                with tempfile.TemporaryFile() as archive_file:
                    _write_tar_from_content(
                        content, posixpath.basename(remote_path), archive_file
                    )
                    ok = await asyncio.to_thread(
                        container.put_archive, remote_dir, archive_file
                    )
                    if not ok:
                        raise RuntimeError(f"Failed to write file to {remote_path}")

    async def read_file(self, remote_path: str) -> str:
        """Read text content from a sandbox file."""
        async with self._control.operation():
            async with self._control.file_lock:
                container = await self._require_container()
                try:
                    stream, _ = await asyncio.to_thread(
                        container.get_archive, remote_path
                    )
                except NotFound as e:
                    raise FileNotFoundError(
                        f"Remote file not found: {remote_path}"
                    ) from e
                with tempfile.TemporaryFile() as archive_file:
                    await asyncio.to_thread(_write_stream_to_file, stream, archive_file)
                    with io.BytesIO() as file_bytes:
                        _extract_single_file_from_tar(archive_file, file_bytes)
                        return file_bytes.getvalue().decode("utf-8")


async def _ensure_image(client: Any, image: str) -> None:
    """Ensure the requested image exists locally before container creation."""
    try:
        await asyncio.to_thread(client.images.get, image)
    except ImageNotFound:
        logger.info("Start pulling sandbox image: %s", image)
        await asyncio.to_thread(client.images.pull, image)
        logger.info("Finish pulling sandbox image: %s", image)


async def _create_container(
    client: Any,
    name: str,
    image: str,
    template: SandboxTemplate,
    config: SandboxConfig,
    extra_labels: Optional[dict[str, str]] = None,
) -> Container:
    """Create a managed Docker container from sandbox template and config.

    ``extra_labels``, when given, is merged into the container's labels.
    With no extra labels the label set is exactly
    managed/name/template-type(/snapshot-id). This parameter exists so the
    ``create()`` lifecycle method can attach the spec fingerprint/version
    attestation labels without this shared helper (also used by the legacy
    ``get_or_create()`` path) needing to know anything about that contract.
    """
    await _ensure_image(client, image)

    volumes: dict[str, dict[str, str]] | None = None
    if config.volumes:
        volumes = {
            host_path: {"bind": guest_path, "mode": mode}
            for host_path, guest_path, mode in config.volumes
        }

    ports: dict[str, int] | None = None
    if config.ports:
        ports = {f"{guest}/tcp": host for host, guest in config.ports}

    labels = {
        LABEL_MANAGED: "true",
        LABEL_SANDBOX_NAME: name,
        LABEL_TEMPLATE_TYPE: template.type or "image",
    }
    if template.type == "snapshot" and template.snapshot_id:
        labels[LABEL_SNAPSHOT_ID] = template.snapshot_id
    if extra_labels:
        labels.update(extra_labels)

    kwargs: dict[str, Any] = {
        "image": image,
        "name": _container_name(name),
        # Keep the container alive
        "command": ["tail", "-f", "/dev/null"],
        "detach": True,
        # Run as root to match the file access behavior of Boxlite.
        "user": "root",
        "working_dir": config.working_dir,
        "environment": config.env,
        "volumes": volumes,
        "ports": ports,
        "nano_cpus": int((config.cpus or 1) * _CPU_NANOS),
        "mem_limit": (config.memory or 512) * 1024 * 1024,
        "network_disabled": bool(config.network_isolated),
        # Security config
        "security_opt": ["no-new-privileges:true"],
        "labels": labels,
    }
    return cast(
        "Container", await asyncio.to_thread(client.containers.create, **kwargs)
    )


class DockerSandboxService(SandboxService):
    """SandboxService implementation backed by Docker containers."""

    def __init__(
        self,
        store: DockerStore,
        client: Optional[Any] = None,
    ) -> None:
        """Initialize the Docker sandbox service and validate daemon access."""
        self._client = client or _create_docker_client()
        self._client.ping()
        self._store = store
        # Per-name lifecycle lock entries, one per sandbox name currently held
        # or waited on.
        self._locks: dict[str, _NamedLockEntry] = {}
        # Sandbox shared runtime control
        self._controls: dict[str, _SandboxControl] = {}

    @asynccontextmanager
    async def _named_lock(self, name: str) -> AsyncIterator[None]:
        """Serialize lifecycle operations (create/delete/snapshot) for one name.

        Entries are dropped once no holder or waiter remains, so the dict
        does not grow with every sandbox name ever seen (names such as
        ``ssh::{task_id}`` come from an unbounded namespace).

        Waiter bookkeeping is deliberately synchronous and unguarded,
        mirroring ``SandboxManager._lifecycle_locked`` in
        ``web/sandbox_manager.py``: every step here is a single dict
        get/set/pop or int increment/decrement with no ``await`` in between,
        so nothing else can interleave on this single-threaded event loop,
        and a guarding lock would only add an extra await point that a
        cancellation could land on mid-rollback. If the acquire is
        cancelled, the waiter count is rolled back in the same
        ``except BaseException`` step so a cancelled waiter never leaks
        either the count or a now-unreferenced entry.
        """
        entry = self._locks.get(name)
        if entry is None:
            entry = _NamedLockEntry()
            self._locks[name] = entry
        entry.waiters += 1

        try:
            await entry.lock.acquire()
        except BaseException:
            entry.waiters -= 1
            self._drop_named_lock_if_unused(name, entry)
            raise

        try:
            yield
        finally:
            entry.lock.release()
            entry.waiters -= 1
            self._drop_named_lock_if_unused(name, entry)

    def _drop_named_lock_if_unused(self, name: str, entry: _NamedLockEntry) -> None:
        """Evict ``name``'s lock entry once it has no holder and no waiter left.

        Called only from the synchronous bookkeeping steps in
        ``_named_lock``, with no ``await`` between the waiter-count update
        and this call, so nothing else can interleave and observe an
        inconsistent count. Identity-checked against ``entry`` so a
        concurrent waiter that already installed a fresh entry for the same
        name is never evicted out from under it.
        """
        if entry.waiters > 0:
            return
        if self._locks.get(name) is entry:
            self._locks.pop(name, None)

    def _get_control(self, name: str) -> _SandboxControl:
        """Get the shared runtime control object for a sandbox."""
        if name not in self._controls:
            self._controls[name] = _SandboxControl(name=name)
        return self._controls[name]

    def _get_live_control(self, name: str) -> _SandboxControl:
        """Return the live control object for a sandbox, replacing it if deleted.

        This is the sole construction point for a fresh ``_SandboxControl``
        used by the create path: if no control exists yet, or the existing
        one was marked deleted by a prior ``delete()``, a new one is
        installed and returned; otherwise the existing live control is
        returned as-is so that in-flight callers sharing it are not split
        across two control objects. Must only be called while holding
        ``_named_lock(name)`` — that per-name mutual exclusion is what makes
        the deleted-check-then-replace race-free. The assertion below can
        only check that *some* task holds ``name``'s lock right now, not
        that it is the caller: ``asyncio.Lock`` has no owner concept, so
        this cannot be a full runtime proof of the contract, only a guard
        against the lock not being held at all.
        """
        entry = self._locks.get(name)
        assert entry is not None and entry.lock.locked(), (
            f"_get_live_control({name!r}) called without holding _named_lock(name)"
        )
        existing = self._controls.get(name)
        if existing is None or existing.deleted:
            existing = _SandboxControl(name=name)
            self._controls[name] = existing
        return existing

    async def _find_container(self, name: str) -> Optional[Container]:
        """Find the managed Docker container for a sandbox name."""
        filters: dict[str, str | list[str] | bool] = {
            "label": [f"{LABEL_MANAGED}=true", f"{LABEL_SANDBOX_NAME}={name}"]
        }
        containers = await asyncio.to_thread(
            self._client.containers.list, all=True, filters=filters
        )
        if not containers:
            return None
        return cast("Container", containers[0])

    async def get_or_create(
        self,
        name: str,
        template: Optional[SandboxTemplate] = None,
        config: Optional[SandboxConfig] = None,
    ) -> DockerSandbox:
        """Get, resume, or create a Docker-backed sandbox."""
        async with self._named_lock(name):
            control = self._get_live_control(name)

            container = await self._find_container(name)
            if container is not None:
                await asyncio.to_thread(container.reload)
                state = _get_state(str(container.attrs.get("State", {}).get("Status")))
                if state != "running":
                    await asyncio.to_thread(container.start)
                    await asyncio.to_thread(container.reload)
                runtime_info = _parse_container_config(container)
                info = _merge_info(runtime_info, self._store.get_info(name))
                self._store.update_info_state(name, "running")
                return DockerSandbox(name, container, info, self._store, control)

            template = template or SandboxTemplate(
                type="image", image=DEFAULT_SANDBOX_IMAGE
            )
            cfg = config or SandboxConfig()
            image = template.image or DEFAULT_SANDBOX_IMAGE
            if template.type == "snapshot":
                snapshot = self._store.get_snapshot(cast(str, template.snapshot_id))
                if snapshot is None:
                    raise FileNotFoundError(
                        f"Snapshot not found: {template.snapshot_id}"
                    )
                image = cast(str, snapshot.metadata.get("image_tag"))

            container = await _create_container(
                self._client,
                name,
                image,
                template,
                cfg,
            )
            try:
                await asyncio.to_thread(container.start)
            except Exception:
                await asyncio.to_thread(container.remove, force=True)
                raise
            await asyncio.to_thread(container.reload)
            runtime_info = _parse_container_config(container)
            stored_info = SandboxInfo(
                name=name,
                state=runtime_info.state,
                template=template,
                config=cfg,
                created_at=runtime_info.created_at,
            )
            info = _merge_info(runtime_info, stored_info)
            self._store.add_info(name, info)
            return DockerSandbox(name, container, info, self._store, control)

    # --- Spec-based reconciliation lifecycle ---
    #
    # supports_runtime_spec/inspect/create/start_existing/stop_existing do
    # not touch the get_or_create()/list_sandboxes()/delete()/
    # create_snapshot() code paths above, aside from sharing
    # `_create_container`'s optional `extra_labels` parameter and the
    # `_named_lock`/`_get_live_control` infrastructure.

    async def supports_runtime_spec(self) -> bool:
        """Docker backs the explicit spec-reconciliation lifecycle."""
        return True

    async def inspect(self, name: str) -> Optional[SandboxInspection]:
        """Observe a sandbox's current state and runtime facts.

        No side effects and no lock/control acquisition: re-finds the
        container and reloads it fresh on every call rather than reusing a
        cached handle, so the only atomicity guaranteed is within this one
        call.

        When multiple containers carry the same sandbox name label (only
        reachable via out-of-band operations against the Docker daemon),
        ``_find_container`` picks whichever one the Docker API lists first;
        which container that is is undefined and callers must not depend
        on it.
        """
        container = await self._find_container(name)
        if container is None:
            return None
        await asyncio.to_thread(container.reload)
        return _build_inspection(container)

    async def create(
        self, name: str, template: SandboxTemplate, config: SandboxConfig
    ) -> DockerSandbox:
        """Create a new sandbox under the explicit, verified lifecycle contract.

        See ``SandboxService.create`` for the full eight-step contract this
        implements: existence check, snapshot resolution, volume-conflict
        validation, labeled container creation, start with raw-remove
        compensation on failure, publish-before-verify against the desired
        spec, store persistence, and returning a live handle.
        """
        async with self._named_lock(name):
            existing = await self._find_container(name)
            if existing is not None:
                raise SandboxAlreadyExistsError(f"Sandbox already exists: {name!r}")

            template_type = template.type or "image"
            if template_type == "snapshot":
                snapshot = self._store.get_snapshot(cast(str, template.snapshot_id))
                if snapshot is None:
                    raise FileNotFoundError(
                        f"Snapshot not found: {template.snapshot_id}"
                    )
                resolved_image = cast(str, snapshot.metadata.get("image_tag"))
                spec_image, spec_snapshot_id = None, template.snapshot_id
            else:
                resolved_image = template.image or DEFAULT_SANDBOX_IMAGE
                spec_image, spec_snapshot_id = resolved_image, None

            _check_no_conflicting_volumes(config.volumes)
            _check_no_conflicting_ports(config.ports)

            desired = ResolvedSandboxRuntimeSpec.from_parts(
                template_type=template_type,
                image=spec_image,
                snapshot_id=spec_snapshot_id,
                working_dir=config.working_dir,
                cpus=config.cpus,
                memory=config.memory,
                env=config.env,
                volumes=config.volumes,
                network_isolated=bool(config.network_isolated),
                ports=config.ports,
            )
            extra_labels = {
                LABEL_SPEC_FINGERPRINT: desired.fingerprint(),
                LABEL_SPEC_VERSION: str(SPEC_CONTRACT_VERSION),
            }

            # Build the container from the same normalized desired spec that
            # publish-before-verify below compares against, so both sides of
            # that check are computed from one canonical source instead of
            # two independent normalizers that could silently diverge (e.g.
            # on a volume's trailing slash or a `..` segment).
            backend_template, backend_config = desired.to_backend_config()

            try:
                container = await _create_container(
                    self._client,
                    name,
                    resolved_image,
                    backend_template,
                    backend_config,
                    extra_labels=extra_labels,
                )
            except APIError as exc:
                if exc.status_code == 409 or "already in use" in str(exc):
                    raise SandboxAlreadyExistsError(
                        f"Sandbox already exists: {name!r}"
                    ) from exc
                raise

            try:
                await asyncio.to_thread(container.start)
            except BaseException as start_exc:
                # Compensate with a raw container removal, never
                # self.delete(): delete() acquires this same _named_lock
                # entry, so calling it here would self-deadlock. The
                # original start failure (including a cancellation) is
                # preserved as the raised exception regardless of whether
                # the compensating remove itself succeeds.
                try:
                    await asyncio.to_thread(container.remove, force=True)
                except Exception as remove_exc:
                    raise start_exc from remove_exc
                raise start_exc

            await asyncio.to_thread(container.reload)
            inspection = _build_inspection(container)
            mismatches = _find_publish_mismatches(desired, resolved_image, inspection)
            if mismatches:
                # A container whose observed facts disagree with what we
                # asked for must never be published: remove it directly
                # (never self.delete(), for the same self-deadlock reason as
                # the start-failure path above) and fail loudly rather than
                # let a lying label reach the store.
                await asyncio.to_thread(container.remove, force=True)
                raise SandboxRuntimeConflictError(
                    f"Sandbox {name!r} failed publish verification; "
                    f"mismatched fields: {', '.join(mismatches)}"
                )

            runtime_info = _parse_container_config(container)
            stored_info = SandboxInfo(
                name=name,
                state=runtime_info.state,
                template=template,
                config=config,
                created_at=runtime_info.created_at,
            )
            info = _merge_info(runtime_info, stored_info)
            # Persisted only after verification passes; a failure here does
            # not roll back the container — it is left running with a
            # verified label and no store row, which the next reconcile pass
            # converges by observing running+MATCH and recreating the row.
            self._store.add_info(name, info)

            control = self._get_live_control(name)
            return DockerSandbox(name, container, info, self._store, control)

    async def start_existing(self, name: str) -> DockerSandbox:
        """Start a previously-created sandbox, idempotent if already running."""
        async with self._named_lock(name):
            control = self._get_live_control(name)
            async with control.exclusive_access(mark_deleted=False):
                container = await self._find_container(name)
                if container is None:
                    raise SandboxNotFoundError(f"Sandbox not found: {name}")
                await asyncio.to_thread(container.reload)
                state = _get_state(str(container.attrs.get("State", {}).get("Status")))
                if state != "running":
                    await asyncio.to_thread(container.start)
                    await asyncio.to_thread(container.reload)
                runtime_info = _parse_container_config(container)
                info = _merge_info(runtime_info, self._store.get_info(name))
                self._store.update_info_state(name, "running")
                return DockerSandbox(name, container, info, self._store, control)

    async def stop_existing(self, name: str) -> None:
        """Stop an existing sandbox, idempotent if already stopped."""
        async with self._named_lock(name):
            control = self._get_live_control(name)
            async with control.exclusive_access(mark_deleted=False):
                container = await self._find_container(name)
                if container is None:
                    raise SandboxNotFoundError(f"Sandbox not found: {name}")
                await asyncio.to_thread(container.reload)
                state = _get_state(str(container.attrs.get("State", {}).get("Status")))
                if state == "running":
                    await asyncio.to_thread(container.stop)
                self._store.update_info_state(name, "stopped")

    async def list_sandboxes(self) -> list[SandboxInfo]:
        """List all managed Docker sandboxes."""
        containers = await asyncio.to_thread(
            lambda: self._client.containers.list(
                all=True,
                filters={"label": f"{LABEL_MANAGED}=true"},
            )
        )
        result: list[SandboxInfo] = []
        for container in containers:
            runtime_info = _parse_container_config(container)
            stored_info = self._store.get_info(runtime_info.name)
            info = _merge_info(runtime_info, stored_info)
            result.append(info)
        return result

    async def delete(self, name: str) -> None:
        """Permanently delete a sandbox container and its metadata."""
        async with self._named_lock(name):
            control = self._get_control(name)
            async with control.exclusive_access(mark_deleted=True):
                container = await self._find_container(name)
                if container is not None:
                    await asyncio.to_thread(container.remove, force=True)
                self._store.delete_info(name)
                if self._controls.get(name) is control:
                    self._controls.pop(name)

    async def supports_snapshots(self) -> bool:
        """Return whether snapshot operations are supported."""
        return True

    async def create_snapshot(self, name: str, snapshot_id: str) -> SandboxSnapshot:
        """Create a snapshot by committing the current container filesystem."""
        async with self._named_lock(name):
            control = self._get_control(name)
            async with control.exclusive_access(mark_deleted=False):
                container = await self._find_container(name)
                if container is None:
                    raise SandboxNotFoundError(f"Sandbox not found: {name}")
                if self._store.get_snapshot(snapshot_id) is not None:
                    raise FileExistsError(f"Snapshot already exists: {snapshot_id}")

                tag = _snapshot_tag(snapshot_id)
                await asyncio.to_thread(
                    container.commit,
                    repository=SNAPSHOT_REPOSITORY,
                    tag=tag.split(":", 1)[1],
                    changes=None,
                )
                image_info = await asyncio.to_thread(self._client.images.get, tag)
                snapshot = SandboxSnapshot(
                    snapshot_id=snapshot_id,
                    metadata={
                        "image_id": image_info.id,
                        "image_tag": tag,
                        "source_sandbox": name,
                    },
                    created_at=str(image_info.attrs.get("Created") or ""),
                )
                self._store.add_snapshot(snapshot)
                return snapshot

    async def list_snapshots(self) -> list[SandboxSnapshot]:
        """List snapshots tracked by the sandbox store."""
        return self._store.list_snapshots()

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a snapshot image and its stored metadata."""
        snapshot = self._store.get_snapshot(snapshot_id)
        if snapshot is None:
            return
        image_tag = cast(Optional[str], snapshot.metadata.get("image_tag"))
        if image_tag:
            try:
                await asyncio.to_thread(self._client.images.remove, image=image_tag)
            except (ImageNotFound, NotFound):
                logger.info(
                    "Snapshot image already absent during delete: snapshot_id=%s tag=%s",
                    snapshot_id,
                    image_tag,
                )
            except APIError as exc:
                raise RuntimeError(
                    f"Failed to delete snapshot {snapshot_id}: {exc}"
                ) from exc
        self._store.delete_snapshot(snapshot_id)
