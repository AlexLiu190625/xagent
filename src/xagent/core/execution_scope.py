"""Execution scope context for scoping sandbox, workspace, and memory.

An :class:`ExecutionScope` is a **cooperative namespace, not a security
boundary**: it partitions sandbox lifecycle keys, workspace/storage paths,
and memory metadata *within* a single platform user. File records, RAG/KB
isolation, and tool credentials remain keyed by the platform ``user_id``
only — a scope must never be relied on to keep one principal's data from
another principal.

Scope fields are consumed **independently** by each subsystem: a consumer
reads exactly the field(s) it needs (``sandbox_key_suffix``,
``workspace_segments``, ``memory_dimensions``, ``strict_memory_isolation``,
``isolate_external_dirs``) and must never gate on "a scope is active" as an
all-or-nothing switch — a scope may set any subset of its fields.

Two activation mechanisms:

1. **Resolver hook** (primary): the embedding application registers a
   resolver via :func:`set_execution_scope_resolver`; the task orchestrator
   enters :func:`turn_execution_scope` at the start of every turn — the same
   place the acting user is resolved — so process restart and task
   resumption re-derive the scope from the embedder's own persistent data
   keyed by ``task_id`` rather than from a long-gone request context.
2. **Contextvar helpers** (secondary): :func:`set_execution_scope` /
   :func:`reset_execution_scope` / :class:`ExecutionScopeContext`, for
   synchronous paths that run inside the request that established the
   scope, mirroring the existing user-context pattern.
"""

from __future__ import annotations

import contextvars
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Optional, Union, cast

logger = logging.getLogger(__name__)

_SCOPE_COMPONENT_RE = re.compile(r"[a-zA-Z0-9_-]{1,63}")


class ExecutionScopeNotProvided:
    """Marker distinguishing an omitted scope from an explicit ``None``."""

    __slots__ = ()


EXECUTION_SCOPE_NOT_PROVIDED = ExecutionScopeNotProvided()

ExecutionScopeInput = Union[
    "ExecutionScope",
    None,
    ExecutionScopeNotProvided,
]


class InvalidScopeComponentError(ValueError):
    """A scope component failed validation.

    Raised instead of sanitizing: silently rewriting an invalid component
    could collapse two distinct inputs into one namespace.
    """


def validate_scope_component(value: Any, *, field_name: str = "scope component") -> str:
    """Validate a single scope component against ``[a-zA-Z0-9_-]{1,63}``.

    No ``:``, ``/``, ``..``, whitespace, or empty strings — components are
    embedded verbatim in sandbox lifecycle keys, filesystem paths, and
    storage keys. Invalid input is rejected with a logged error, never
    silently sanitized.

    Args:
        value: The candidate component.
        field_name: Name used in the log/error message.

    Returns:
        ``value`` unchanged, if valid.

    Raises:
        InvalidScopeComponentError: if ``value`` is not a string matching
            ``[a-zA-Z0-9_-]{1,63}``.
    """
    if not isinstance(value, str) or not _SCOPE_COMPONENT_RE.fullmatch(value):
        logger.error(
            "Invalid %s %r: must be a string matching [a-zA-Z0-9_-]{1,63}",
            field_name,
            value,
        )
        raise InvalidScopeComponentError(
            f"invalid {field_name} {value!r}: "
            "must be a string matching [a-zA-Z0-9_-]{1,63}"
        )
    return value


# Shape version stamped by ``to_dict`` / read back by ``from_dict``. Bump
# whenever a namespace-affecting field is added so ``resolve_execution_scope``
# can tell a snapshot written against a different shape (``version`` missing,
# lower, or -- during a mixed-version rollout -- higher than this constant)
# from one written against the current shape.
# ``from_dict`` cannot distinguish "field absent because pre-dates it" from
# "field explicitly at its default" any other way -- it fills both the same.
EXECUTION_SCOPE_SHAPE_VERSION = 1


@dataclass(frozen=True)
class ExecutionScope:
    """Immutable execution scope. All fields default to current behavior.

    Attributes:
        sandbox_key_suffix: Appended to the sandbox lifecycle key
            (``user:{owner_id}`` becomes ``user:{owner_id}:{suffix}``).
        workspace_segments: Extra path segments inserted after the user root
            in workspace paths and storage keys.
        sandbox_mount_segments: When set, the sandbox bind-mount root covers
            only this **prefix** of ``workspace_segments`` instead of the full
            tuple. Two scopes that share ``sandbox_key_suffix`` and this prefix
            then produce an identical mount and can share one container, while
            their deeper ``workspace_segments`` place them in distinct subtrees
            of that shared mount. **Security note:** those subtrees are *not*
            an isolation boundary. The mount is read-write and the
            code-execution tools (shell/python executors) run directly in the
            sandbox with no ``scoped_user_root`` path check, so code in one
            scope's task can read and write a co-mounted sibling's subtree.
            Only the orchestrator-side file/workspace API enforces
            ``scoped_user_root``. Therefore this field must only group scopes
            that already share one **runtime trust domain**; never use it to
            co-mount scopes across distinct runtime trust domains. Must be a prefix
            of ``workspace_segments``. ``None`` (the default) means the mount
            covers the full ``workspace_segments`` — byte-identical to
            pre-existing behavior. Consumed only by the sandbox-mount
            composition; workspace paths and storage keys always use the full
            ``workspace_segments``.
        memory_dimensions: Extra metadata stamped on memory notes on add and
            filtered on scoped search.
        strict_memory_isolation: When True, unscoped searches also exclude
            any note carrying scope dimensions (default is one-way
            visibility: scoped searches are isolated, unscoped searches see
            everything under the user). Consumed even when every other field
            is empty.
        isolate_external_dirs: When True, KB/upload external dirs become
            scope-local instead of shared across the user's scopes.
        version: Shape version this instance was constructed against
            (see :data:`EXECUTION_SCOPE_SHAPE_VERSION`). Excluded from
            equality (``compare=False``): it is bookkeeping for
            :func:`resolve_execution_scope`'s snapshot-vs-resolver
            comparison, not a namespace-affecting field, and two otherwise
            identical scopes built at different times must still compare
            equal. Not intended for callers to set explicitly.
    """

    sandbox_key_suffix: Optional[str] = None
    workspace_segments: tuple[str, ...] = ()
    sandbox_mount_segments: Optional[tuple[str, ...]] = None
    memory_dimensions: Mapping[str, str] = field(default_factory=dict)
    strict_memory_isolation: bool = False
    isolate_external_dirs: bool = False
    version: int = field(default=EXECUTION_SCOPE_SHAPE_VERSION, compare=False)

    @property
    def effective_mount_segments(self) -> tuple[str, ...]:
        """Segments the sandbox bind-mount root covers.

        Defaults to the full ``workspace_segments`` (mount root == workspace
        root), so an unset prefix reproduces today's behavior exactly. When
        ``sandbox_mount_segments`` is set, the mount root covers only that
        prefix and scopes sharing ``sandbox_key_suffix`` + this prefix share
        one container.
        """
        if self.sandbox_mount_segments is None:
            return self.workspace_segments
        return self.sandbox_mount_segments

    @property
    def durable_storage_segments(self) -> tuple[str, ...]:
        """Segments a durable-storage handle should confine to.

        Mirrors the filesystem external-dir allowlist
        (``_build_allowed_external_dirs``): a ``ScopedFileStorage`` handle is
        narrowed to the scope subtree only when ``isolate_external_dirs`` is
        set, so a scoped-but-not-isolated execution keeps its legitimate
        shared owner-level reads. When the flag is off this returns ``()`` — a
        handle bound to ``users/{owner}`` exactly as before.
        """
        return self.workspace_segments if self.isolate_external_dirs else ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot (see :func:`ExecutionScope.from_dict`).

        Used to persist a scope into a task's ``agent_config`` so internally
        created tasks (workforce runs) stay scoped across process restarts
        without the embedder's resolver knowing their task ids.
        """
        return {
            "sandbox_key_suffix": self.sandbox_key_suffix,
            "workspace_segments": list(self.workspace_segments),
            "sandbox_mount_segments": (
                None
                if self.sandbox_mount_segments is None
                else list(self.sandbox_mount_segments)
            ),
            "memory_dimensions": dict(self.memory_dimensions),
            "strict_memory_isolation": self.strict_memory_isolation,
            "isolate_external_dirs": self.isolate_external_dirs,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionScope":
        """Rebuild a scope from :meth:`to_dict` output (re-validated).

        ``version`` defaults to ``0`` (not
        :data:`EXECUTION_SCOPE_SHAPE_VERSION`) when the key is absent: a
        snapshot persisted before this field existed must decode as
        distinguishably older, not silently pass as current-shape.
        """
        raw_mount = data.get("sandbox_mount_segments")
        return cls(
            sandbox_key_suffix=data.get("sandbox_key_suffix"),
            workspace_segments=tuple(data.get("workspace_segments") or ()),
            sandbox_mount_segments=(None if raw_mount is None else tuple(raw_mount)),
            memory_dimensions=dict(data.get("memory_dimensions") or {}),
            strict_memory_isolation=bool(data.get("strict_memory_isolation", False)),
            isolate_external_dirs=bool(data.get("isolate_external_dirs", False)),
            version=int(data.get("version") or 0),
        )

    def __post_init__(self) -> None:
        if self.sandbox_key_suffix is not None:
            validate_scope_component(
                self.sandbox_key_suffix, field_name="sandbox_key_suffix"
            )

        if self.workspace_segments is None:
            raise ValueError(
                "workspace_segments cannot be None; pass () for a scope "
                "without workspace segments"
            )
        if self.memory_dimensions is None:
            raise ValueError(
                "memory_dimensions cannot be None; pass {} for a scope "
                "without memory dimensions"
            )

        segments = tuple(self.workspace_segments)
        for segment in segments:
            validate_scope_component(segment, field_name="workspace_segments entry")
        object.__setattr__(self, "workspace_segments", segments)

        if self.sandbox_mount_segments is not None:
            mount_segments = tuple(self.sandbox_mount_segments)
            for segment in mount_segments:
                validate_scope_component(
                    segment, field_name="sandbox_mount_segments entry"
                )
            # The mount root must be a prefix of the workspace root: the
            # workspace subtree (full segments) has to live *inside* the
            # mounted directory to be visible in the container, and a
            # non-prefix mount could expose an unrelated subtree.
            if mount_segments != segments[: len(mount_segments)]:
                logger.error(
                    "sandbox_mount_segments %r is not a prefix of "
                    "workspace_segments %r",
                    mount_segments,
                    segments,
                )
                raise InvalidScopeComponentError(
                    f"sandbox_mount_segments {mount_segments!r} must be a "
                    f"prefix of workspace_segments {segments!r}"
                )
            object.__setattr__(self, "sandbox_mount_segments", mount_segments)

        dimensions = dict(self.memory_dimensions)
        for key, dim_value in dimensions.items():
            validate_scope_component(key, field_name="memory_dimensions key")
            if not isinstance(dim_value, str) or not dim_value:
                logger.error(
                    "Invalid memory_dimensions value %r for key %r: "
                    "must be a non-empty string",
                    dim_value,
                    key,
                )
                raise InvalidScopeComponentError(
                    f"invalid memory_dimensions value {dim_value!r} for key "
                    f"{key!r}: must be a non-empty string"
                )
        object.__setattr__(self, "memory_dimensions", MappingProxyType(dimensions))


# Reserved key under which a task's ``agent_config`` JSON carries a
# persisted scope snapshot (ExecutionScope.to_dict()). Internally created
# tasks (workforce runs) have task ids the embedder's resolver cannot map;
# the snapshot is written at task creation and read back as a corroborating
# candidate when a resolver is registered (see resolve_execution_scope), or
# as the sole answer when no resolver is registered.
EXECUTION_SCOPE_AGENT_CONFIG_KEY = "execution_scope"


def execution_scope_from_agent_config(
    agent_config: Any,
) -> Optional[ExecutionScope]:
    """Decode the persisted scope snapshot owned by a task config.

    ``None`` means the task has no persisted snapshot. The canonical
    resolution flow (:func:`resolve_execution_scope`) always calls a
    registered resolver first; this snapshot is only a corroborating
    candidate for the resolver's answer, or the sole answer when no
    resolver is registered -- it is never consulted ahead of the resolver.
    Invalid snapshots propagate instead of silently degrading to an
    unscoped namespace.
    """

    if not isinstance(agent_config, Mapping):
        return None
    scope_data = agent_config.get(EXECUTION_SCOPE_AGENT_CONFIG_KEY)
    if not isinstance(scope_data, Mapping):
        return None
    return ExecutionScope.from_dict(scope_data)


# Metadata-key prefix under which ExecutionScope.memory_dimensions are
# stamped onto memory notes (flat, string-valued entries — the memory
# backends apply plain string-equality filters). The prefix keeps dimension
# keys from colliding with system metadata such as ``user_id``.
MEMORY_DIMENSION_METADATA_PREFIX = "execution_scope_"


def memory_dimension_metadata(scope: Optional[ExecutionScope]) -> dict[str, str]:
    """Prefixed metadata entries for a scope's memory dimensions.

    Empty when unscoped or when the scope carries no dimensions — fields
    are consumed independently.
    """
    if scope is None:
        return {}
    return {
        f"{MEMORY_DIMENSION_METADATA_PREFIX}{key}": value
        for key, value in scope.memory_dimensions.items()
    }


def metadata_carries_scope_dimensions(metadata: Mapping[str, Any]) -> bool:
    """True when a note's metadata was stamped with any scope dimension.

    Used by ``strict_memory_isolation`` post-filters to exclude scoped
    notes from unscoped searches.
    """
    return any(key.startswith(MEMORY_DIMENSION_METADATA_PREFIX) for key in metadata)


# Hashable identity of a scope's namespace-affecting fields:
# (sandbox_key_suffix, workspace_segments, effective_mount_segments,
#  sorted memory_dimensions items, isolate_external_dirs).
ScopeFingerprint = tuple[
    Optional[str],
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    bool,
]


def scope_fingerprint(scope: Optional[ExecutionScope]) -> Optional[ScopeFingerprint]:
    """Hashable fingerprint of the namespaces a scope selects.

    Per-task caches that bake scope-derived state in at build time (sandbox
    keys, workspace paths, sandbox mount root, memory dimensions,
    ``allowed_external_dirs``) key their eviction checks on this. The mount
    root is captured via ``effective_mount_segments`` so a changed mount
    prefix invalidates the cache instead of silently reusing a stale
    ``base_dir`` (which a later rebuild would then reject in
    ``SandboxManager._ensure_config_equivalent``). ``isolate_external_dirs``
    is included for the same reason: it is baked
    into the cached ``AgentService``'s ``Workspace.allowed_external_dirs``
    at build time (``_build_allowed_external_dirs`` -> ``AgentService.
    __init__`` -> ``WorkspaceManager.get_or_create_workspace``) rather than
    read fresh per call, so an isolate_external_dirs-only change across
    turns must evict the cache or the stale allowed-dirs list (shared root
    vs. scope-local) keeps being enforced. ``strict_memory_isolation`` is
    intentionally excluded: it is read fresh from the contextvar on every
    memory operation (``UserIsolatedMemoryStore``), so nothing cached here
    goes stale when only that flag changes. ``None`` is the sentinel for
    unscoped, distinct from an empty scope's fingerprint.
    """
    if scope is None:
        return None
    return (
        scope.sandbox_key_suffix,
        scope.workspace_segments,
        scope.effective_mount_segments,
        tuple(sorted(scope.memory_dimensions.items())),
        scope.isolate_external_dirs,
    )


current_execution_scope: contextvars.ContextVar[Optional[ExecutionScope]] = (
    contextvars.ContextVar("current_execution_scope", default=None)
)


def get_execution_scope() -> Optional[ExecutionScope]:
    """Get the execution scope active in the current context, if any."""
    return current_execution_scope.get()


def set_execution_scope(scope: Optional[ExecutionScope]) -> contextvars.Token:
    """Set the current execution scope.

    Args:
        scope: Scope to activate, or None for explicitly-unscoped.

    Returns:
        Context token for :func:`reset_execution_scope`.
    """
    return current_execution_scope.set(scope)


def reset_execution_scope(token: contextvars.Token) -> None:
    """Reset the execution scope to its previous state.

    Args:
        token: Context token from :func:`set_execution_scope`.
    """
    current_execution_scope.reset(token)


class ExecutionScopeContext:
    """Context manager for setting the execution scope."""

    def __init__(self, scope: Optional[ExecutionScope]) -> None:
        self.scope = scope
        self.token: Optional[contextvars.Token] = None

    def __enter__(self) -> "ExecutionScopeContext":
        self.token = set_execution_scope(self.scope)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[object],
    ) -> None:
        if self.token is not None:
            reset_execution_scope(self.token)


class DeferToSnapshot:
    """Resolver return value meaning "defer to the persisted snapshot".

    Not an :class:`ExecutionScope`: it never enters
    :func:`resolve_execution_scope`'s return value directly, is never
    activated on the ``current_execution_scope`` contextvar, and carries no
    scope attributes of its own. Construct via :func:`defer_to_snapshot`;
    the ``fallback`` is used only when no snapshot is persisted for the
    task (a task id the resolver's embedder does not itself recognize, with
    no Task-table snapshot either).
    """

    __slots__ = ("fallback",)

    def __init__(self, fallback: ExecutionScope) -> None:
        if not isinstance(fallback, ExecutionScope):
            raise TypeError(
                "defer_to_snapshot(fallback) requires an ExecutionScope, "
                f"got {fallback!r}"
            )
        self.fallback = fallback


def defer_to_snapshot(fallback: ExecutionScope) -> DeferToSnapshot:
    """Build a resolver return value that defers to the persisted snapshot.

    ``fallback`` is mandatory (not ``Optional``, no default): it is the
    scope used when the task carries no persisted snapshot, and must be the
    resolver's own most conservative answer for that task (e.g. today's
    creator-direct scope) so a defer never resolves *wider* than the
    resolver's own answer would have. This cannot be enforced by
    ``resolve_execution_scope`` itself (a resolver can pass anything as
    ``fallback``); it is the resolver author's obligation. An implicit
    ``None`` fallback would silently mean "authoritative unscoped on a
    snapshot miss" and must instead be spelled out explicitly by the caller.
    """
    return DeferToSnapshot(fallback)


# The embedding application injects a scope resolver via
# set_execution_scope_resolver() (same injection pattern as
# set_user_tool_overrides_hook in the web layer). Three return values:
#
# - ``ExecutionScope``: authoritative for this task.
# - ``None``: authoritative unscoped for this task (not "abstain").
# - ``DeferToSnapshot`` (see :func:`defer_to_snapshot`): abstain in favor of
#   the persisted snapshot, with a fallback for when none is persisted.
ExecutionScopeResolver = Callable[[str], Union[ExecutionScope, None, DeferToSnapshot]]

_execution_scope_resolver: Optional[ExecutionScopeResolver] = None


def execution_scope_resolver_registered() -> bool:
    """Whether an embedder has registered a scope resolver.

    Used at startup to log which authority mode is active: with a resolver
    registered, persisted snapshots are a corroborating candidate (see
    :func:`resolve_execution_scope`); with none, they are the sole answer.
    """
    return _execution_scope_resolver is not None


def set_execution_scope_resolver(
    resolver: Optional[ExecutionScopeResolver],
    *,
    acknowledges_snapshot_candidate_contract: bool = False,
) -> None:
    """Register the resolver that maps a ``task_id`` to its ExecutionScope.

    Resolver contract:

    - **Idempotent per task**: called at the start of every turn of a task
      (including resumed turns after a process restart), it must return an
      equal scope for the same ``task_id`` every time. Reassigning a task to
      a different scope between turns is possible but expensive (per-task
      caches rebuild); a resolver that flaps A -> B -> A is a bug.
    - **Three-valued return**: an ``ExecutionScope`` is authoritative for
      the task; ``None`` is authoritative *unscoped* for the task (not an
      abstention); :func:`defer_to_snapshot` abstains in favor of the
      persisted snapshot, with a mandatory fallback for tasks that carry
      none. No registered resolver means every task runs unscoped.
    - **Provenance, not existence**: whether to defer must be decided from
      the task's own provenance (e.g. whether the embedder's own records
      show it as one of its tasks), never from whether a snapshot happens
      to exist for it. Deferring merely because a snapshot is present
      degrades this contract back into "snapshot always wins".
    - Scope fields are consumed independently by subsystems; the resolver
      may populate any subset.
    - An exception from the resolver fails the turn: falling back to
      unscoped on error would silently merge namespaces.
    - When a persisted snapshot also exists for the task,
      :func:`resolve_execution_scope` treats it as a corroborating
      candidate, not an override: a namespace-affecting disagreement with
      an authoritative (non-defer) answer fails the turn instead of
      silently picking one side.

    Args:
        resolver: The resolver, or ``None`` to clear it.
        acknowledges_snapshot_candidate_contract: Must be ``True`` to
            register a non-``None`` resolver.

    Raises:
        TypeError: a non-``None`` resolver is registered without passing
            ``acknowledges_snapshot_candidate_contract=True``. This fails at
            registration time (e.g. embedder import/startup) instead of the
            first turn silently treating a persisted snapshot as an override
            by an embedder that has not read the contract above.
    """
    if resolver is not None and not acknowledges_snapshot_candidate_contract:
        raise TypeError(
            "set_execution_scope_resolver(resolver) requires "
            "acknowledges_snapshot_candidate_contract=True: a persisted "
            "scope snapshot is now a corroborating candidate rather than an "
            "override, and callers must confirm they have read the "
            "resolver's three-valued / provenance contract in this "
            "function's docstring before registering one"
        )
    global _execution_scope_resolver
    _execution_scope_resolver = resolver


# Loader for persisted scope snapshots (EXECUTION_SCOPE_AGENT_CONFIG_KEY in
# a task's agent_config). The web layer registers an implementation backed
# by the Task table; None means no snapshot support.
ExecutionScopeSnapshotLoader = Callable[[str], Optional[ExecutionScope]]

_execution_scope_snapshot_loader: Optional[ExecutionScopeSnapshotLoader] = None


def set_execution_scope_snapshot_loader(
    loader: Optional[ExecutionScopeSnapshotLoader],
) -> None:
    """Register the loader for persisted per-task scope snapshots.

    The loader returns the snapshot persisted at task creation, or None for
    tasks without one. With a resolver registered, the snapshot is a
    corroborating candidate for the resolver's authoritative answer (see
    :func:`resolve_execution_scope`); with no resolver registered, it is the
    sole answer -- this is what keeps internally created tasks (workforce
    runs, whose ids the embedder's resolver cannot map) scoped across
    process restarts. Loader exceptions fail the turn when no resolver is
    registered; otherwise a broken candidate is logged and ignored rather
    than vetoing the resolver's already-given authoritative answer.
    """
    global _execution_scope_snapshot_loader
    _execution_scope_snapshot_loader = loader


class ExecutionScopeResolverContractError(Exception):
    """A registered resolver returned something outside its three-valued contract.

    Deliberately not a subclass of ``RuntimeError``, ``ValueError``, or
    ``TypeError``: several websocket handlers catch
    ``except (ValueError, KeyError, TypeError)`` around the turn-execution
    path and fold anything in that tuple into a generic "client message
    format error" response. A resolver author's bug at the ``resolve_execution_scope``
    boundary is a server-side contract violation, not a malformed client
    message, and must not be swallowed by that handler as if it were one.

    Only used for the return-type check inside ``resolve_execution_scope``
    itself. The ``TypeError`` that :func:`set_execution_scope_resolver`
    raises for a missing acknowledgment token is unrelated -- that happens
    at registration time (embedder import/startup, before any request
    handler exists) and is intentionally left as a plain ``TypeError``.
    """


class ExecutionScopeAuthorityError(Exception):
    """A persisted snapshot disagrees with the resolver's authoritative scope.

    Deliberately not a subclass of ``RuntimeError`` or ``ValueError``:
    ``resolve_execution_scope`` already raises plain ``ValueError`` for a
    ``None`` task_id, and callers/tests must be able to tell an authority
    conflict apart from that structural error instead of both being folded
    by the same ``except ValueError`` (or a broad ``except RuntimeError``)
    clause.

    Carries the resolver's answer (``resolver_scope``) so a caller that must
    not fail a turn that has already ended or never started (see
    :func:`resolve_execution_scope_off_turn`) can log a structured warning
    and continue with the authoritative value instead of raising.

    ``str()`` on this exception deliberately carries only the ``task_id``
    and the names of the mismatched fields, never the scope values: it
    ends up in ``task.error_message`` and in the client's terminal error
    event (see ``task_orchestrator``'s setup/run failure handling, which
    formats durable/broadcast error strings from ``str(exc)``), and the
    scope values include ``sandbox_key_suffix``, ``workspace_segments``,
    and ``memory_dimensions`` -- namespace components that can carry
    end-user/client identifiers. The full scopes and the field-level value
    diff are logged via ``logger.error`` at the raise site instead, which
    stays server-side.
    """

    def __init__(
        self,
        task_id: str,
        *,
        resolver_scope: ExecutionScope,
        snapshot_scope: ExecutionScope,
        mismatched_fields: Mapping[str, tuple[Any, Any]],
    ) -> None:
        self.task_id = task_id
        self.resolver_scope = resolver_scope
        self.snapshot_scope = snapshot_scope
        self.mismatched_fields = dict(mismatched_fields)
        super().__init__(
            f"execution scope authority mismatch for task {task_id!r}: "
            f"mismatched_fields={sorted(self.mismatched_fields)!r}"
        )


# Namespace-affecting fields: a disagreement here changes which sandbox key,
# workspace path, mount root, or memory-dimension notes a task's execution
# actually touches, so resolve_execution_scope fails the turn rather than
# silently picking one side.
_EXECUTION_SCOPE_NAMESPACE_FIELDS: tuple[str, ...] = (
    "sandbox_key_suffix",
    "workspace_segments",
    "sandbox_mount_segments",
    "memory_dimensions",
    "isolate_external_dirs",
)

# Policy fields: change post-filter behavior on an otherwise-identical
# namespace, never which key/path is touched. A disagreement here is logged
# and the resolver's value wins, but does not fail the turn.
_EXECUTION_SCOPE_POLICY_FIELDS: tuple[str, ...] = ("strict_memory_isolation",)


def _execution_scope_field_diff(
    snapshot: ExecutionScope, resolver: ExecutionScope
) -> dict[str, tuple[Any, Any]]:
    """Per-field ``(snapshot_value, resolver_value)`` for differing fields.

    Compares every namespace/policy field (excludes ``version``, which is
    shape bookkeeping, not a scope field).
    """
    diff: dict[str, tuple[Any, Any]] = {}
    for name in _EXECUTION_SCOPE_NAMESPACE_FIELDS + _EXECUTION_SCOPE_POLICY_FIELDS:
        snapshot_value = getattr(snapshot, name)
        resolver_value = getattr(resolver, name)
        if snapshot_value != resolver_value:
            diff[name] = (snapshot_value, resolver_value)
    return diff


def _load_execution_scope_snapshot(
    task_id: str | int,
    persisted_snapshot: ExecutionScopeInput,
) -> Optional[ExecutionScope]:
    """Fetch the persisted snapshot, honoring an explicitly-passed override.

    A database owner that already read the persisted snapshot in its own
    Session may pass it via ``persisted_snapshot`` to skip the registered
    loader (see :func:`resolve_execution_scope`).
    """
    if persisted_snapshot is EXECUTION_SCOPE_NOT_PROVIDED:
        return (
            _execution_scope_snapshot_loader(str(task_id))
            if _execution_scope_snapshot_loader is not None
            else None
        )
    return cast(Optional[ExecutionScope], persisted_snapshot)


def resolve_execution_scope(
    task_id: str | int,
    *,
    persisted_snapshot: ExecutionScopeInput = EXECUTION_SCOPE_NOT_PROVIDED,
) -> Optional[ExecutionScope]:
    """Resolve the scope for ``task_id``.

    With no resolver registered, the persisted snapshot (see
    :func:`set_execution_scope_snapshot_loader`) is the sole answer,
    byte-identical to the pre-authority behavior standalone/workforce
    deployments rely on for restart/resume. With a resolver registered, the
    resolver is authoritative and runs first; the snapshot (when the
    resolver's answer needs one) is a corroborating candidate only:

    - Resolver raises: propagates immediately: the snapshot is not consulted.
    - Resolver returns ``None``: authoritative unscoped; the snapshot is not
      consulted.
    - Resolver returns :func:`defer_to_snapshot`'s carrier: the snapshot is
      used if present, else the carrier's fallback. A broken snapshot loader
      here is logged and treated as absent (falls back), matching the
      principle below that a broken candidate cannot veto an answer the
      resolver has already committed to (the fallback *is* that answer).
    - Resolver returns an ``ExecutionScope``: authoritative. A snapshot
      loader exception is logged and ignored (the candidate is corrupt, but
      an authoritative answer already exists). A snapshot whose shape
      version does not match the current one (:data:`EXECUTION_SCOPE_SHAPE_VERSION`,
      including one missing the field entirely, and also a snapshot stamped
      by a *newer* process during a mixed-version rollout) is logged
      (field-level diff) and ignored rather than compared -- a field the
      current shape added or changed always looks "different" from a
      freshly-resolved scope built against a different shape, and that is
      not a real conflict. Otherwise: a namespace-affecting difference (see
      :data:`_EXECUTION_SCOPE_NAMESPACE_FIELDS`) raises
      :class:`ExecutionScopeAuthorityError`; a policy-only difference (see
      :data:`_EXECUTION_SCOPE_POLICY_FIELDS`) is logged and the resolver's
      value still wins.

    A database owner that already read the persisted snapshot may pass it
    explicitly via ``persisted_snapshot``; this skips the registered loader
    while preserving the same precedence. Explicit ``None`` means "snapshot
    absent", not "force an unscoped task".

    Raises:
        ValueError: ``task_id`` is None — ``str(None)`` would silently
            query the loader/resolver for the literal string ``"None"``.
            Callers that legitimately have no task identity must treat
            that as unscoped themselves instead of passing None.
        ExecutionScopeAuthorityError: a persisted snapshot disagrees with
            the resolver's authoritative scope on a namespace-affecting
            field.
    """
    if task_id is None:
        raise ValueError(
            "task_id cannot be None; a caller without a task identity "
            "must treat the execution as unscoped instead"
        )

    if _execution_scope_resolver is None:
        return _load_execution_scope_snapshot(task_id, persisted_snapshot)

    resolved = _execution_scope_resolver(str(task_id))

    if isinstance(resolved, DeferToSnapshot):
        try:
            snapshot = _load_execution_scope_snapshot(task_id, persisted_snapshot)
        except Exception:
            logger.warning(
                "Snapshot loader failed while resolver deferred for task %s; "
                "using the resolver's fallback",
                task_id,
                exc_info=True,
            )
            return resolved.fallback
        return snapshot if snapshot is not None else resolved.fallback

    if resolved is None:
        return None

    if not isinstance(resolved, ExecutionScope):
        raise ExecutionScopeResolverContractError(
            f"execution scope resolver returned {resolved!r}; expected an "
            "ExecutionScope, None, or defer_to_snapshot(...)"
        )

    try:
        snapshot = _load_execution_scope_snapshot(task_id, persisted_snapshot)
    except Exception:
        logger.warning(
            "Snapshot loader failed while the resolver returned an "
            "authoritative scope for task %s; ignoring the candidate",
            task_id,
            exc_info=True,
        )
        return resolved

    if snapshot is None:
        return resolved

    if snapshot.version != EXECUTION_SCOPE_SHAPE_VERSION:
        # Not just "older": a mixed-version rollout can also see a snapshot
        # stamped by a newer process than this one. Either direction means
        # the snapshot's shape cannot be safely compared field-by-field
        # against a freshly-resolved scope built against *this* process's
        # shape, so the candidate is ignored rather than raised on.
        logger.warning(
            "Ignoring execution scope snapshot candidate for task %s: shape "
            "version %s does not match the current version %s; diff=%s",
            task_id,
            snapshot.version,
            EXECUTION_SCOPE_SHAPE_VERSION,
            _execution_scope_field_diff(snapshot, resolved),
        )
        return resolved

    diff = _execution_scope_field_diff(snapshot, resolved)
    if not diff:
        return resolved

    namespace_diff = {
        name: values
        for name, values in diff.items()
        if name in _EXECUTION_SCOPE_NAMESPACE_FIELDS
    }
    if namespace_diff:
        logger.error(
            "Execution scope authority mismatch for task %s: %s",
            task_id,
            diff,
        )
        raise ExecutionScopeAuthorityError(
            str(task_id),
            resolver_scope=resolved,
            snapshot_scope=snapshot,
            mismatched_fields=diff,
        )

    logger.warning(
        "Execution scope policy-only mismatch for task %s (resolver wins): %s",
        task_id,
        diff,
    )
    return resolved


def resolve_execution_scope_off_turn(task_id: str | int) -> Optional[ExecutionScope]:
    """Resolve scope for a consumer outside the turn lifecycle.

    Used by off-turn storage-key/workspace-segment composition (legacy
    preview backfill, a not-yet-persisted durable object) that cannot fail a
    turn that has already ended or never started. An
    :class:`ExecutionScopeAuthorityError` here would otherwise surface as a
    misleading "file not found" or a bulk endpoint's 500, at a point where
    the resolver has already produced an authoritative answer -- so it is
    downgraded to that answer plus a structured warning instead of raised.
    Every other exception (resolver/loader failure, ``task_id`` None) still
    propagates unchanged.
    """
    try:
        return resolve_execution_scope(task_id)
    except ExecutionScopeAuthorityError as exc:
        logger.warning(
            "Execution scope authority mismatch resolved off-turn for task "
            "%s (using the resolver's answer): %s",
            exc.task_id,
            exc.mismatched_fields,
        )
        return exc.resolver_scope


@contextmanager
def turn_execution_scope(task_id: str | int) -> Iterator[Optional[ExecutionScope]]:
    """Resolve and activate the execution scope for one turn of ``task_id``.

    The task orchestrator enters this at the start of every turn, at the
    same place the acting user is resolved, so restart/resume re-derive the
    scope correctly. The scope (or explicit None) is set for the duration of
    the turn and restored on exit.
    """
    scope = resolve_execution_scope(task_id)
    token = set_execution_scope(scope)
    try:
        yield scope
    finally:
        reset_execution_scope(token)
