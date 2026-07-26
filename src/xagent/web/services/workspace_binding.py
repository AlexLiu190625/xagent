"""Chat workspace binding: Actor-logical access policy + CA-physical mount intent.

Two different concerns share one conceptual workspace root, and this module
is the single construction point for both:

- what file tools are allowed to read/write (the *Actor*-logical view: the
  full workspace root under all ``ExecutionScope.workspace_segments``, plus
  the external directory allowlist -- ``WorkspaceAccessPolicy``, mirrored by
  ``chat._build_allowed_external_dirs`` / ``WebToolConfig.workspace_config``,
  which ``chat.py`` still builds independently rather than consuming
  ``ChatWorkspaceBinding.policy`` directly; the two are pinned equivalent by
  test, see ``tests/web/test_execution_scope_workspace_web.py``);
- what the sandbox container actually gets bind-mounted (the *CA*-physical
  view: one mount root plus any genuinely separate extra mounts --
  ``ChatWorkspaceBinding.mount_intent``, which ``chat.py`` does consume
  directly when creating/reusing the task's sandbox).

:func:`build_chat_workspace_binding` returns the Actor policy untouched, and
folds the CA mount candidates (the computed mount root plus every allowlist
entry) through ``SandboxMountIntent``'s covered/covering/disjoint
classification so a redundant nested mount never becomes a second bind:

- an allowlist entry the mount root already covers (equal to or a
  descendant of it) is dropped -- nothing is lost, the root's bind already
  exposes it;
- an allowlist entry that covers the mount root (a proper ancestor) absorbs
  it: the entry becomes the new root and the narrower original root is
  dropped, for the same reason.

The invariant this exists to enforce: when a scope isolates its external
dirs and narrows the mount to a prefix of ``workspace_segments`` (the
"CA root, Actor subtree" shape -- an org-level container shared by several
per-Actor scopes), the Actor's own subtree is *covered by* the CA mount root
and is dropped rather than surfacing as a second, Actor-specific bind. Two
Actors under the same CA then compute byte-identical mount intents and can
share one container -- keeping the Actor subtree as a separate bind would
make their desired configs diverge and is the root cause (#296) this
projection removes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from ...config import get_external_upload_dirs, get_uploads_dir
from ...core.execution_scope import ExecutionScope
from ...core.workspace import scoped_user_root
from ...sandbox import SandboxMountIntent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspaceAccessPolicy:
    """Actor-logical workspace access: full-segment root + external allowlist.

    ``workspace_root`` always uses the scope's full ``workspace_segments``
    (never ``effective_mount_segments``) -- it is the path file tools read
    and write against, independent of how narrow the sandbox's own mount is.
    ``external_allowlist`` mirrors today's ``chat._build_allowed_external_dirs``
    (``only_existing=False``): the user's upload dir -- scope-narrowed only
    when ``isolate_external_dirs`` is set -- plus any deployment-level
    external upload dirs.
    """

    workspace_root: str
    external_allowlist: tuple[str, ...]


@dataclass(frozen=True)
class ChatWorkspaceBinding:
    """Result of :func:`build_chat_workspace_binding`.

    ``mount_intent`` is the folded set actually handed to the sandbox
    manager. ``prepare_root`` is deliberately a separate field: the mount
    root computed *before* folding (``scoped_user_root`` at the scope's
    ``effective_mount_segments``). Folding can re-root ``mount_intent`` onto
    a covering ancestor already in the allowlist (see module docstring), but
    the directory the task's own files actually live in is always
    ``prepare_root`` -- that is the ``mkdir -p`` target, never
    ``mount_intent.mount_root`` directly, because in the re-rooted case the
    ancestor may already exist while the deeper subtree does not.
    """

    policy: WorkspaceAccessPolicy
    mount_intent: SandboxMountIntent
    prepare_root: str


# Process-level, once-per-kind guard for the known-limitation warnings below.
# Deliberately never cleared: the goal is one log line per limitation *kind*
# per process, enough to point at a producer, not a per-call trace.
_warned_limitation_kinds: set[str] = set()


def _warn_once(kind: str, message: str, *args: object) -> None:
    if kind in _warned_limitation_kinds:
        return
    _warned_limitation_kinds.add(kind)
    logger.warning(message, *args)


def _build_external_allowlist(
    owner_id: int, scope: Optional[ExecutionScope]
) -> tuple[str, ...]:
    """Mirror ``chat._build_allowed_external_dirs`` (``only_existing=False``).

    The user's upload dir is scope-narrowed only when
    ``isolate_external_dirs`` is set; deployment-level external upload dirs
    (``XAGENT_EXTERNAL_UPLOAD_DIRS``) are never user-root derived and are
    always included.
    """
    segments = (
        scope.workspace_segments
        if scope is not None and scope.isolate_external_dirs
        else ()
    )
    user_upload_dir = scoped_user_root(get_uploads_dir(), owner_id, segments)
    dirs = [str(user_upload_dir)]
    dirs.extend(str(d) for d in get_external_upload_dirs())
    return tuple(dirs)


def _fold_mount_paths(
    mount_root: str, candidates: Sequence[str]
) -> tuple[str, tuple[str, ...]]:
    """Collapse a mount root and allowlist candidates into one physical set.

    Repeatedly classifies ``candidates`` against the current root with
    ``SandboxMountIntent`` (purely lexical -- see ``xagent.sandbox.base``):

    - a candidate the root already covers (equal to it or a descendant) is
      redundant and dropped;
    - a candidate that covers the root (a proper ancestor) absorbs it: the
      candidate is promoted to root and the old root is dropped (it is now
      implied by the promoted one). Covering candidates are always a
      lexical chain -- all are prefixes of the same root, hence prefixes of
      each other -- so promoting the shortest one is unambiguous and a
      single promotion reclassifies everything else against the new root.
    - anything left over is disjoint and kept as its own mount.

    Returns the final root and the deduplicated, sorted disjoint extras.
    """
    root = mount_root
    remaining: Sequence[str] = tuple(candidates)
    while True:
        probe = SandboxMountIntent(mount_root=root, extra_mounts=tuple(remaining))
        covering = probe.covering_extras
        if not covering:
            covered = set(probe.covered_extras)
            disjoint = tuple(p for p in probe.extra_mounts if p not in covered)
            return root, disjoint
        new_root = min(covering, key=len)
        remaining = tuple(p for p in probe.extra_mounts if p != new_root)
        root = new_root


def build_chat_workspace_binding(
    owner_id: int, scope: Optional[ExecutionScope]
) -> ChatWorkspaceBinding:
    """Build the Actor-logical policy and CA-physical mount intent for a task.

    Called from ``chat.py`` (task creation and agent reconstruction alike)
    to build ``mount_intent`` for the task's sandbox lease provider; see the
    module docstring for why ``.policy`` is not yet consumed the same way.
    """
    workspace_segments = scope.workspace_segments if scope is not None else ()
    mount_segments = scope.effective_mount_segments if scope is not None else ()

    workspace_root = scoped_user_root(get_uploads_dir(), owner_id, workspace_segments)
    external_allowlist = _build_external_allowlist(owner_id, scope)
    policy = WorkspaceAccessPolicy(
        workspace_root=str(workspace_root),
        external_allowlist=external_allowlist,
    )

    prepare_root = scoped_user_root(get_uploads_dir(), owner_id, mount_segments)
    folded_root, folded_extras = _fold_mount_paths(
        str(prepare_root), external_allowlist
    )

    # Known limitation (pending PR-2 scope authority): a suffix-less scope
    # shares the unscoped sandbox lifecycle key (``user:{owner}``) purely
    # from ``sandbox_key_suffix`` being absent, but ``isolate_external_dirs``
    # still narrows this call's own mount root away from the unscoped path.
    # The same lifecycle key then sees a different desired mount depending on
    # which scope happened to build it -- a config-equivalence hazard, not
    # something this projection can resolve on its own.
    if (
        scope is not None
        and scope.sandbox_key_suffix is None
        and scope.isolate_external_dirs
        and scope.workspace_segments
    ):
        _warn_once(
            "suffixless_isolate",
            "ExecutionScope has isolate_external_dirs=True and "
            "workspace_segments=%r but no sandbox_key_suffix: it shares the "
            "unscoped sandbox lifecycle key for owner %s while this call's "
            "own mount root (%s) diverges from the unscoped path. Known "
            "limitation pending PR-2 scope authority -- give this scope a "
            "sandbox_key_suffix.",
            scope.workspace_segments,
            owner_id,
            prepare_root,
        )

    # Known limitation (pending PR-2 scope authority): an explicit
    # sandbox_mount_segments prefix declares a narrower mount, but with
    # isolate_external_dirs left False the allowlist still carries the
    # unscoped user root, which always covers (and folds away) the
    # narrower mount root -- the requested narrowing has no physical effect.
    if (
        scope is not None
        and scope.sandbox_mount_segments is not None
        and not scope.isolate_external_dirs
    ):
        _warn_once(
            "mount_prefix_without_isolate",
            "ExecutionScope sets sandbox_mount_segments=%r without "
            "isolate_external_dirs: the narrower mount root is folded away "
            "into the unscoped user-root allowlist entry, so the requested "
            "mount narrowing has no effect. Known limitation pending PR-2 "
            "scope authority -- set isolate_external_dirs=True to keep the "
            "narrower mount.",
            scope.sandbox_mount_segments,
        )

    mount_intent = SandboxMountIntent(
        mount_root=folded_root, extra_mounts=folded_extras
    )
    return ChatWorkspaceBinding(
        policy=policy,
        mount_intent=mount_intent,
        prepare_root=str(prepare_root),
    )
