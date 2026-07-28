"""Chat workspace binding: the CA-physical sandbox mount intent.

Two different concerns share one conceptual workspace root:

- what file tools are allowed to read/write (the *Actor*-logical view: the
  full workspace root under all ``ExecutionScope.workspace_segments``, plus
  the external directory allowlist). ``chat.py`` owns that view and builds
  it itself, through ``chat._build_allowed_external_dirs`` /
  ``WebToolConfig.workspace_config``; :func:`_build_external_allowlist`
  here recomputes the same allowlist purely as folding input, and the two
  are pinned equivalent by test (see
  ``tests/web/test_execution_scope_workspace_web.py``);
- what the sandbox container actually gets bind-mounted (the *CA*-physical
  view: one mount root plus any genuinely separate extra mounts --
  ``ChatWorkspaceBinding.mount_intent``, which ``chat.py`` consumes
  directly when creating/reusing the task's sandbox). This module is that
  view's single construction point.

:func:`build_chat_workspace_binding` folds the CA mount candidates (the
computed mount root plus every allowlist entry) through
``SandboxMountIntent``'s covered/covering/disjoint classification so a
redundant nested mount never becomes a second bind:

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

Because that classification is lexical, every candidate is absolutized and
symlink-resolved in the backend path domain first, and a candidate whose
lexical and resolved verdicts disagree is never folded away -- see
:func:`_fold_mount_paths`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from ...config import get_external_upload_dirs, get_uploads_dir
from ...core.execution_scope import ExecutionScope
from ...core.workspace import scoped_user_root
from ...sandbox import SandboxMountIntent
from ..sandbox_manager import absolute_backend_mount_path, resolve_backend_mount_path

logger = logging.getLogger(__name__)


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

    Entries keep chat's exact spelling -- this mirrors an Actor-logical
    allowlist that is pinned equivalent to chat's own by test. Backend-domain
    absolutization belongs to :func:`_fold_mount_paths`, which is where these
    values stop being an allowlist and become mount candidates.
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


def _lexical_relation(root: str, path: str) -> str:
    """``SandboxMountIntent``'s verdict for one path against one root."""
    probe = SandboxMountIntent(mount_root=root, extra_mounts=(path,))
    if probe.covered_extras:
        return "covered"
    if probe.covering_extras:
        return "covering"
    return "disjoint"


def _fold_relation(root: str, resolved_root: str, path: str, resolved_path: str) -> str:
    """Fold verdict for one candidate: lexical, vetoed by the resolved view.

    ``SandboxMountIntent`` classifies lexically, which is blind to symlinks
    (its own docstring says so and requires callers to resolve first). Both
    folding directions are only sound when the two views agree, because
    each of them keeps the *unresolved* spelling as the surviving mount:

    - dropping a covered candidate assumes the root's bind already exposes
      it at its own path. A symlink lexically under the root but resolving
      outside it is exposed by nothing once the root is bind-mounted, so
      dropping it silently loses the mount;
    - promoting a covering candidate assumes mounting it still exposes the
      old root at the old root's path. A symlink that only *resolves* to an
      ancestor does not contain the old root at all.

    So a disagreement demotes the candidate to disjoint: it keeps its own
    bind, which is exactly the unfolded behavior and can never lose access.
    """
    lexical = _lexical_relation(root, path)
    if lexical != _lexical_relation(resolved_root, resolved_path):
        return "disjoint"
    return lexical


def _fold_mount_paths(
    mount_root: str, candidates: Sequence[str]
) -> tuple[str, tuple[str, ...]]:
    """Collapse a mount root and allowlist candidates into one physical set.

    Root and candidates are first absolutized in the backend path domain
    (``absolute_backend_mount_path``): they come from raw configuration
    (``XAGENT_UPLOADS_DIR``, ``XAGENT_EXTERNAL_UPLOAD_DIRS``) and may be
    relative or ``~``-prefixed, which the mount-path contract rejects and
    which the pre-projection path mapper used to absolutize for them.

    Then, repeatedly, each candidate is classified against the current root
    by :func:`_fold_relation`:

    - a candidate the root already covers (equal to it or a descendant) is
      redundant and dropped;
    - a candidate that covers the root (a proper ancestor) absorbs it: the
      candidate is promoted to root and the old root is dropped (it is now
      implied by the promoted one). Covering candidates are always a
      lexical chain -- all are prefixes of the same root, hence prefixes of
      each other -- so promoting the shortest one is unambiguous and a
      single promotion reclassifies everything else against the new root.
    - anything left over is disjoint and kept as its own mount.

    Returns the final root and the deduplicated, sorted disjoint extras,
    each in its absolutized (never symlink-resolved) spelling.
    """
    root = str(absolute_backend_mount_path(mount_root))
    remaining = tuple(str(absolute_backend_mount_path(p)) for p in candidates)
    resolved = {p: resolve_backend_mount_path(p) for p in remaining}
    resolved_root = resolve_backend_mount_path(root)

    while True:
        verdicts = {
            p: _fold_relation(root, resolved_root, p, resolved[p]) for p in remaining
        }
        covering = [p for p in remaining if verdicts[p] == "covering"]
        if not covering:
            disjoint = tuple(sorted({p for p in remaining if verdicts[p] != "covered"}))
            return root, disjoint
        new_root = min(covering, key=len)
        remaining = tuple(p for p in remaining if p != new_root)
        root, resolved_root = new_root, resolved[new_root]


def build_chat_workspace_binding(
    owner_id: int, scope: Optional[ExecutionScope]
) -> ChatWorkspaceBinding:
    """Build the CA-physical mount intent for a task's sandbox.

    Called from ``chat.py`` (task creation and agent reconstruction alike)
    to build ``mount_intent`` for the task's sandbox lease provider; the
    Actor-logical allowlist stays chat-owned, see the module docstring.
    """
    mount_segments = scope.effective_mount_segments if scope is not None else ()

    external_allowlist = _build_external_allowlist(owner_id, scope)

    prepare_root = absolute_backend_mount_path(
        scoped_user_root(get_uploads_dir(), owner_id, mount_segments)
    )
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
        mount_intent=mount_intent,
        prepare_root=str(prepare_root),
    )
