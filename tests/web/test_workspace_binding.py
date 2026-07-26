"""Golden tests for the chat workspace projection split (PR-1b stage 1).

``build_chat_workspace_binding`` is the single builder both of chat.py's
inline ``sandbox_workspace_config`` dicts will collapse onto in a later
stage; this stage only builds and pins it against a frozen reimplementation
of *today's* computation (chat.py's ``_build_allowed_external_dirs`` +
inline ``sandbox_workspace_config`` dicts, and
``SandboxManager._workspace_mount_paths``'s "mount base_dir + every
allowed_external_dirs entry, deduplicated only by exact string" behavior).

Six-row physical-set matrix (unscoped / scoped isolate=False / external CA
scoped / internal scoped / two known-limitation shapes), plus an Actor-path
invariant and an external-dir ancestor/descendant boundary check.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pytest

import xagent.web.services.workspace_binding as workspace_binding
from xagent.core.execution_scope import ExecutionScope
from xagent.core.workspace import scoped_user_root
from xagent.web.services.workspace_binding import (
    ChatWorkspaceBinding,
    build_chat_workspace_binding,
)

OWNER_ID = 42


@pytest.fixture(autouse=True)
def _uploads_dir(tmp_path, monkeypatch):
    """Point the builder's uploads dir at an isolated tmp tree."""
    monkeypatch.setattr(workspace_binding, "get_uploads_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _external_dirs(tmp_path, monkeypatch):
    """A single deployment-level external upload dir, disjoint from uploads.

    A sibling of ``tmp_path`` (not nested under it), matching real
    deployments where ``XAGENT_EXTERNAL_UPLOAD_DIRS`` points at a shared KB
    location outside the per-user uploads tree.
    """
    ext_dir = tmp_path.parent / f"{tmp_path.name}-shared-kb"
    monkeypatch.setattr(
        workspace_binding, "get_external_upload_dirs", lambda: [ext_dir]
    )
    return ext_dir


@pytest.fixture(autouse=True)
def _reset_warn_guard(monkeypatch):
    """Isolate the process-level once-per-kind warn guard per test."""
    monkeypatch.setattr(workspace_binding, "_warned_limitation_kinds", set())


def _legacy_today_paths(
    owner_id: int,
    scope: Optional[ExecutionScope],
    *,
    uploads_dir: Path,
    ext_dirs: list[Path],
) -> set[str]:
    """Frozen reimplementation of today's physical mount set.

    Mirrors chat.py's ``sandbox_workspace_config`` dict (``base_dir`` from
    ``scoped_user_root`` at ``effective_mount_segments``,
    ``allowed_external_dirs`` from ``_build_allowed_external_dirs``) fed
    through ``SandboxManager._workspace_mount_paths``, which mounts
    ``base_dir`` plus every ``allowed_external_dirs`` entry with no
    covered/covering folding -- only exact-string dedup (a plain ``set``).
    """
    mount_segments = scope.effective_mount_segments if scope is not None else ()
    base_dir = scoped_user_root(uploads_dir, owner_id, mount_segments)

    ext_segments = (
        scope.workspace_segments
        if scope is not None and scope.isolate_external_dirs
        else ()
    )
    user_upload_dir = scoped_user_root(uploads_dir, owner_id, ext_segments)
    allowed_external_dirs = [str(user_upload_dir)] + [str(d) for d in ext_dirs]

    return {str(base_dir)} | set(allowed_external_dirs)


def _new_physical_paths(binding: ChatWorkspaceBinding) -> set[str]:
    intent = binding.mount_intent
    assert intent.mount_root is not None
    return {intent.mount_root} | set(intent.extra_mounts)


class TestUnscopedRow:
    """Row a: scope=None -- byte-identical to today, no folding needed."""

    def test_physical_set_matches_today(self, _uploads_dir, _external_dirs):
        binding = build_chat_workspace_binding(OWNER_ID, None)
        old = _legacy_today_paths(
            OWNER_ID, None, uploads_dir=_uploads_dir, ext_dirs=[_external_dirs]
        )
        assert _new_physical_paths(binding) == old

    def test_prepare_root_is_user_root(self, _uploads_dir):
        binding = build_chat_workspace_binding(OWNER_ID, None)
        assert binding.prepare_root == str(scoped_user_root(_uploads_dir, OWNER_ID, ()))

    def test_policy_workspace_root_is_user_root(self, _uploads_dir):
        binding = build_chat_workspace_binding(OWNER_ID, None)
        assert binding.policy.workspace_root == str(
            scoped_user_root(_uploads_dir, OWNER_ID, ())
        )


class TestScopedIsolateFalseRow:
    """Row b: scoped, isolate_external_dirs=False (default).

    The mount root (a scope subtree) is covered by an ancestor already in
    the allowlist (the unscoped user root, present because isolate=False
    keeps the shared, un-narrowed allowlist entry) -- that ancestor absorbs
    the mount, replacing it. The original mount root disappears from the
    physical set, but it stays fully reachable through the promoted
    ancestor's mount: this is the one documented, harmless reduction for
    this row (not a byte-exact match to today's raw, non-folding mount
    list).
    """

    def _scope(self) -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix="tenantA", workspace_segments=("proj1",)
        )

    def test_covering_ancestor_absorbs_mount_root(self, _uploads_dir, _external_dirs):
        scope = self._scope()
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        user_root = str(scoped_user_root(_uploads_dir, OWNER_ID, ()))
        scoped_root = str(scoped_user_root(_uploads_dir, OWNER_ID, ("proj1",)))

        old = _legacy_today_paths(
            OWNER_ID, scope, uploads_dir=_uploads_dir, ext_dirs=[_external_dirs]
        )
        new = _new_physical_paths(binding)

        assert old == {scoped_root, user_root, str(_external_dirs)}
        assert new == {user_root, str(_external_dirs)}
        assert old - new == {scoped_root}, (
            "only the absorbed (now-redundant) root drops"
        )
        assert new - old == set(), "folding never introduces a path absent from today"
        assert binding.mount_intent.mount_root == user_root

    def test_prepare_root_stays_the_unfolded_scope_subtree(self, _uploads_dir):
        scope = self._scope()
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        assert binding.prepare_root == str(
            scoped_user_root(_uploads_dir, OWNER_ID, ("proj1",))
        )
        # mkdir target is the pre-fold root, distinct from the folded
        # mount_intent.mount_root (the promoted ancestor).
        assert binding.prepare_root != binding.mount_intent.mount_root

    def test_no_warning_emitted(self, caplog):
        caplog.set_level(logging.WARNING, logger=workspace_binding.logger.name)
        build_chat_workspace_binding(OWNER_ID, self._scope())
        assert caplog.records == []


class TestExternalCAScopedRow:
    """Row c: suffix + mount prefix + isolate=True -- the #296 fix itself.

    The Actor's own subtree (full workspace_segments, present in the
    allowlist because isolate=True) is covered by the CA mount root and is
    dropped. Unlike row b, the root itself is unchanged -- only a covered
    extra disappears. This is the sole row where the physical-set diff is
    the deliberate fix target, pinned exactly to the Actor subtree.
    """

    def _scope(self) -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix="ca-1",
            workspace_segments=("ca1", "actor7"),
            sandbox_mount_segments=("ca1",),
            isolate_external_dirs=True,
        )

    def test_actor_child_is_the_only_diff_from_today(
        self, _uploads_dir, _external_dirs
    ):
        scope = self._scope()
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        ca_root = str(scoped_user_root(_uploads_dir, OWNER_ID, ("ca1",)))
        actor_child = str(scoped_user_root(_uploads_dir, OWNER_ID, ("ca1", "actor7")))

        old = _legacy_today_paths(
            OWNER_ID, scope, uploads_dir=_uploads_dir, ext_dirs=[_external_dirs]
        )
        new = _new_physical_paths(binding)

        assert old == {ca_root, actor_child, str(_external_dirs)}
        assert new == {ca_root, str(_external_dirs)}
        assert old - new == {actor_child}, (
            "the Actor subtree is exactly what disappears"
        )
        assert new - old == set()
        assert binding.mount_intent.mount_root == ca_root

    def test_prepare_root_is_the_ca_root(self, _uploads_dir):
        scope = self._scope()
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        assert binding.prepare_root == str(
            scoped_user_root(_uploads_dir, OWNER_ID, ("ca1",))
        )
        assert binding.prepare_root == binding.mount_intent.mount_root

    def test_two_actors_under_same_ca_fold_to_identical_intent(self, _uploads_dir):
        """The multi-Actor collision #296 is about: same CA, different Actor
        subtrees must fold to a byte-identical intent to share one container.
        """
        scope_a = self._scope()
        scope_b = ExecutionScope(
            sandbox_key_suffix="ca-1",
            workspace_segments=("ca1", "actor9"),
            sandbox_mount_segments=("ca1",),
            isolate_external_dirs=True,
        )
        binding_a = build_chat_workspace_binding(OWNER_ID, scope_a)
        binding_b = build_chat_workspace_binding(OWNER_ID, scope_b)
        assert binding_a.mount_intent == binding_b.mount_intent


class TestInternalScopedRow:
    """Row d: suffix + mount=None (full segments) + isolate=True.

    base_dir and the isolate-narrowed allowlist entry are the *same* path
    already -- today's raw list already carries an exact-string duplicate
    that a plain ``set()`` collapses, so this row is byte-identical to
    today with no caveat.
    """

    def _scope(self) -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix="proj2",
            workspace_segments=("proj2",),
            isolate_external_dirs=True,
        )

    def test_physical_set_matches_today_exactly(self, _uploads_dir, _external_dirs):
        scope = self._scope()
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        workspace_root = str(scoped_user_root(_uploads_dir, OWNER_ID, ("proj2",)))

        old = _legacy_today_paths(
            OWNER_ID, scope, uploads_dir=_uploads_dir, ext_dirs=[_external_dirs]
        )
        new = _new_physical_paths(binding)

        assert old == {workspace_root, str(_external_dirs)}
        assert new == old
        assert binding.mount_intent.mount_root == workspace_root
        assert binding.prepare_root == workspace_root

    def test_no_warning_emitted(self, caplog):
        caplog.set_level(logging.WARNING, logger=workspace_binding.logger.name)
        build_chat_workspace_binding(OWNER_ID, self._scope())
        assert caplog.records == []


class TestSuffixlessIsolateRow:
    """Row e (known limitation): isolate=True with no sandbox_key_suffix.

    This scope shares the *unscoped* sandbox lifecycle key (no suffix) yet
    still computes a scoped, non-unscoped mount root -- the same container
    identity would see a different desired mount depending on which scope
    built it. Root cause: scope authority isn't fully closed until PR-2;
    this builder can only flag it, not fix it.
    """

    def _scope(self) -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix=None,
            workspace_segments=("solo",),
            isolate_external_dirs=True,
        )

    def test_result_differs_from_unscoped(self, _uploads_dir, _external_dirs):
        scoped_binding = build_chat_workspace_binding(OWNER_ID, self._scope())
        unscoped_binding = build_chat_workspace_binding(OWNER_ID, None)

        scoped_root = str(scoped_user_root(_uploads_dir, OWNER_ID, ("solo",)))
        assert _new_physical_paths(scoped_binding) == {scoped_root, str(_external_dirs)}
        assert _new_physical_paths(scoped_binding) != _new_physical_paths(
            unscoped_binding
        )

    def test_emits_one_structured_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=workspace_binding.logger.name)
        build_chat_workspace_binding(OWNER_ID, self._scope())
        matching = [r for r in caplog.records if "sandbox_key_suffix" in r.getMessage()]
        assert len(matching) == 1

    def test_warning_fires_only_once_per_process(self, caplog):
        caplog.set_level(logging.WARNING, logger=workspace_binding.logger.name)
        scope = self._scope()
        build_chat_workspace_binding(OWNER_ID, scope)
        build_chat_workspace_binding(OWNER_ID, scope)
        build_chat_workspace_binding(99, scope)
        matching = [r for r in caplog.records if "sandbox_key_suffix" in r.getMessage()]
        assert len(matching) == 1


class TestDivergentMountPrefixRow:
    """Row f (known limitation): explicit sandbox_mount_segments without
    isolate_external_dirs.

    The narrower mount root the caller asked for is folded away into the
    unscoped user-root allowlist entry (isolate=False keeps that entry
    un-narrowed), so the requested narrowing has zero physical effect --
    the result collapses to the same set as the plain unscoped row.
    """

    def _scope(self) -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix="ca-2",
            workspace_segments=("ca2", "actorX"),
            sandbox_mount_segments=("ca2",),
            isolate_external_dirs=False,
        )

    def test_narrowing_has_no_physical_effect(self, _uploads_dir, _external_dirs):
        binding = build_chat_workspace_binding(OWNER_ID, self._scope())
        unscoped_binding = build_chat_workspace_binding(OWNER_ID, None)
        assert _new_physical_paths(binding) == _new_physical_paths(unscoped_binding)

    def test_emits_one_structured_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=workspace_binding.logger.name)
        build_chat_workspace_binding(OWNER_ID, self._scope())
        matching = [
            r for r in caplog.records if "sandbox_mount_segments" in r.getMessage()
        ]
        assert len(matching) == 1

    def test_warning_fires_only_once_per_process(self, caplog):
        caplog.set_level(logging.WARNING, logger=workspace_binding.logger.name)
        scope = self._scope()
        build_chat_workspace_binding(OWNER_ID, scope)
        build_chat_workspace_binding(OWNER_ID, scope)
        matching = [
            r for r in caplog.records if "sandbox_mount_segments" in r.getMessage()
        ]
        assert len(matching) == 1


class TestActorPathNeverEntersIntent:
    """Invariant: with isolate=True and a genuine mount/workspace split, the
    Actor's own (deeper) path never surfaces as a mount -- neither as an
    extra mount nor as the mount root -- at any workspace_segments depth.
    """

    @pytest.mark.parametrize(
        "workspace_segments,mount_segments",
        [
            (("ca", "actor"), ("ca",)),
            (("ca", "team", "actor"), ("ca",)),
            (("ca", "team", "actor"), ("ca", "team")),
        ],
    )
    def test_actor_subtree_excluded_at_every_depth(
        self, _uploads_dir, workspace_segments, mount_segments
    ):
        scope = ExecutionScope(
            sandbox_key_suffix="ca",
            workspace_segments=workspace_segments,
            sandbox_mount_segments=mount_segments,
            isolate_external_dirs=True,
        )
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        actor_path = str(scoped_user_root(_uploads_dir, OWNER_ID, workspace_segments))
        mount_root_path = str(scoped_user_root(_uploads_dir, OWNER_ID, mount_segments))

        assert actor_path != mount_root_path, "fixture must exercise a genuine split"
        physical = _new_physical_paths(binding)
        assert actor_path not in physical
        assert binding.mount_intent.mount_root != actor_path


class TestExternalDirBoundary:
    """ext contains the mount root's ancestor or descendant (deployment-level
    external dir specifically -- isolate=True keeps the isolate-driven
    allowlist candidate equal to the mount root so it cannot itself act as
    the covering/covered path under test).
    """

    def test_ancestor_external_dir_becomes_new_root(self, _uploads_dir, monkeypatch):
        user_root = scoped_user_root(_uploads_dir, OWNER_ID, ())
        monkeypatch.setattr(
            workspace_binding, "get_external_upload_dirs", lambda: [user_root]
        )
        scope = ExecutionScope(
            sandbox_key_suffix="s",
            workspace_segments=("proj",),
            isolate_external_dirs=True,
        )
        binding = build_chat_workspace_binding(OWNER_ID, scope)

        assert binding.mount_intent.mount_root == str(user_root)
        assert binding.mount_intent.extra_mounts == ()

    def test_descendant_external_dir_is_dropped(self, _uploads_dir, monkeypatch):
        scoped_root = scoped_user_root(_uploads_dir, OWNER_ID, ("proj",))
        descendant = scoped_root / "kb"
        monkeypatch.setattr(
            workspace_binding, "get_external_upload_dirs", lambda: [descendant]
        )
        scope = ExecutionScope(
            sandbox_key_suffix="s",
            workspace_segments=("proj",),
            isolate_external_dirs=True,
        )
        binding = build_chat_workspace_binding(OWNER_ID, scope)

        assert binding.mount_intent.mount_root == str(scoped_root)
        assert binding.mount_intent.extra_mounts == ()
