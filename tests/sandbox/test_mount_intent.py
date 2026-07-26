"""
Unit tests for ``SandboxMountIntent``.

Classification is purely lexical: it compares already-normalized path
strings and never touches the filesystem. These tests pin that behavior,
including the never-raises contract and the exact descendant/ancestor/
disjoint split.
"""

from __future__ import annotations

from xagent.sandbox.base import SandboxMountIntent


class TestMountIntentNormalization:
    def test_normalizes_mount_root(self):
        intent = SandboxMountIntent(mount_root="/data/../data2", extra_mounts=())
        assert intent.mount_root == "/data2"

    def test_normalizes_extra_mounts(self):
        intent = SandboxMountIntent(mount_root=None, extra_mounts=("/a/./b", "/a/../c"))
        assert intent.extra_mounts == ("/a/b", "/c")

    def test_deduplicates_after_normalization(self):
        intent = SandboxMountIntent(
            mount_root=None,
            extra_mounts=("/a/b", "/a/./b", "/a/b"),
        )
        assert intent.extra_mounts == ("/a/b",)

    def test_extra_mounts_are_sorted(self):
        intent = SandboxMountIntent(mount_root=None, extra_mounts=("/z", "/a", "/m"))
        assert intent.extra_mounts == ("/a", "/m", "/z")

    def test_none_mount_root_stays_none(self):
        intent = SandboxMountIntent(mount_root=None, extra_mounts=("/a",))
        assert intent.mount_root is None


class TestMountIntentNeverRaises:
    def test_empty_intent_does_not_raise(self):
        intent = SandboxMountIntent(mount_root=None, extra_mounts=())
        assert intent.mount_root is None
        assert intent.extra_mounts == ()

    def test_relative_paths_do_not_raise(self):
        # Malformed input relative to contract (paths should be absolute) but
        # construction itself must not raise.
        intent = SandboxMountIntent(mount_root="relative/root", extra_mounts=("x/y",))
        assert intent.mount_root == "relative/root"
        assert intent.extra_mounts == ("x/y",)

    def test_root_equal_to_slash_does_not_raise(self):
        intent = SandboxMountIntent(mount_root="/", extra_mounts=("/etc",))
        assert intent.mount_root == "/"


class TestMountIntentClassification:
    def test_descendant_is_covered(self):
        intent = SandboxMountIntent(mount_root="/data", extra_mounts=("/data/sub",))
        assert intent.covered_extras == ("/data/sub",)
        assert intent.covering_extras == ()
        assert intent.disjoint_extras == ()

    def test_equal_path_is_covered(self):
        intent = SandboxMountIntent(mount_root="/data", extra_mounts=("/data",))
        assert intent.covered_extras == ("/data",)
        assert intent.covering_extras == ()
        assert intent.disjoint_extras == ()

    def test_ancestor_is_covering(self):
        intent = SandboxMountIntent(mount_root="/data/sub", extra_mounts=("/data",))
        assert intent.covered_extras == ()
        assert intent.covering_extras == ("/data",)
        assert intent.disjoint_extras == ()

    def test_root_of_filesystem_is_covering(self):
        intent = SandboxMountIntent(mount_root="/data/sub", extra_mounts=("/",))
        assert intent.covering_extras == ("/",)

    def test_unrelated_path_is_disjoint(self):
        intent = SandboxMountIntent(mount_root="/data", extra_mounts=("/other",))
        assert intent.covered_extras == ()
        assert intent.covering_extras == ()
        assert intent.disjoint_extras == ("/other",)

    def test_sibling_with_shared_prefix_is_disjoint_not_covered(self):
        # "/database" is not a descendant of "/data" despite the string
        # prefix match; classification must respect path-segment boundaries.
        intent = SandboxMountIntent(mount_root="/data", extra_mounts=("/database",))
        assert intent.covered_extras == ()
        assert intent.covering_extras == ()
        assert intent.disjoint_extras == ("/database",)

    def test_mixed_matrix(self):
        intent = SandboxMountIntent(
            mount_root="/data",
            extra_mounts=("/data/sub", "/data", "/", "/other", "/database"),
        )
        assert set(intent.covered_extras) == {"/data/sub", "/data"}
        assert set(intent.covering_extras) == {"/"}
        assert set(intent.disjoint_extras) == {"/other", "/database"}

    def test_no_mount_root_means_all_disjoint(self):
        intent = SandboxMountIntent(mount_root=None, extra_mounts=("/a", "/b"))
        assert intent.covered_extras == ()
        assert intent.covering_extras == ()
        assert intent.disjoint_extras == ("/a", "/b")
