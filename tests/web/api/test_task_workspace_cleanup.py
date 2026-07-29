"""Deleting a task's workspace finds it whichever spelling wrote it.

``_cleanup_workspace_directory`` runs when no agent is in memory, so it has
to locate the workspace from configuration alone. Two things make that
non-trivial: the uploads dir can be spelled so that the canonical path and
the raw one name different directories (a symlink followed by ``..``), and
``TaskWorkspace``'s constructor creates the tree it is pointed at -- so the
candidate probe has to be side-effect free or the first candidate always
"exists" and the real workspace is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xagent.core.execution_scope import set_execution_scope_resolver
from xagent.web.api.chat import AgentServiceManager

OWNER_ID = 7
TASK_ID = 42
WORKSPACE_ID = f"web_task_{TASK_ID}"


@pytest.fixture(autouse=True)
def _no_resolver():
    set_execution_scope_resolver(None)
    yield
    set_execution_scope_resolver(None)


@pytest.fixture(autouse=True)
def _no_external_dirs(monkeypatch):
    monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", "")


def _make_workspace(base: Path) -> Path:
    workspace = base / WORKSPACE_ID
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "result.txt").write_text("payload")
    return workspace


def test_cleans_the_workspace_at_the_current_spelling(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    workspace = _make_workspace(tmp_path / "uploads" / f"user_{OWNER_ID}")

    AgentServiceManager()._cleanup_workspace_directory(TASK_ID, OWNER_ID)

    assert not workspace.exists()


def test_cleans_a_workspace_written_under_the_pre_migration_spelling(
    tmp_path, monkeypatch
):
    """Upgrade path: the files are where the raw spelling resolved to.

    A deployment whose uploads dir is ``<base>/link/..`` wrote workspaces
    under ``link``'s target, because every consumer resolved the raw spelling
    itself. The canonical path names ``<base>`` instead, so cleanup that only
    knows the current spelling would leave those files behind forever.
    """
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside" / "nested"
    outside.mkdir(parents=True)
    (base / "link").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", f"{base}/link/..")
    legacy_workspace = _make_workspace(outside.parent / f"user_{OWNER_ID}")
    canonical_root = base / f"user_{OWNER_ID}"

    AgentServiceManager()._cleanup_workspace_directory(TASK_ID, OWNER_ID)

    assert not legacy_workspace.exists()
    assert not canonical_root.exists(), "no candidate may be created to be deleted"


def test_probing_candidates_creates_nothing(tmp_path, monkeypatch):
    """The probe cannot be the constructor.

    With nothing on disk, cleanup must leave nothing on disk: constructing a
    ``TaskWorkspace`` per candidate would create the first one's tree, report
    it as found, and delete that instead of searching on.
    """
    uploads = tmp_path / "uploads"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads))

    AgentServiceManager()._cleanup_workspace_directory(TASK_ID, OWNER_ID)

    assert not (uploads / f"user_{OWNER_ID}").exists()
