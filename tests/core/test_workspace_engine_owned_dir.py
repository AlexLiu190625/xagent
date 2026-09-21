"""The engine-owned tool-results directory is invisible to every listing."""

import errno
import os
from pathlib import Path

import pytest

from xagent.core.tools.core.workspace_file_tool import WorkspaceFileOperations
from xagent.core.tools.tool_result_spill import SPILL_DIR_NAME
from xagent.core.workspace import TaskWorkspace


@pytest.fixture
def workspace(tmp_path):
    return TaskWorkspace("task_w", str(tmp_path))


@pytest.fixture
def spilled(workspace):
    """One engine file in the spill directory and one ordinary output file."""
    spill_dir = workspace.output_dir / SPILL_DIR_NAME
    spill_dir.mkdir(parents=True)
    engine_file = spill_dir / "acme-stored-result.json"
    engine_file.write_text("[]", encoding="utf-8")
    user_file = workspace.output_dir / "report.txt"
    user_file.write_text("report", encoding="utf-8")
    return engine_file, user_file


def test_get_all_files_omits_the_engine_directory(workspace, spilled):
    engine_file, user_file = spilled
    paths = {entry["file_path"] for entry in workspace.get_all_files()["output"]}
    assert str(user_file) in paths
    assert str(engine_file) not in paths


def test_get_output_files_omits_the_engine_directory(workspace, spilled):
    engine_file, user_file = spilled
    paths = {entry["file_path"] for entry in workspace.get_output_files()}
    assert str(user_file) in paths
    assert str(engine_file) not in paths


def test_get_output_files_non_recursive_omits_a_look_alike_file(workspace):
    """The non-recursive branch only scans the top of output/, where the
    engine directory can only be met as a regular file that happens to be
    named exactly like it, not as the directory itself."""
    look_alike = workspace.output_dir / SPILL_DIR_NAME
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    look_alike.write_text("not the engine directory", encoding="utf-8")
    sibling = workspace.output_dir / "ok.txt"
    sibling.write_text("ok", encoding="utf-8")

    paths = {
        entry["file_path"]
        for entry in workspace.get_output_files(include_subdirs=False)
    }
    assert str(sibling) in paths
    assert str(look_alike) not in paths


def test_scan_all_files_omits_the_engine_directory(workspace, spilled):
    engine_file, user_file = spilled
    scanned = workspace._scan_all_files()
    assert user_file in scanned
    assert engine_file not in scanned


# A bare top-level segment such as "output" does not resolve to that
# directory itself: the resolver behind the named-directory branch only reads
# a leading "output"/"input"/"temp" segment when the string contains a slash,
# so a lone "output" is instead treated as a name to look up inside the
# resolver's own default output directory, and raises FileNotFoundError
# because no such nested directory exists. The spellings below are the ones
# that reach the scan for real.
ROOT_OUTPUT_DIRECTORY_SPELLINGS = [
    pytest.param("absolute", id="absolute-output-dir"),
]


def _root_output_directory_path(workspace: TaskWorkspace, spelling: str) -> str:
    if spelling == "absolute":
        return str(workspace.output_dir)
    raise AssertionError(f"unhandled spelling: {spelling}")


@pytest.mark.parametrize("spelling", ROOT_OUTPUT_DIRECTORY_SPELLINGS)
@pytest.mark.parametrize("show_hidden", [False, True])
def test_named_directory_listing_of_output_root_omits_the_engine_directory(
    workspace, spilled, spelling, show_hidden
):
    engine_file, user_file = spilled
    ops = WorkspaceFileOperations(workspace)
    directory_path = _root_output_directory_path(workspace, spelling)
    listing = ops.list_files(directory_path, show_hidden=show_hidden, recursive=True)
    paths = {entry["path"] for entry in listing["files"]}
    assert str(user_file) in paths
    assert str(engine_file) not in paths
    # The directory entry itself is gone too, which is what stops the descent.
    assert str(workspace.output_dir / SPILL_DIR_NAME) not in paths


@pytest.mark.parametrize("show_hidden", [False, True])
def test_named_directory_listing_of_the_engine_directory_itself_is_empty(
    workspace, spilled, show_hidden
):
    """Addressing the engine directory by its own path returns nothing."""
    engine_file, _ = spilled
    ops = WorkspaceFileOperations(workspace)
    listing = ops.list_files(
        f"output/{SPILL_DIR_NAME}", show_hidden=show_hidden, recursive=True
    )
    assert listing["files"] == []
    assert engine_file.exists()


SYMLINK_LOOP_CASES = [
    pytest.param("direct", "symlink_loop", False, True, id="direct-show-hidden-false"),
    pytest.param("direct", "symlink_loop", True, True, id="direct-show-hidden-true"),
    pytest.param("nested", "symlink_loop", False, True, id="nested-show-hidden-false"),
    pytest.param("nested", "symlink_loop", True, True, id="nested-show-hidden-true"),
    pytest.param(
        "direct", ".symlink_loop", True, True, id="direct-dotted-show-hidden-true"
    ),
    pytest.param(
        "direct", ".symlink_loop", False, False, id="direct-dotted-show-hidden-false"
    ),
]


@pytest.mark.parametrize(
    "location, entry_name, show_hidden, reaches_stat", SYMLINK_LOOP_CASES
)
def test_named_directory_listing_reports_a_symlink_loop_as_a_filesystem_error(
    workspace, location, entry_name, show_hidden, reaches_stat
):
    # The ownership check resolves each entry, and a symlink loop cannot be
    # resolved. An entry that reaches the check this way still reaches
    # item.stat() below, so the listing fails with the filesystem's own
    # ELOOP error rather than one raised by the check. A dotted entry with
    # show_hidden off is skipped by the hidden-name rule before the check
    # ever runs, so the listing succeeds and simply omits it.
    parent = (
        workspace.output_dir if location == "direct" else workspace.output_dir / "sub"
    )
    parent.mkdir(parents=True, exist_ok=True)
    loop_path = parent / entry_name
    try:
        os.symlink(loop_path, loop_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available on this platform/user")

    ops = WorkspaceFileOperations(workspace)
    if reaches_stat:
        with pytest.raises(OSError) as raised:
            ops.list_files(
                str(workspace.output_dir), show_hidden=show_hidden, recursive=True
            )
        assert raised.value.errno == errno.ELOOP
    else:
        listing = ops.list_files(
            str(workspace.output_dir), show_hidden=show_hidden, recursive=True
        )
        assert listing["files"] == []


def test_spill_temp_files_are_hidden_too(workspace):
    """The writer's .tmp name is not a dotfile; the directory rule is what hides it."""
    spill_dir = workspace.output_dir / SPILL_DIR_NAME
    spill_dir.mkdir(parents=True)
    leftover = spill_dir / "acme-stored-result.json.4242.partial.tmp"
    leftover.write_text("partial", encoding="utf-8")
    assert leftover not in workspace._scan_all_files()


def test_ordinary_output_subdirectory_is_still_listed(workspace):
    """The rule is a directory rule, not a name-prefix rule."""
    near_miss = workspace.output_dir / f"{SPILL_DIR_NAME}-mine"
    near_miss.mkdir(parents=True)
    mine = near_miss / "x.json"
    mine.write_text("{}", encoding="utf-8")
    assert mine in workspace._scan_all_files()
    assert str(mine) in {e["file_path"] for e in workspace.get_output_files()}


def test_auto_registration_ignores_the_engine_directory(workspace, mocker):
    """A file the engine drops in its own directory never becomes a file record."""
    registered: list[str] = []
    mocker.patch.object(
        TaskWorkspace,
        "register_file",
        lambda self, path, db_session=None: registered.append(str(path)) or "fid",
    )
    spill_dir = workspace.output_dir / SPILL_DIR_NAME
    spill_dir.mkdir(parents=True)
    with workspace.auto_register_files():
        (spill_dir / "acme-stored-result.json").write_text("[]", encoding="utf-8")
        (workspace.output_dir / "report.txt").write_text("report", encoding="utf-8")

    assert registered == [str(workspace.output_dir / "report.txt")]


def test_ownership_holds_when_the_output_dir_itself_is_a_symlink(tmp_path):
    """The check resolves the root too, so a symlinked output/ still matches.

    ``output_dir`` is built by appending names to a resolved ``base_dir``; no
    construction step resolves it, so only the comparison site can see through
    a symlink standing where ``output/`` does. Drop ``.resolve()`` from the
    parent of the reserved segment and both the listing and the write refusal
    go silently permissive.
    """

    workspace = TaskWorkspace("task_symlinked_output", str(tmp_path / "base"))
    elsewhere = workspace.workspace_dir / "real-output"
    (elsewhere / SPILL_DIR_NAME).mkdir(parents=True)
    workspace.output_dir.rmdir()
    try:
        workspace.output_dir.symlink_to(elsewhere, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available on this platform/user")

    engine_file = workspace.output_dir / SPILL_DIR_NAME / "acme-stored-result.json"
    engine_file.write_text("[]", encoding="utf-8")
    user_file = workspace.output_dir / "report.txt"
    user_file.write_text("report", encoding="utf-8")

    assert workspace.is_engine_owned_path(engine_file) is True
    listed = {entry["file_path"] for entry in workspace.get_output_files()}
    assert str(user_file) in listed
    assert str(engine_file) not in listed
    with pytest.raises(ValueError, match="engine-owned"):
        WorkspaceFileOperations(workspace).write_file(
            str(workspace.output_dir / SPILL_DIR_NAME / "mine.txt"), "x"
        )


def _symlink(target, link):
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available on this platform/user")


def test_a_symlink_standing_in_for_the_reserved_name_does_not_move_the_directory(
    workspace,
):
    """The reserved directory is a name under output/, not its symlink target.

    The model's code execution tools run with output/ as their working
    directory, so a direct writer can put a symlink where the reserved name
    would go. Following it would hand that writer the choice of which
    directory is protected: every file under the target would drop out of the
    deliverables and become unwritable.
    """
    reports = workspace.output_dir / "reports"
    reports.mkdir(parents=True)
    report = reports / "quarterly.pdf"
    report.write_text("report", encoding="utf-8")
    _symlink("reports", workspace.output_dir / SPILL_DIR_NAME)

    assert workspace.is_engine_owned_path(report) is False
    assert str(report) in {e["file_path"] for e in workspace.get_output_files()}
    assert str(report) in {e["file_path"] for e in workspace.get_all_files()["output"]}
    assert report in workspace._scan_all_files()
    WorkspaceFileOperations(workspace).write_file(
        f"output/reports/{report.name}", "rewritten"
    )
    assert report.read_text(encoding="utf-8") == "rewritten"


def test_a_loop_at_the_reserved_name_does_not_fail_unrelated_callers(workspace):
    """A loop where the reserved directory would be is one entry's problem.

    Resolving the reserved name itself put the failure on the right-hand side
    of the comparison, where it had nothing to do with the path being asked
    about, so every caller failed about every path.
    """
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    ordinary = workspace.output_dir / "plain.txt"
    ordinary.write_text("plain", encoding="utf-8")
    loop = workspace.output_dir / SPILL_DIR_NAME
    _symlink(loop, loop)

    assert workspace.is_engine_owned_path(ordinary) is False
    assert str(ordinary) in {e["file_path"] for e in workspace.get_output_files()}
    assert str(ordinary) in {
        e["file_path"] for e in workspace.get_all_files()["output"]
    }
    assert ordinary in workspace._scan_all_files()
    WorkspaceFileOperations(workspace).write_file("output/plain.txt", "rewritten")
    assert ordinary.read_text(encoding="utf-8") == "rewritten"


def test_a_loop_at_the_reserved_temp_name_does_not_fail_the_listings(workspace):
    """The temp reserved root is compared the same way, for the same reason."""
    workspace.temp_dir.mkdir(parents=True, exist_ok=True)
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    ordinary = workspace.output_dir / "plain.txt"
    ordinary.write_text("plain", encoding="utf-8")
    loop = workspace.internal_temp_dir
    _symlink(loop, loop)

    all_files = workspace.get_all_files()
    assert str(ordinary) in {e["file_path"] for e in workspace.get_output_files()}
    assert str(ordinary) in {
        e["file_path"] for e in workspace.get_output_files(include_subdirs=False)
    }
    assert str(ordinary) in {e["file_path"] for e in all_files["output"]}
    assert all_files["temp"] == []
    assert ordinary in workspace._scan_all_files()


def test_an_aliased_reserved_temp_name_still_hides_its_target(workspace):
    """Unlike the engine-owned output subtree, this name's target stays hidden.

    The engine creates this directory itself, so the only way its name
    points elsewhere is an alias placed by something else; the files
    already living there stay internal rather than surfacing as new user
    files, because the check answers by name and not by resolving to find
    where the alias physically points.
    """
    workspace.temp_dir.mkdir(parents=True, exist_ok=True)
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    alias_target = workspace.temp_dir / "drafts"
    alias_target.mkdir(parents=True)
    note = alias_target / "note.txt"
    note.write_text("note", encoding="utf-8")
    _symlink(alias_target, workspace.internal_temp_dir)

    all_files = workspace.get_all_files()
    assert all_files["temp"] == []
    assert note not in workspace._scan_all_files()


ESCAPING_ALIAS_TARGETS = [
    pytest.param("..", id="workspace-root"),
    pytest.param("../output", id="output-dir"),
]


@pytest.mark.parametrize("relative_target", ESCAPING_ALIAS_TARGETS)
def test_an_alias_escaping_temp_does_not_hide_the_rest_of_the_workspace(
    workspace, relative_target
):
    """A directory this name points at is reserved only when it sits inside temp/.

    Nothing the engine writes ever lives outside temp/, so an alias that
    escapes it names no scratch data to protect; honouring it anyway would
    let a symlink placed under temp/ blank out listings anywhere else in
    the workspace.
    """
    workspace.temp_dir.mkdir(parents=True, exist_ok=True)
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    plain = workspace.output_dir / "plain.txt"
    plain.write_text("plain", encoding="utf-8")
    _symlink(relative_target, workspace.internal_temp_dir)

    assert str(plain) in {e["file_path"] for e in workspace.get_output_files()}
    assert str(plain) in {
        e["file_path"] for e in workspace.get_output_files(include_subdirs=False)
    }
    assert str(plain) in {e["file_path"] for e in workspace.get_all_files()["output"]}


def test_an_alias_outside_the_workspace_does_not_hide_output(workspace, tmp_path):
    workspace.temp_dir.mkdir(parents=True, exist_ok=True)
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    plain = workspace.output_dir / "plain.txt"
    plain.write_text("plain", encoding="utf-8")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    _symlink(outside, workspace.internal_temp_dir)

    assert str(plain) in {e["file_path"] for e in workspace.get_output_files()}
    assert str(plain) in {
        e["file_path"] for e in workspace.get_output_files(include_subdirs=False)
    }
    assert str(plain) in {e["file_path"] for e in workspace.get_all_files()["output"]}


CASE_SPELLINGS = [
    SPILL_DIR_NAME.upper(),
    SPILL_DIR_NAME.capitalize(),
    SPILL_DIR_NAME.title(),
]


@pytest.mark.parametrize("spelling", CASE_SPELLINGS)
def test_a_case_variant_of_the_reserved_name_is_reserved(workspace, spilled, spelling):
    """One rule on every operating system, so one expectation in one test.

    On a case-insensitive file system this spelling is the engine's own
    directory, and a segment-by-segment comparison would let the write reach
    the engine's bytes. On a case-sensitive file system it is a different
    directory that the rule reserves anyway. Either way the answer, and this
    assertion, are the same.
    """
    engine_file, _ = spilled
    assert spelling != SPILL_DIR_NAME
    target = workspace.output_dir / spelling / engine_file.name
    assert workspace.is_engine_owned_path(target) is True
    with pytest.raises(ValueError, match="engine-owned"):
        WorkspaceFileOperations(workspace).write_file(
            f"output/{spelling}/{engine_file.name}", "rewritten"
        )
    assert engine_file.read_text(encoding="utf-8") == "[]"


OUTPUT_SEGMENT_SPELLINGS = ["OUTPUT", "Output"]


def _respell_output_segment(
    workspace: TaskWorkspace, spelling: str, *rest: str
) -> Path:
    """Build an absolute path with the output/ segment itself respelled.

    Every entry a listing returns is an absolute path (get_output_files and
    friends), so a direct writer already holds one; respelling the segment
    the workspace itself calls "output" is a shape that writer can produce
    without any other knowledge of the workspace layout.
    """
    respelled_output = workspace.output_dir.resolve().with_name(spelling)
    return respelled_output.joinpath(*rest)


@pytest.mark.parametrize("spelling", OUTPUT_SEGMENT_SPELLINGS)
def test_a_case_variant_of_the_output_segment_still_reserves_the_directory(
    workspace, spilled, spelling
):
    """The whole path down to the reserved name is compared case folded,
    not only the reserved name's own segment."""
    engine_file, _ = spilled
    target = _respell_output_segment(
        workspace, spelling, SPILL_DIR_NAME, engine_file.name
    )
    assert workspace.is_engine_owned_path(target) is True
    with pytest.raises(ValueError, match="engine-owned"):
        WorkspaceFileOperations(workspace).write_file(str(target), "rewritten")
    assert engine_file.read_text(encoding="utf-8") == "[]"


@pytest.mark.parametrize("spelling", OUTPUT_SEGMENT_SPELLINGS)
def test_a_case_variant_of_the_output_segment_refuses_delete_too(
    workspace, spilled, spelling
):
    engine_file, _ = spilled
    target = _respell_output_segment(
        workspace, spelling, SPILL_DIR_NAME, engine_file.name
    )
    with pytest.raises(ValueError, match="engine-owned"):
        WorkspaceFileOperations(workspace).delete_file(str(target))
    assert engine_file.exists()
    assert engine_file.read_text(encoding="utf-8") == "[]"


@pytest.mark.parametrize("spelling", OUTPUT_SEGMENT_SPELLINGS)
def test_a_case_variant_of_the_output_segment_is_refused_before_the_directory_exists(
    workspace, spelling
):
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    target = _respell_output_segment(workspace, spelling, SPILL_DIR_NAME, "x.json")
    with pytest.raises(ValueError, match="engine-owned"):
        WorkspaceFileOperations(workspace).write_file(str(target), "planted")
    assert not (workspace.output_dir / SPILL_DIR_NAME).exists()


def test_output_dir_itself_is_not_engine_owned(workspace):
    """The reserved subtree sits inside output/; output/ is not the subtree."""
    assert workspace.is_engine_owned_path(workspace.output_dir) is False


def test_a_plain_file_directly_in_output_is_not_engine_owned(workspace):
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    plain = workspace.output_dir / "report.txt"
    plain.write_text("report", encoding="utf-8")
    assert workspace.is_engine_owned_path(plain) is False


def test_the_reserved_name_only_matters_directly_under_output(workspace):
    """The rule reserves a name one level under output/, not the name anywhere."""
    workspace.temp_dir.mkdir(parents=True, exist_ok=True)
    workspace.input_dir.mkdir(parents=True, exist_ok=True)
    under_temp = workspace.temp_dir / SPILL_DIR_NAME / "x.json"
    under_input = workspace.input_dir / SPILL_DIR_NAME / "x.json"

    assert workspace.is_engine_owned_path(under_temp) is False
    assert workspace.is_engine_owned_path(under_input) is False


def test_a_same_named_directory_nested_deeper_than_the_first_level_is_ordinary(
    workspace,
):
    """The rule matches only the first segment under output/, not any depth."""
    nested = workspace.output_dir / "sub" / SPILL_DIR_NAME
    nested.mkdir(parents=True)
    deep_file = nested / "x.json"
    deep_file.write_text("{}", encoding="utf-8")

    assert workspace.is_engine_owned_path(deep_file) is False
    assert deep_file in workspace._scan_all_files()
    assert str(deep_file) in {e["file_path"] for e in workspace.get_output_files()}
    WorkspaceFileOperations(workspace).write_file(
        f"output/sub/{SPILL_DIR_NAME}/rewritten.json", "mine"
    )
    assert (nested / "rewritten.json").read_text(encoding="utf-8") == "mine"


def test_a_case_variant_nested_deeper_than_the_first_level_is_also_ordinary(workspace):
    """A case variant of the reserved name only matters at the first segment."""
    nested = workspace.output_dir / "sub" / SPILL_DIR_NAME.upper()
    nested.mkdir(parents=True)
    deep_file = nested / "x.json"
    deep_file.write_text("{}", encoding="utf-8")

    assert workspace.is_engine_owned_path(deep_file) is False
    assert deep_file in workspace._scan_all_files()


NEAR_MISS_DIRECTORY_NAMES = [f"{SPILL_DIR_NAME}-backup", f"a{SPILL_DIR_NAME}"]


@pytest.mark.parametrize("name", NEAR_MISS_DIRECTORY_NAMES)
def test_a_directory_whose_name_only_resembles_the_reserved_name_is_ordinary(
    workspace, name
):
    near_miss = workspace.output_dir / name
    near_miss.mkdir(parents=True)
    sibling = near_miss / "x.json"
    sibling.write_text("{}", encoding="utf-8")

    assert workspace.is_engine_owned_path(sibling) is False
    assert sibling in workspace._scan_all_files()
    assert str(sibling) in {e["file_path"] for e in workspace.get_output_files()}


def test_a_file_whose_name_only_resembles_the_reserved_name_is_ordinary(workspace):
    workspace.output_dir.mkdir(parents=True, exist_ok=True)
    look_alike = workspace.output_dir / f"{SPILL_DIR_NAME}.txt"
    look_alike.write_text("not the engine directory", encoding="utf-8")

    assert workspace.is_engine_owned_path(look_alike) is False
    assert look_alike in workspace._scan_all_files()
