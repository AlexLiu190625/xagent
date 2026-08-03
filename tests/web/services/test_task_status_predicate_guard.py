"""Forward guard and consumption proof for Task.status storage comparisons.

See task_status_predicate_guard.py for the four scanned AST shapes and the
exemption table's provenance. Two independent claims are pinned here:

1. Every raw Task.status comparison/write anywhere under src/xagent is
   either absent or an explicitly reasoned, still-live exemption -- a new
   file (or new code in an existing file) that adds one fails closed.
2. The two production call sites this PR actually converts
   (task_lease_service.py, monitor.py) have zero raw comparisons at all --
   full conversion, not "mostly converted".
"""

from __future__ import annotations

import ast

from tests.web.services.task_status_predicate_guard import (
    A2A_PROTOCOL_VALUE_LOOKUP_NOTE,
    EXEMPTIONS,
    REPO_ROOT,
    SCAN_ROOT,
    ViolationKind,
    scan_file,
    scan_source,
    scan_tree,
)


def test_forward_guard_flags_only_exempted_sites() -> None:
    exempt_keys = {(e.path, e.lineno, e.kind) for e in EXEMPTIONS}
    violations = scan_tree()
    unexempted = [v for v in violations if v.key not in exempt_keys]
    assert unexempted == [], (
        "raw Task.status comparison/write outside the typed binding "
        "(task_status_predicate) with no exemption entry -- route it "
        f"through the binding or add a reasoned exemption: {unexempted}"
    )


def test_exemptions_still_resolve_to_live_call_sites() -> None:
    """Every exemption row must still point at the violation it claims.

    Without this, the exemption table degrades into a write-only
    suppression list: a line renumbers or the code is rewritten to no
    longer need the exemption, and nobody notices because the forward
    guard above only ever complains about *new* unexempted violations.
    """
    stale: list[str] = []
    for exemption in EXEMPTIONS:
        path = REPO_ROOT / exemption.path
        violations = {v.key: v for v in scan_file(path)}
        key = (exemption.path, exemption.lineno, exemption.kind)
        if key not in violations:
            stale.append(
                f"{exemption.path}:{exemption.lineno} ({exemption.kind.value}) "
                f"owner={exemption.owner!r} no longer resolves to a live "
                "Task.status violation -- update or remove this exemption"
            )
    assert stale == [], stale


def test_task_lease_service_and_monitor_have_no_raw_status_comparisons() -> None:
    """File-scoped consumption proof for the two converted production sites.

    Compiled-SQL equality (elsewhere in test_task_status_storage.py) proves
    the binding produces the same predicate as the code it replaced; it
    does not prove the raw comparison is actually gone. This asserts the
    AST shape itself is absent, with zero exemptions allowed here -- these
    two files are the PR's production consumers, not pre-existing debt.
    """
    for relative in (
        "src/xagent/web/services/task_lease_service.py",
        "src/xagent/web/api/monitor.py",
    ):
        violations = scan_file(REPO_ROOT / relative)
        assert violations == [], (
            f"{relative} still has a raw Task.status comparison/write not "
            f"routed through task_status_predicate: {violations}"
        )


# --- Meta-tests: prove the guard itself actually fires (R-F3 fixture) -----

_FIXTURE_ALL_FOUR_SHAPES = """
from sqlalchemy import update
from xagent.web.models.task import Task, TaskStatus

def compare_shape():
    return Task.status == TaskStatus.RUNNING

def in_like_call_shape():
    return Task.status.in_([TaskStatus.RUNNING, TaskStatus.PENDING])

def values_keyword_shape(db):
    return db.execute(
        update(Task).where(Task.id == 1).values(status=TaskStatus.FAILED)
    )

def values_dict_key_shape(db):
    values = {"status": TaskStatus.FAILED, "runner_id": None}
    return db.execute(update(Task).where(Task.id == 1).values(**values))
"""

_FIXTURE_NEGATIVE_CONTROLS = """
from sqlalchemy import update
from xagent.web.models.task import Task as AliasedTask, TaskStatus
from xagent.web.models.agent import Agent

def aliased_import_is_out_of_scan_scope(db):
    # Fail-open by design: only the literal name "Task" is matched.
    return AliasedTask.status == TaskStatus.RUNNING

def projection_is_not_a_comparison(db):
    return db.query(Task.status).all()

def unrelated_status_dict_is_not_a_task_write():
    return {"status": "success", "message": "ok"}

def other_model_values_call_is_not_flagged(db):
    return db.execute(update(Agent).where(Agent.id == 1).values(status="draft"))

def wrapped_write_side_is_compliant(db):
    return db.execute(
        update(Task)
        .where(Task.id == 1)
        .values(status=task_status_predicate.value(TaskStatus.FAILED))
    )
"""


def test_guard_detects_all_four_node_shapes() -> None:
    violations = scan_source(_FIXTURE_ALL_FOUR_SHAPES, "fixture_all_four_shapes.py")
    kinds_found = {v.kind for v in violations}
    assert kinds_found == {
        ViolationKind.COMPARE,
        ViolationKind.IN_LIKE_CALL,
        ViolationKind.VALUES_KEYWORD,
        ViolationKind.VALUES_DICT_KEY,
    }, (
        "the forward-guard fixture must exercise every scanned shape; a "
        f"shape stopped firing (found: {kinds_found}) -- this is the "
        "guard's own regression test, not a scan of real source"
    )


def test_guard_ignores_documented_negative_controls() -> None:
    violations = scan_source(_FIXTURE_NEGATIVE_CONTROLS, "fixture_negative_controls.py")
    assert violations == [], (
        "the guard must not flag: aliased Task imports (fail-open by "
        "design), bare Task.status projections, unrelated dicts with a "
        "'status' key, other models' .values(status=...) calls, or a "
        f"write already routed through task_status_predicate.value(): {violations}"
    )


def test_fixtures_are_syntactically_valid() -> None:
    # Guard the guard's own test fixtures against a typo silently turning a
    # meta-test into a vacuous pass (ast.parse would just raise and fail the
    # test loudly, but assert it explicitly so the intent is on the page).
    ast.parse(_FIXTURE_ALL_FOUR_SHAPES)
    ast.parse(_FIXTURE_NEGATIVE_CONTROLS)


def test_a2a_protocol_value_lookup_note_still_targets_live_code() -> None:
    """FREEZE record item 2: a2a_protocol.py:104 is not a guard violation
    (TaskStatus(task.status) is a Call, not one of the four flagged
    shapes), so its safety reasoning is recorded here instead, and pinned
    against the line it describes so a refactor there does not leave a
    stale claim unwritten anywhere.
    """
    path = REPO_ROOT / "src/xagent/web/services/a2a_protocol.py"
    lines = path.read_text().splitlines()
    target_line = lines[103]  # 1-indexed line 104
    assert "TaskStatus(task.status)" in target_line, (
        "a2a_protocol.py:104 changed -- update A2A_PROTOCOL_VALUE_LOOKUP_NOTE "
        f"(and this test) to match. Found: {target_line!r}"
    )
    assert "TaskStatus(task.status)" in A2A_PROTOCOL_VALUE_LOOKUP_NOTE


def test_scan_root_is_src_xagent() -> None:
    # Cheap sanity pin: a future refactor that narrows SCAN_ROOT to one
    # subpackage would silently shrink the forward guard's coverage without
    # any test going red elsewhere.
    assert SCAN_ROOT == REPO_ROOT / "src" / "xagent"
