"""Forward guard and consumption proof for Task.status storage comparisons.

See task_status_predicate_guard.py for the five scanned AST shapes and the
exemption table's provenance (21 rows). Two independent claims are pinned
here:

1. Every raw Task.status comparison/write anywhere under src/xagent is
   either absent or an explicitly reasoned, still-live exemption -- a new
   file (or new code in an existing file) that adds one fails closed.
2. The two production call sites routed through the binding
   (task_lease_service.py, monitor.py) have zero raw comparisons at all --
   full conversion, not "mostly converted".
"""

from __future__ import annotations

import ast

from tests.web.services.task_status_predicate_guard import (
    _BINDING_OWNER_FILE,
    _BINDING_OWNER_SCOPE,
    EXEMPTIONS,
    REPO_ROOT,
    SCAN_ROOT,
    TaskStatusViolation,
    ViolationKind,
    _is_binding_internal,
    format_violations,
    scan_file,
    scan_source,
    scan_tree,
)


def test_forward_guard_flags_only_exempted_sites() -> None:
    exempt_ids = {e.identity for e in EXEMPTIONS}
    violations = scan_tree()
    unexempted = [v for v in violations if v.identity not in exempt_ids]
    assert unexempted == [], format_violations(unexempted)


def test_exemptions_still_resolve_to_live_call_sites() -> None:
    """Every exemption row must still point at the violation it claims.

    Matching is by stable identity (path, kind, enclosing scope, ast.unparse
    signature, occurrence) -- not line number, which drifts every time
    unrelated code moves in the same file. Without this test, the
    exemption table degrades into a write-only suppression list: a site is
    rewritten or removed, and nobody notices because the forward guard
    above only ever complains about *new* unexempted violations.
    """
    stale: list[str] = []
    for exemption in EXEMPTIONS:
        path = REPO_ROOT / exemption.path
        live_ids = {v.identity for v in scan_file(path)}
        if exemption.identity not in live_ids:
            stale.append(
                f"{exemption.path} scope={exemption.scope!r} "
                f"sig={exemption.signature!r} occ={exemption.occurrence} "
                f"({exemption.kind.value}) owner={exemption.owner!r} "
                f"[advisory_lineno={exemption.advisory_lineno}] no longer "
                "resolves to a live Task.status violation -- update or "
                "remove this exemption"
            )
    assert stale == [], stale


def test_task_lease_service_and_monitor_have_no_raw_status_comparisons() -> None:
    """File-scoped consumption proof for the two converted production sites.

    Compiled-SQL equality (elsewhere in test_task_status_storage.py) proves
    the binding produces the same predicate as the code it replaced; it
    does not prove the raw comparison is actually gone. This asserts the
    AST shape itself is absent, with zero exemptions allowed here -- these
    two files are the binding's production consumers, not existing debt.
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


def test_scan_root_is_src_xagent() -> None:
    # Cheap sanity pin: a future refactor that narrows SCAN_ROOT to one
    # subpackage would silently shrink the forward guard's coverage without
    # any test going red elsewhere.
    assert SCAN_ROOT == REPO_ROOT / "src" / "xagent"


def test_guard_detects_attribute_style_update_calls() -> None:
    source = (
        "import sqlalchemy as sa\n"
        "def f(db):\n"
        "    db.execute(sa.update(Task).values(status=TaskStatus.RUNNING))\n"
    )
    violations = scan_source(source, "fixture_attribute_update.py")
    assert {v.kind for v in violations} == {ViolationKind.VALUES_KEYWORD}, (
        "an update(Task) call reached through an attribute (sa.update, "
        "sqlalchemy.update) must be scanned the same as a bare update(Task)"
    )


# --- Meta-tests: prove the guard itself actually fires --------------------

_FIXTURE_ALL_SHAPES = """
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

def values_column_key_shape(db, status):
    values = {Task.control_state: "idle"}
    values[Task.status] = status
    return db.execute(update(Task).where(Task.id == 1).values(**values))
"""

_FIXTURE_IN_LIKE_SPELLINGS = """
from xagent.web.models.task import Task, TaskStatus

def in_spelling():
    return Task.status.in_([TaskStatus.RUNNING])

def not_in_spelling():
    return Task.status.not_in([TaskStatus.RUNNING])

def notin_spelling():
    return Task.status.notin_([TaskStatus.RUNNING])

def is_spelling():
    return Task.status.is_(None)

def is_not_spelling():
    return Task.status.is_not(None)

def isnot_spelling():
    return Task.status.isnot(None)
"""

_FIXTURE_STAGED_UPDATE = """
from sqlalchemy import update
from xagent.web.models.task import Task, TaskStatus

def staged_values_after_where(db):
    stmt = update(Task).where(Task.id == 1)
    stmt = stmt.values(status=TaskStatus.FAILED)
    return db.execute(stmt)

def staged_where_after_update_then_values(db):
    stmt = update(Task)
    stmt = stmt.where(Task.id == 1)
    return db.execute(stmt.values(status=TaskStatus.FAILED))

def staged_values_dict_supplied_by_name(db):
    stmt = update(Task).where(Task.id == 1)
    payload = {"status": TaskStatus.FAILED}
    return db.execute(stmt.values(payload))
"""

_FIXTURE_POSITIONAL_AND_INLINE_DICTS = """
from sqlalchemy import update
from xagent.web.models.task import Task, TaskStatus

def positional_dict_literal(db):
    return db.execute(
        update(Task).where(Task.id == 1).values({"status": TaskStatus.FAILED})
    )

def inline_spread_dict_literal(db):
    return db.execute(
        update(Task).where(Task.id == 1).values(**{"status": TaskStatus.FAILED})
    )

def positional_dict_by_name(db):
    payload = {"status": TaskStatus.FAILED}
    return db.execute(update(Task).where(Task.id == 1).values(payload))
"""

_FIXTURE_QUERY_UPDATE = """
from xagent.web.models.task import Task, TaskStatus

def string_keyed_query_update(db):
    return (
        db.query(Task)
        .filter(Task.id == 1)
        .update({"status": TaskStatus.FAILED}, synchronize_session=False)
    )

def column_keyed_query_update(db):
    return (
        db.query(Task)
        .filter(Task.id == 1)
        .update({Task.status: TaskStatus.FAILED}, synchronize_session=False)
    )
"""

_FIXTURE_COLUMN_KEYED_WRITE = """
from sqlalchemy import update
from xagent.web.models.task import Task, TaskStatus

def column_keyed_write(db, status):
    values = {Task.control_state: "idle"}
    values[Task.status] = status
    statement = update(Task).where(Task.id == 1)
    return db.execute(statement.values(values))
"""

_FIXTURE_NEGATIVE_CONTROLS = """
from sqlalchemy import update
from xagent.web.models.task import (
    Task as AliasedTask,
    TaskStatus,
    task_status_predicate,
)
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

def plain_dict_update_is_not_a_task_write():
    payload = {}
    payload.update({"status": "ok"})
    return payload

def null_check_through_the_binding_is_compliant():
    return task_status_predicate.is_null()

def column_keyed_write_through_the_binding_is_compliant(db, status):
    return db.query(Task).update({Task.status: task_status_predicate.value(status)})

def staged_write_through_the_binding_is_compliant(db, status):
    stmt = update(Task).where(Task.id == 1)
    stmt = stmt.values(status=task_status_predicate.value(status))
    return db.execute(stmt)
"""


def test_guard_detects_every_scanned_shape() -> None:
    violations = scan_source(_FIXTURE_ALL_SHAPES, "fixture_all_shapes.py")
    kinds_found = {v.kind for v in violations}
    # Hard-coded literal set, not set(ViolationKind) alone -- asserting only
    # against set(ViolationKind) would stay green if a kind and its
    # detection were deleted together.
    assert kinds_found == {
        ViolationKind.COMPARE,
        ViolationKind.IN_LIKE_CALL,
        ViolationKind.VALUES_KEYWORD,
        ViolationKind.VALUES_DICT_KEY,
        ViolationKind.VALUES_COLUMN_KEY,
    }, (
        "the forward-guard fixture must exercise every scanned shape; a "
        f"shape stopped firing (found: {kinds_found}) -- this is the "
        "guard's own regression test, not a scan of real source"
    )
    assert kinds_found == set(ViolationKind), (
        "a ViolationKind exists with no fixture case exercising it: "
        f"{set(ViolationKind) - kinds_found}"
    )


def test_guard_detects_all_in_like_spellings() -> None:
    violations = scan_source(_FIXTURE_IN_LIKE_SPELLINGS, "fixture_in_like_spellings.py")
    in_like = [v for v in violations if v.kind == ViolationKind.IN_LIKE_CALL]
    spellings = {v.signature.split(".status.", 1)[1].split("(", 1)[0] for v in in_like}
    # Hard-coded literal frozenset, not _IN_LIKE_METHODS -- against
    # _IN_LIKE_METHODS both sides would shrink together on a bad edit and
    # the test would stay green, which is the exact vacuity being guarded
    # against.
    assert spellings == {"in_", "not_in", "notin_", "is_", "is_not", "isnot"}, spellings


def test_guard_detects_staged_update_statements() -> None:
    violations = scan_source(_FIXTURE_STAGED_UPDATE, "fixture_staged_update.py")
    pairs = {(v.scope, v.kind) for v in violations}
    assert pairs == {
        ("staged_values_after_where", ViolationKind.VALUES_KEYWORD),
        ("staged_where_after_update_then_values", ViolationKind.VALUES_KEYWORD),
        ("staged_values_dict_supplied_by_name", ViolationKind.VALUES_DICT_KEY),
    }, pairs


def test_guard_detects_positional_and_inline_values_dicts() -> None:
    violations = scan_source(
        _FIXTURE_POSITIONAL_AND_INLINE_DICTS, "fixture_positional_and_inline.py"
    )
    assert len(violations) == 3, violations
    assert {v.kind for v in violations} == {ViolationKind.VALUES_DICT_KEY}


def test_guard_detects_orm_query_update_writes() -> None:
    violations = scan_source(_FIXTURE_QUERY_UPDATE, "fixture_query_update.py")
    assert len(violations) == 2, violations
    assert {v.kind for v in violations} == {
        ViolationKind.VALUES_DICT_KEY,
        ViolationKind.VALUES_COLUMN_KEY,
    }


def test_guard_detects_column_keyed_writes() -> None:
    """Replaces the old test_guard_does_not_flag_the_column_keyed_write_shape
    -- the guard now covers this shape instead of documenting it as a gap.
    """
    violations = scan_source(_FIXTURE_COLUMN_KEYED_WRITE, "fixture_column_keyed.py")
    assert len(violations) == 1, violations
    (violation,) = violations
    assert violation.kind == ViolationKind.VALUES_COLUMN_KEY
    assert violation.signature == "status"


def test_guard_ignores_documented_negative_controls() -> None:
    violations = scan_source(_FIXTURE_NEGATIVE_CONTROLS, "fixture_negative_controls.py")
    assert violations == [], (
        "the guard must not flag: aliased Task imports (fail-open by "
        "design), bare Task.status projections, unrelated dicts with a "
        "'status' key, other models' .values(status=...) calls, a write "
        "already routed through task_status_predicate.value() (string or "
        "Column key, inline or staged), a null check through the binding, "
        f"or dict.update() on an unrelated object: {violations}"
    )


def test_fixtures_are_syntactically_valid() -> None:
    # Guard the guard's own test fixtures against a typo silently turning a
    # meta-test into a vacuous pass (ast.parse would just raise and fail the
    # test loudly, but assert it explicitly so the intent is on the page).
    ast.parse(_FIXTURE_ALL_SHAPES)
    ast.parse(_FIXTURE_IN_LIKE_SPELLINGS)
    ast.parse(_FIXTURE_STAGED_UPDATE)
    ast.parse(_FIXTURE_POSITIONAL_AND_INLINE_DICTS)
    ast.parse(_FIXTURE_QUERY_UPDATE)
    ast.parse(_FIXTURE_COLUMN_KEYED_WRITE)
    ast.parse(_FIXTURE_NEGATIVE_CONTROLS)


def test_binding_owner_exclusion_is_scoped_to_the_predicate_class() -> None:
    file_violations = scan_file(REPO_ROOT / _BINDING_OWNER_FILE)
    assert file_violations != [], (
        "task.py should still contain the binding's own raw expressions"
    )
    assert all(
        v.scope == _BINDING_OWNER_SCOPE
        or v.scope.startswith(_BINDING_OWNER_SCOPE + ".")
        for v in file_violations
    ), file_violations

    assert all(v.path != _BINDING_OWNER_FILE for v in scan_tree())

    def _violation(scope: str, path: str = _BINDING_OWNER_FILE) -> TaskStatusViolation:
        return TaskStatusViolation(
            path=path,
            kind=ViolationKind.COMPARE,
            scope=scope,
            signature="x",
            occurrence=1,
            lineno=1,
            detail="",
        )

    assert _is_binding_internal(_violation("TaskStatusPredicate.eq"))
    assert not _is_binding_internal(_violation("_require_task_status_members"))
    assert not _is_binding_internal(_violation("TaskStatusPredicateHelper.foo"))
    assert not _is_binding_internal(
        _violation("TaskStatusPredicate.eq", path="src/xagent/web/api/other.py")
    )


def test_guard_failure_message_points_at_the_binding_and_exemption_table() -> None:
    violations = scan_source(_FIXTURE_ALL_SHAPES, "fixture_all_shapes.py")
    message = format_violations(violations)
    for token in (
        "task_status_predicate",
        "is_null",
        ".value(",
        "src/xagent/web/models/task.py",
        "EXEMPTIONS",
        "task_status_predicate_guard.py",
    ):
        assert token in message, f"failure message missing {token!r}: {message}"
