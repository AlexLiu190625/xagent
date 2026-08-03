"""Static AST guard for raw ``Task.status`` storage comparisons.

``xagent.web.models.task.task_status_predicate`` is the one typed entry
point for every SQL predicate and write value against ``Task.status`` (see
that module for why: the column stores enum member *names*, and a raw
literal built from the member value silently miscompiles). This module
implements the forward guard that keeps new code from reintroducing a raw
comparison, and the file-scoped consumption proof used to show that the two
converted production call sites (task_lease_service.py, monitor.py) no
longer contain one.

Four AST shapes count as a raw ``Task.status`` storage comparison:

1. ``Compare`` -- ``Task.status`` appears on either side of ``==``/``!=``/
   ``in``/``not in``.
2. ``Call`` -- ``Task.status.in_(...)``, ``.notin_(...)``, ``.is_(...)``, or
   ``.isnot_(...)``.
3. A ``status=`` keyword argument in a call to a method literally named
   ``values`` whose call chain contains a literal ``update(Task)``.
4. A dict literal with a ``"status"`` string key that is later spread
   (``**name``) into a ``.values(...)`` call matching shape 3.

Shapes 3 and 4 are about the *keyword/key*, which the write-side binding
(``task_status_predicate.value(...)``) cannot hide -- SQLAlchemy requires
the column name there. So a value expression that is exactly a call to
``task_status_predicate.value(...)`` (any object name -- aliasing is out of
scan scope, see below) is treated as compliant, not a violation.

Identity, not line number: a violation's matching key is
``(path, kind, scope, signature, occurrence)`` --
``scope`` is the dotted name of the enclosing function/class (or
``"<module>"``), ``signature`` is ``ast.unparse()`` of the offending
expression (stable across reformatting and unrelated edits elsewhere in the
file), and ``occurrence`` is a 1-based index disambiguating two genuinely
identical expressions in the same scope (this happens for real:
a2a_protocol.py's ``a2a_task_state_filter`` compares
``Task.status == TaskStatus.WAITING_FOR_USER`` twice). ``lineno`` is kept on
``TaskStatusViolation`` and ``Exemption`` as advisory metadata for humans
reading a diff or a failure message -- it is never part of the matching key,
because it drifts every time unrelated code moves in the same file and a
line-keyed exemption table would go stale on every unrelated commit to
``main``, not just ones that touch the exempted site.

Known, accepted scope limits:

- Only the literal identifier ``Task`` is matched
  (``ast.Name(id="Task")``). An aliased import
  (``from ...models.task import Task as T``) is out of scan scope and will
  not be flagged. Fail-open by design: a clean guard run is not proof the
  whole repository has zero raw literals, only that none were found in the
  scanned shapes under the imported name ``Task``.
- The scan is purely syntactic. It does not resolve whether an in-scope
  ``Task`` name actually binds to
  ``xagent.web.models.task.Task``.
- ``db.query(Task.status)`` (a bare projection, not a comparison) is not one
  of the four shapes and is never flagged.
- A write keyed by the *Column object* instead of the column name --
  ``values[Task.status] = status`` spread into
  ``update(Task).values(**values)`` -- is a fifth shape, outside the four
  above, and is not flagged. One such site exists at
  ``src/xagent/web/services/task_execution_controller.py:139``; see
  ``COLUMN_KEYED_WRITE_NOTE`` below for why it is safe and what would break
  that reasoning.
"""

from __future__ import annotations

import ast
import enum
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCAN_ROOT = REPO_ROOT / "src" / "xagent"

# The binding's own definition file. Every method body on TaskStatusPredicate
# necessarily contains "Task.status == status" etc; excluding the file it is
# defined in is a structural exclusion (this is the binding), not a policy
# exemption, so it does not go in EXEMPTIONS below.
_BINDING_OWNER_FILE = "src/xagent/web/models/task.py"

_IN_LIKE_METHODS = {"in_", "notin_", "is_", "isnot_"}


class ViolationKind(str, enum.Enum):
    COMPARE = "compare"
    IN_LIKE_CALL = "in_like_call"
    VALUES_KEYWORD = "values_keyword"
    VALUES_DICT_KEY = "values_dict_key"


@dataclass(frozen=True)
class TaskStatusViolation:
    path: str
    kind: ViolationKind
    scope: str
    signature: str
    occurrence: int
    lineno: int  # advisory only -- not part of identity, see module docstring
    detail: str

    @property
    def identity(self) -> tuple[str, ViolationKind, str, str, int]:
        return (self.path, self.kind, self.scope, self.signature, self.occurrence)


def _is_task_status_attr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "status"
        and isinstance(node.value, ast.Name)
        and node.value.id == "Task"
    )


def _compare_touches_task_status(node: ast.Compare) -> bool:
    sides = [node.left, *node.comparators]
    return any(_is_task_status_attr(side) for side in sides)


def _is_update_task_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "update"
        and any(isinstance(arg, ast.Name) and arg.id == "Task" for arg in node.args)
    )


def _chain_contains_update_task(node: ast.AST) -> bool:
    """Whether ``node``'s subtree contains a literal ``update(Task)`` call."""
    return any(_is_update_task_call(sub) for sub in ast.walk(node))


def _is_task_status_predicate_value_call(node: ast.AST) -> bool:
    """Whether ``node`` is a call to a ``.value(...)`` write-side entry.

    Matches on the attribute name only (not the receiver's identifier) --
    the receiver is expected to be ``task_status_predicate`` in every
    real call site, but pinning the exact identifier would just move the
    same alias fail-open gap onto the write side without adding coverage.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "value"
    )


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Dotted name of the nearest enclosing def(s), or "<module>"."""
    names: list[str] = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
        current = parents.get(current)
    names.reverse()
    return ".".join(names) if names else "<module>"


# A single raw hit before occurrence-disambiguation: keyed by exact source
# position so the module-scope walk and each enclosing function's walk in
# _scan_values_calls (see its docstring) collapse back to one entry per real
# AST node, instead of accidentally colliding two distinct sites that happen
# to share the same scope+signature before occurrence numbers are assigned.
@dataclass(frozen=True)
class _RawHit:
    path: str
    lineno: int
    col_offset: int
    kind: ViolationKind
    scope: str
    signature: str
    detail: str


def _scan_values_calls(
    func: ast.AST, path: str, parents: dict[ast.AST, ast.AST]
) -> list[_RawHit]:
    hits: list[_RawHit] = []

    # Shape 4 requires two passes: collect {"status": ...} dict literals
    # assigned to a name, then check whether that name is later spread into
    # a matching update(Task).values(**name) call in the same function body.
    status_dict_assigns: dict[str, tuple[ast.Dict, ast.AST]] = {}
    for node in ast.walk(func):
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        for key_node, value_node in zip(node.value.keys, node.value.values):
            if isinstance(key_node, ast.Constant) and key_node.value == "status":
                status_dict_assigns[node.targets[0].id] = (node.value, value_node)
                break

    for node in ast.walk(func):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "values"
            and _chain_contains_update_task(node.func.value)
        ):
            continue
        for kw in node.keywords:
            if kw.arg == "status":
                if not _is_task_status_predicate_value_call(kw.value):
                    hits.append(
                        _RawHit(
                            path,
                            kw.value.lineno,
                            kw.value.col_offset,
                            ViolationKind.VALUES_KEYWORD,
                            _enclosing_scope(kw.value, parents),
                            ast.unparse(kw.value),
                            "status= keyword in .values() on update(Task) "
                            "is not routed through task_status_predicate.value(...)",
                        )
                    )
            elif kw.arg is None and isinstance(kw.value, ast.Name):
                assign = status_dict_assigns.get(kw.value.id)
                if assign is not None:
                    dict_node, status_value_node = assign
                    if not _is_task_status_predicate_value_call(status_value_node):
                        hits.append(
                            _RawHit(
                                path,
                                dict_node.lineno,
                                dict_node.col_offset,
                                ViolationKind.VALUES_DICT_KEY,
                                _enclosing_scope(dict_node, parents),
                                # Signature is the "status" entry's value, not
                                # the whole dict literal -- an unrelated key
                                # (e.g. adding "runner_id") must not change
                                # the identity of this violation.
                                ast.unparse(status_value_node),
                                "'status' dict key spread into "
                                f"update(Task).values(**{kw.value.id}) is not "
                                "routed through task_status_predicate.value(...)",
                            )
                        )
    return hits


def _scan_source(source: str, path: str) -> list[TaskStatusViolation]:
    tree = ast.parse(source, filename=path)
    parents = _build_parent_map(tree)
    raw_hits: list[_RawHit] = []

    # Module scope plus every function scope are each walked independently
    # for shape 3/4 (status_dict_assigns is scoped per walk target), which
    # revisits a nested function's own values() calls once via its own scan
    # and again as part of its enclosing function's/module's subtree walk.
    # Every real hit has a unique (lineno, col_offset) -- the dedup below
    # collapses those re-visits back to one _RawHit per AST node, rather
    # than building a scope-correct single-pass walker for what is a debt
    # inventory, not a hot path.
    for func in ast.walk(tree):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            raw_hits.extend(_scan_values_calls(func, path, parents))

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and _compare_touches_task_status(node):
            raw_hits.append(
                _RawHit(
                    path,
                    node.lineno,
                    node.col_offset,
                    ViolationKind.COMPARE,
                    _enclosing_scope(node, parents),
                    ast.unparse(node),
                    "Task.status compare",
                )
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _IN_LIKE_METHODS
            and _is_task_status_attr(node.func.value)
        ):
            raw_hits.append(
                _RawHit(
                    path,
                    node.lineno,
                    node.col_offset,
                    ViolationKind.IN_LIKE_CALL,
                    _enclosing_scope(node, parents),
                    ast.unparse(node),
                    f"Task.status.{node.func.attr}(...)",
                )
            )

    deduped: dict[tuple[str, int, int, ViolationKind], _RawHit] = {}
    for hit in raw_hits:
        deduped[(hit.path, hit.lineno, hit.col_offset, hit.kind)] = hit

    # Assign occurrence numbers within each (path, kind, scope, signature)
    # group, ordered by source position, so two textually identical
    # expressions in the same scope (real example:
    # a2a_protocol.py:a2a_task_state_filter compares
    # "Task.status == TaskStatus.WAITING_FOR_USER" twice) get distinct,
    # stable identities instead of colliding into one.
    groups: dict[tuple[str, ViolationKind, str, str], list[_RawHit]] = {}
    for hit in deduped.values():
        groups.setdefault((hit.path, hit.kind, hit.scope, hit.signature), []).append(
            hit
        )

    violations: list[TaskStatusViolation] = []
    for group_hits in groups.values():
        for occurrence, hit in enumerate(
            sorted(group_hits, key=lambda h: (h.lineno, h.col_offset)), start=1
        ):
            violations.append(
                TaskStatusViolation(
                    path=hit.path,
                    kind=hit.kind,
                    scope=hit.scope,
                    signature=hit.signature,
                    occurrence=occurrence,
                    lineno=hit.lineno,
                    detail=hit.detail,
                )
            )
    return violations


def scan_source(source: str, path: str) -> list[TaskStatusViolation]:
    """Scan one in-memory source string. ``path`` is a label only."""
    return _scan_source(source, path)


def scan_file(path: Path) -> list[TaskStatusViolation]:
    relative = path.resolve().relative_to(REPO_ROOT).as_posix()
    return _scan_source(path.read_text(), relative)


def scan_tree(root: Path = SCAN_ROOT) -> list[TaskStatusViolation]:
    violations: list[TaskStatusViolation] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.resolve().relative_to(REPO_ROOT).as_posix()
        if relative == _BINDING_OWNER_FILE:
            continue
        violations.extend(scan_file(path))
    return violations


@dataclass(frozen=True)
class Exemption:
    path: str
    kind: ViolationKind
    scope: str
    signature: str
    reason: str
    owner: str
    occurrence: int = 1
    # Informational only, for humans reading this table -- never matched
    # against a live violation. See the module docstring for why line
    # numbers cannot be the matching key.
    advisory_lineno: int | None = None

    @property
    def identity(self) -> tuple[str, ViolationKind, str, str, int]:
        return (self.path, self.kind, self.scope, self.signature, self.occurrence)


# Typed Task.status call sites that predate task_status_predicate and are not
# routed through it. New lifecycle code must go through the binding; this
# table is a debt inventory of what is already there, not an endorsement to
# add more. Each entry is checked by
# test_exemptions_still_resolve_to_live_call_sites in
# test_task_status_predicate_guard.py -- a stale scope/signature turns that
# test red so the table cannot silently rot into a blanket suppression list.
# advisory_lineno reflects the 478333c0 baseline audit and is not
# re-verified; it is not part of matching (see module docstring).
EXEMPTIONS: tuple[Exemption, ...] = (
    Exemption(
        "src/xagent/web/api/a2a.py",
        ViolationKind.COMPARE,
        "_acquire_a2a_resume_prelease_sync",
        "Task.status == resumable_status",
        "typed comparison that predates the binding; conversion is confined to "
        "task.py, task_lease_service.py and monitor.py",
        "a2a",
        advisory_lineno=210,
    ),
    Exemption(
        "src/xagent/web/api/a2a.py",
        ViolationKind.COMPARE,
        "_update_a2a_resume_input_sync",
        "Task.status == TaskStatus.RUNNING",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "a2a",
        advisory_lineno=288,
    ),
    Exemption(
        "src/xagent/web/api/a2a.py",
        ViolationKind.VALUES_KEYWORD,
        "_finalize_a2a_cancel_sync",
        "TaskStatus.FAILED",
        "typed .values(status=...) write that predates the binding; outside the "
        "three files routed through it",
        "a2a",
        advisory_lineno=1449,
    ),
    Exemption(
        "src/xagent/web/api/a2a.py",
        ViolationKind.COMPARE,
        "_finalize_a2a_cancel_sync",
        "Task.status == TaskStatus.FAILED",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "a2a",
        advisory_lineno=1465,
    ),
    Exemption(
        "src/xagent/web/api/a2a.py",
        ViolationKind.IN_LIKE_CALL,
        "_finalize_a2a_cancel_sync",
        "Task.status.notin_(_TERMINAL_STATUSES)",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "a2a",
        advisory_lineno=1471,
    ),
    Exemption(
        "src/xagent/web/services/triggers.py",
        ViolationKind.COMPARE,
        "_get_pending_trigger_run_ids",
        "Task.status == TaskStatus.PENDING",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "triggers",
        advisory_lineno=1934,
    ),
    Exemption(
        "src/xagent/web/api/trace_handlers.py",
        ViolationKind.COMPARE,
        "DatabaseTraceHandler._save_trace_event",
        "Task.status == TaskStatus.RUNNING",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "trace_handlers",
        advisory_lineno=328,
    ),
    Exemption(
        "src/xagent/web/api/websocket.py",
        ViolationKind.VALUES_KEYWORD,
        "_terminal_task_error_payload",
        "TaskStatus.FAILED",
        "typed .values(status=...) write that predates the binding; outside the "
        "three files routed through it",
        "websocket",
        advisory_lineno=369,
    ),
    Exemption(
        "src/xagent/web/api/websocket.py",
        ViolationKind.COMPARE,
        "_terminal_task_error_payload",
        "Task.status != TaskStatus.FAILED",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "websocket",
        advisory_lineno=376,
    ),
    Exemption(
        "src/xagent/web/api/websocket.py",
        ViolationKind.COMPARE,
        "_terminal_task_error_payload",
        "Task.status == TaskStatus.RUNNING",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "websocket",
        advisory_lineno=389,
    ),
    Exemption(
        "src/xagent/web/api/websocket.py",
        ViolationKind.COMPARE,
        "_reconcile_websocket_acceptance_graph",
        "Task.status == expected_status",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "websocket",
        advisory_lineno=4489,
    ),
    Exemption(
        "src/xagent/web/api/websocket.py",
        ViolationKind.COMPARE,
        "_apply_pause_requested_isolated",
        "Task.status == TaskStatus.RUNNING",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "websocket",
        advisory_lineno=7167,
    ),
    Exemption(
        "src/xagent/web/services/task_orchestrator.py",
        ViolationKind.COMPARE,
        "_claim_turn_no_commit",
        "Task.status == TaskStatus.PENDING",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "task_orchestrator",
        advisory_lineno=839,
    ),
    Exemption(
        "src/xagent/web/services/task_orchestrator.py",
        ViolationKind.IN_LIKE_CALL,
        "_claim_turn_no_commit",
        "Task.status.in_(_APPENDABLE_STATUSES)",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "task_orchestrator",
        advisory_lineno=841,
    ),
    Exemption(
        "src/xagent/web/services/task_orchestrator.py",
        ViolationKind.COMPARE,
        "_claim_turn_no_commit",
        "Task.status == TaskStatus.RUNNING",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "task_orchestrator",
        advisory_lineno=933,
    ),
    Exemption(
        "src/xagent/web/services/task_orchestrator.py",
        ViolationKind.COMPARE,
        "_reconcile_claimed_turn_after_commit_ack_failure",
        "Task.status == TaskStatus.RUNNING",
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "task_orchestrator",
        advisory_lineno=1040,
    ),
    # a2a_protocol.py:479-512 (a2a_task_state_filter) is an existing typed
    # predicate factory that stays on its own typed surface: only new
    # lifecycle code is required to use the binding. It compares
    # Task.status == TaskStatus.WAITING_FOR_USER twice in the same function
    # (the TASK_STATE_WORKING and TASK_STATE_INPUT_REQUIRED branches), hence
    # occurrence=1 and occurrence=2 below for otherwise-identical entries.
    # It also has a separate, distinctly-signatured Compare inside a list
    # comprehension ("Task.status == status" -- ast.unparse() of a Compare
    # node yields only the comparison itself, not the enclosing
    # comprehension clauses).
    Exemption(
        "src/xagent/web/services/a2a_protocol.py",
        ViolationKind.COMPARE,
        "a2a_task_state_filter",
        "Task.status == status",
        "typed predicate factory (a2a_task_state_filter) that predates the "
        "binding; outside the three files routed through it -- this Compare "
        "is inside a list comprehension over projected_statuses",
        "a2a_protocol",
        advisory_lineno=491,
    ),
    Exemption(
        "src/xagent/web/services/a2a_protocol.py",
        ViolationKind.COMPARE,
        "a2a_task_state_filter",
        "Task.status == TaskStatus.WAITING_FOR_USER",
        "typed predicate factory (a2a_task_state_filter) that predates the "
        "binding; outside the three files routed through it",
        "a2a_protocol",
        occurrence=1,
        advisory_lineno=495,
    ),
    Exemption(
        "src/xagent/web/services/a2a_protocol.py",
        ViolationKind.COMPARE,
        "a2a_task_state_filter",
        "Task.status == TaskStatus.WAITING_FOR_USER",
        "typed predicate factory (a2a_task_state_filter) that predates the "
        "binding; outside the three files routed through it -- second, "
        "textually identical occurrence in the same function (the "
        "TASK_STATE_INPUT_REQUIRED branch)",
        "a2a_protocol",
        occurrence=2,
        advisory_lineno=502,
    ),
)

# a2a_protocol.py:104 is a fragile point, not a guard violation
# (TaskStatus(task.status) is a Call, not one of the four flagged shapes), so
# its safety reasoning is recorded here rather than left unwritten: it is
# correct only because ``task.status`` on a live ORM instance is already a
# ``TaskStatus`` member (the column's Python-side value), and
# ``TaskStatus(member)`` is an identity lookup for a member argument. Feeding
# it a raw storage string (the enum *name*, e.g. "WAITING_FOR_USER") raises
# ValueError instead of returning a member.
A2A_PROTOCOL_VALUE_LOOKUP_NOTE = (
    "src/xagent/web/services/a2a_protocol.py:104 -- "
    "TaskStatus(task.status) is safe only while task.status is sourced from "
    "an ORM-loaded Task instance (already a TaskStatus member); it must "
    "never be called with a raw storage string."
)

# task_execution_controller.py:139 writes Task.status through a Column-object
# dict key (``values[Task.status] = status``), the fifth shape named in this
# module's docstring, which the four scanned shapes do not reach. It is not
# routed through task_status_predicate.value(...), so its only enforcement is
# the column itself: Enum(TaskStatus, validate_strings=True) rejects a raw
# string at bind time. apply_task_control_transition annotates the parameter
# as TaskStatus but performs no runtime check of its own, so widening the
# guard to Column-object keys -- or routing this site through the binding --
# is what would restore construction-time failure here.
COLUMN_KEYED_WRITE_NOTE = (
    "src/xagent/web/services/task_execution_controller.py:139 -- "
    "values[Task.status] = status is a Column-keyed write outside the four "
    "scanned shapes; it fails closed only at the column's validate_strings "
    "bind check, not at construction time."
)
