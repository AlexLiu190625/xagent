"""Static AST guard for raw ``Task.status`` storage comparisons.

``xagent.web.models.task.task_status_predicate`` is the one typed entry
point for every SQL predicate and write value against ``Task.status`` (see
that module for why: the column stores enum member *names*, and a raw
literal built from the member value silently miscompiles). This module
implements the forward guard that keeps new code from reintroducing a raw
comparison, and the file-scoped consumption proof used to show that the two
converted production call sites (task_lease_service.py, monitor.py) no
longer contain one.

Five AST shapes count as a raw ``Task.status`` storage comparison:

1. ``Compare`` -- ``Task.status`` appears on either side of ``==``/``!=``/
   ``in``/``not in``.
2. ``Call`` -- ``Task.status.in_(...)``, ``.not_in(...)``, ``.notin_(...)``,
   ``.is_(...)``, ``.is_not(...)``, or ``.isnot(...)``.
3. A ``status=`` keyword argument in a call to a method literally named
   ``values`` or ``update`` whose receiver chain contains -- directly, or
   through a local name assigned from -- a literal ``update(Task)`` or
   ``db.query(Task)``.
4. A ``"status"`` string-key entry reaching such a call, whether the dict is
   supplied inline, by keyword ``**`` spread (name or literal), or as a
   positional argument (literal or name); the entry may come from a dict
   literal or from a later ``name["status"] = value`` subscript assignment.
5. The same as (4) but keyed by the Column object, ``Task.status``, instead
   of the string ``"status"`` -- ``values[Task.status] = status`` and
   ``.update({Task.status: ...})`` both count.

Shapes 3-5 are about the *keyword/key*, which the write-side binding
(``task_status_predicate.value(...)``) cannot hide -- SQLAlchemy requires
the column name there. So a value expression that is exactly a call to
``task_status_predicate.value(...)`` (any object name -- aliasing is out of
scan scope, see below) is treated as compliant, not a violation, for both
string and Column keys.

The receiver-chain analysis in (3)-(5) is a bounded, order-insensitive
fixpoint over local single-target assignments in the enclosing scope: it is
deliberately an over-approximation (a name rebound away from a tracked
statement stays tracked) rather than a reaching-definitions analysis, which
is the safe direction of error for a debt inventory.

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
  of the five shapes and is never flagged.
- ``a2a_protocol.py``'s ``TaskStatus(task.status)`` value lookup is a
  ``Call``, not a scanned shape; it is correct only while ``task.status``
  comes from an ORM-loaded instance (already a member), and raises
  ``ValueError`` if handed a raw storage string.
"""

from __future__ import annotations

import ast
import enum
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCAN_ROOT = REPO_ROOT / "src" / "xagent"

# The binding's own definition file and the class that owns it. Every method
# body on TaskStatusPredicate necessarily contains "Task.status == status"
# etc; excluding that class scope is a structural exclusion (this is the
# binding), not a policy exemption, so it does not go in EXEMPTIONS below.
# Scoped to the class rather than the whole file, so a future module-level
# helper or a second class added to this file is still scanned.
_BINDING_OWNER_FILE = "src/xagent/web/models/task.py"
_BINDING_OWNER_SCOPE = "TaskStatusPredicate"

_IN_LIKE_METHODS = {"in_", "not_in", "notin_", "is_", "is_not", "isnot"}
_WRITE_METHODS = {"values", "update"}


class ViolationKind(str, enum.Enum):
    COMPARE = "compare"
    IN_LIKE_CALL = "in_like_call"
    VALUES_KEYWORD = "values_keyword"
    VALUES_DICT_KEY = "values_dict_key"
    VALUES_COLUMN_KEY = "values_column_key"


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


def _is_task_statement_source(node: ast.AST) -> bool:
    """``update(Task)``, ``sa.update(Task)``, ``db.query(Task)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    is_name = isinstance(func, ast.Name) and func.id in ("update", "query")
    is_attr = isinstance(func, ast.Attribute) and func.attr in ("update", "query")
    if not (is_name or is_attr):
        return False
    return any(isinstance(arg, ast.Name) and arg.id == "Task" for arg in node.args)


def _subtree_has_task_statement(node: ast.AST) -> bool:
    """Whether ``node``'s subtree contains a literal ``update(Task)`` /
    ``query(Task)`` call."""
    return any(_is_task_statement_source(sub) for sub in ast.walk(node))


def _receiver_root_name(node: ast.AST) -> str | None:
    """Innermost ``Name`` at the head of an attribute/call/subscript chain."""
    current = node
    while True:
        if isinstance(current, ast.Name):
            return current.id
        if isinstance(current, ast.Attribute):
            current = current.value
        elif isinstance(current, ast.Call):
            current = current.func
        elif isinstance(current, ast.Subscript):
            current = current.value
        else:
            return None


def _task_statement_names(scope: ast.AST) -> set[str]:
    """Local names holding an ``update(Task)`` / ``query(Task)`` statement.

    Deliberately an over-approximation: no reaching-definitions is done, so a
    name rebound away from a statement stays tracked for the rest of the
    scope. For a debt inventory that direction of error is the safe one. The
    fixpoint is order-insensitive by design (a bounded loop of 4 passes is
    sufficient) so ``stmt = stmt.where(...)`` is caught regardless of walk
    order relative to the original ``stmt = update(Task)`` assignment.
    """
    names: set[str] = set()
    for _ in range(4):
        changed = False
        for node in ast.walk(scope):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                continue
            target = node.targets[0].id
            if target in names:
                continue
            if _subtree_has_task_statement(node.value):
                names.add(target)
                changed = True
                continue
            root = _receiver_root_name(node.value)
            if root is not None and root in names:
                names.add(target)
                changed = True
        if not changed:
            break
    return names


def _status_key_kind(key_node: ast.AST) -> str | None:
    """'str' for a ``"status"`` key, 'col' for a ``Task.status`` key, else
    ``None``. Shared between dict-literal entries and a
    ``name[<key>] = value`` subscript assignment so both recognize the same
    two key shapes."""
    if isinstance(key_node, ast.Constant) and key_node.value == "status":
        return "str"
    if _is_task_status_attr(key_node):
        return "col"
    return None


def _dict_status_entries(node: ast.Dict) -> list[tuple[str, ast.expr]]:
    """('str'|'col', value_node) for ``"status"`` and ``Task.status`` keys."""
    entries: list[tuple[str, ast.expr]] = []
    for key_node, value_node in zip(node.keys, node.values):
        if key_node is None:
            continue  # a ** spread inside the literal, not a direct key
        kind = _status_key_kind(key_node)
        if kind is not None:
            entries.append((kind, value_node))
    return entries


def _status_dict_entries(scope: ast.AST) -> dict[str, list[tuple[str, ast.expr]]]:
    """Local dict names -> their status entries.

    Collects from two statement forms: ``name = {...}`` (every ``"status"`` /
    ``Task.status`` key in the literal) and ``name[<key>] = value`` where
    ``<key>`` is ``"status"`` or ``Task.status`` -- the subscript-assign form
    is what reaches ``task_execution_controller.py``'s
    ``values[Task.status] = status``.
    """
    entries: dict[str, list[tuple[str, ast.expr]]] = {}
    for node in ast.walk(scope):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            found = _dict_status_entries(node.value)
            if found:
                entries.setdefault(node.targets[0].id, []).extend(found)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
        ):
            name = node.targets[0].value.id
            kind = _status_key_kind(node.targets[0].slice)
            if kind is not None:
                entries.setdefault(name, []).append((kind, node.value))
    return entries


def _is_task_status_predicate_value_call(node: ast.AST) -> bool:
    """Whether ``node`` is a call to a ``.value(...)`` write-side entry.

    Matches on the attribute name only (not the receiver's identifier) --
    the receiver is expected to be ``task_status_predicate`` in every
    real call site, but pinning the exact identifier would just move the
    same alias fail-open gap onto the write side without adding coverage.
    This holds for Column keys too, so ``{Task.status:
    task_status_predicate.value(status)}`` is the compliant form of the
    Column-keyed shape.
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
# _scan_write_calls (see its docstring) collapse back to one entry per real
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


def _scan_write_calls(
    scope: ast.AST, path: str, parents: dict[ast.AST, ast.AST]
) -> list[_RawHit]:
    """Shapes 3-5: a ``status``/``Task.status`` value reaching a write call.

    A call is a Task write call when its method is named ``values`` or
    ``update`` and its receiver chain contains -- directly, or through a
    locally tracked name -- an ``update(Task)`` / ``query(Task)`` statement.
    Status targets are collected from the call's ``status=`` keyword, from an
    inline or name-resolved ``**`` spread dict, and from an inline or
    name-resolved positional dict argument.
    """
    hits: list[_RawHit] = []
    statement_names = _task_statement_names(scope)
    dict_entries = _status_dict_entries(scope)

    def _is_task_write_receiver(receiver: ast.AST) -> bool:
        if _subtree_has_task_statement(receiver):
            return True
        root = _receiver_root_name(receiver)
        return root is not None and root in statement_names

    def _kind_for(key_kind: str) -> ViolationKind:
        return (
            ViolationKind.VALUES_DICT_KEY
            if key_kind == "str"
            else ViolationKind.VALUES_COLUMN_KEY
        )

    def _emit(value_node: ast.expr, kind: ViolationKind, detail: str) -> None:
        if _is_task_status_predicate_value_call(value_node):
            return
        hits.append(
            _RawHit(
                path,
                value_node.lineno,
                value_node.col_offset,
                kind,
                _enclosing_scope(value_node, parents),
                ast.unparse(value_node),
                f"{detail} is not routed through task_status_predicate.value(...)",
            )
        )

    for node in ast.walk(scope):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _WRITE_METHODS
            and _is_task_write_receiver(node.func.value)
        ):
            continue

        for kw in node.keywords:
            if kw.arg == "status":
                _emit(kw.value, ViolationKind.VALUES_KEYWORD, "status= keyword")
            elif kw.arg is None:
                if isinstance(kw.value, ast.Dict):
                    for key_kind, value_node in _dict_status_entries(kw.value):
                        _emit(
                            value_node,
                            _kind_for(key_kind),
                            "'status' entry in an inline ** spread dict",
                        )
                elif isinstance(kw.value, ast.Name):
                    for key_kind, value_node in dict_entries.get(kw.value.id, []):
                        _emit(
                            value_node,
                            _kind_for(key_kind),
                            f"'status' entry in {kw.value.id!r}, ** spread",
                        )

        for arg in node.args:
            if isinstance(arg, ast.Dict):
                for key_kind, value_node in _dict_status_entries(arg):
                    _emit(
                        value_node,
                        _kind_for(key_kind),
                        "'status' entry in an inline positional dict",
                    )
            elif isinstance(arg, ast.Name):
                for key_kind, value_node in dict_entries.get(arg.id, []):
                    _emit(
                        value_node,
                        _kind_for(key_kind),
                        f"'status' entry in {arg.id!r}, positional argument",
                    )

    return hits


def _scan_source(source: str, path: str) -> list[TaskStatusViolation]:
    tree = ast.parse(source, filename=path)
    parents = _build_parent_map(tree)
    raw_hits: list[_RawHit] = []

    # Module scope plus every function scope are each walked independently
    # for shapes 3-5 (statement/dict name tracking is scoped per walk
    # target), which revisits a nested function's own write calls once via
    # its own scan and again as part of its enclosing function's/module's
    # subtree walk. Every real hit has a unique (lineno, col_offset) -- the
    # dedup below collapses those re-visits back to one _RawHit per AST
    # node, rather than building a scope-correct single-pass walker for what
    # is a debt inventory, not a hot path.
    for func in ast.walk(tree):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            raw_hits.extend(_scan_write_calls(func, path, parents))

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


def _is_binding_internal(violation: TaskStatusViolation) -> bool:
    """A violation inside the TaskStatusPredicate class body itself."""
    return violation.path == _BINDING_OWNER_FILE and (
        violation.scope == _BINDING_OWNER_SCOPE
        or violation.scope.startswith(_BINDING_OWNER_SCOPE + ".")
    )


def scan_tree(root: Path = SCAN_ROOT) -> list[TaskStatusViolation]:
    violations: list[TaskStatusViolation] = []
    for path in sorted(root.rglob("*.py")):
        violations.extend(scan_file(path))
    return [v for v in violations if not _is_binding_internal(v)]


def format_violations(violations: Sequence[TaskStatusViolation]) -> str:
    """Failure text for an unexempted violation set."""
    return (
        "found a raw Task.status comparison or write not routed through the "
        "typed binding. Use task_status_predicate.eq / ne / in_ / not_in / "
        "is_null / is_not_null for predicates and "
        "task_status_predicate.value(...) for writes -- both defined in "
        "src/xagent/web/models/task.py. If this site predates the binding "
        "and converting it is out of scope, add a reasoned Exemption row to "
        "EXEMPTIONS in tests/web/services/task_status_predicate_guard.py "
        f"instead. Violations: {list(violations)}"
    )


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
# advisory_lineno reflects the baseline audit at the time each row was added
# and is not re-verified; it is not part of matching (see module docstring).
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
    Exemption(
        "src/xagent/web/services/task_execution_controller.py",
        ViolationKind.VALUES_COLUMN_KEY,
        "apply_task_control_transition",
        "status",
        "Column-keyed write (values[Task.status] = status) that predates the "
        "binding; the parameter is annotated TaskStatus but not checked at "
        "runtime, so this site fails closed only at the column's "
        "validate_strings bind check, not at construction time",
        "task_execution_controller",
        advisory_lineno=139,
    ),
    Exemption(
        "src/xagent/web/services/task_orchestrator.py",
        ViolationKind.VALUES_COLUMN_KEY,
        "_claim_turn_no_commit",
        "TaskStatus.RUNNING",
        "Column-keyed ORM bulk update -- Query.update({Task.status: ...}) -- "
        "that predates the binding; the value is a TaskStatus constant today, "
        "so the exposure is a future edit substituting a raw string, caught "
        "only by the column's validate_strings bind check",
        "task_orchestrator",
        advisory_lineno=853,
    ),
)
