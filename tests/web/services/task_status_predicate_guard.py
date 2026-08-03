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
    lineno: int
    kind: ViolationKind
    detail: str

    @property
    def key(self) -> tuple[str, int, ViolationKind]:
        return (self.path, self.lineno, self.kind)


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


def _scan_values_calls(func: ast.AST, path: str) -> list[TaskStatusViolation]:
    violations: list[TaskStatusViolation] = []

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
                    violations.append(
                        TaskStatusViolation(
                            path,
                            kw.value.lineno,
                            ViolationKind.VALUES_KEYWORD,
                            "status= keyword in .values() on update(Task) "
                            "is not routed through task_status_predicate.value(...)",
                        )
                    )
            elif kw.arg is None and isinstance(kw.value, ast.Name):
                assign = status_dict_assigns.get(kw.value.id)
                if assign is not None:
                    dict_node, status_value_node = assign
                    if not _is_task_status_predicate_value_call(status_value_node):
                        violations.append(
                            TaskStatusViolation(
                                path,
                                dict_node.lineno,
                                ViolationKind.VALUES_DICT_KEY,
                                "'status' dict key spread into "
                                f"update(Task).values(**{kw.value.id}) is not "
                                "routed through task_status_predicate.value(...)",
                            )
                        )
    return violations


def _scan_source(source: str, path: str) -> list[TaskStatusViolation]:
    tree = ast.parse(source, filename=path)
    violations: list[TaskStatusViolation] = []

    # Module scope plus every function scope are each walked independently
    # for shape 3/4 (status_dict_assigns is scoped per walk target), which
    # revisits a nested function's own values() calls once via its own scan
    # and again as part of its enclosing function's/module's subtree walk.
    # De-duplicate by (path, lineno, kind) below rather than building a
    # scope-correct single-pass walker for what is a debt inventory, not a
    # hot path.
    for func in ast.walk(tree):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            violations.extend(_scan_values_calls(func, path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and _compare_touches_task_status(node):
            violations.append(
                TaskStatusViolation(
                    path, node.lineno, ViolationKind.COMPARE, "Task.status compare"
                )
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _IN_LIKE_METHODS
            and _is_task_status_attr(node.func.value)
        ):
            violations.append(
                TaskStatusViolation(
                    path,
                    node.lineno,
                    ViolationKind.IN_LIKE_CALL,
                    f"Task.status.{node.func.attr}(...)",
                )
            )

    deduped: dict[tuple[str, int, ViolationKind], TaskStatusViolation] = {}
    for violation in violations:
        deduped[violation.key] = violation
    return list(deduped.values())


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
    lineno: int
    kind: ViolationKind
    reason: str
    owner: str


# Typed Task.status call sites that predate task_status_predicate and are not
# routed through it. New lifecycle code must go through the binding; this
# table is a debt inventory of what is already there, not an endorsement to
# add more. Each entry is checked by
# test_exemptions_still_resolve_to_live_call_sites in
# test_task_status_predicate_guard.py -- a stale line/shape turns that test
# red so the table cannot silently rot into a blanket suppression list.
EXEMPTIONS: tuple[Exemption, ...] = (
    Exemption(
        "src/xagent/web/api/a2a.py",
        210,
        ViolationKind.COMPARE,
        "typed comparison that predates the binding; conversion is confined to "
        "task.py, task_lease_service.py and monitor.py",
        "a2a",
    ),
    Exemption(
        "src/xagent/web/api/a2a.py",
        288,
        ViolationKind.COMPARE,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "a2a",
    ),
    Exemption(
        "src/xagent/web/api/a2a.py",
        1449,
        ViolationKind.VALUES_KEYWORD,
        "typed .values(status=...) write that predates the binding; outside the "
        "three files routed through it",
        "a2a",
    ),
    Exemption(
        "src/xagent/web/api/a2a.py",
        1465,
        ViolationKind.COMPARE,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "a2a",
    ),
    Exemption(
        "src/xagent/web/api/a2a.py",
        1471,
        ViolationKind.IN_LIKE_CALL,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "a2a",
    ),
    Exemption(
        "src/xagent/web/services/triggers.py",
        1934,
        ViolationKind.COMPARE,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "triggers",
    ),
    Exemption(
        "src/xagent/web/api/trace_handlers.py",
        328,
        ViolationKind.COMPARE,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "trace_handlers",
    ),
    Exemption(
        "src/xagent/web/api/websocket.py",
        369,
        ViolationKind.VALUES_KEYWORD,
        "typed .values(status=...) write that predates the binding; outside the "
        "three files routed through it",
        "websocket",
    ),
    Exemption(
        "src/xagent/web/api/websocket.py",
        376,
        ViolationKind.COMPARE,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "websocket",
    ),
    Exemption(
        "src/xagent/web/api/websocket.py",
        389,
        ViolationKind.COMPARE,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "websocket",
    ),
    Exemption(
        "src/xagent/web/api/websocket.py",
        4489,
        ViolationKind.COMPARE,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "websocket",
    ),
    Exemption(
        "src/xagent/web/api/websocket.py",
        7167,
        ViolationKind.COMPARE,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "websocket",
    ),
    Exemption(
        "src/xagent/web/services/task_orchestrator.py",
        839,
        ViolationKind.COMPARE,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "task_orchestrator",
    ),
    Exemption(
        "src/xagent/web/services/task_orchestrator.py",
        841,
        ViolationKind.IN_LIKE_CALL,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "task_orchestrator",
    ),
    Exemption(
        "src/xagent/web/services/task_orchestrator.py",
        933,
        ViolationKind.COMPARE,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "task_orchestrator",
    ),
    Exemption(
        "src/xagent/web/services/task_orchestrator.py",
        1040,
        ViolationKind.COMPARE,
        "typed comparison that predates the binding; outside the three files "
        "routed through it",
        "task_orchestrator",
    ),
    # a2a_protocol.py:479-512 (a2a_task_state_filter) is an existing typed
    # predicate factory that stays on its own typed surface: only new
    # lifecycle code is required to use the binding. Listed per node rather
    # than per function because the guard matches per node.
    Exemption(
        "src/xagent/web/services/a2a_protocol.py",
        491,
        ViolationKind.COMPARE,
        "typed predicate factory (a2a_task_state_filter) that predates the "
        "binding; outside the three files routed through it",
        "a2a_protocol",
    ),
    Exemption(
        "src/xagent/web/services/a2a_protocol.py",
        495,
        ViolationKind.COMPARE,
        "typed predicate factory (a2a_task_state_filter) that predates the "
        "binding; outside the three files routed through it",
        "a2a_protocol",
    ),
    Exemption(
        "src/xagent/web/services/a2a_protocol.py",
        502,
        ViolationKind.COMPARE,
        "typed predicate factory (a2a_task_state_filter) that predates the "
        "binding; outside the three files routed through it",
        "a2a_protocol",
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
