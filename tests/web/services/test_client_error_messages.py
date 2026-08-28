"""Contracts for the wire-safe projection of connector-runtime failures.

The projection has two halves and both are pinned here: the message adapter
(fail-closed on anything that is not a ``ConnectorRuntimeError``) and the
``(code, details)`` projector whose reason whitelist lives inside
``PublicErrorDetails.__post_init__``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from xagent.core.tools.adapters.vibe.config import RequiredMCPUnavailableError
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.services import client_error_messages
from xagent.web.services.client_error_messages import (
    CLIENT_SAFE_TASK_FAILURE,
    CONNECTOR_RUNTIME_PUBLIC_REASON_PREFIXES,
    CONNECTOR_RUNTIME_PUBLIC_REASONS,
    PublicErrorDetails,
    connector_runtime_client_message,
    connector_runtime_public_error,
)

# Anchored on a real module file rather than on the package: xagent is a
# namespace package, so it has no __file__ of its own and may span trees.
SRC_ROOT = Path(client_error_messages.__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# connector_runtime_client_message
# --------------------------------------------------------------------------


def test_client_message_returns_the_curated_safe_message() -> None:
    error = ConnectorRuntimeError(
        "missing_runtime_context",
        "Required connector runtime context is missing.",
    )

    assert (
        connector_runtime_client_message(error)
        == "Required connector runtime context is missing."
    )


@pytest.mark.parametrize("safe_message", ["", "   ", "\n\t"])
def test_client_message_falls_back_on_a_blank_safe_message(safe_message: str) -> None:
    error = ConnectorRuntimeError("missing_runtime_context", safe_message)

    assert connector_runtime_client_message(error) == CLIENT_SAFE_TASK_FAILURE


@pytest.mark.parametrize(
    "error",
    [
        ValueError("secret-token-xyz"),
        KeyError("secret-token-xyz"),
        RuntimeError("secret-token-xyz"),
        RequiredMCPUnavailableError("secret-token-xyz"),
    ],
)
def test_client_message_is_fail_closed_for_an_incidental_exception(
    error: BaseException,
) -> None:
    """The specific name is not the gate; the isinstance check is."""

    assert connector_runtime_client_message(error) == CLIENT_SAFE_TASK_FAILURE


# --------------------------------------------------------------------------
# I-3b: the reason whitelist lives in the constructor
# --------------------------------------------------------------------------


ILLEGAL_REASONS: list[object] = [
    # The real str(exc) product of validate_runtime_source_key.
    "runtime input key must match [A-Za-z0-9_-]+",
    "The connector could not resolve tenant acme-corp",
    "missing_context.auth_token\nSELECT * FROM connectors",
    "missing_context.'auth_token'",
    "/etc/xagent/connectors/acme.yaml",
    "SELECT value FROM task_connector_runtime_contexts WHERE task_id = 1",
    "x" * 5120,
    object(),
    # Shape-legal, deliberately withheld: both state something about who owns
    # the task or how an authorization check resolved, and this frame reaches
    # anonymous widget and share-link visitors.
    "runtime_owner_mismatch",
    "runtime_task_identity_mismatch",
]


@pytest.mark.parametrize("reason", ILLEGAL_REASONS)
def test_public_error_details_normalizes_reason(reason: object) -> None:
    details = PublicErrorDetails(reason=reason)  # type: ignore[arg-type]

    assert details.reason is None
    assert details.to_wire() == {}


LEGAL_REASONS = [
    "not_provided",
    "store_lost",
    "connector_not_selected",
    "undeclared_context_key",
    "team_env_resolution_failed",
    "team_scope_resolution_failed",
    "runtime_view_resolution_failed",
    "custom_api_config_load_failed",
    "missing_context.auth_token",
    "type_mismatch.context.tenant_id",
    "conflict.secrets.authorization",
]


@pytest.mark.parametrize("reason", LEGAL_REASONS)
def test_public_error_details_keeps_a_listed_reason(reason: str) -> None:
    details = PublicErrorDetails(reason=reason)

    assert details.reason == reason
    assert details.to_wire() == {"reason": reason}


def test_public_error_details_accepts_an_absent_reason() -> None:
    assert PublicErrorDetails(reason=None).to_wire() == {}


# --------------------------------------------------------------------------
# connector_runtime_public_error: the three read tiers of exc.details
# --------------------------------------------------------------------------


def test_public_error_projects_code_and_whitelisted_reason() -> None:
    error = ConnectorRuntimeError(
        "missing_runtime_context",
        "Required connector runtime context is missing.",
        details={"reason": "missing_context.auth_token"},
    )

    assert connector_runtime_public_error(error) == (
        "missing_runtime_context",
        PublicErrorDetails(reason="missing_context.auth_token"),
    )


@pytest.mark.parametrize(
    "details",
    [
        {},
        {"reason": "the connector could not be reached"},
        {"connector_ref": {"id": 7}},
    ],
)
def test_public_error_reads_an_empty_reason_as_a_present_code(
    details: dict[str, object],
) -> None:
    """Read-empty is not read-failed: the code still reaches the client."""

    error = ConnectorRuntimeError("missing_runtime_context", "x", details=details)

    assert connector_runtime_public_error(error) == (
        "missing_runtime_context",
        PublicErrorDetails(reason=None),
    )


def test_public_error_refuses_a_tampered_details_payload() -> None:
    """A details of the wrong shape means the whole instance is untrusted."""

    error = ConnectorRuntimeError("missing_runtime_context", "x")
    error.details = "not a mapping"  # type: ignore[assignment]

    assert connector_runtime_public_error(error) is None


@pytest.mark.parametrize(
    "error",
    [
        ValueError("boom"),
        RuntimeError("boom"),
        RequiredMCPUnavailableError("boom"),
    ],
)
def test_public_error_does_not_project_an_incidental_exception(
    error: BaseException,
) -> None:
    assert connector_runtime_public_error(error) is None


# --------------------------------------------------------------------------
# I-3: nothing but reason can reach the wire
# --------------------------------------------------------------------------


def test_public_error_drops_every_field_but_reason() -> None:
    error = ConnectorRuntimeError(
        "missing_runtime_context",
        "x",
        details={
            "reason": "missing_context.auth_token",
            "internal_sql": "SELECT 1",
            "raw_value": "tenant-secret",
            "connector_ref": {"id": 7, "name": "acme"},
        },
    )

    projected = connector_runtime_public_error(error)

    assert projected is not None
    assert set(projected[1].to_wire()) == {"reason"}


# --------------------------------------------------------------------------
# I-34: ownership of the type
# --------------------------------------------------------------------------


def _python_sources() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def test_public_error_details_is_constructed_in_one_module_only() -> None:
    construction_sites: set[str] = set()
    subclass_sites: set[str] = set()

    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = path.relative_to(SRC_ROOT).as_posix()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "PublicErrorDetails"
            ):
                construction_sites.add(relative)
            if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id == "PublicErrorDetails"
                for base in node.bases
            ):
                subclass_sites.add(relative)

    # Not a security boundary -- __post_init__ and the frame builder's
    # type(...) is check are. This is an ownership boundary: the semantics of
    # this type belong to the projector, so a second construction site or any
    # subclass is a design drift that should be seen in review.
    assert construction_sites == {"web/services/client_error_messages.py"}
    assert subclass_sites == set()


# --------------------------------------------------------------------------
# I-31: the whitelist and the real raise sites stay in step
# --------------------------------------------------------------------------


# Literal reasons that the derivation below finds and that are deliberately
# kept off the wire. The first two state the task's ownership and the outcome
# of an authorization check; the audience of this frame includes anonymous
# widget and share-link visitors. The last two are safe fixed strings but are
# English sentences rather than enum values, and rewriting them would mean
# touching an old path this change has no bearing on. The rest are built from
# an exception message, so their content is not controlled.
DELIBERATELY_NOT_PUBLIC_REASONS = frozenset(
    {
        "runtime_owner_mismatch",
        "runtime_task_identity_mismatch",
        "runtime section must be an object",
        "stored selected refs must be a list",
    }
)


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value
    return constants


def _string_bindings(
    tree: ast.Module, module_constants: dict[str, str]
) -> dict[str, set[str]]:
    """Every ``name = <str>`` binding in the module, flattened across scopes.

    ``module_constants`` is repo-wide so that a reason passed as an imported
    constant (``reason=RUNTIME_SECRET_REASON_NOT_PROVIDED``) still resolves
    without this scan having to follow imports. Flattening scopes is
    deliberate for the same reason: this answers "which literal strings can
    end up in a reason", and over-approximating there is the safe direction.
    """

    bindings: dict[str, set[str]] = {
        name: {value} for name, value in module_constants.items()
    }

    def resolve(node: ast.expr) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            value = module_constants.get(node.id)
            return {value} if value is not None else set()
        if isinstance(node, ast.IfExp):
            return resolve(node.body) | resolve(node.orelse)
        return set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            values = resolve(node.value)
            if not values:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings.setdefault(target.id, set()).update(values)
    return bindings


def _fstring_pattern(node: ast.JoinedStr) -> str | None:
    """Turn an f-string reason into a regex covering everything it can build."""

    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(re.escape(value.value))
        elif isinstance(value, ast.FormattedValue):
            parts.append(".+")
        else:
            return None
    return "^" + "".join(parts) + "$"


def _reason_expressions(tree: ast.Module) -> list[ast.expr]:
    """Every expression that becomes a reason on a ConnectorRuntimeError.

    Derived from the construction target, not from a list of modules: a module
    list would silently stop covering a raise site added somewhere new.
    """

    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "_raise_runtime_error":
            for keyword in node.keywords:
                if keyword.arg == "reason":
                    found.append(keyword.value)
        elif node.func.id == "ConnectorRuntimeError":
            for keyword in node.keywords:
                if keyword.arg != "details" or not isinstance(keyword.value, ast.Dict):
                    continue
                for key, value in zip(keyword.value.keys, keyword.value.values):
                    if isinstance(key, ast.Constant) and key.value == "reason":
                        found.append(value)
    return found


def _derive_reasons() -> tuple[set[str], set[str]]:
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8")) for path in _python_sources()
    }
    module_constants: dict[str, str] = {}
    for tree in trees.values():
        module_constants.update(_module_string_constants(tree))

    literals: set[str] = set()
    patterns: set[str] = set()
    for tree in trees.values():
        bindings = _string_bindings(tree, module_constants)
        for expression in _reason_expressions(tree):
            if isinstance(expression, ast.Constant) and isinstance(
                expression.value, str
            ):
                literals.add(expression.value)
            elif isinstance(expression, ast.Name):
                literals.update(bindings.get(expression.id, set()))
            elif isinstance(expression, ast.JoinedStr):
                pattern = _fstring_pattern(expression)
                if pattern is not None:
                    patterns.add(pattern)
    return literals, patterns


def _is_listed(reason: str) -> bool:
    if reason in CONNECTOR_RUNTIME_PUBLIC_REASONS:
        return True
    prefix, separator, key = reason.rpartition(".")
    return bool(separator) and prefix in CONNECTOR_RUNTIME_PUBLIC_REASON_PREFIXES


def test_public_reason_whitelist_covers_every_raise_site() -> None:
    literals, _ = _derive_reasons()

    assert literals, "the reason derivation found nothing; the scan is broken"

    unclassified = {
        reason
        for reason in literals
        if not _is_listed(reason) and reason not in DELIBERATELY_NOT_PUBLIC_REASONS
    }
    assert not unclassified, (
        "these reasons are raised but neither whitelisted nor listed as "
        f"deliberately withheld: {sorted(unclassified)}"
    )


def test_public_reason_whitelist_has_no_member_without_a_raise_site() -> None:
    """Every listed reason is produced somewhere, with no exemptions.

    Zero exemptions is the point of this assertion. A listed reason nothing
    raises is a standing allowance with no expiry, and by the time the code
    raising it arrives nobody remembers which audience it was judged against.
    A reason therefore enters the whitelist in the same change as the site
    that raises it.
    """

    literals, patterns = _derive_reasons()
    compiled = [re.compile(pattern) for pattern in patterns]

    ungrounded = {
        reason
        for reason in CONNECTOR_RUNTIME_PUBLIC_REASONS
        if reason not in literals
        and not any(expression.match(reason) for expression in compiled)
    }
    assert not ungrounded, (
        f"these whitelisted reasons are produced nowhere in src/: {sorted(ungrounded)}"
    )


def test_the_withheld_reasons_are_really_raised_somewhere() -> None:
    """A withheld entry that nothing raises is a stale exemption."""

    literals, _ = _derive_reasons()

    assert DELIBERATELY_NOT_PUBLIC_REASONS <= literals


def test_knowledge_base_scope_reason_is_not_in_the_derived_surface() -> None:
    """A same-named literal on a different exception class must stay out.

    ``knowledge_base_team_scope`` raises ``KnowledgeBaseScopeError`` with the
    same ``team_scope_resolution_failed`` string. Deriving by construction
    target rather than by module list is what keeps it out on its own.
    """

    path = SRC_ROOT / "web" / "services" / "knowledge_base_team_scope.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    assert _reason_expressions(tree) == []
