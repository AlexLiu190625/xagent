"""Optional application hooks for team-owned MCP and Custom API connectors.

Standalone xagent keeps connectors user-owned. A multi-tenant application can
install these hooks to overlay team visibility without teaching xagent about
the application's team tables.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from sqlalchemy.sql.elements import ColumnElement

if TYPE_CHECKING:
    from ..models.custom_api import UserCustomApi
    from ..models.mcp import UserMCPServer

from ...core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ConnectorRuntimeError,
)

logger = logging.getLogger(__name__)

ConnectorType = Literal["mcp", "custom_api"]
ConnectorRenamedHook = Callable[[Any, int, ConnectorType, int, str, str], None]


@dataclass(frozen=True)
class ConnectorDeleteDecision:
    team_owned: bool = False
    authorized: bool = False
    delete_definition: bool = False
    # Set when the delete is refused because the connector is still selected by a
    # team agent. The endpoint surfaces this as a 403 before any mutation, mirroring
    # the unshare path's "still used by a team agent" guard.
    blocked_reason: str | None = None


ConnectorDeletedHook = Callable[[Any, int, ConnectorType, int], ConnectorDeleteDecision]


@dataclass(frozen=True)
class ConnectorAccess:
    """Whether the caller's team links a connector, and may edit it.

    A verdict that reaches a caller always carries ``team_owned=True``:
    the only way to say "the caller's team does not link this connector"
    is to leave its ref out of the hook's answer map entirely, not to
    return a verdict with ``team_owned=False``. ``can_edit`` is otherwise
    independent -- a team can link a connector without granting edit
    rights to it, which is a legal answer on its own, not an intermediate
    or partial state. Both fields are validated as exact bools on the way
    in (see ``_validate_connector_access_answer``); the dataclass defaults
    below stay ``False``/``False`` on purpose so that constructing a bare
    ``ConnectorAccess()`` remains the shape the validator rejects, rather
    than quietly becoming a legitimate "not linked" answer.
    """

    team_owned: bool = False
    can_edit: bool = False


ConnectorRef = tuple[ConnectorType, int]

ConnectorAccessHook = Callable[
    [Any, int, "Collection[ConnectorRef]"], "dict[ConnectorRef, ConnectorAccess]"
]

ConnectorVisibilityHook = Callable[[Any, int], dict[str, set[int]]]


class TeamConnectorVisibilityHook(Protocol):
    """Resolver for the connectors a team owns.

    Typed as a ``Protocol`` with a keyword-only ``team_id`` so it cannot be
    bound where ``ConnectorVisibilityHook`` (user-keyed) is expected, or vice
    versa: the two shapes are otherwise identical and a swapped install would
    type-check while resolving an unrelated team's connectors.

    Keyed strictly on the team that owns the *governing agent* -- never on
    the running user's own team membership, and the hook has no way to
    express "is the runner a member of this team": there is no ``user_id``
    parameter, and the hook is a process-global singleton with no
    per-request state, so the hook body cannot learn which runner a
    resolution is for. An application that wants runner-scoped narrowing
    enforces it at its agent-access policy layer -- deciding who may run a
    team's agents at all -- not at this seam, because a published team
    agent is meant to serve runners -- including anonymous end users --
    who are not members of the team that owns it.

    A platform admin's tasks reach this seam through the agent-access
    policy layer's admin short-circuit, which returns every agent
    regardless of team -- so the runner of a team-governed agent is not
    necessarily a member of that team even in deployments whose policy
    layer is otherwise membership-scoped. This is accepted platform
    behavior, not a defect this hook is meant to close.

    Every connector id a hook returns becomes executable by *any* runner of
    that team's agents, not only members: for stdio and other static-auth
    MCP transports, a team-owned server with no personal link for the
    running user executes using the shared definition row's own stored
    credentials, not the runner's. (OAuth transports fail closed instead,
    for lack of a token to resolve.) For stdio specifically, **once an
    application has also installed the separate, credential-side hook**
    (``mcp_runtime.set_mcp_team_env_hook`` -- a different socket from this
    one, installed independently), the shared env layer a team-owned
    server resolves is keyed on the team this hook answers for -- the team
    that owns the *governing agent* -- never on any other team, including
    one the running user happens to belong to: when the governing team has
    no stored row for that server, there is no shared layer at all, and a
    runner whose own env-source pick is "shared" falls back to their own
    stored key instead of silently borrowing another team's row. An
    application that installs only this visibility hook, without also
    installing ``set_mcp_team_env_hook``, gets none of that: the shared env
    layer then still resolves however the pre-existing, user-keyed
    ``set_mcp_shared_env_hook`` was wired, which can and typically does key
    on the *running user's own* team rather than the governing one --
    exactly the cross-team leak the paragraph above describes as closed.
    Deciding to share a team-owned MCP server through this hook is a
    visibility decision only; closing the credential leak is a second,
    separate installation step. Custom APIs are starker: there is no
    per-member credential override layer at all, and no transport-level
    exception either -- every custom API always executes with whatever is
    stored on its shared definition row (headers, a static API key, and so
    on), for every runner, member or not. Sharing a custom API with a team
    means sharing whatever credential it holds with everyone who can run
    the team's agents.
    """

    def __call__(self, db: Any, *, team_id: int) -> dict[str, set[int]]: ...


_connector_deleted_hook: ConnectorDeletedHook | None = None
_connector_renamed_hook: ConnectorRenamedHook | None = None
_connector_visibility_hook: ConnectorVisibilityHook | None = None
_team_connector_visibility_hook: TeamConnectorVisibilityHook | None = None
_connector_access_hook: ConnectorAccessHook | None = None


def set_connector_team_hooks(
    *,
    deleted: ConnectorDeletedHook | None = None,
    renamed: ConnectorRenamedHook | None = None,
    visibility: ConnectorVisibilityHook | None = None,
    team_visibility: TeamConnectorVisibilityHook | None = None,
    access: ConnectorAccessHook | None = None,
) -> None:
    """Install application-owned connector lifecycle hooks.

    Installs the complete hook set: every hook not supplied to a given call
    is cleared, even one a previous call installed. This is a reset-all
    setter, not a merge -- an application that installs hooks across more
    than one call must pass its complete set each time.
    """

    global _connector_deleted_hook, _connector_renamed_hook
    global _connector_visibility_hook, _team_connector_visibility_hook
    global _connector_access_hook
    _connector_deleted_hook = deleted
    _connector_renamed_hook = renamed
    _connector_visibility_hook = visibility
    _team_connector_visibility_hook = team_visibility
    _connector_access_hook = access


def visible_team_connector_ids(db: Any, user_id: int) -> dict[str, set[int]]:
    """Team-shared connector ids visible to user; empty when no hook/standalone."""
    if _connector_visibility_hook is None:
        return {"mcp": set(), "custom_api": set()}
    return _connector_visibility_hook(db, int(user_id))


def _validate_team_connector_answer(answer: Any) -> dict[str, set[int]]:
    """Validate the team-visibility hook's answer shape.

    This is an authorization input, not user-facing data: a malformed
    answer must fail loudly, never be normalized, coerced, or defaulted to
    empty. The hook must return a ``dict`` carrying both ``"mcp"`` and
    ``"custom_api"`` keys, each mapped to a ``set`` whose members are all
    ``int`` (``bool`` is a subclass of ``int`` in Python but is rejected
    here -- a truthy/falsy value is never a legitimate connector id). The
    ``set`` requirement is exact and deliberate: ``frozenset`` and other
    iterables are rejected too, matching the hook Protocol's declared
    ``set[int]`` rather than duck-typing around it.
    Extra keys beyond the two required ones are accepted and silently
    ignored: only ``"mcp"`` and ``"custom_api"`` are ever read by callers,
    so this function does not iterate the answer's keys at all, only probe
    the two it needs.
    """
    if not isinstance(answer, dict):
        raise ValueError(
            "team visibility hook returned a malformed answer: expected a "
            f"dict, got {type(answer).__name__}"
        )
    for key in ("mcp", "custom_api"):
        if key not in answer:
            raise ValueError(
                f"team visibility hook returned a malformed answer: missing key {key!r}"
            )
        value = answer[key]
        if not isinstance(value, set):
            raise ValueError(
                "team visibility hook returned a malformed answer: key "
                f"{key!r} must be a set, got {type(value).__name__}"
            )
        for member in value:
            if isinstance(member, bool) or not isinstance(member, int):
                raise ValueError(
                    "team visibility hook returned a malformed answer: key "
                    f"{key!r} contains a member {member!r} that is not a "
                    "connector id (must be int, not bool)"
                )
    # Defensive copy at the authorization boundary: callers must not be able
    # to mutate the installing application's own answer object (or vice
    # versa). Extra keys are dropped by the copy -- only the two probed keys
    # are part of the contract.
    return {"mcp": set(answer["mcp"]), "custom_api": set(answer["custom_api"])}


def team_connector_ids(db: Any, *, team_id: int | None) -> dict[str, set[int]]:
    """Connector ids owned by ``team_id``; empty for no team and for standalone.

    ``team_id`` is the team that owns the *governing agent*, never the
    running user's current team. ``None`` resolves empty without calling the
    hook. A non-``None`` answer is shape-validated (see
    ``_validate_team_connector_answer``) before it reaches any caller.
    """
    if team_id is None or _team_connector_visibility_hook is None:
        return {"mcp": set(), "custom_api": set()}
    answer = _team_connector_visibility_hook(db, team_id=int(team_id))
    return _validate_team_connector_answer(answer)


def team_connector_hook_installed() -> bool:
    """Whether an application installed a team-keyed visibility hook.

    Callers select on this, never on an empty return value: an installed
    hook legitimately answers with empty sets for a team that owns nothing.
    """
    return _team_connector_visibility_hook is not None


def _validate_connector_access_answer(
    answer: Any, requested: "frozenset[ConnectorRef]"
) -> "dict[ConnectorRef, ConnectorAccess]":
    """Validate the access hook's batch answer shape.

    An authorization input, not user-facing data: a malformed answer must
    fail loudly, never be normalized, coerced, or defaulted to empty. The
    hook answers a ``dict`` keyed on the connectors it was asked about; a
    connector the caller's team does not link is expressed by leaving its
    ref out of the answer entirely, never by a verdict with
    ``team_owned=False`` -- unlike the team-visibility hook's
    ``_validate_team_connector_answer`` above, where extra keys beyond the
    two required ones are silently accepted because nothing ever reads
    them, here the keys of the answer *are* the question: a verdict for a
    ref that was never asked about means the hook answered a different
    question than the one it was asked, and silently dropping it would
    hide that the hook and the caller have gone out of sync.

    Each value must be a ``ConnectorAccess`` with ``team_owned`` exactly
    ``True`` (an identity check, not a truthiness check, matching
    ``knowledge_base_team_scope.py``'s ``element.team_owned is not True``)
    and ``can_edit`` exactly ``True`` or ``False`` -- ``bool`` is a
    subclass of ``int`` in Python, so a merely truthy value is never
    accepted as a legitimate grant.
    """
    if not isinstance(answer, dict):
        raise ValueError(
            "connector access hook returned a malformed answer: expected a "
            f"dict, got {type(answer).__name__}"
        )
    validated: "dict[ConnectorRef, ConnectorAccess]" = {}
    for key, verdict in answer.items():
        if key not in requested:
            raise ValueError(
                "connector access hook returned a malformed answer: a "
                f"verdict for {key!r}, which was not among the connectors "
                "asked about"
            )
        if not isinstance(verdict, ConnectorAccess):
            raise ValueError(
                "connector access hook returned a malformed answer: "
                f"expected ConnectorAccess values, got "
                f"{type(verdict).__name__} for {key!r}"
            )
        if verdict.team_owned is not True:
            raise ValueError(
                "connector access hook returned a malformed answer for "
                f"{key!r}: team_owned must be True -- a connector the "
                "caller's team does not link is expressed by leaving it "
                f"out of the answer, not by a verdict, got {verdict.team_owned!r}"
            )
        if verdict.can_edit is not True and verdict.can_edit is not False:
            raise ValueError(
                "connector access hook returned a malformed answer for "
                f"{key!r}: can_edit must be exactly True or False (bool is "
                "a subclass of int in Python, and a truthy value is never "
                f"a legitimate grant), got {verdict.can_edit!r}"
            )
        validated[key] = verdict
    return validated


def resolve_connector_access(
    db: Any, user_id: int, refs: "Collection[ConnectorRef]"
) -> "dict[ConnectorRef, ConnectorAccess]":
    """Whether the caller's team links each of ``refs``, and may edit it.

    Asks the installed access hook, if any, at most once per call
    regardless of how many refs are passed -- batching is the point of
    this signature, not an incidental property, because the seam's whole
    reason to exist is to answer "what is this caller's team's
    relationship to these connectors" without paying one hook call per
    connector. Returns ``{}`` immediately, without calling the hook at
    all, when no hook is installed or when ``refs`` is empty: an empty
    request is never worth a call, and a standalone deployment with no
    hook installed sees zero queries and zero behavior change.

    A ref missing from the returned map means "the caller's team does not
    link this connector" -- the only way that fact is ever expressed (see
    ``_validate_connector_access_answer``). The answer is shape-validated
    before it reaches any caller.
    """
    requested = frozenset(
        (connector_type, int(connector_id)) for connector_type, connector_id in refs
    )
    if _connector_access_hook is None or not requested:
        return {}
    answer = _connector_access_hook(db, int(user_id), requested)
    return _validate_connector_access_answer(answer, requested)


def _restore_session_after_hook_failure(db: Any) -> None:
    """Roll back whatever a failed hook left on the shared session.

    Hooks are handed the endpoint's own live session (see
    ``delete_team_connector``'s contract note). A hook whose own statement
    failed leaves that transaction unusable on PostgreSQL, and an ORM
    ``flush`` failure leaves it unusable on every backend -- so every
    later statement in the request, including the ones a degradation path
    needs to build its response, would be refused. Rolling back here, at
    the one door application code passes through, is what keeps the
    degradation contract true; the roll back happens after the route's own
    ``db.commit()`` on the post-commit decoration paths, so it never
    discards durable work.

    A rollback that itself fails is logged and swallowed: this runs on an
    already-failing path, the original failure is re-raised by the caller
    either way, and there is no further recovery available.
    """
    rollback = getattr(db, "rollback", None)
    if rollback is None:
        return
    try:
        rollback()
    except Exception:
        logger.warning(
            "Rolling back after a failed connector hook failed", exc_info=True
        )


def resolve_team_connector_ids_or_raise(
    db: Any, *, team_id: int | None, log_subject: int | None
) -> dict[str, set[int]]:
    """``team_connector_ids(db, team_id=team_id)``, with every non-typed
    failure converted into the seam's one typed 503.

    Every team-scope resolution call site -- both of ``WebToolConfig``'s
    custom-API read points, its MCP read point, and the runtime-context
    view loader -- wraps ``team_connector_ids`` the same way: a
    ``ConnectorRuntimeError`` passes through unchanged, and any other
    exception is logged at ``WARNING`` and converted into
    ``ConnectorRuntimeError(ERROR_CONNECTOR_RUNTIME_UNAVAILABLE, "Connector
    team scope is unavailable.", details={"reason":
    "team_scope_resolution_failed"}, status_code=503)``. This function is
    that wrap, written once. Returns the full ``{"mcp": ..., "custom_api":
    ...}`` dict unmodified -- callers that need one connector kind extract
    it themselves (e.g. ``result["mcp"]``); the runtime-context view loader
    needs both and takes the dict as-is.

    ``log_subject`` is the user id that identifies the caller in the
    warning log line -- ``None`` at any call site whose own identity guard
    the caller has not yet reached (the MCP read point's private loader has
    no identity guard of its own; production only reaches it through its
    guarded public wrapper). It is only ever formatted into the log
    message, never interpreted.

    Both failure arms roll back the shared session first (see
    ``_restore_session_after_hook_failure``), including the arm that
    passes a typed error straight through: a hook can leave a statement
    failed on the session and *then* raise its own ``ConnectorRuntimeError``,
    so restoring the session cannot be confined to the generic-exception
    arm alone.
    """
    try:
        return team_connector_ids(db, team_id=team_id)
    except ConnectorRuntimeError:
        _restore_session_after_hook_failure(db)
        raise
    except Exception as exc:
        _restore_session_after_hook_failure(db)
        logger.warning(
            "Failed to resolve team connector scope for user %s",
            log_subject,
            exc_info=True,
        )
        raise ConnectorRuntimeError(
            ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
            "Connector team scope is unavailable.",
            details={"reason": "team_scope_resolution_failed"},
            status_code=503,
        ) from exc


def resolve_connector_access_or_raise(
    db: Any, user_id: int, refs: "Collection[ConnectorRef]"
) -> "dict[ConnectorRef, ConnectorAccess]":
    """``resolve_connector_access(db, user_id, refs)``, with every
    non-typed failure converted into the seam's one typed 503.

    A ``ConnectorRuntimeError`` -- whether raised by the hook itself or by
    ``resolve_connector_access``'s own answer validation -- passes through
    unchanged (same object, not re-wrapped). Any other exception is logged
    at ``WARNING`` and converted into
    ``ConnectorRuntimeError(ERROR_CONNECTOR_RUNTIME_UNAVAILABLE, "Connector
    access is unavailable.", details={"reason":
    "connector_access_resolution_failed"}, status_code=503)``. Unlike
    ``resolve_team_connector_ids_or_raise``, there is no separate
    ``log_subject`` parameter: ``user_id`` here already identifies the
    caller directly, so it doubles as the value logged. The logged refs
    are the plain ``(connector_type, id)`` tuples the caller passed in,
    sorted for a stable log line -- never an ORM attribute read off a row,
    which could itself fail if the session is left unusable by whatever
    just failed.

    Both failure arms roll back the shared session first (see
    ``_restore_session_after_hook_failure``), including the arm that
    passes a typed error straight through: a hook can leave a statement
    failed on the session and *then* raise its own ``ConnectorRuntimeError``,
    so restoring the session cannot be confined to the generic-exception
    arm alone.
    """
    requested = frozenset(
        (connector_type, int(connector_id)) for connector_type, connector_id in refs
    )
    try:
        return resolve_connector_access(db, user_id, requested)
    except ConnectorRuntimeError:
        _restore_session_after_hook_failure(db)
        raise
    except Exception as exc:
        _restore_session_after_hook_failure(db)
        logger.warning(
            "Failed to resolve connector access for user %s across %s connectors: %s",
            user_id,
            len(requested),
            sorted(requested),
            exc_info=True,
        )
        raise ConnectorRuntimeError(
            ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
            "Connector access is unavailable.",
            details={"reason": "connector_access_resolution_failed"},
            status_code=503,
        ) from exc


def resolve_one_connector_access_or_raise(
    db: Any, user_id: int, ref: "ConnectorRef"
) -> "ConnectorAccess | None":
    """Single-``ref`` convenience wrapper around
    ``resolve_connector_access_or_raise``: wraps ``ref`` in a one-element
    collection, calls the batch resolver, and unwraps the answer for that
    ref. ``None`` means the same thing it means for any ref missing from a
    batch answer -- not linked, or a legitimate answer the hook chose to
    omit -- never a failure, which still raises ``ConnectorRuntimeError``
    same as the batch form. Exists so item GET/PUT call sites do not each
    repeat the wrap-then-``.get(ref)`` shape by hand.
    """
    return resolve_connector_access_or_raise(db, user_id, [ref]).get(ref)


@contextmanager
def snapshot_connector_team_hooks() -> Iterator[None]:
    """Save every module-level hook slot, restore it on exit.

    Intended for tests: entering the block, replacing any slot (through
    ``set_connector_team_hooks`` or a direct module-attribute monkeypatch),
    and leaving restores every slot to the exact object it held before the
    block, including a slot the block never touched. A slot added to this
    module later must be added here too, or a snapshot taken before that
    slot exists will silently fail to restore it -- covered by the
    discovery-based coverage test in
    tests/web/services/test_connector_team_scope.py, which enumerates every
    module global ending in ``_hook`` and asserts this snapshot restores
    each one by identity. Saving and restoring lives on the module because
    the state being saved lives on the module: a test-side helper would
    have to name and reach these globals from outside, and would go stale
    the moment a slot is added here.
    """
    global _connector_deleted_hook, _connector_renamed_hook
    global _connector_visibility_hook, _team_connector_visibility_hook
    global _connector_access_hook
    saved = (
        _connector_deleted_hook,
        _connector_renamed_hook,
        _connector_visibility_hook,
        _team_connector_visibility_hook,
        _connector_access_hook,
    )
    try:
        yield
    finally:
        (
            _connector_deleted_hook,
            _connector_renamed_hook,
            _connector_visibility_hook,
            _team_connector_visibility_hook,
            _connector_access_hook,
        ) = saved


def connector_visible_to_user(
    *,
    association: "UserMCPServer | UserCustomApi | None",
    connector_id: int,
    team_ids: Collection[int],
) -> bool:
    """In-Python twin of the ``visible_*_clause`` predicates below.

    Same rule, expressed for callers that hold already-loaded ORM rows rather
    than a query to filter: an active personal association, unioned with the
    team-owned ids. ``is_active`` gates only the personal arm, so a
    team-visible connector stays visible through a deactivated personal link
    -- the corner that separates this from a naive ``association.is_active``
    check, and the one ``/api/mcp/apps`` got wrong before #1384.

    ``association`` is None when the caller resolved the connector through the
    team overlay alone. Kept beside the clauses it mirrors so the declarative
    and imperative forms of the rule move together;
    ``_load_visible_runtime_connectors`` expresses the same union a third time
    in connector_runtime.py, against its own queries.
    """

    if association is not None and bool(association.is_active):
        return True
    return connector_id in team_ids


def visible_mcp_server_clause(
    owner_user_id: int | None, team_mcp_ids: Collection[int]
) -> ColumnElement[bool]:
    """Predicate selecting the MCP servers visible to ``owner_user_id``.

    Personal servers (an active ``UserMCPServer`` link) unioned with the
    team-owned servers named in ``team_mcp_ids``. ``is_active`` gates only
    the personal arm -- a team-owned server is visible regardless of the
    member's own personal link state, which is intentional (see the
    module's callers for the credential consequence). Pure function over
    ORM constructs -- no I/O. An empty ``team_mcp_ids`` reduces to exactly
    the personal semi-join, so a deployment with no team overlay compiles
    the same query shape it always has. A ``None`` owner also reduces to
    the personal arm regardless of ``team_mcp_ids`` -- defense in depth so
    this function is fail-closed on its own terms, not only because of how
    its one caller happens to be gated today.
    """

    from sqlalchemy import or_, select

    from ..models.mcp import MCPServer, UserMCPServer

    personal = MCPServer.id.in_(
        select(UserMCPServer.mcpserver_id).where(
            UserMCPServer.user_id == owner_user_id,
            UserMCPServer.is_active,
        )
    )
    if not team_mcp_ids or owner_user_id is None:
        return personal
    return or_(personal, MCPServer.id.in_(set(team_mcp_ids)))


def visible_custom_api_clause(
    owner_user_id: int | None, team_api_ids: Collection[int]
) -> ColumnElement[bool]:
    """Predicate selecting the custom APIs visible to ``owner_user_id``.

    Personal custom APIs (an active ``UserCustomApi`` link) unioned with the
    team-owned APIs named in ``team_api_ids``. ``is_active`` gates only the
    personal arm -- a team-owned API is visible regardless of the member's
    own personal link state, which is intentional (see the module's callers
    for the credential consequence). Pure function over ORM constructs -- no
    I/O. An empty ``team_api_ids`` reduces to exactly the personal
    semi-join, so a deployment with no team overlay compiles the same query
    shape it always has. A ``None`` owner also reduces to the personal arm
    regardless of ``team_api_ids`` -- defense in depth so this function is
    fail-closed on its own terms, not only because of how its callers
    happen to be gated today.
    """

    from sqlalchemy import or_, select

    from ..models.custom_api import CustomApi, UserCustomApi

    personal = CustomApi.id.in_(
        select(UserCustomApi.custom_api_id).where(
            UserCustomApi.user_id == owner_user_id,
            UserCustomApi.is_active,
        )
    )
    if not team_api_ids or owner_user_id is None:
        return personal
    return or_(personal, CustomApi.id.in_(set(team_api_ids)))


def delete_team_connector(
    db: Any, user_id: int, connector_type: ConnectorType, connector_id: int
) -> ConnectorDeleteDecision:
    """Remove team ownership before a global delete.

    Returns whether the application recognized the connector as team-owned.
    Hooks must use the passed session and must not commit independently, so a
    refused endpoint request can discard all hook-side mutations atomically.
    """

    if _connector_deleted_hook is None:
        return ConnectorDeleteDecision()
    return _connector_deleted_hook(db, user_id, connector_type, connector_id)


def rename_team_connector(
    db: Any,
    user_id: int,
    connector_type: ConnectorType,
    connector_id: int,
    old_name: str,
    new_name: str,
) -> None:
    """Keep application-owned connector selectors aligned after a rename."""

    if _connector_renamed_hook is not None and old_name != new_name:
        _connector_renamed_hook(
            db, user_id, connector_type, connector_id, old_name, new_name
        )
