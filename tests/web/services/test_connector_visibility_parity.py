"""The runtime-context view and the tool loader agree on MCP visibility.

One fixture seeded with all four connector categories (owner-active,
owner-inactive, team-only, stranger) simultaneously, plus one
OAuth-without-grant row so the ``config["server_id"]`` fallback below is
actually exercised (the tool loader's "unavailable" shape has no top-level
``id``). The matrix varies the team parameter over ``{T1, None}`` on the run
owner, plus one ``T2`` cell -- trimmed to the cells that each discriminate a
distinct case rather than repeating one already covered.

Scope note: this asserts ``connector_type == "mcp"`` parity only. The
custom-API half is deliberately NOT team-keyed on either side: the new-hook
branch of ``_load_visible_runtime_connectors`` narrows its custom-API set
to the junction-only view (pinned by
``test_new_hook_branch_does_not_union_team_custom_api``), matching the
custom-API tool loaders in ``config.py``, until both sides are team-keyed
together. Custom-API parity therefore holds trivially today; the cells
here pin the MCP half, where the union is live.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.web.models import Base, MCPServer, User, UserMCPServer
from xagent.web.services import agent_team_scope, connector_team_scope
from xagent.web.services.connector_runtime import _load_visible_runtime_connectors
from xagent.web.tools.config import WebToolConfig

T1 = 101
T2 = 102


@pytest.fixture()
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture(autouse=True)
def _reset_hooks() -> Iterator[None]:
    yield
    connector_team_scope.set_connector_team_hooks()
    agent_team_scope.set_agent_team_scope_hook(None)


def _create_user(db: Session, username: str) -> User:
    user = User(username=username, password_hash="hash")
    db.add(user)
    db.flush()
    return user


def _create_mcp(
    db: Session,
    name: str,
    *,
    owner: User | None = None,
    active: bool = True,
    transport: str = "streamable_http",
) -> MCPServer:
    server = MCPServer(
        name=name,
        description=f"{name} description",
        managed="external",
        transport=transport,
        url="https://example.com/mcp" if transport != "oauth" else None,
    )
    db.add(server)
    db.flush()
    if owner is not None:
        db.add(
            UserMCPServer(
                user_id=owner.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=active,
            )
        )
        db.flush()
    return server


@pytest.fixture()
def seed(db_session: Session):
    c = _create_user(db_session, "run-owner")
    stranger_owner = _create_user(db_session, "stranger-owner")
    active_own = _create_mcp(db_session, "active-own", owner=c, active=True)
    inactive_own = _create_mcp(db_session, "inactive-own", owner=c, active=False)
    stranger = _create_mcp(db_session, "stranger", owner=stranger_owner)
    team_s = _create_mcp(db_session, "team-s")
    team_x = _create_mcp(db_session, "team-x")
    # OAuth server with no matching catalog app: exercises the tool loader's
    # "unavailable" shape, whose id lives at config["server_id"], not "id".
    oauth_no_grant = _create_mcp(
        db_session, "oauth-no-grant", owner=c, transport="oauth"
    )
    connector_team_scope.set_connector_team_hooks(
        # inactive_own also carries a live T1 team grant here, pinning the
        # deliberate "no is_active veto" decision: a member's deactivated
        # personal link must not silently hide a connector the team still
        # shares. Without this, ``inactive_own`` only ever pinned "inactive
        # personal link alone -> not visible", never the combination that
        # actually matters.
        team_visibility=lambda db, *, team_id: (
            {"mcp": {int(team_s.id), int(inactive_own.id)}, "custom_api": set()}
            if team_id == T1
            else {"mcp": {int(team_x.id)}, "custom_api": set()}
            if team_id == T2
            else {"mcp": set(), "custom_api": set()}
        )
    )
    # Negative-control fixture (see test_mcp_team_visibility.py's
    # ``_install_env_t`` docstring): maps the run owner to T1 so a
    # runner-keyed misimplementation is distinguishable from the correct
    # agent-keyed one on the ``team=None`` cell.
    agent_team_scope.set_agent_team_scope_hook(
        lambda db, user_id: agent_team_scope.AgentTeamScope(
            team_id=T1, is_team_admin=False
        )
        if user_id == int(c.id)
        else None
    )
    return SimpleNamespace(
        c=c,
        active_own=active_own,
        inactive_own=inactive_own,
        stranger=stranger,
        team_s=team_s,
        team_x=team_x,
        oauth_no_grant=oauth_no_grant,
    )


async def _parity_ids(
    db_session: Session, seed, *, team: int | None
) -> tuple[set[int], set[int]]:
    visible = _load_visible_runtime_connectors(
        db_session, user_id=int(seed.c.id), agent_team_id=team
    )
    runtime_ids = {ref.connector_id for ref in visible if ref.connector_type == "mcp"}

    cfg = WebToolConfig(
        db=db_session,
        request=None,
        user_id=int(seed.c.id),
        connector_team_id=team,
        include_mcp_tools=True,
    )
    configs = await cfg._load_mcp_server_configs()
    loader_ids = {c.get("id") or c.get("config", {}).get("server_id") for c in configs}
    return runtime_ids, loader_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("team", [T1, None, T2])
async def test_runtime_view_and_tool_config_agree(db_session, seed, team):
    runtime_ids, loader_ids = await _parity_ids(db_session, seed, team=team)
    assert runtime_ids == loader_ids

    # Every seeded category is represented in the agreed-on set so the
    # parity assertion isn't vacuous for any row of the trimmed matrix.
    assert int(seed.active_own.id) in loader_ids
    assert int(seed.oauth_no_grant.id) in loader_ids
    assert int(seed.stranger.id) not in loader_ids
    if team == T1:
        assert int(seed.team_s.id) in loader_ids
        assert int(seed.team_x.id) not in loader_ids
        # Plan §6's "no is_active veto": a deactivated personal link plus a
        # live T1 team grant still resolves visible, in parity on both
        # read points -- the team grant is not blocked by the member's own
        # inactive association.
        assert int(seed.inactive_own.id) in loader_ids
    elif team == T2:
        assert int(seed.team_x.id) in loader_ids
        assert int(seed.team_s.id) not in loader_ids
        # inactive_own has no T2 grant -- the deactivated personal link
        # alone stays not-visible.
        assert int(seed.inactive_own.id) not in loader_ids
    else:
        assert int(seed.team_s.id) not in loader_ids
        assert int(seed.team_x.id) not in loader_ids
        assert int(seed.inactive_own.id) not in loader_ids
