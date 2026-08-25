"""The edit right on a team-linked MCP connector: ``GET``/``PUT
/api/mcp/servers/{server_id}`` resolve a caller with no personal row
through the connector access hook instead of 404ing outright, the edit
branch of ``_check_mcp_permission`` falls back to that verdict, the two
per-user fields reject outright for a caller with no row to hold them, and
a raising hook surfaces as its declared status rather than a 500.

Every test installs the access hook through
``snapshot_connector_team_hooks`` so no hook state leaks between tests or
into suites that run after this one.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.api import mcp as mcp_module
from xagent.web.api.mcp import (
    MCPAppConnectRequest,
    MCPServerUpdate,
    _check_mcp_permission,
    connect_mcp_app,
    get_mcp_server,
    toggle_mcp_server,
    update_mcp_server,
)
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services.connector_team_scope import (
    ConnectorAccess,
    set_connector_team_hooks,
    snapshot_connector_team_hooks,
)


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def _make_user(db, user_id: int, *, is_admin: bool = False) -> User:
    user = User(
        id=user_id, username=f"user-{user_id}", password_hash="x", is_admin=is_admin
    )
    db.add(user)
    db.commit()
    return user


def _make_owned_server(db, owner_id: int, *, name: str = "shared-server") -> MCPServer:
    server = MCPServer(name=name, transport="stdio", managed="external", command="true")
    db.add(server)
    db.flush()
    db.add(
        UserMCPServer(
            user_id=owner_id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()
    return server


class TestCheckMcpPermissionTeamAccessFallback:
    """New assertions only -- ``test_check_mcp_permission`` in
    test_mcp_api.py is left untouched by design."""

    def test_owner_wins_the_edit_branch_without_consulting_the_verdict(self):
        from unittest.mock import MagicMock

        owner = MagicMock(is_owner=True, can_delete=False)
        # A verdict that would deny edit rights on its own is still beaten
        # by is_owner -- the verdict is a fallback, never an override.
        denying_access = ConnectorAccess(team_owned=True, can_edit=False)
        assert (
            _check_mcp_permission(
                owner, is_admin=False, require="edit", team_access=denying_access
            )
            is True
        )

    def test_non_owner_falls_back_to_a_granting_verdict(self):
        from unittest.mock import MagicMock

        guest = MagicMock(is_owner=False, can_delete=False)
        granting_access = ConnectorAccess(team_owned=True, can_edit=True)
        assert (
            _check_mcp_permission(
                guest, is_admin=False, require="edit", team_access=granting_access
            )
            is True
        )

    def test_non_owner_stays_denied_by_a_linked_but_not_editable_verdict(self):
        from unittest.mock import MagicMock

        guest = MagicMock(is_owner=False, can_delete=False)
        linked_only = ConnectorAccess(team_owned=True, can_edit=False)
        assert (
            _check_mcp_permission(
                guest, is_admin=False, require="edit", team_access=linked_only
            )
            is False
        )

    def test_missing_team_access_keyword_behaves_exactly_as_before(self):
        from unittest.mock import MagicMock

        owner = MagicMock(is_owner=True, can_delete=False)
        guest = MagicMock(is_owner=False, can_delete=False)
        assert _check_mcp_permission(owner, is_admin=False, require="edit") is True
        assert _check_mcp_permission(guest, is_admin=False, require="edit") is False

    def test_delete_branch_ignores_team_access_entirely(self):
        """Delete stays exactly as it is today: a granting verdict changes
        nothing on the ``delete`` branch, which reads only ``can_delete``."""
        from unittest.mock import MagicMock

        guest = MagicMock(is_owner=False, can_delete=False)
        granting_access = ConnectorAccess(team_owned=True, can_edit=True)
        assert (
            _check_mcp_permission(
                guest, is_admin=False, require="delete", team_access=granting_access
            )
            is False
        )


class TestGateHelperOnGetAndPut:
    def test_get_404s_for_an_unrelated_user_with_no_link_and_no_team_access(self, db):
        owner = _make_user(db, 1)
        stranger = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=lambda db, user_id, refs: {})
            with pytest.raises(HTTPException) as exc:
                get_mcp_server(server.id, current_user=stranger, db=db)
        assert exc.value.status_code == 404

    def test_get_returns_the_stand_in_for_a_team_member_with_no_personal_row(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = get_mcp_server(server.id, current_user=member, db=db)

        assert response.id == server.id
        assert response.user_id == member.id

    def test_get_owner_behaviour_is_unchanged_with_no_hook_installed(self, db):
        owner = _make_user(db, 1)
        server = _make_owned_server(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks()
            response = get_mcp_server(server.id, current_user=owner, db=db)

        assert response.id == server.id
        assert response.can_edit_global is True


class TestPutWiringForATeamEditor:
    def test_team_editor_edit_is_durable_and_creates_no_association_row(self, db):
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = update_mcp_server(
                server_id,
                MCPServerUpdate(description="edited by the team"),
                current_user=editor,
                db=db,
            )

        assert response.description == "edited by the team"

        # I5: durability, not staging -- a same-session query would still
        # see an uncommitted UPDATE even if the route never committed.
        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.description == "edited by the team"

        # I6: the edit did not fabricate a personal association for the
        # team editor -- that would be a get-or-create write on an
        # authorization path.
        assert (
            db.query(UserMCPServer).filter(UserMCPServer.user_id == editor.id).first()
            is None
        )

    def test_view_only_team_member_cannot_tamper_the_shared_config(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=False)
                    for ref in refs
                }
            )
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server.id,
                    MCPServerUpdate(description="should not land"),
                    current_user=member,
                    db=db,
                )
        assert exc.value.status_code == 403

    def test_rename_propagates_to_team_agent_selectors(self, db, monkeypatch):
        """I10, and the mutation check the design requires for it: deleting
        the ``rename_team_connector`` call must turn this red."""
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="old-name")
        server_id = server.id

        calls: list[tuple[str, str]] = []

        def fake_renamed_hook(_db, _user_id, _connector_type, _connector_id, old, new):
            calls.append((old, new))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                },
                renamed=fake_renamed_hook,
            )
            update_mcp_server(
                server_id,
                MCPServerUpdate(name="new-name"),
                current_user=editor,
                db=db,
            )

        assert calls == [("old-name", "new-name")]


class TestUserEnvAndIsActiveRejectionForAStandIn:
    def test_user_env_from_a_caller_with_no_personal_row_is_400_not_a_silent_drop(
        self, db
    ):
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="unchanged-name")
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(user_env={"API_KEY": "x"}),
                    current_user=editor,
                    db=db,
                )

        assert exc.value.status_code == 400
        assert "personal connection" in str(exc.value.detail)

        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.name == "unchanged-name"
        assert (
            db.query(UserMCPServer).filter(UserMCPServer.user_id == editor.id).first()
            is None
        )

    def test_is_active_from_a_caller_with_no_personal_row_is_400_not_a_silent_drop(
        self, db
    ):
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="still-unchanged")
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(is_active=False),
                    current_user=editor,
                    db=db,
                )

        assert exc.value.status_code == 400

        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.name == "still-unchanged"
        assert (
            db.query(UserMCPServer).filter(UserMCPServer.user_id == editor.id).first()
            is None
        )


class TestTypedErrorArm:
    """A raising hook still surfaces its declared status for a caller whose
    own personal row does not already decide the answer -- the verdict is
    genuinely the gate for that population, and must stay fail-closed. An
    owner's row already decides the answer on its own, so a hook is never
    called for it at all; that population is pinned separately, below, in
    ``TestOwnerIsImmuneToAHookFailure``."""

    def test_get_surfaces_a_raising_hooks_declared_status(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            with pytest.raises(HTTPException) as exc:
                get_mcp_server(server.id, current_user=member, db=db)

        assert exc.value.status_code == 503

    def test_put_surfaces_a_raising_hooks_declared_status_and_leaves_the_row_unchanged(
        self, db
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="pristine")
        server_id = server.id

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(name="should-not-land"),
                    current_user=member,
                    db=db,
                )

        assert exc.value.status_code == 503

        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.name == "pristine"

    def test_put_passes_through_a_planted_connector_runtime_error_by_its_own_status(
        self, db
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)

        def boom(*_a, **_k):
            raise ConnectorRuntimeError("planted", "planted failure", status_code=409)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server.id,
                    MCPServerUpdate(name="irrelevant"),
                    current_user=member,
                    db=db,
                )

        assert exc.value.status_code == 409
        assert exc.value.detail == "planted failure"


class TestOwnerIsImmuneToAHookFailure:
    """An owner's row already decides the edit answer on its own -- the
    edit branch returns True on ``is_owner`` without ever consulting a
    verdict -- so ``GET``/``PUT`` never call the hook for an owner's row at
    all. A hook that would raise must therefore never surface: both routes
    return their normal success status, unaffected by whatever the hook
    would have done."""

    def test_get_and_put_succeed_for_an_owner_even_though_the_hook_would_raise(
        self, db
    ):
        owner = _make_user(db, 1)
        server = _make_owned_server(db, owner.id, name="owner-immune")
        server_id = server.id

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            get_response = get_mcp_server(server_id, current_user=owner, db=db)
            put_response = update_mcp_server(
                server_id,
                MCPServerUpdate(description="edited by the owner"),
                current_user=owner,
                db=db,
            )

        assert get_response.can_edit_global is True
        assert put_response.can_edit_global is True
        assert put_response.description == "edited by the owner"


def _make_catalog_app(db, app_id: str) -> None:
    db.add(
        PublicMCPApp(
            app_id=app_id,
            name=app_id,
            transport="stdio",
            launch_config={"command": "true", "args": []},
        )
    )
    db.commit()


class TestDecorationDegradesAfterTheWriteCommits:
    """``toggle`` and ``connect`` both commit their write before resolving
    the verdict, purely to decorate the response's ``can_edit_global`` --
    a hook failure there must degrade that field to False rather than fail
    a request whose write already landed."""

    async def test_toggle_degrades_and_keeps_its_effect_when_the_hook_raises(
        self, db, monkeypatch
    ):
        # A non-owner personal row, not the owner's: an owner's
        # can_edit_global cannot be moved by any verdict at all (is_owner
        # wins outright), so only a non-owner's reported field actually
        # depends on whether the verdict resolved or degraded.
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)
        server_id = server.id
        db.add(
            UserMCPServer(
                user_id=editor.id,
                mcpserver_id=server_id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()
        before = (
            db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == editor.id,
                UserMCPServer.mcpserver_id == server_id,
            )
            .one()
            .is_active
        )

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        fake_logger = MagicMock()
        monkeypatch.setattr(mcp_module, "logger", fake_logger)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            response = await toggle_mcp_server(server_id, current_user=editor, db=db)

        assert response.can_edit_global is False
        fake_logger.warning.assert_called_once()

        db.rollback()
        refreshed = (
            db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == editor.id,
                UserMCPServer.mcpserver_id == server_id,
            )
            .one()
        )
        assert refreshed.is_active is (not before)

    def test_connect_degrades_and_keeps_its_effect_when_the_hook_raises(
        self, db, monkeypatch
    ):
        user = _make_user(db, 1)
        _make_catalog_app(db, "decorate-only-app")

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        fake_logger = MagicMock()
        monkeypatch.setattr(mcp_module, "logger", fake_logger)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            response = connect_mcp_app(
                "decorate-only-app",
                MCPAppConnectRequest(),
                current_user=user,
                db=db,
            )

        assert response.can_edit_global is False
        fake_logger.warning.assert_called_once()

        db.rollback()
        server = db.query(MCPServer).filter(MCPServer.name == "decorate-only-app").one()
        assoc = (
            db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == user.id,
                UserMCPServer.mcpserver_id == server.id,
            )
            .one()
        )
        assert assoc.is_owner is False
