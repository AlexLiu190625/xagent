"""The reported ``can_edit_global``/``can_configure`` fields agree with what
the gates in earlier stages actually enforce, across every response-builder
call site and both connector kinds -- and the four MCP OAuth routes, the
rename call's scope, and every route's no-hook-installed shape are all
unchanged by threading that verdict through.

Every test installs hooks (or explicitly installs none) through
``snapshot_connector_team_hooks`` so no hook state leaks between tests or
into suites that run after this one.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from xagent.web.api.custom_api import CustomApiUpdate, get_custom_api, update_custom_api
from xagent.web.api.mcp import (
    MCPOAuthConnectRequest,
    MCPOAuthDiscoverRequest,
    MCPServerUpdate,
    connect_mcp_oauth,
    delete_mcp_oauth_grant,
    discover_mcp_oauth,
    get_mcp_oauth_status,
    get_mcp_server,
    get_mcp_servers,
    list_mcp_apps,
    toggle_mcp_server,
    update_mcp_server,
)
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
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


def _make_owned_api(db, owner_id: int, *, name: str = "shared-api") -> CustomApi:
    api = CustomApi(name=name, url="https://example.com/api", method="GET")
    db.add(api)
    db.flush()
    db.add(
        UserCustomApi(
            user_id=owner_id,
            custom_api_id=api.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=True,
        )
    )
    db.commit()
    return api


class TestListEndpointAccessHookCallBudget:
    """The list endpoint calls the access hook zero times for an
    is_owner=True row, once for an is_owner=False row, and once for every
    stand-in row -- pinned with a counting test double. Counting hook calls
    alone would hide the query cost the gate helper and the per-row
    definition lookups add, so a SQLAlchemy ``before_cursor_execute``
    listener additionally pins the number of SQL statements the endpoint
    issues for this exact population."""

    def test_hook_is_called_exactly_once_per_non_owner_row_and_never_for_owner_rows(
        self, db
    ):
        caller = _make_user(db, 1)
        other_owner = _make_user(db, 2)

        # P = 2 personal rows the caller owns outright.
        owned = [_make_owned_server(db, caller.id, name=f"owned-{i}") for i in range(2)]

        # Q = 3 personal rows the caller holds but does not own (a second
        # link on a connector someone else owns).
        shared_personal = []
        for i in range(3):
            server = _make_owned_server(db, other_owner.id, name=f"shared-personal-{i}")
            db.add(
                UserMCPServer(
                    user_id=caller.id,
                    mcpserver_id=server.id,
                    is_owner=False,
                    is_active=True,
                )
            )
            db.commit()
            shared_personal.append(server)

        # R = 2 rows the caller has no personal row for at all, made visible
        # through the separate visibility hook (not the access hook under
        # test here).
        stand_in = [
            _make_owned_server(db, other_owner.id, name=f"stand-in-{i}")
            for i in range(2)
        ]

        # Read every id the hooks below will need before the query listener
        # attaches: the objects above were expired by their own setup
        # commits (session default expire_on_commit=True), so reading
        # .id for the first time inside the measured window would count as
        # a query the *endpoint* issues, when it is really just this test's
        # own setup catching up. caller.id specifically: get_mcp_servers
        # reads current_user.id as its very first act.
        _ = caller.id
        owned_ids = {s.id for s in owned}
        shared_personal_ids = {s.id for s in shared_personal}
        stand_in_ids = {s.id for s in stand_in}

        calls: list[tuple[int, str, int]] = []

        def counting_access_hook(_db, user_id, connector_type, connector_id):
            calls.append((user_id, connector_type, connector_id))
            return None

        def visibility_hook(_db, _user_id):
            return {"mcp": set(stand_in_ids), "custom_api": set()}

        queries: list[str] = []

        def record_query(conn, cursor, statement, parameters, context, executemany):
            del conn, cursor, parameters, context, executemany
            queries.append(statement)

        engine = db.get_bind()
        event.listen(engine, "before_cursor_execute", record_query)
        try:
            with snapshot_connector_team_hooks():
                set_connector_team_hooks(
                    access=counting_access_hook, visibility=visibility_hook
                )
                get_mcp_servers(current_user=caller, db=db)
        finally:
            event.remove(engine, "before_cursor_execute", record_query)

        assert len(calls) == 3 + 2
        # Sanity: never called for an owned row's id.
        called_ids = {connector_id for _uid, _kind, connector_id in calls}
        assert called_ids.isdisjoint(owned_ids)
        assert called_ids == shared_personal_ids | stand_in_ids

        # The hook-call count above cannot see the SQL the gate helper and
        # the per-row definition lookups issue on top of it. Observed by
        # running this exact population and reading the recorded
        # statements, not derived from a formula: one query for the
        # caller's own MCP rows (P + Q personal rows in one join), one for
        # the OAuth-account lookup, one for the caller's own Custom API
        # rows (none here), and one for the stand-in rows' MCPServer lookup
        # (one statement, batched with an IN clause over both stand-in
        # ids) -- 4 statements total for this population, none of them
        # growing per row within P, Q or R.
        assert len(queries) == 4, queries


class TestReportedEditPermissionConsistencyMcp:
    """The response's can_edit_global must agree across every surface that
    reports it, for the same (user, connector) -- for MCP connectors, across
    the list, GET, PUT's response and toggle's response."""

    @pytest.mark.parametrize(
        "population,access_answer,has_personal_row",
        [
            ("owner", None, True),
            ("personal_non_owner_no_team_link", None, True),
            (
                "stand_in_granting_edit",
                ConnectorAccess(team_owned=True, can_edit=True),
                False,
            ),
            (
                "stand_in_denying_edit",
                ConnectorAccess(team_owned=True, can_edit=False),
                False,
            ),
            (
                # The admin bypass in _check_mcp_permission wins even over a
                # verdict that itself denies edit -- this is the one
                # population where the two connector kinds genuinely
                # diverge (Custom API's own gate has no admin bypass at
                # all), so it is pinned per kind, not by cross-kind equality.
                "platform_admin",
                ConnectorAccess(team_owned=True, can_edit=False),
                False,
            ),
        ],
    )
    async def test_can_edit_global_agrees_across_list_get_put_and_toggle(
        self, db, population, access_answer, has_personal_row
    ):
        owner = _make_user(db, 10)
        if population == "owner":
            caller = owner
        elif population == "platform_admin":
            caller = _make_user(db, 12, is_admin=True)
        else:
            caller = _make_user(db, 11)
        server = _make_owned_server(db, owner.id, name=f"consistency-mcp-{population}")
        server_id = server.id

        if population == "personal_non_owner_no_team_link":
            db.add(
                UserMCPServer(
                    user_id=caller.id,
                    mcpserver_id=server_id,
                    is_owner=False,
                    is_active=True,
                )
            )
            db.commit()

        expected = population in ("owner", "platform_admin") or bool(
            access_answer is not None and access_answer.can_edit
        )

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda *_a, **_k: access_answer,
                visibility=lambda _db, _uid: {"mcp": {server_id}, "custom_api": set()},
            )

            list_entries = get_mcp_servers(current_user=caller, db=db)
            list_entry = next(r for r in list_entries if r.id == server_id)

            get_response = get_mcp_server(server_id, current_user=caller, db=db)
            put_response = update_mcp_server(
                server_id, MCPServerUpdate(), current_user=caller, db=db
            )

            toggle_response = None
            if has_personal_row:
                toggle_response = await toggle_mcp_server(
                    server_id, current_user=caller, db=db
                )

        assert list_entry.can_edit_global == expected
        assert get_response.can_edit_global == expected
        assert put_response.can_edit_global == expected
        if toggle_response is not None:
            assert toggle_response.can_edit_global == expected


class TestReportedEditPermissionConsistencyCustomApi:
    """The same agreement, for the Custom API kind: ``_custom_api_to_mcp_response`` has no
    ``_check_mcp_permission``-shaped gate to compare against and Custom
    API's own ``GET``/``PUT`` response model carries no ``can_edit_global``
    field at all -- so the surface to agree with is not a second reported
    field but ``update_custom_api``'s actual 2xx/403 outcome, exactly the
    motivating case: the list must not report ``False`` for a connector
    whose ``PUT`` now succeeds."""

    @pytest.mark.parametrize(
        "population,access_answer",
        [
            ("owner", None),
            ("personal_non_owner_no_team_link", None),
            (
                "stand_in_granting_edit",
                ConnectorAccess(team_owned=True, can_edit=True),
            ),
            (
                "stand_in_denying_edit",
                ConnectorAccess(team_owned=True, can_edit=False),
            ),
            (
                # Unlike the MCP kind, update_custom_api's own gate has no
                # admin bypass at all -- so a platform admin with no
                # personal row and a denying verdict is refused just like
                # any other caller, and the list must agree by reporting
                # False, not by copying MCP's True.
                "platform_admin",
                ConnectorAccess(team_owned=True, can_edit=False),
            ),
        ],
    )
    async def test_list_can_edit_global_agrees_with_whether_put_actually_succeeds(
        self, db, population, access_answer
    ):
        owner = _make_user(db, 20)
        if population == "owner":
            caller = owner
        elif population == "platform_admin":
            caller = _make_user(db, 22, is_admin=True)
        else:
            caller = _make_user(db, 21)
        api = _make_owned_api(db, owner.id, name=f"consistency-api-{population}")
        api_id = api.id

        if population == "personal_non_owner_no_team_link":
            db.add(
                UserCustomApi(
                    user_id=caller.id,
                    custom_api_id=api_id,
                    is_owner=False,
                    can_edit=False,
                    is_active=True,
                )
            )
            db.commit()

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda *_a, **_k: access_answer,
                visibility=lambda _db, _uid: {"mcp": set(), "custom_api": {api_id}},
            )

            list_entries = get_mcp_servers(current_user=caller, db=db)
            list_entry = next(
                r
                for r in list_entries
                if r.id == api_id and r.transport == "custom_api"
            )

            try:
                await update_custom_api(
                    api_id,
                    CustomApiUpdate(description="edited by the consistency test"),
                    current_user=caller,
                    db=db,
                )
                put_succeeded = True
            except HTTPException as exc:
                assert exc.status_code == 403
                put_succeeded = False

        assert list_entry.can_edit_global == put_succeeded
        if population == "platform_admin":
            # Pinned by value, not only by cross-surface agreement: a
            # regression that adds an admin bypass to the list's formula
            # alone would leave this False on one side and True on the
            # other, which the equality assertion above already catches --
            # this makes the intended, current answer explicit too.
            assert list_entry.can_edit_global is False
            assert put_succeeded is False


class TestLocalCanConfigureWidening:
    """``_local_mcp_can_configure`` answers True for a stand-in whose team
    access verdict links the connector but denies edit -- visible and
    reachable rather than invisible on ``association is None`` alone, for
    both connector kinds."""

    def test_mcp_stand_in_with_a_linked_but_not_editable_verdict_is_configurable(
        self, db
    ):
        owner = _make_user(db, 30)
        member = _make_user(db, 31)
        server = _make_owned_server(db, owner.id, name="visible-not-editable-mcp")
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda *_a, **_k: ConnectorAccess(
                    team_owned=True, can_edit=False
                ),
                visibility=lambda _db, _uid: {"mcp": {server_id}, "custom_api": set()},
            )
            entries = list_mcp_apps(location="local", current_user=member, db=db)
            entry = next(e for e in entries if e["server_id"] == server_id)
            assert entry["can_configure"] is True

            # The actual route (fixed independently of this UI hint) already
            # resolves for this population -- this proves the hint agrees.
            response = get_mcp_server(server_id, current_user=member, db=db)
        assert response.id == server_id

    async def test_custom_api_stand_in_with_a_linked_but_not_editable_verdict_is_configurable(
        self, db
    ):
        owner = _make_user(db, 32)
        member = _make_user(db, 33)
        api = _make_owned_api(db, owner.id, name="visible-not-editable-api")
        api_id = api.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda *_a, **_k: ConnectorAccess(
                    team_owned=True, can_edit=False
                ),
                visibility=lambda _db, _uid: {"mcp": set(), "custom_api": {api_id}},
            )
            entries = list_mcp_apps(location="local", current_user=member, db=db)
            entry = next(
                e
                for e in entries
                if e["server_id"] == api_id and e["transport"] == "custom_api"
            )
            assert entry["can_configure"] is True

            response = await get_custom_api(api_id, current_user=member, db=db)
        assert response.id == api_id


class TestOAuthRoutesKeepTheirOwnGate:
    """The four MCP OAuth routes keep the old personal-row-only helper and
    still 404 a team member with no personal row, verdict or not."""

    async def test_all_four_oauth_routes_404_a_team_member_with_no_personal_row(
        self, db
    ):
        owner = _make_user(db, 40)
        member = _make_user(db, 41)
        server = _make_owned_server(db, owner.id, name="oauth-gate-untouched")
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda *_a, **_k: ConnectorAccess(team_owned=True, can_edit=True)
            )

            with pytest.raises(HTTPException) as exc:
                await discover_mcp_oauth(
                    server_id, MCPOAuthDiscoverRequest(), current_user=member, db=db
                )
            assert exc.value.status_code == 404

            with pytest.raises(HTTPException) as exc:
                await connect_mcp_oauth(
                    server_id,
                    MCPOAuthConnectRequest(),
                    current_user=member,
                    db=db,
                    accept=None,
                )
            assert exc.value.status_code == 404

            with pytest.raises(HTTPException) as exc:
                await get_mcp_oauth_status(server_id, current_user=member, db=db)
            assert exc.value.status_code == 404

            with pytest.raises(HTTPException) as exc:
                await delete_mcp_oauth_grant(server_id, 1, current_user=member, db=db)
            assert exc.value.status_code == 404


class TestDenyingVerdictIsFalseEverywhere:
    """A connector whose verdict denies edit reports can_edit_global False
    in the list, in the response from GET, and in the response from PUT
    alike."""

    async def test_a_denying_verdict_yields_false_in_the_list_get_and_put_response(
        self, db
    ):
        owner = _make_user(db, 50)
        member = _make_user(db, 51)
        server = _make_owned_server(db, owner.id, name="denied-everywhere")
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda *_a, **_k: ConnectorAccess(
                    team_owned=True, can_edit=False
                ),
                visibility=lambda _db, _uid: {"mcp": {server_id}, "custom_api": set()},
            )
            list_entries = get_mcp_servers(current_user=member, db=db)
            list_entry = next(r for r in list_entries if r.id == server_id)
            get_response = get_mcp_server(server_id, current_user=member, db=db)
            put_response = update_mcp_server(
                server_id, MCPServerUpdate(), current_user=member, db=db
            )

        assert list_entry.can_edit_global is False
        assert get_response.can_edit_global is False
        assert put_response.can_edit_global is False


class TestRenameStaysScopedToItsOwnConnector:
    """Renaming one connector must not touch an outsider's own, unrelated
    connector -- a regression guard on the rename call's scope, exercised
    again here alongside the response now carrying the verdict too."""

    def test_renaming_one_connector_does_not_touch_an_outsiders_own_connector(self, db):
        owner_a = _make_user(db, 60)
        editor = _make_user(db, 61)
        outsider = _make_user(db, 62)

        server_a = _make_owned_server(db, owner_a.id, name="rename-target")
        server_b = _make_owned_server(db, outsider.id, name="outsiders-own-connector")
        server_a_id, server_b_id = server_a.id, server_b.id

        renamed_calls: list[tuple[int, str, str]] = []

        def spy_renamed_hook(_db, _user_id, _connector_type, connector_id, old, new):
            renamed_calls.append((connector_id, old, new))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda *_a, **_k: ConnectorAccess(
                    team_owned=True, can_edit=True
                ),
                renamed=spy_renamed_hook,
            )
            update_mcp_server(
                server_a_id,
                MCPServerUpdate(name="renamed-target"),
                current_user=editor,
                db=db,
            )

        assert renamed_calls == [(server_a_id, "rename-target", "renamed-target")]

        db.rollback()
        outsiders_server = db.query(MCPServer).filter(MCPServer.id == server_b_id).one()
        assert outsiders_server.name == "outsiders-own-connector"


class TestStandaloneParityWithNoHookInstalled:
    """With no hook installed at all, every route touched by this work --
    both GETs, both PUTs, toggle, and the list -- behaves exactly as it did
    before any of it started."""

    async def test_every_route_in_scope_behaves_as_before_with_no_hook_installed(
        self, db
    ):
        owner = _make_user(db, 70)
        stranger = _make_user(db, 71)
        server = _make_owned_server(db, owner.id, name="standalone-parity-mcp")
        server_id = server.id
        api = _make_owned_api(db, owner.id, name="standalone-parity-api")
        api_id = api.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks()  # explicit reset: no hooks installed

            get_response = get_mcp_server(server_id, current_user=owner, db=db)
            assert get_response.can_edit_global is True

            with pytest.raises(HTTPException) as exc:
                get_mcp_server(server_id, current_user=stranger, db=db)
            assert exc.value.status_code == 404

            put_response = update_mcp_server(
                server_id,
                MCPServerUpdate(description="parity"),
                current_user=owner,
                db=db,
            )
            assert put_response.can_edit_global is True

            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(description="x"),
                    current_user=stranger,
                    db=db,
                )
            assert exc.value.status_code == 404

            toggle_response = await toggle_mcp_server(
                server_id, current_user=owner, db=db
            )
            assert toggle_response.can_edit_global is True

            list_entries = get_mcp_servers(current_user=owner, db=db)
            mcp_entry = next(r for r in list_entries if r.id == server_id)
            assert mcp_entry.can_edit_global is True
            custom_api_entry = next(
                r
                for r in list_entries
                if r.id == api_id and r.transport == "custom_api"
            )
            assert custom_api_entry.can_edit_global is True

            api_get_response = await get_custom_api(api_id, current_user=owner, db=db)
            assert api_get_response.id == api_id

            with pytest.raises(HTTPException) as exc:
                await get_custom_api(api_id, current_user=stranger, db=db)
            assert exc.value.status_code == 404

            api_put_response = await update_custom_api(
                api_id,
                CustomApiUpdate(description="parity"),
                current_user=owner,
                db=db,
            )
            assert api_put_response.id == api_id

            with pytest.raises(HTTPException) as exc:
                await update_custom_api(
                    api_id,
                    CustomApiUpdate(description="x"),
                    current_user=stranger,
                    db=db,
                )
            assert exc.value.status_code == 404


class TestListMcpAppsPerRowDegradation:
    """A raising hook inside ``list_mcp_apps``'s local-connector loop must
    not blank the whole response list -- only the affected row's
    ``can_configure`` degrades to False, and every other row keeps
    reporting its correct value."""

    def test_a_raising_hook_for_one_mcp_connector_degrades_only_that_row(self, db):
        owner = _make_user(db, 80)
        member = _make_user(db, 81)
        broken = _make_owned_server(db, owner.id, name="broken-connector")
        healthy = _make_owned_server(db, owner.id, name="healthy-connector")
        broken_id, healthy_id = broken.id, healthy.id

        def selective_access(_db, _user_id, _connector_type, connector_id):
            if connector_id == broken_id:
                raise ValueError("hook exploded")
            return ConnectorAccess(team_owned=True, can_edit=True)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=selective_access,
                visibility=lambda _db, _uid: {
                    "mcp": {broken_id, healthy_id},
                    "custom_api": set(),
                },
            )
            entries = list_mcp_apps(location="local", current_user=member, db=db)

        broken_entry = next(e for e in entries if e["server_id"] == broken_id)
        healthy_entry = next(e for e in entries if e["server_id"] == healthy_id)
        assert broken_entry["can_configure"] is False
        assert healthy_entry["can_configure"] is True

    def test_a_raising_hook_for_one_custom_api_degrades_only_that_row(self, db):
        owner = _make_user(db, 82)
        member = _make_user(db, 83)
        broken = _make_owned_api(db, owner.id, name="broken-api")
        healthy = _make_owned_api(db, owner.id, name="healthy-api")
        broken_id, healthy_id = broken.id, healthy.id

        def selective_access(_db, _user_id, _connector_type, connector_id):
            if connector_id == broken_id:
                raise ValueError("hook exploded")
            return ConnectorAccess(team_owned=True, can_edit=True)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=selective_access,
                visibility=lambda _db, _uid: {
                    "mcp": set(),
                    "custom_api": {broken_id, healthy_id},
                },
            )
            entries = list_mcp_apps(location="local", current_user=member, db=db)

        broken_entry = next(
            e
            for e in entries
            if e["server_id"] == broken_id and e["transport"] == "custom_api"
        )
        healthy_entry = next(
            e
            for e in entries
            if e["server_id"] == healthy_id and e["transport"] == "custom_api"
        )
        assert broken_entry["can_configure"] is False
        assert healthy_entry["can_configure"] is True
