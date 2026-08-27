"""The edit right on a team-linked Custom API: ``GET``/``PUT
/api/custom-apis/{api_id}`` resolve a caller with no personal row through
the connector access hook instead of 404ing outright, ``can_edit`` falls
back to that verdict for a caller with no personal row, an ``is_active``
payload from such a caller rejects outright instead of writing a shadow
attribute the response then reads back, a raising hook surfaces as its
declared status rather than a 500, and the verdict is re-resolved once
more after the definition row's lock is taken, refusing the write if it
no longer grants what the pre-lock answer granted.

Every test installs the access hook through ``snapshot_connector_team_hooks``
so no hook state leaks between tests or into suites that run after this one.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.api.custom_api import (
    CustomApiUpdate,
    get_custom_api,
    update_custom_api,
)
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import Base
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


def _make_owned_api(db, owner_id: int, *, name: str = "shared-api") -> CustomApi:
    api = CustomApi(name=name, url="https://example.test/api", method="GET")
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


async def _get(api_id, current_user, db):
    return get_custom_api(api_id, current_user=current_user, db=db)


async def _put(api_id, payload, current_user, db):
    return update_custom_api(api_id, payload, current_user=current_user, db=db)


def _sequenced_access_hook(*answers):
    """An access hook that answers differently on successive calls, so a
    test can make the second (post-lock) resolution disagree with the
    first. ``None`` in the sequence means an empty answer -- the batch
    contract's way of saying "the caller's team does not link this". An
    entry that is an exception instance is raised instead of returned, so a
    test can make the second resolution fail outright. The last entry
    repeats for any further call. Records every call's ``refs`` on
    ``.calls`` so a test can pin how many round trips the route pays."""
    calls: list[object] = []

    def hook(db, user_id, refs):
        calls.append(refs)
        index = min(len(calls) - 1, len(answers) - 1)
        answer = answers[index]
        if isinstance(answer, BaseException):
            raise answer
        if answer is None:
            return {}
        return {ref: answer for ref in refs}

    hook.calls = calls
    return hook


class TestGateHelperOnGetAndPut:
    @pytest.mark.asyncio
    async def test_get_404s_for_an_unrelated_user_with_no_link_and_no_team_access(
        self, db
    ):
        owner = _make_user(db, 1)
        stranger = _make_user(db, 2)
        api = _make_owned_api(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=lambda db, user_id, refs: {})
            with pytest.raises(HTTPException) as exc:
                await _get(api.id, stranger, db)
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_returns_the_stand_in_for_a_team_member_with_no_personal_row(
        self, db
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = await _get(api.id, member, db)

        assert response.id == api.id
        assert response.user_id == member.id

    @pytest.mark.asyncio
    async def test_get_owner_behaviour_is_unchanged_with_no_hook_installed(self, db):
        owner = _make_user(db, 1)
        api = _make_owned_api(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks()
            response = await _get(api.id, owner, db)

        assert response.id == api.id
        assert response.user_id == owner.id


class TestPutWiringForATeamEditor:
    @pytest.mark.asyncio
    async def test_team_editor_edit_is_durable_and_creates_no_association_row(self, db):
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        api = _make_owned_api(db, owner.id)
        api_id = api.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = await _put(
                api_id,
                CustomApiUpdate(description="edited by the team"),
                editor,
                db,
            )

        assert response.description == "edited by the team"

        # Durability, not staging -- a same-session query would still see
        # an uncommitted UPDATE even if the route never committed.
        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.description == "edited by the team"

        # The edit did not fabricate a personal association for the team
        # editor -- that would be a get-or-create write on an
        # authorization path.
        assert (
            db.query(UserCustomApi).filter(UserCustomApi.user_id == editor.id).first()
            is None
        )

    @pytest.mark.asyncio
    async def test_view_only_team_member_cannot_tamper_the_shared_config(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=False)
                    for ref in refs
                }
            )
            with pytest.raises(HTTPException) as exc:
                await _put(
                    api.id,
                    CustomApiUpdate(description="should not land"),
                    member,
                    db,
                )
        assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_a_denying_verdict_stand_in_is_403_on_an_empty_payload_too(db):
    """The MCP side needed a new guard for this (see
    TestADenyingStandInIsRefusedRatherThanReportedSuccessful in
    test_mcp_team_connector_edit.py) because its personal-field guard and
    tamper check only fire for specific payload shapes. This route's own
    gate (custom_api.py's ``can_edit`` check) has no such carve-out: it
    requires the edit right for every payload, including an empty one, so
    a stand-in whose verdict denies edit is already 403 here without any
    new code. This test exists to pin that so it cannot be changed out
    from under this route's contract unnoticed.
    """
    owner = _make_user(db, 1)
    member = _make_user(db, 2)
    api = _make_owned_api(db, owner.id, name="denying-stand-in-target")
    api_id = api.id
    # Captured as plain values, not read off ``api`` after the call: ``api``
    # and the ``refreshed`` row below share the same identity-mapped Python
    # object in this session, so comparing one against the other after the
    # call would be comparing the object with itself and could never fail.
    original_name = str(api.name)
    original_description = str(api.description) if api.description is not None else None

    with snapshot_connector_team_hooks():
        set_connector_team_hooks(
            access=lambda db, user_id, refs: {
                ref: ConnectorAccess(team_owned=True, can_edit=False) for ref in refs
            }
        )
        with pytest.raises(HTTPException) as exc:
            await _put(api_id, CustomApiUpdate(), member, db)
    assert exc.value.status_code == 403

    db.rollback()
    refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
    assert refreshed.name == original_name
    assert refreshed.description == original_description
    assert (
        db.query(UserCustomApi).filter(UserCustomApi.user_id == member.id).count() == 0
    )


class TestIsActiveRejectionForAStandIn:
    @pytest.mark.asyncio
    async def test_is_active_from_a_caller_with_no_personal_row_is_400_not_a_silent_drop(
        self, db
    ):
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="unchanged-name")
        api_id = api.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            with pytest.raises(HTTPException) as exc:
                await _put(
                    api_id,
                    CustomApiUpdate(is_active=False),
                    editor,
                    db,
                )

        # 1. the declared status.
        assert exc.value.status_code == 400
        assert "personal connection" in str(exc.value.detail)

        # 2. nothing persisted -- the exception was raised before any
        # commit, so a same-session rollback-then-requery must still show
        # no personal association row for this caller.
        db.rollback()
        assert (
            db.query(UserCustomApi).filter(UserCustomApi.user_id == editor.id).first()
            is None
        )

        # 3. the response body does not claim the change -- the call
        # raised rather than returning, so no ``CustomApiResponse`` ever
        # left the route carrying an ``is_active`` value nothing wrote.
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.name == "unchanged-name"


class TestTypedErrorArm:
    """A raising hook still surfaces its declared status for a caller with
    no working personal row -- the verdict is genuinely the gate for that
    population and must stay fail-closed. An owner's row already decides
    ``GET``'s answer (it never reads the verdict at all) and ``PUT``'s
    (``can_edit`` is already ``True``), so neither ever calls the hook for
    an owner's row; that population is pinned separately, below, in
    ``TestOwnerIsImmuneToAHookFailure``."""

    @pytest.mark.asyncio
    async def test_get_surfaces_a_raising_hooks_declared_status(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id)

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            with pytest.raises(HTTPException) as exc:
                await _get(api.id, member, db)

        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_put_surfaces_a_raising_hooks_declared_status_and_leaves_the_row_unchanged(
        self, db
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="pristine")
        api_id = api.id

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            with pytest.raises(HTTPException) as exc:
                await _put(
                    api_id,
                    CustomApiUpdate(name="should-not-land"),
                    member,
                    db,
                )

        assert exc.value.status_code == 503

        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.name == "pristine"

    @pytest.mark.asyncio
    async def test_put_passes_through_a_planted_connector_runtime_error_by_its_own_status(
        self, db
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id)

        def boom(*_a, **_k):
            raise ConnectorRuntimeError("planted", "planted failure", status_code=409)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            with pytest.raises(HTTPException) as exc:
                await _put(
                    api.id,
                    CustomApiUpdate(description="irrelevant"),
                    member,
                    db,
                )

        assert exc.value.status_code == 409
        assert exc.value.detail == "planted failure"


class TestOwnerIsImmuneToAHookFailure:
    """An owner's row already decides both routes' answers on its own --
    ``GET`` never reads the verdict at all, and ``PUT``'s ``can_edit`` is
    already ``True`` -- so neither ever calls the hook for an owner's row.
    A hook that would raise must therefore never surface: both routes
    return their normal success status, unaffected by whatever the hook
    would have done."""

    @pytest.mark.asyncio
    async def test_get_and_put_succeed_for_an_owner_even_though_the_hook_would_raise(
        self, db
    ):
        owner = _make_user(db, 1)
        api = _make_owned_api(db, owner.id, name="owner-immune")
        api_id = api.id

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            get_response = await _get(api_id, owner, db)
            put_response = await _put(
                api_id,
                CustomApiUpdate(description="edited by the owner"),
                owner,
                db,
            )

        assert get_response.id == api_id
        assert put_response.description == "edited by the owner"


class TestTheVerdictIsRevalidatedUnderTheDefinitionLock:
    """The same re-check as the MCP side's PUT (see
    TestTheVerdictIsRevalidatedUnderTheDefinitionLock in
    test_mcp_team_connector_edit.py), for the same reason: the verdict
    granting a stand-in edit access was resolved before this route's own
    row lock existed, and the installing application can revoke the link
    at any moment through its own tables, which this lock does not cover.
    No personal-field exemption here: this route's gate requires can_edit
    for every payload, including an is_active-only one, so the verdict is
    the authority for everything this route admits.
    """

    async def _run(self, db, *, hook):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="revalidated-under-lock")
        api_id = api.id
        # Captured as plain values before the call, not read off ``api``
        # afterwards: ``api`` and the requery below share the same
        # identity-mapped Python object in this session, so comparing one
        # against the other after the call would be comparing the object
        # with itself and could never fail.
        original_name = str(api.name)
        original_description = (
            str(api.description) if api.description is not None else None
        )

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            result = {}
            try:
                result["response"] = await _put(
                    api_id,
                    CustomApiUpdate(description="edited-while-in-flight"),
                    member,
                    db,
                )
            except HTTPException as exc:
                result["error"] = exc
            return api, api_id, result, original_name, original_description

    @pytest.mark.asyncio
    async def test_revoked_between_resolution_and_lock_is_refused(self, db):
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True), None
        )
        _api, api_id, result, original_name, original_description = await self._run(
            db, hook=hook
        )

        assert result["error"].status_code == 403
        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.name == original_name
        assert refreshed.description == original_description
        assert db.query(UserCustomApi).filter(UserCustomApi.user_id == 2).count() == 0

    @pytest.mark.asyncio
    async def test_downgraded_to_not_editable_between_resolution_and_lock_is_refused(
        self, db
    ):
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True),
            ConnectorAccess(team_owned=True, can_edit=False),
        )
        _api, api_id, result, original_name, original_description = await self._run(
            db, hook=hook
        )

        assert result["error"].status_code == 403
        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.name == original_name
        assert refreshed.description == original_description
        assert db.query(UserCustomApi).filter(UserCustomApi.user_id == 2).count() == 0

    @pytest.mark.asyncio
    async def test_still_granted_on_recheck_commits_durably(self, db):
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True),
            ConnectorAccess(team_owned=True, can_edit=True),
        )
        (
            _api,
            api_id,
            result,
            _original_name,
            _original_description,
        ) = await self._run(db, hook=hook)

        assert "error" not in result
        assert result["response"].description == "edited-while-in-flight"

        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.description == "edited-while-in-flight"

    @pytest.mark.asyncio
    async def test_recheck_that_raises_surfaces_the_hooks_own_status_with_zero_side_effects(
        self, db
    ):
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True),
            ValueError("hook exploded during recheck"),
        )
        _api, api_id, result, original_name, original_description = await self._run(
            db, hook=hook
        )

        assert result["error"].status_code == 503
        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.name == original_name
        assert refreshed.description == original_description
        assert db.query(UserCustomApi).filter(UserCustomApi.user_id == 2).count() == 0


class TestTheRecheckCostsExactlyOneExtraHookCall:
    """The Custom API halves of cells i and j in the design's call-count
    table -- MCP's own halves (cells e-h) live in
    test_mcp_team_connector_edit.py."""

    @pytest.mark.asyncio
    async def test_a_granting_stand_in_editing_the_shared_config_pays_two_calls(
        self, db
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="cost-stand-in-shared")
        api_id = api.id
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            await _put(api_id, CustomApiUpdate(description="shared-edit"), member, db)

        assert len(hook.calls) == 2

    @pytest.mark.asyncio
    async def test_an_owner_pays_zero_calls(self, db):
        owner = _make_user(db, 1)
        api = _make_owned_api(db, owner.id, name="cost-owner")
        api_id = api.id
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            await _put(api_id, CustomApiUpdate(description="owner-edit"), owner, db)

        assert len(hook.calls) == 0
