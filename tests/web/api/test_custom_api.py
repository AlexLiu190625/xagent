import ast
import importlib
import inspect
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.api.custom_api import (
    CustomApiCreate,
    CustomApiResponse,
    CustomApiUpdate,
    _process_env_vars,
    create_custom_api,
    delete_custom_api,
    get_custom_api,
    list_custom_apis,
    update_custom_api,
)
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.services.connector_team_scope import (
    ConnectorDeleteDecision,
    set_connector_team_hooks,
    snapshot_connector_team_hooks,
)


def test_custom_api_models_env_validation():
    # Valid creation
    api = CustomApiCreate(name="test", env={"key": "val"})
    assert api.env == {"key": "val"}

    # Missing env is allowed (handled by database default or just none)
    api = CustomApiCreate(name="test")
    assert api.env is None

    # Empty env dict is allowed to clear secrets
    api_empty = CustomApiCreate(name="test", env={})
    assert api_empty.env == {}

    # Same for update
    api_update = CustomApiUpdate(name="test", env={})
    assert api_update.env == {}

    runtime_api = CustomApiCreate(
        name="runtime",
        runtime_input_schema={"context": {"account_id": {"type": "string"}}},
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
    )
    assert runtime_api.runtime_input_schema == {
        "context": {"account_id": {"type": "string"}}
    }
    assert runtime_api.runtime_bindings is not None


def test_custom_api_response_requires_runtime_projection_fields():
    """Custom API response mappers must project every persisted runtime field."""
    response_data = {
        "id": 1,
        "user_id": 1,
        "name": "runtime",
        "description": None,
        "url": None,
        "method": "GET",
        "headers": None,
        "body": None,
        "env": None,
        "is_active": True,
        "is_default": False,
        "created_at": "2026-07-14T00:00:00",
        "updated_at": "2026-07-14T00:00:00",
    }

    with pytest.raises(ValidationError) as exc_info:
        CustomApiResponse(**response_data)

    missing_fields = {error["loc"] for error in exc_info.value.errors()}
    assert missing_fields == {
        ("runtime_input_schema",),
        ("runtime_bindings",),
        ("allow_delegated_authorization",),
    }


def test_process_env_vars():
    with patch(
        "xagent.web.api.custom_api.encrypt_value", side_effect=lambda x: f"enc_{x}"
    ):
        # Test None
        assert _process_env_vars(None) is None

        # Test encrypting new values
        env = {"key1": "val1", "key2": "val2"}
        res = _process_env_vars(env)
        assert res == {"key1": "enc_val1", "key2": "enc_val2"}

        # Test keeping masked values
        env_with_mask = {"key1": "********", "key3": "val3"}
        existing = {"key1": "enc_old1", "key2": "enc_old2"}
        res_masked = _process_env_vars(env_with_mask, existing)
        assert res_masked == {"key1": "enc_old1", "key3": "enc_val3"}

        # A mask cannot be moved to a new key identity.
        with pytest.raises(ValueError, match="new_key"):
            _process_env_vars({"new_key": "********"}, existing)


@pytest.mark.asyncio
async def test_list_custom_apis():
    db = MagicMock(spec=Session)
    user = User(id=1)

    mock_api = CustomApi(
        id=10, name="test_api", created_at=datetime.now(), updated_at=datetime.now()
    )
    mock_user_api = UserCustomApi(
        user_id=1,
        custom_api_id=10,
        is_active=True,
        is_default=False,
        custom_api=mock_api,
    )

    db.query().filter().all.return_value = [mock_user_api]

    res = await list_custom_apis(current_user=user, db=db)
    assert len(res) == 1
    assert res[0].name == "test_api"
    assert res[0].id == 10


@pytest.mark.asyncio
async def test_create_custom_api():
    db = MagicMock(spec=Session)
    user = User(id=1)

    api_data = CustomApiCreate(
        name="new_api",
        description="desc",
        env={"k1": "v1"},
        runtime_input_schema={"context": {"account_id": {"type": "string"}}},
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
        is_active=True,
    )

    # Mock no existing api
    db.query().filter().first.return_value = None

    # Create mock CustomApi object with datetimes so isoformat() doesn't fail
    CustomApi(
        id=1, name="new_api", created_at=datetime.now(), updated_at=datetime.now()
    )

    # Create mock UserCustomApi object to pair with our custom api mock
    UserCustomApi(
        user_id=1, custom_api_id=1, is_owner=True, is_active=True, is_default=False
    )

    # Update db.add to populate created_at/updated_at fields on our mock
    def mock_add(obj):
        if isinstance(obj, CustomApi):
            obj.id = 1
            obj.created_at = datetime.now()
            obj.updated_at = datetime.now()
        elif isinstance(obj, UserCustomApi):
            obj.user_id = 1
            obj.custom_api_id = 1
            obj.is_active = True
            obj.is_default = False

    db.add.side_effect = mock_add

    with patch(
        "xagent.web.api.custom_api.encrypt_value", side_effect=lambda x: f"enc_{x}"
    ):
        res = await create_custom_api(api_data, current_user=user, db=db)

        assert res.name == "new_api"
        assert res.env == {"k1": "********"}  # Response should mask env
        assert res.runtime_input_schema == {
            "context": {"account_id": {"type": "string"}}
        }
        assert res.runtime_bindings == [
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ]
        db.add.assert_called()
        db.commit.assert_called()


@pytest.mark.asyncio
async def test_create_custom_api_duplicate_name():
    db = MagicMock(spec=Session)
    user = User(id=1)

    api_data = CustomApiCreate(name="existing_api")

    # Mock existing api
    db.query().filter().first.return_value = CustomApi(name="existing_api")

    with pytest.raises(HTTPException) as exc_info:
        await create_custom_api(api_data, current_user=user, db=db)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_create_custom_api_rejects_runtime_static_header_conflict():
    db = MagicMock(spec=Session)
    user = User(id=1)
    api_data = CustomApiCreate(
        name="runtime_api",
        headers={"X-Account-ID": "static"},
        runtime_input_schema={"context": {"account_id": {"type": "string"}}},
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
    )
    db.query().filter().first.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await create_custom_api(api_data, current_user=user, db=db)

    assert exc_info.value.status_code == 400
    assert "Invalid runtime configuration" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_get_custom_api():
    db = MagicMock(spec=Session)
    user = User(id=1)

    mock_api = CustomApi(
        id=10, name="test_api", created_at=datetime.now(), updated_at=datetime.now()
    )
    mock_user_api = UserCustomApi(
        user_id=1,
        custom_api_id=10,
        is_active=True,
        is_default=False,
        custom_api=mock_api,
    )

    db.query().filter().first.return_value = mock_user_api

    res = get_custom_api(10, current_user=user, db=db)
    assert res.id == 10
    assert res.name == "test_api"


@pytest.mark.asyncio
async def test_get_custom_api_not_found():
    db = MagicMock(spec=Session)
    user = User(id=1)
    db.query().filter().first.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        get_custom_api(99, current_user=user, db=db)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_custom_api():
    db = MagicMock(spec=Session)
    user = User(id=1)

    mock_api = CustomApi(
        id=10,
        name="old_name",
        env={"k1": "enc_old1"},
        runtime_input_schema={"context": {"account_id": {"type": "string"}}},
        runtime_bindings=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_user_api = UserCustomApi(
        user_id=1,
        custom_api_id=10,
        can_edit=True,
        is_active=True,
        is_default=False,
        custom_api=mock_api,
    )

    # Return user api on first query
    # Return None for existing name check
    db.query().filter().first.side_effect = [mock_user_api, None]
    # The row lock's own fresh query is a separate mock chain
    # (.populate_existing().with_for_update() sits between .filter() and
    # .first()), so it needs its own return value rather than sharing the
    # side_effect list above.
    db.query().filter().populate_existing().with_for_update().first.return_value = (
        mock_api
    )

    api_data = CustomApiUpdate(
        name="new_name",
        env={"k1": "********", "k2": "v2"},
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
    )

    with patch(
        "xagent.web.api.custom_api.encrypt_value", side_effect=lambda x: f"enc_{x}"
    ):
        update_custom_api(10, api_data, current_user=user, db=db)

        assert mock_api.name == "new_name"
        assert mock_api.env == {"k1": "enc_old1", "k2": "enc_v2"}
        assert mock_api.runtime_input_schema == {
            "context": {"account_id": {"type": "string"}}
        }
        assert mock_api.runtime_bindings == [
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ]
        db.commit.assert_called()


@pytest.mark.asyncio
async def test_update_custom_api_env_replacement_deletes_only_the_omitted_secret():
    db = MagicMock(spec=Session)
    user = User(id=1)
    mock_api = CustomApi(
        id=10,
        name="records",
        env={"BEARER_TOKEN": "enc_bearer", "TENANT": "enc_tenant"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_user_api = UserCustomApi(
        user_id=1,
        custom_api_id=10,
        can_edit=True,
        is_active=True,
        is_default=False,
        custom_api=mock_api,
    )
    db.query().filter().first.return_value = mock_user_api
    # The row lock's own fresh query is a separate mock chain -- see the
    # comment in test_update_custom_api.
    db.query().filter().populate_existing().with_for_update().first.return_value = (
        mock_api
    )

    with patch(
        "xagent.web.api.custom_api.encrypt_value", side_effect=lambda x: f"enc_{x}"
    ):
        update_custom_api(
            10,
            CustomApiUpdate(env={"TENANT": "********"}),
            current_user=user,
            db=db,
        )

    assert mock_api.env == {"TENANT": "enc_tenant"}


@pytest.mark.asyncio
async def test_update_custom_api_rejects_renamed_masked_secret():
    db = MagicMock(spec=Session)
    user = User(id=1)
    mock_api = CustomApi(
        id=10,
        name="records",
        env={"TOKEN": "encrypted-token"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_user_api = UserCustomApi(
        user_id=1,
        custom_api_id=10,
        can_edit=True,
        is_active=True,
        is_default=False,
        custom_api=mock_api,
    )
    db.query().filter().first.return_value = mock_user_api
    # The row lock's own fresh query is a separate mock chain -- see the
    # comment in test_update_custom_api.
    db.query().filter().populate_existing().with_for_update().first.return_value = (
        mock_api
    )

    with pytest.raises(HTTPException) as exc_info:
        update_custom_api(
            10,
            CustomApiUpdate(env={"RENAMED_TOKEN": "********"}),
            current_user=user,
            db=db,
        )

    assert exc_info.value.status_code == 400
    assert mock_api.env == {"TOKEN": "encrypted-token"}
    db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_update_custom_api_explicit_null_clears_runtime_config():
    db = MagicMock(spec=Session)
    user = User(id=1)

    mock_api = CustomApi(
        id=10,
        name="old_name",
        runtime_input_schema={"context": {"account_id": {"type": "string"}}},
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
        allow_delegated_authorization=True,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_user_api = UserCustomApi(
        user_id=1,
        custom_api_id=10,
        can_edit=True,
        is_active=True,
        is_default=False,
        custom_api=mock_api,
    )
    db.query().filter().first.return_value = mock_user_api
    # The row lock's own fresh query is a separate mock chain -- see the
    # comment in test_update_custom_api.
    db.query().filter().populate_existing().with_for_update().first.return_value = (
        mock_api
    )

    api_data = CustomApiUpdate(
        runtime_input_schema=None,
        runtime_bindings=None,
        allow_delegated_authorization=False,
    )

    update_custom_api(10, api_data, current_user=user, db=db)

    assert mock_api.runtime_input_schema is None
    assert mock_api.runtime_bindings is None
    assert mock_api.allow_delegated_authorization is False
    db.commit.assert_called()


@pytest.mark.asyncio
async def test_delete_custom_api():
    db = MagicMock(spec=Session)
    user = User(id=1)

    mock_api = CustomApi(id=10)
    mock_user_api = UserCustomApi(
        user_id=1, custom_api_id=10, can_delete=True, custom_api=mock_api
    )

    db.query().filter().first.return_value = mock_user_api
    db.query().filter().populate_existing().with_for_update().first.return_value = (
        mock_api
    )

    delete_custom_api(10, current_user=user, db=db)

    db.delete.assert_called_once_with(mock_api)
    db.commit.assert_called()


@pytest.mark.asyncio
async def test_delete_team_custom_api_flushes_only_current_user_link():
    db = MagicMock(spec=Session)
    user = User(id=1)
    mock_api = CustomApi(id=10)
    mock_user_api = UserCustomApi(
        user_id=1, custom_api_id=10, can_delete=True, custom_api=mock_api
    )
    db.query().filter().first.side_effect = [mock_user_api, None]
    db.query().filter().populate_existing().with_for_update().first.return_value = (
        mock_api
    )

    decision = ConnectorDeleteDecision(
        team_owned=True,
        authorized=True,
        delete_definition=True,
    )
    with patch(
        "xagent.web.services.connector_team_scope.delete_team_connector",
        return_value=decision,
    ):
        delete_custom_api(10, current_user=user, db=db)

    db.flush.assert_called_once_with([mock_user_api])
    assert db.no_autoflush.__enter__.called
    assert db.delete.call_args_list == [call(mock_user_api), call(mock_api)]
    db.commit.assert_called_once()


def test_the_locking_routes_are_sync_defs_so_a_lock_wait_never_holds_the_event_loop():
    """Both routes below run a ``SELECT ... FOR UPDATE`` that can wait
    indefinitely on a concurrent writer. FastAPI runs a coroutine route on
    the event loop thread itself, so such a wait inside an ``async def``
    route stalls every other request the process is serving. Declaring them
    as plain ``def`` puts them in the threadpool instead, which is what the
    MCP side's own PUT already does."""
    import inspect

    from xagent.web.api import custom_api as custom_api_api

    assert not inspect.iscoroutinefunction(custom_api_api.update_custom_api)
    assert not inspect.iscoroutinefunction(custom_api_api.delete_custom_api)


_SEAM_MODULES = ("xagent.web.api.custom_api", "xagent.web.api.mcp")

# The one function that reaches the connector team seam and is still a
# coroutine, with the fact that makes it impossible to convert. Its own
# await -- one that is not the seam call itself -- is asserted below, so
# this entry cannot be claimed by a route whose coroutine is only the
# seam's doing.
_COROUTINE_EXEMPTIONS = {("xagent.web.api.mcp", "delete_mcp_server")}

_SEAM_REACHING_FUNCTIONS = {
    ("xagent.web.api.custom_api", "_resolve_custom_api_for_request"),
    ("xagent.web.api.custom_api", "get_custom_api"),
    ("xagent.web.api.custom_api", "update_custom_api"),
    ("xagent.web.api.custom_api", "delete_custom_api"),
    ("xagent.web.api.mcp", "_resolve_mcp_server_for_request"),
    ("xagent.web.api.mcp", "_local_mcp_can_attach"),
    ("xagent.web.api.mcp", "list_mcp_apps"),
    ("xagent.web.api.mcp", "get_mcp_servers"),
    ("xagent.web.api.mcp", "get_mcp_server"),
    ("xagent.web.api.mcp", "connect_mcp_app"),
    ("xagent.web.api.mcp", "update_mcp_server"),
    ("xagent.web.api.mcp", "delete_mcp_server"),
    ("xagent.web.api.mcp", "toggle_mcp_server"),
}


def _functions_reaching_the_connector_seam(module_name: str) -> dict[str, ast.AST]:
    """Every top-level function in ``module_name`` that can reach an
    installed connector team hook.

    Seeded on the functions that import ``connector_team_scope`` in their
    own body -- which is how every call site in these two modules reaches
    the seam -- then closed transitively over plain-name calls, because
    two of the routes reach it only through a helper (``get_custom_api``
    through ``_resolve_custom_api_for_request``, ``get_mcp_server``
    through ``_resolve_mcp_server_for_request``). A seed-only check would
    miss exactly the route this test exists for.
    """
    module = importlib.import_module(module_name)
    tree = ast.parse(inspect.getsource(module))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reaching = {
        name
        for name, node in functions.items()
        if any(
            isinstance(child, ast.ImportFrom)
            and child.module is not None
            and child.module.endswith("connector_team_scope")
            for child in ast.walk(node)
        )
    }
    changed = True
    while changed:
        changed = False
        for name, node in functions.items():
            if name in reaching:
                continue
            called = {
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            }
            if called & reaching:
                reaching.add(name)
                changed = True
    return {name: functions[name] for name in reaching}


def _seam_names_imported_by(node: ast.AST) -> set[str]:
    """The names this function imports from ``connector_team_scope``.

    Read off the function's own body because that is how every call site in
    these two modules reaches the seam -- the same fact
    ``_functions_reaching_the_connector_seam`` above is seeded on.
    """
    return {
        alias.asname or alias.name
        for child in ast.walk(node)
        if isinstance(child, ast.ImportFrom)
        and child.module is not None
        and child.module.endswith("connector_team_scope")
        for alias in child.names
    }


def test_the_discovery_of_seam_reaching_functions_is_not_vacuous():
    """Pins the enumeration itself, so the assertion below cannot pass by
    finding nothing."""
    found = {
        (module_name, name)
        for module_name in _SEAM_MODULES
        for name in _functions_reaching_the_connector_seam(module_name)
    }
    assert found == _SEAM_REACHING_FUNCTIONS


def test_no_function_that_reaches_the_connector_seam_is_a_coroutine():
    """An installed connector team hook may be slow -- this repo's own
    design assumes it does database-backed work. FastAPI runs a coroutine
    route on the event loop thread itself, so a slow hook call inside an
    ``async def`` stalls every other request the process is serving, not
    just this one; a plain ``def`` goes to the threadpool instead, where a
    slow call occupies one worker.

    Enumerated by reachability rather than by a hand-written list of
    routes: the earlier fix for this same risk class swept siblings along
    the "takes a row lock" axis and therefore missed two routes that call
    a hook without taking one.
    """
    offenders = []
    for module_name in _SEAM_MODULES:
        for name, node in _functions_reaching_the_connector_seam(module_name).items():
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            if (module_name, name) in _COROUTINE_EXEMPTIONS:
                # An exemption is only legitimate for a function that
                # genuinely cannot be converted, so it must carry an await
                # that is NOT the seam call itself. A function whose only
                # await IS the seam call is a coroutine of the seam's own
                # making -- convertible by making that call synchronous --
                # and "contains some await" would still wave it through.
                seam_names = _seam_names_imported_by(node)
                non_seam_awaits = [
                    child
                    for child in ast.walk(node)
                    if isinstance(child, ast.Await)
                    and not (
                        isinstance(child.value, ast.Call)
                        and isinstance(child.value.func, ast.Name)
                        and child.value.func.id in seam_names
                    )
                ]
                assert non_seam_awaits, (
                    f"{module_name}.{name} is exempted from this invariant, but "
                    "every await it has is a seam call -- the coroutine is the "
                    "seam's own doing, so make that call synchronous instead of "
                    "exempting the route"
                )
                continue
            offenders.append(f"{module_name}.{name}")
    assert offenders == [], (
        "these functions can reach an installed connector team hook while "
        f"running on the event loop thread: {offenders}"
    )


def _lock_order_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine), engine


def _seed_owned_api_for_lock_order(session_factory, *, name: str) -> tuple[int, int]:
    db = session_factory()
    owner = User(username=f"user-{name}", password_hash="x", is_admin=False)
    db.add(owner)
    db.flush()
    api = CustomApi(name=name, url="https://example.test/api", method="GET")
    db.add(api)
    db.flush()
    db.add(
        UserCustomApi(
            user_id=owner.id,
            custom_api_id=api.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=True,
        )
    )
    db.commit()
    owner_id, api_id = int(owner.id), int(api.id)
    db.close()
    return owner_id, api_id


def _count_custom_apis_selects_before_first_delete(statements: list[str]) -> int:
    """How many ``SELECT``s against ``custom_apis`` land before the first
    ``DELETE`` of either table.

    The route's own not-found guard (``not user_api or not
    user_api.custom_api``) always lazy-loads the ``custom_api`` relationship,
    which is one such ``SELECT`` on its own -- with or without the lock
    statement this test exists to pin. So *presence* of a ``custom_apis``
    ``SELECT`` before the delete is true either way and proves nothing; the
    *count* is what distinguishes them -- one without the lock statement,
    two with it, because ``populate_existing()`` forces the lock's query to
    hit the database again rather than reuse the already-loaded row.
    """
    count = 0
    for statement in statements:
        upper = statement.strip().upper()
        if upper.startswith("DELETE"):
            break
        if upper.startswith("SELECT") and "FROM CUSTOM_APIS" in upper:
            count += 1
    return count


class TestDeleteLockOrderMatchesThePutsLockOrder:
    """``update_custom_api`` locks the ``CustomApi`` definition row first and
    writes the ``UserCustomApi`` link row afterwards. For the two routes to
    share one global lock order, ``delete_custom_api`` must take the same
    definition-row lock before it deletes the link row, in both of its
    branches.

    SQLite silently drops ``FOR UPDATE`` (it is a no-op on this dialect), so
    nothing here demonstrates that the lock actually blocks a second writer
    -- that proof lives in test_custom_api_edit_lock_postgresql.py, against
    a real server. What this proves instead is statement *order*, which is
    dialect-independent and exercisable without one.
    """

    def _run(self, *, team_owned: bool) -> list[str]:
        session_factory, engine = _lock_order_session_factory()
        owner_id, api_id = _seed_owned_api_for_lock_order(
            session_factory,
            name="lock-order-team" if team_owned else "lock-order-cascade",
        )
        db = session_factory()
        current_user = SimpleNamespace(id=owner_id, is_admin=False)

        statements: list[str] = []

        def record_query(conn, cursor, statement, parameters, context, executemany):
            del conn, cursor, parameters, context, executemany
            statements.append(statement)

        event.listen(engine, "before_cursor_execute", record_query)
        try:
            if team_owned:

                def deleted_hook(_db, _user_id, _connector_type, _connector_id):
                    return ConnectorDeleteDecision(
                        team_owned=True, authorized=True, delete_definition=True
                    )

                with snapshot_connector_team_hooks():
                    set_connector_team_hooks(deleted=deleted_hook)
                    delete_custom_api(api_id, current_user=current_user, db=db)
            else:
                delete_custom_api(api_id, current_user=current_user, db=db)
        finally:
            event.remove(engine, "before_cursor_execute", record_query)
            db.close()
        return statements

    def test_lock_order_team_owned_branch(self):
        statements = self._run(team_owned=True)
        assert _count_custom_apis_selects_before_first_delete(statements) == 2, (
            "expected the not-found guard's relationship load AND the new "
            "lock statement's own SELECT against custom_apis, both before "
            "the first DELETE"
        )

    def test_lock_order_cascade_branch(self):
        statements = self._run(team_owned=False)
        assert _count_custom_apis_selects_before_first_delete(statements) == 2, (
            "expected the not-found guard's relationship load AND the new "
            "lock statement's own SELECT against custom_apis, both before "
            "the first DELETE"
        )
