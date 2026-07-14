import os
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import xagent.web.api.mcp as mcp_api
from xagent.web.api.admin_mcp import admin_mcp_router
from xagent.web.api.auth import (
    AppNotOAuthError,
    _ensure_user_mcp_server,
    auth_router,
)
from xagent.web.api.mcp import mcp_router
from xagent.web.models.database import Base, get_db, get_engine
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth


def override_get_db():
    db = None
    try:
        db = next(get_db())
        yield db
    finally:
        if db is not None:
            db.close()


app_for_tests = FastAPI()
app_for_tests.include_router(auth_router)
app_for_tests.include_router(mcp_router)
app_for_tests.include_router(admin_mcp_router)
app_for_tests.dependency_overrides[get_db] = override_get_db
client = TestClient(app_for_tests)


def _setup_test_db() -> str:
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    db_url = f"sqlite:///{temp_db_path}"
    init_db(db_url=db_url)
    return temp_dir


def _setup_admin() -> None:
    status_response = client.get("/api/auth/setup-status")
    assert status_response.status_code == 200
    if status_response.json().get("needs_setup", True):
        setup_response = client.post(
            "/api/auth/setup-admin",
            json={
                "username": "admin",
                "email": "admin@example.com",
                "password": "admin123",
            },
        )
        assert setup_response.status_code == 200


def _login(username: str, password: str) -> dict[str, str]:
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _create_public_app(
    headers: dict[str, str],
    app_id: str,
    name: str,
    is_visible_in_connector: bool,
    transport: str = "oauth",
) -> None:
    response = client.post(
        "/api/admin/mcp/apps",
        headers=headers,
        json={
            "app_id": app_id,
            "name": name,
            "description": f"{name} description",
            "icon": "",
            "transport": transport,
            "provider_name": None,
            "category": "Communication",
            "oauth_scopes": [],
            "is_visible_in_connector": is_visible_in_connector,
            "launch_config": {},
        },
    )
    assert response.status_code == 200


def _connect_app_for_user(username: str, server_name: str) -> None:
    db = next(get_db())
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None

        server = MCPServer(
            name=server_name,
            description="connected hidden app",
            managed="external",
            transport="oauth",
        )
        db.add(server)
        db.flush()

        db.add(
            UserMCPServer(
                user_id=user.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        db.commit()
    finally:
        db.close()


def _connect_custom_stdio_mcp_for_user(username: str, server_name: str) -> None:
    db = next(get_db())
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None

        server = MCPServer(
            name=server_name,
            description="custom stdio connector",
            managed="external",
            transport="stdio",
            command="npx",
            args=["-y", "@floriscornel/teams-mcp@latest"],
        )
        db.add(server)
        db.flush()

        db.add(
            UserMCPServer(
                user_id=user.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        db.commit()
    finally:
        db.close()


def _connect_oauth_account_for_user(username: str, provider: str) -> None:
    db = next(get_db())
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None

        db.add(
            UserOAuth(
                user_id=user.id,
                provider=provider,
                access_token="access-token",
                provider_user_id=f"{provider}-user",
                email=f"{provider}@example.com",
            )
        )
        db.commit()
    finally:
        db.close()


def test_connected_non_oauth_public_app_is_marked_connected() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        _create_public_app(
            admin_headers,
            "public-stdio",
            "Public Stdio",
            True,
            transport="stdio",
        )
        _connect_custom_stdio_mcp_for_user("regular", "Public Stdio")

        response = client.get(
            "/api/mcp/apps?location=remote&search=public",
            headers=regular_headers,
        )
        assert response.status_code == 200

        public_app = next(app for app in response.json() if app["id"] == "public-stdio")
        assert public_app["is_connected"] is True
        assert isinstance(public_app["server_id"], int)
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_local_custom_mcp_app_is_a_self_contained_actor_scoped_edit_reference() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")
        _connect_custom_stdio_mcp_for_user("regular", "Private Local MCP")

        response = client.get(
            "/api/mcp/apps?location=local&search=private",
            headers=regular_headers,
        )
        assert response.status_code == 200, response.text
        local_app = next(
            app for app in response.json() if app["id"] == "Private Local MCP"
        )
        assert isinstance(local_app["server_id"], int)
        assert local_app["transport"] == "stdio"
        assert local_app["is_custom"] is True
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_non_oauth_public_app_matches_space_hyphen_name_variant() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        _create_public_app(
            admin_headers,
            "space-hyphen",
            "Different Display Name",
            True,
            transport="stdio",
        )
        _connect_custom_stdio_mcp_for_user("regular", "Space Hyphen")

        response = client.get(
            "/api/mcp/apps?location=remote&search=different",
            headers=regular_headers,
        )
        assert response.status_code == 200

        public_app = next(app for app in response.json() if app["id"] == "space-hyphen")
        assert public_app["is_connected"] is True
        assert isinstance(public_app["server_id"], int)
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_remote_connector_builds_oauth_connectability_once_per_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")
        _connect_oauth_account_for_user("regular", "microsoft")

        checked_providers: list[str] = []

        def count_connectability_check(oauth_account: object) -> bool:
            checked_providers.append(str(getattr(oauth_account, "provider")))
            return True

        monkeypatch.setattr(
            mcp_api, "_oauth_account_can_connect", count_connectability_check
        )

        response = client.get("/api/mcp/apps?location=remote", headers=regular_headers)
        assert response.status_code == 200

        assert checked_providers == ["microsoft"]
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_oauth_account_can_connect_with_sqlite_naive_utc_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            if tz is timezone.utc:
                return cls(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
            return cls(2026, 1, 1, 16, 0)

    monkeypatch.setattr(mcp_api, "datetime", FixedDateTime)

    oauth_account = SimpleNamespace(
        access_token="access-token",
        refresh_token=None,
        expires_at=FixedDateTime(2026, 1, 1, 9, 0),
    )

    assert mcp_api._oauth_account_can_connect(oauth_account) is True


def test_hidden_public_mcp_app_is_excluded_from_remote_connector_list() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        _create_public_app(admin_headers, "visible-app", "Visible App", True)
        _create_public_app(admin_headers, "hidden-app", "Hidden App", False)

        response = client.get("/api/mcp/apps?location=remote", headers=regular_headers)
        assert response.status_code == 200

        app_ids = {app["id"] for app in response.json()}
        assert "visible-app" in app_ids
        assert "hidden-app" not in app_ids
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_custom_stdio_mcp_with_same_name_does_not_mark_builtin_oauth_app_connected() -> (
    None
):
    temp_dir = _setup_test_db()
    try:
        _setup_admin()

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        _connect_custom_stdio_mcp_for_user("regular", "Teams")

        response = client.get(
            "/api/mcp/apps?location=remote&search=teams",
            headers=regular_headers,
        )
        assert response.status_code == 200

        teams_app = next(app for app in response.json() if app["id"] == "teams")
        assert teams_app["is_connected"] is False
        assert "server_id" not in teams_app
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_oauth_connection_does_not_reuse_same_name_custom_stdio_mcp() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        _connect_custom_stdio_mcp_for_user("regular", "Teams")

        db = next(get_db())
        try:
            user = db.query(User).filter(User.username == "regular").first()
            assert user is not None

            with pytest.raises(
                ValueError, match="conflicts with an existing MCP server"
            ) as exc:
                _ensure_user_mcp_server(
                    db,
                    str(user.id),
                    {
                        "id": "teams",
                        "name": "Teams",
                        "description": "Connect to Microsoft Teams.",
                        "provider": "microsoft",
                        "auth_type": "builtin_oauth",
                    },
                )
            # A genuine metadata conflict on a real OAuth app is a plain
            # ValueError, NOT AppNotOAuthError — so the batch loop's narrowed
            # except surfaces it instead of misreporting it as "non-oauth".
            assert not isinstance(exc.value, AppNotOAuthError)
        finally:
            db.close()
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_init_db_seeds_builtin_oauth_and_microsoft_graph_public_apps() -> None:
    temp_dir = _setup_test_db()
    db = next(get_db())
    try:
        provider_names = {row.provider_name for row in db.query(OAuthProvider).all()}
        assert {"google", "linkedin", "microsoft", "meta"}.issubset(provider_names)

        microsoft_provider = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == "microsoft")
            .first()
        )
        assert microsoft_provider is not None
        assert microsoft_provider.default_scopes == ["User.Read"]
        meta_provider = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == "meta")
            .first()
        )
        assert meta_provider is not None
        assert meta_provider.default_scopes == ["public_profile"]

        app_ids = {row.app_id for row in db.query(PublicMCPApp).all()}
        assert {
            "linkedin",
            "gmail",
            "google-drive",
            "google-calendar",
            "teams",
            "outlook",
            "onedrive",
            "facebook",
            "instagram",
        }.issubset(app_ids)

        teams_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "teams").first()
        )
        outlook_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "outlook").first()
        )
        onedrive_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "onedrive").first()
        )
        facebook_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "facebook").first()
        )
        instagram_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "instagram").first()
        )

        assert teams_app is not None
        assert teams_app.provider_name == "microsoft"
        assert teams_app.oauth_scopes == [
            "Team.ReadBasic.All",
            "Channel.ReadBasic.All",
            "TeamMember.Read.All",
            "ChannelMessage.Read.All",
            "ChannelMessage.Send",
            "Chat.ReadWrite",
        ]
        assert teams_app.launch_config == {
            "command": "uv",
            "args": ["run", "python", "-m", "xagent.web.tools.mcp.teams"],
            "env_mapping": {"AUTH_TOKEN": "access_token"},
        }

        assert outlook_app is not None
        assert outlook_app.provider_name == "microsoft"
        assert outlook_app.oauth_scopes == [
            "Mail.Read",
            "Mail.Send",
            "Calendars.ReadWrite",
            "Contacts.Read",
        ]
        assert outlook_app.launch_config == {
            "command": "uv",
            "args": ["run", "python", "-m", "xagent.web.tools.mcp.outlook"],
            "env_mapping": {"AUTH_TOKEN": "access_token"},
        }

        assert onedrive_app is not None
        assert onedrive_app.provider_name == "microsoft"
        assert onedrive_app.oauth_scopes == ["Files.ReadWrite"]
        assert onedrive_app.launch_config == {
            "command": "uv",
            "args": ["run", "python", "-m", "xagent.web.tools.mcp.onedrive"],
            "env_mapping": {"AUTH_TOKEN": "access_token"},
        }

        assert facebook_app is not None
        assert facebook_app.provider_name == "meta"
        assert facebook_app.category == "Marketing"
        assert facebook_app.oauth_scopes == [
            "pages_show_list",
            "pages_read_engagement",
            "pages_manage_posts",
        ]
        assert facebook_app.launch_config == {
            "command": "uv",
            "args": ["run", "python", "-m", "xagent.web.tools.mcp.facebook"],
            "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
        }

        assert instagram_app is not None
        assert instagram_app.provider_name == "meta"
        assert instagram_app.category == "Marketing"
        assert instagram_app.oauth_scopes == [
            "pages_show_list",
            "pages_read_engagement",
            "instagram_basic",
            "instagram_content_publish",
        ]
        assert instagram_app.launch_config == {
            "command": "uv",
            "args": ["run", "python", "-m", "xagent.web.tools.mcp.instagram"],
            "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
        }
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_init_db_does_not_reseed_deleted_builtin_app_on_existing_database() -> None:
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    db_url = f"sqlite:///{temp_db_path}"

    init_db(db_url=db_url)

    db = next(get_db())
    try:
        gmail_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").first()
        )
        assert gmail_app is not None
        db.delete(gmail_app)
        db.commit()
    finally:
        db.close()

    init_db(db_url=db_url)

    db = next(get_db())
    try:
        recreated_gmail_app = (
            db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "gmail").first()
        )
        assert recreated_gmail_app is None
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_connected_hidden_public_mcp_app_is_excluded_in_strong_hide_mode() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        _create_public_app(admin_headers, "hidden-app", "Hidden App", False)
        _connect_app_for_user("regular", "Hidden App")

        response = client.get("/api/mcp/apps?location=remote", headers=regular_headers)
        assert response.status_code == 200

        app_ids = {app["id"] for app in response.json()}
        assert "hidden-app" not in app_ids
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_mixed_case_oauth_transport_app_is_marked_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the mixed-case transport fix: list_mcp_apps now routes on
    auth_type (which lowercases), so a "OAuth"-cased catalog entry is treated as
    builtin_oauth. Before, the exact-case `transport == "oauth"` branch stranded
    it in the non-oauth path, leaving is_connected always False."""
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        register_response = client.post(
            "/api/auth/register",
            json={
                "username": "regular",
                "email": "regular@example.com",
                "password": "password123",
            },
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        # Admin API accepts arbitrary transport casing.
        resp = client.post(
            "/api/admin/mcp/apps",
            headers=admin_headers,
            json={
                "app_id": "mixed-oauth",
                "name": "MixedOauth",
                "description": "",
                "icon": "",
                "transport": "OAuth",
                "provider_name": "microsoft",
                "category": "Communication",
                "oauth_scopes": [],
                "is_visible_in_connector": True,
                "launch_config": {},
            },
        )
        assert resp.status_code == 200

        _connect_oauth_account_for_user("regular", "microsoft")
        monkeypatch.setattr(mcp_api, "_oauth_account_can_connect", lambda _a: True)

        db = next(get_db())
        try:
            user = db.query(User).filter(User.username == "regular").first()
            assert user is not None
            server = MCPServer(
                name="MixedOauth",
                description="",
                managed="external",
                transport="oauth",
                auth={"app_id": "mixed-oauth"},
            )
            db.add(server)
            db.flush()
            db.add(
                UserMCPServer(
                    user_id=user.id,
                    mcpserver_id=server.id,
                    is_owner=True,
                    can_edit=True,
                    can_delete=True,
                    is_active=True,
                )
            )
            db.commit()
        finally:
            db.close()

        response = client.get("/api/mcp/apps?location=remote", headers=regular_headers)
        assert response.status_code == 200
        app = next(a for a in response.json() if a["id"] == "mixed-oauth")
        assert app["is_connected"] is True
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_create_app_rejects_keyless_command_entry() -> None:
    """Write-time constraint (#764): a non-oauth entry with a launch command but
    no required_env would classify as "unconnectable", so the admin API rejects
    it instead of silently persisting an unconnectable row."""
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        # Shape 1: command without required_env.
        resp = client.post(
            "/api/admin/mcp/apps",
            headers=admin_headers,
            json={
                "app_id": "bad-keyless",
                "name": "BadKeyless",
                "transport": "stdio",
                "launch_config": {"command": "npx", "args": ["-y", "x"]},
            },
        )
        assert resp.status_code == 422

        # Shape 2 (the reverse asymmetric shape): required_env without command.
        resp = client.post(
            "/api/admin/mcp/apps",
            headers=admin_headers,
            json={
                "app_id": "bad-nocommand",
                "name": "BadNoCommand",
                "transport": "stdio",
                "launch_config": {"required_env": ["KEY"]},
            },
        )
        assert resp.status_code == 422
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_update_app_enforces_auth_classification() -> None:
    """The write-time constraint fires on PUT too, not just POST (both use the
    same PublicMCPAppCreate model)."""
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        created = client.post(
            "/api/admin/mcp/apps",
            headers=admin_headers,
            json={
                "app_id": "good-keyed",
                "name": "GoodKeyed",
                "transport": "stdio",
                "launch_config": {"command": "npx", "required_env": ["KEY"]},
            },
        )
        assert created.status_code == 200
        app_pk = created.json()["id"]

        updated = client.put(
            f"/api/admin/mcp/apps/{app_pk}",
            headers=admin_headers,
            json={
                "app_id": "good-keyed",
                "name": "GoodKeyed",
                "transport": "stdio",
                "launch_config": {"command": "npx"},
            },
        )
        assert updated.status_code == 422
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_admin_list_apps_does_not_500_on_partial_launch_config_row() -> None:
    """The write-time validator lives on the create model only, so listing must
    not re-validate on response serialization. A legacy/direct-DB row with a
    partial launch_config (classifies "unconnectable") must be returned, not turn
    the whole admin list into a 500."""
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        db = next(get_db())
        try:
            db.add(
                PublicMCPApp(
                    app_id="legacy-bad",
                    name="LegacyBad",
                    transport="stdio",
                    launch_config={"command": "npx"},
                )
            )
            db.commit()
        finally:
            db.close()

        resp = client.get("/api/admin/mcp/apps", headers=admin_headers)
        assert resp.status_code == 200
        assert any(a["app_id"] == "legacy-bad" for a in resp.json())
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass
