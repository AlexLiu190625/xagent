from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.tools.config import (
    ResolvedToken,
    TokenRequest,
    WebToolConfig,
    set_oauth_token_resolver_hook,
)


@pytest.fixture(autouse=True)
def clear_oauth_token_resolver_hook():
    set_oauth_token_resolver_hook(None)
    yield
    set_oauth_token_resolver_hook(None)


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "oauth-token-resolver.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _launch_config(
    env_key: str = "GOOGLE_ACCESS_TOKEN", resource: str | None = None
) -> dict:
    launch_config = {
        "command": "npx",
        "args": ["-y", "@mcp-servers/google-drive"],
        "env_mapping": {env_key: "access_token", "IGNORED": "refresh_token"},
    }
    if resource is not None:
        launch_config["resource"] = resource
    return launch_config


def _add_oauth_server(
    db,
    user: User,
    *,
    name: str = "Google Drive",
    app_id: str | None = "google-drive",
    provider: str | None = "google",
    launch_config: object | None = None,
    register_app: bool = True,
) -> MCPServer:
    server = MCPServer(
        name=name,
        description=f"{name} server",
        managed="external",
        transport="oauth",
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    if register_app:
        db.add(
            PublicMCPApp(
                app_id=app_id or f"{name.lower()}-app",
                name=name,
                description=f"{name} app",
                transport="oauth",
                provider_name=provider,
                launch_config=launch_config if launch_config is not None else {},
            )
        )
    db.commit()
    return server


def _add_stdio_server(db, user: User, *, name: str) -> MCPServer:
    server = MCPServer(
        name=name,
        description=f"{name} server",
        managed="external",
        transport="stdio",
        command="echo",
        args=["ok"],
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()
    return server


def _add_user_oauth(
    db,
    user: User,
    *,
    provider: str = "google",
    access_token: str = "user-token",
) -> UserOAuth:
    account = UserOAuth(
        user_id=user.id,
        provider=provider,
        access_token=access_token,
        provider_user_id=f"{provider}-user",
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


def _tool_config(db, user: User, **kwargs) -> WebToolConfig:
    return WebToolConfig(
        db=db,
        request=None,
        user=user,
        user_id=user.id,
        workspace_config={"base_dir": "/tmp", "task_id": "task-1"},
        **kwargs,
    )


def _access_token_env(config: dict, key: str = "GOOGLE_ACCESS_TOKEN") -> str:
    return config["config"]["env"][key]


def _assert_same_oauth_config_except_token(
    hook_config: dict,
    user_config: dict,
    *,
    token_key: str,
    hook_token: str,
    user_token: str,
) -> None:
    hook_env = dict(hook_config["config"]["env"])
    user_env = dict(user_config["config"]["env"])
    assert hook_env.pop(token_key) == hook_token
    assert user_env.pop(token_key) == user_token
    assert hook_config["transport"] == user_config["transport"] == "stdio"
    assert hook_config["config"]["transport"] == user_config["config"]["transport"]
    assert hook_config["config"]["command"] == user_config["config"]["command"]
    assert hook_config["config"]["args"] == user_config["config"]["args"]
    assert hook_env == user_env


@pytest.mark.asyncio
async def test_hook_receives_provider_candidates_in_order_and_first_hit_wins(
    db_session,
):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    seen: list[str] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen.append(request.provider)
        if request.provider == "google-drive":
            return ResolvedToken(
                access_token="hook-token",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
        return None

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert seen == ["google", "google-drive"]
    assert len(configs) == 1
    assert configs[0]["transport"] == "stdio"
    assert _access_token_env(configs[0]) == "hook-token"


@pytest.mark.asyncio
async def test_hook_accepts_sync_resolver(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    seen: list[str] = []

    def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen.append(request.provider)
        return ResolvedToken(
            access_token="sync-hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert seen == ["google"]
    assert configs[0]["transport"] == "stdio"
    assert _access_token_env(configs[0]) == "sync-hook-token"


@pytest.mark.asyncio
async def test_hook_dedupes_provider_candidates(db_session):
    db, user = db_session
    _add_oauth_server(
        db,
        user,
        app_id="google",
        provider="google",
        launch_config=_launch_config(),
    )
    seen: list[str] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen.append(request.provider)
        return None

    set_oauth_token_resolver_hook(resolver)
    await _tool_config(db, user).get_mcp_server_configs()

    assert seen == ["google"]


@pytest.mark.asyncio
async def test_hook_request_receives_provider_resource_and_scope_verbatim(db_session):
    db, user = db_session
    scope = object()
    resource = "https://MCP.EXAMPLE.com:443/mcp/%7Euser/?Q=1#Fragment"
    _add_oauth_server(db, user, launch_config=_launch_config(resource=resource))
    seen: list[tuple[str, str | None, object | None]] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen.append((request.provider, request.resource, request.scope))
        if request.provider == "google-drive":
            return ResolvedToken(
                access_token="hook-token",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
        return None

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(
        db, user, execution_scope=scope
    ).get_mcp_server_configs()

    assert seen == [
        ("google", resource, scope),
        ("google-drive", resource, scope),
    ]
    assert _access_token_env(configs[0]) == "hook-token"


@pytest.mark.asyncio
async def test_hook_decline_falls_back_to_user_oauth(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    _add_user_oauth(db, user, provider="google", access_token="user-token")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return None

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert configs[0]["transport"] == "stdio"
    assert _access_token_env(configs[0]) == "user-token"


@pytest.mark.asyncio
async def test_hook_supply_fallback_npx_env_matches_user_oauth_shape(db_session):
    db, user = db_session
    _add_oauth_server(db, user, name="Legacy App", app_id="legacy-app")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    hook_config = (await _tool_config(db, user).get_mcp_server_configs())[0]
    set_oauth_token_resolver_hook(None)
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    user_config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    _assert_same_oauth_config_except_token(
        hook_config,
        user_config,
        token_key="LEGACY_APP_ACCESS_TOKEN",
        hook_token="hook-token",
        user_token="user-token",
    )


@pytest.mark.asyncio
async def test_hook_supply_launch_config_env_matches_user_oauth_shape(db_session):
    db, user = db_session
    _add_oauth_server(
        db,
        user,
        launch_config=_launch_config(env_key="CUSTOM_ACCESS_TOKEN"),
    )

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    hook_config = (await _tool_config(db, user).get_mcp_server_configs())[0]
    set_oauth_token_resolver_hook(None)
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    user_config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    _assert_same_oauth_config_except_token(
        hook_config,
        user_config,
        token_key="CUSTOM_ACCESS_TOKEN",
        hook_token="hook-token",
        user_token="user-token",
    )


@pytest.mark.asyncio
async def test_launch_config_args_none_matches_user_oauth_shape(db_session):
    db, user = db_session
    launch_config = _launch_config()
    launch_config["args"] = None
    _add_oauth_server(db, user, launch_config=launch_config)

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    hook_config = (await _tool_config(db, user).get_mcp_server_configs())[0]
    set_oauth_token_resolver_hook(None)
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    user_config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    assert hook_config["config"]["args"] == []
    _assert_same_oauth_config_except_token(
        hook_config,
        user_config,
        token_key="GOOGLE_ACCESS_TOKEN",
        hook_token="hook-token",
        user_token="user-token",
    )


@pytest.mark.asyncio
async def test_launch_config_args_string_matches_user_oauth_shape(db_session):
    db, user = db_session
    launch_config = _launch_config()
    launch_config["args"] = '--flag "two words"'
    _add_oauth_server(db, user, launch_config=launch_config)

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    hook_config = (await _tool_config(db, user).get_mcp_server_configs())[0]
    set_oauth_token_resolver_hook(None)
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    user_config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    assert hook_config["config"]["args"] == ["--flag", "two words"]
    _assert_same_oauth_config_except_token(
        hook_config,
        user_config,
        token_key="GOOGLE_ACCESS_TOKEN",
        hook_token="hook-token",
        user_token="user-token",
    )


@pytest.mark.asyncio
async def test_launch_config_args_unsupported_type_warning_matches_contract(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    launch_config = _launch_config()
    launch_config["args"] = {"not": "supported"}
    _add_oauth_server(db, user, launch_config=launch_config)

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    assert config["config"]["args"] == []
    assert (
        "Ignoring OAuth MCP launch config args because args must be a list or a string"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_launch_config_env_mapping_none_matches_user_oauth_shape(db_session):
    db, user = db_session
    launch_config = _launch_config()
    launch_config["env_mapping"] = None
    _add_oauth_server(db, user, launch_config=launch_config)

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    hook_config = (await _tool_config(db, user).get_mcp_server_configs())[0]
    set_oauth_token_resolver_hook(None)
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    user_config = (await _tool_config(db, user).get_mcp_server_configs())[0]

    assert "GOOGLE_ACCESS_TOKEN" not in hook_config["config"]["env"]
    assert hook_config == user_config


@pytest.mark.asyncio
async def test_launch_config_missing_command_skips_only_that_server_for_hook(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    _add_stdio_server(db, user, name="before")
    launch_config = _launch_config()
    launch_config.pop("command")
    _add_oauth_server(db, user, launch_config=launch_config)
    _add_stdio_server(db, user, name="after")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert [config["name"] for config in configs] == ["before", "after"]
    assert "launch_config.command is invalid" in caplog.text


@pytest.mark.asyncio
async def test_launch_config_missing_command_skips_only_that_server_for_user_oauth(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    _add_stdio_server(db, user, name="before")
    launch_config = _launch_config()
    launch_config.pop("command")
    _add_oauth_server(db, user, launch_config=launch_config)
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    _add_stdio_server(db, user, name="after")

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert [config["name"] for config in configs] == ["before", "after"]
    assert "launch_config.command is invalid" in caplog.text


@pytest.mark.asyncio
async def test_launch_config_non_mapping_skips_only_that_server_for_hook(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    _add_stdio_server(db, user, name="before")
    _add_oauth_server(db, user, launch_config="not-a-mapping")
    _add_stdio_server(db, user, name="after")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert [config["name"] for config in configs] == ["before", "after"]
    assert "launch_config.type is invalid" in caplog.text


@pytest.mark.asyncio
async def test_launch_config_non_mapping_skips_only_that_server_for_user_oauth(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    _add_stdio_server(db, user, name="before")
    _add_oauth_server(db, user, launch_config=["not", "a", "mapping"])
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    _add_stdio_server(db, user, name="after")

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert [config["name"] for config in configs] == ["before", "after"]
    assert "launch_config.type is invalid" in caplog.text


@pytest.mark.asyncio
async def test_hook_is_not_asked_without_app_info(db_session):
    db, user = db_session
    _add_oauth_server(db, user, name="Unregistered OAuth", register_app=False)
    seen: list[str] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen.append(request.provider)
        raise AssertionError("resolver should not be called")

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert seen == []
    assert len(configs) == 1
    assert configs[0]["name"] == "Unregistered OAuth"
    assert configs[0]["transport"] == "oauth"


@pytest.mark.asyncio
async def test_hook_failure_does_not_fallback_and_later_servers_still_build(
    db_session,
    caplog,
):
    db, user = db_session
    caplog.set_level(logging.WARNING)
    _add_stdio_server(db, user, name="before")
    oauth_server = _add_oauth_server(db, user, launch_config=_launch_config())
    _add_user_oauth(db, user, provider="google", access_token="user-token")
    _add_stdio_server(db, user, name="after")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        raise RuntimeError("secret-token-in-exception")

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert [config["name"] for config in configs] == [
        "before",
        "Google Drive",
        "after",
    ]
    unavailable = configs[1]
    assert unavailable["transport"] == "unavailable"
    assert unavailable["config"]["reason"] == "oauth_token_resolver_failed"
    assert unavailable["config"]["server_id"] == oauth_server.id
    assert "secret-token-in-exception" not in str(unavailable)
    assert "secret-token-in-exception" not in caplog.text
    assert "runtime_input_schema" not in unavailable
    assert "runtime_bindings" not in unavailable
    assert "allow_delegated_authorization" not in unavailable
    assert "connector_runtime" not in unavailable
    assert cfg.get_mcp_oauth_diagnostics() == [
        {
            "code": "oauth_token_resolver_failed",
            "message": "OAuth token resolver failed",
            "server_id": oauth_server.id,
            "server_name": "Google Drive",
            "resource_owner_key": None,
            "resource": None,
            "scope": "",
            "issuer": None,
            "providers": ["google", "google-drive"],
            "exception_type": "RuntimeError",
        }
    ]


@pytest.mark.asyncio
async def test_hook_connector_runtime_error_propagates(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    _add_user_oauth(db, user, provider="google", access_token="user-token")

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        raise ConnectorRuntimeError(
            "connector_runtime_unavailable",
            "Connector runtime context is unavailable.",
            status_code=503,
        )

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    with pytest.raises(ConnectorRuntimeError):
        await cfg.get_mcp_server_configs()

    assert cfg.get_mcp_oauth_diagnostics() == []


@pytest.mark.asyncio
async def test_hook_failure_diagnostic_includes_bounded_resource(db_session):
    db, user = db_session
    resource = "https://mcp.example.com/" + "r" * 160
    _add_oauth_server(db, user, launch_config=_launch_config(resource=resource))

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        raise RuntimeError("resolver failed")

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    diagnostic = cfg.get_mcp_oauth_diagnostics()[0]
    assert diagnostic["resource"] == f"{resource[:125]}..."
    assert len(diagnostic["resource"]) == 128


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolved", "exception_type"),
    [
        (object(), "object"),
        (ResolvedToken(access_token="", expires_at=None), "InvalidAccessToken"),
        (ResolvedToken(access_token=123, expires_at=None), "InvalidAccessToken"),
        (
            ResolvedToken(access_token="hook-token", expires_at="soon"),
            "InvalidExpiresAt",
        ),
    ],
)
async def test_hook_malformed_token_creates_unavailable_config(
    db_session,
    resolved,
    exception_type,
):
    db, user = db_session
    server = _add_oauth_server(db, user, launch_config=_launch_config())

    async def resolver(request: TokenRequest):
        return resolved

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["server_id"] == server.id
    assert cfg.get_mcp_oauth_diagnostics()[0]["exception_type"] == exception_type


@pytest.mark.asyncio
async def test_hook_is_skipped_when_user_id_is_none(db_session):
    db, user = db_session
    seen: list[TokenRequest] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen.append(request)
        return ResolvedToken(access_token="hook-token", expires_at=None)

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    cfg._user_id = None
    resolved = await cfg._resolve_oauth_token_from_hook(
        providers=["google"],
        resource=None,
    )

    assert resolved is None
    assert seen == []


@pytest.mark.asyncio
async def test_invalid_launch_config_with_uncacheable_hook_token_is_not_cached(
    db_session,
):
    db, user = db_session
    _add_oauth_server(db, user, launch_config="not-a-mapping")
    calls = 0

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        nonlocal calls
        calls += 1
        return ResolvedToken(access_token=f"hook-token-{calls}", expires_at=None)

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    first = await cfg.get_mcp_server_configs()
    app = db.query(PublicMCPApp).filter(PublicMCPApp.name == "Google Drive").one()
    app.launch_config = _launch_config()
    db.commit()
    second = await cfg.get_mcp_server_configs()

    assert first == []
    assert calls == 2
    assert _access_token_env(second[0]) == "hook-token-2"


@pytest.mark.asyncio
async def test_hook_expires_at_none_is_used_but_not_cached(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    calls = 0

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        nonlocal calls
        calls += 1
        return ResolvedToken(access_token=f"hook-token-{calls}", expires_at=None)

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    first = await cfg.get_mcp_server_configs()
    second = await cfg.get_mcp_server_configs()

    assert calls == 2
    assert _access_token_env(first[0]) == "hook-token-1"
    assert _access_token_env(second[0]) == "hook-token-2"


@pytest.mark.asyncio
async def test_hook_future_expiry_is_cached_until_generation_changes(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    calls = 0

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        nonlocal calls
        calls += 1
        return ResolvedToken(
            access_token=f"hook-token-{calls}",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    first = await cfg.get_mcp_server_configs()
    second = await cfg.get_mcp_server_configs()
    set_oauth_token_resolver_hook(resolver)
    third = await cfg.get_mcp_server_configs()

    assert calls == 2
    assert _access_token_env(first[0]) == "hook-token-1"
    assert _access_token_env(second[0]) == "hook-token-1"
    assert _access_token_env(third[0]) == "hook-token-2"


@pytest.mark.asyncio
async def test_hook_naive_expiry_is_interpreted_as_utc(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc).replace(tzinfo=None)
            + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(db, user).get_mcp_server_configs()

    assert _access_token_env(configs[0]) == "hook-token"


@pytest.mark.asyncio
async def test_hook_near_expiry_token_is_used_but_not_cached(db_session):
    db, user = db_session
    _add_oauth_server(db, user, launch_config=_launch_config())
    calls = 0

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        nonlocal calls
        calls += 1
        return ResolvedToken(
            access_token=f"hook-token-{calls}",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        )

    set_oauth_token_resolver_hook(resolver)
    cfg = _tool_config(db, user)

    first = await cfg.get_mcp_server_configs()
    second = await cfg.get_mcp_server_configs()

    assert calls == 2
    assert _access_token_env(first[0]) == "hook-token-1"
    assert _access_token_env(second[0]) == "hook-token-2"


@pytest.mark.asyncio
async def test_hook_expired_token_creates_unavailable_config(db_session):
    db, user = db_session
    server = _add_oauth_server(db, user, launch_config=_launch_config())

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )

    set_oauth_token_resolver_hook(resolver)

    cfg = _tool_config(db, user)
    configs = await cfg.get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["server_id"] == server.id
    assert cfg.get_mcp_oauth_diagnostics()[0]["exception_type"] == (
        "ExpiredAccessToken"
    )


@pytest.mark.asyncio
async def test_hook_request_receives_execution_scope(db_session):
    db, user = db_session
    scope = object()
    _add_oauth_server(db, user, launch_config=_launch_config())
    seen_scope: list[object] = []

    async def resolver(request: TokenRequest) -> ResolvedToken | None:
        seen_scope.append(request.scope)
        return ResolvedToken(
            access_token="hook-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    set_oauth_token_resolver_hook(resolver)

    configs = await _tool_config(
        db, user, execution_scope=scope
    ).get_mcp_server_configs()

    assert seen_scope == [scope]
    assert _access_token_env(configs[0]) == "hook-token"
