import httpx
import pytest

from xagent.web.services.mcp_oauth import (
    MCPOAuthDiscoveryError,
    authorization_server_metadata_urls,
    discover_mcp_oauth_metadata,
    parse_www_authenticate_bearer,
    protected_resource_metadata_urls,
)


def test_parse_www_authenticate_bearer_challenge():
    challenge = parse_www_authenticate_bearer(
        'Basic realm="ignored", Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="records.read records.write"'
    )

    assert challenge is not None
    assert (
        challenge.resource_metadata_url
        == "https://mcp.example.com/.well-known/oauth-protected-resource"
    )
    assert challenge.scope == "records.read records.write"


def test_protected_resource_metadata_urls_use_endpoint_path_before_root():
    assert protected_resource_metadata_urls("https://mcp.example.com/public/mcp") == (
        "https://mcp.example.com/.well-known/oauth-protected-resource/public/mcp",
        "https://mcp.example.com/.well-known/oauth-protected-resource",
    )


def test_authorization_server_metadata_urls_for_path_issuer():
    assert authorization_server_metadata_urls("https://auth.example.com/org1") == (
        "https://auth.example.com/.well-known/oauth-authorization-server/org1",
        "https://auth.example.com/.well-known/openid-configuration/org1",
        "https://auth.example.com/org1/.well-known/openid-configuration",
    )


@pytest.mark.asyncio
async def test_discover_uses_challenge_resource_metadata_and_scope():
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(
                401,
                headers={
                    "WWW-Authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="records.read"'
                },
            )
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource"
        ):
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example.com/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                    "scopes_supported": ["records.read", "records.write"],
                },
            )
        if (
            str(request.url)
            == "https://auth.example.com/.well-known/oauth-authorization-server"
        ):
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example.com",
                    "authorization_endpoint": "https://auth.example.com/authorize",
                    "token_endpoint": "https://auth.example.com/token",
                    "client_id_metadata_document_supported": True,
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await discover_mcp_oauth_metadata(
            "https://mcp.example.com/mcp", client=client
        )

    assert result.resource == "https://mcp.example.com/mcp"
    assert result.scopes == ("records.read",)
    assert result.authorization_server.issuer == "https://auth.example.com"
    assert result.authorization_server.client_id_metadata_document_supported is True
    assert requested_urls[:3] == [
        "https://mcp.example.com/mcp",
        "https://mcp.example.com/.well-known/oauth-protected-resource",
        "https://auth.example.com/.well-known/oauth-authorization-server",
    ]


@pytest.mark.asyncio
async def test_discover_falls_back_to_well_known_resource_metadata():
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if str(request.url) == "https://mcp.example.com/public/mcp":
            return httpx.Response(401)
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource/public/mcp"
        ):
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example.com/public/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                    "scopes_supported": ["records.read"],
                },
            )
        if (
            str(request.url)
            == "https://auth.example.com/.well-known/oauth-authorization-server"
        ):
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example.com",
                    "authorization_endpoint": "https://auth.example.com/authorize",
                    "token_endpoint": "https://auth.example.com/token",
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await discover_mcp_oauth_metadata(
            "https://mcp.example.com/public/mcp", client=client
        )

    assert result.scopes == ("records.read",)
    assert (
        "https://mcp.example.com/.well-known/oauth-protected-resource/public/mcp"
        in requested_urls
    )


@pytest.mark.asyncio
async def test_discover_rejects_configured_resource_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(401)
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
        ):
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example.com/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await discover_mcp_oauth_metadata(
                "https://mcp.example.com/mcp",
                configured_resource="https://other.example.com/mcp",
                client=client,
            )

    assert exc.value.code == "resource_mismatch"


@pytest.mark.asyncio
async def test_discover_rejects_configured_issuer_not_advertised():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(401)
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
        ):
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example.com/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await discover_mcp_oauth_metadata(
                "https://mcp.example.com/mcp",
                configured_issuer="https://login.example.com",
                client=client,
            )

    assert exc.value.code == "issuer_mismatch"


@pytest.mark.asyncio
async def test_discover_accepts_case_variant_configured_resource_and_issuer():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(401)
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
        ):
            return httpx.Response(
                200,
                json={
                    "resource": "https://MCP.EXAMPLE.com/mcp/",
                    "authorization_servers": ["https://AUTH.EXAMPLE.com/"],
                },
            )
        if (
            str(request.url)
            == "https://auth.example.com/.well-known/oauth-authorization-server"
        ):
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example.com",
                    "authorization_endpoint": "https://auth.example.com/authorize",
                    "token_endpoint": "https://auth.example.com/token",
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await discover_mcp_oauth_metadata(
            "https://mcp.example.com/mcp",
            configured_resource="https://mcp.example.com/mcp",
            configured_issuer="https://auth.example.com",
            client=client,
        )

    assert result.resource == "https://MCP.EXAMPLE.com/mcp/"
    assert result.authorization_server.issuer == "https://auth.example.com"


@pytest.mark.asyncio
async def test_discover_rejects_missing_authorization_servers():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(401)
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
        ):
            return httpx.Response(
                200,
                json={"resource": "https://mcp.example.com/mcp"},
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await discover_mcp_oauth_metadata(
                "https://mcp.example.com/mcp", client=client
            )

    assert exc.value.code == "authorization_server_not_found"


@pytest.mark.asyncio
async def test_discover_surfaces_probe_network_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await discover_mcp_oauth_metadata(
                "https://mcp.example.com/mcp", client=client
            )

    assert exc.value.code == "metadata_not_found"
