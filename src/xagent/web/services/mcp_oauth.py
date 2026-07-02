from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit
from urllib.request import parse_http_list, parse_keqv_list

import httpx


DEFAULT_MCP_OAUTH_DISCOVERY_TIMEOUT = 10.0


class MCPOAuthDiscoveryError(RuntimeError):
    """Raised when MCP OAuth discovery cannot produce usable metadata."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class MCPAuthorizationChallenge:
    """Bearer challenge advertised by a protected MCP resource."""

    resource_metadata_url: str | None
    scope: str | None
    params: dict[str, str]


@dataclass(frozen=True)
class MCPProtectedResourceMetadata:
    """OAuth Protected Resource Metadata for an MCP endpoint."""

    url: str
    resource: str | None
    authorization_servers: tuple[str, ...]
    scopes_supported: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class OAuthAuthorizationServerMetadata:
    """OAuth/OIDC authorization server metadata needed for MCP OAuth."""

    url: str
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None
    client_id_metadata_document_supported: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class MCPOAuthDiscoveryResult:
    """Complete metadata selected for an MCP OAuth authorization flow."""

    challenge: MCPAuthorizationChallenge | None
    protected_resource: MCPProtectedResourceMetadata
    authorization_server: OAuthAuthorizationServerMetadata
    resource: str
    scopes: tuple[str, ...]


def parse_www_authenticate_bearer(
    headers: str | Sequence[str] | None,
) -> MCPAuthorizationChallenge | None:
    """Parse the first Bearer challenge from one or more WWW-Authenticate headers."""
    for header_value in _iter_header_values(headers):
        challenge = _parse_bearer_challenge(header_value)
        if challenge is not None:
            return challenge
    return None


def protected_resource_metadata_urls(endpoint_url: str) -> tuple[str, ...]:
    """Return MCP protected-resource metadata candidates in spec priority order."""
    parts = urlsplit(endpoint_url)
    if not parts.scheme or not parts.netloc:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "MCP endpoint URL must be an absolute HTTP(S) URL",
        )

    root_metadata = urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            "/.well-known/oauth-protected-resource",
            "",
            "",
        )
    )
    path = parts.path.rstrip("/")
    if not path:
        return (root_metadata,)

    path_metadata = urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            f"/.well-known/oauth-protected-resource{path}",
            "",
            "",
        )
    )
    return _dedupe_urls((path_metadata, root_metadata))


def authorization_server_metadata_urls(issuer_url: str) -> tuple[str, ...]:
    """Return OAuth/OIDC metadata candidates for an authorization server issuer."""
    parts = urlsplit(issuer_url)
    if not parts.scheme or not parts.netloc:
        raise MCPOAuthDiscoveryError(
            "authorization_server_not_found",
            "Authorization server URL must be absolute",
        )

    base = (parts.scheme.lower(), parts.netloc.lower())
    issuer_path = parts.path.rstrip("/")
    if issuer_path:
        return _dedupe_urls(
            (
                urlunsplit(
                    (
                        *base,
                        f"/.well-known/oauth-authorization-server{issuer_path}",
                        "",
                        "",
                    )
                ),
                urlunsplit(
                    (
                        *base,
                        f"/.well-known/openid-configuration{issuer_path}",
                        "",
                        "",
                    )
                ),
                urlunsplit(
                    (
                        *base,
                        f"{issuer_path}/.well-known/openid-configuration",
                        "",
                        "",
                    )
                ),
            )
        )

    return _dedupe_urls(
        (
            urlunsplit((*base, "/.well-known/oauth-authorization-server", "", "")),
            urlunsplit((*base, "/.well-known/openid-configuration", "", "")),
        )
    )


async def discover_mcp_oauth_metadata(  # noqa: PLR0913
    endpoint_url: str,
    *,
    headers: dict[str, str] | None = None,
    configured_resource_metadata_url: str | None = None,
    configured_issuer: str | None = None,
    configured_resource: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> MCPOAuthDiscoveryResult:
    """Discover MCP protected-resource and OAuth authorization-server metadata."""
    if client is not None:
        return await _discover_with_client(
            endpoint_url,
            headers=headers,
            configured_resource_metadata_url=configured_resource_metadata_url,
            configured_issuer=configured_issuer,
            configured_resource=configured_resource,
            client=client,
        )

    async with httpx.AsyncClient(
        timeout=DEFAULT_MCP_OAUTH_DISCOVERY_TIMEOUT,
        follow_redirects=True,
    ) as owned_client:
        return await _discover_with_client(
            endpoint_url,
            headers=headers,
            configured_resource_metadata_url=configured_resource_metadata_url,
            configured_issuer=configured_issuer,
            configured_resource=configured_resource,
            client=owned_client,
        )


async def _discover_with_client(  # noqa: PLR0913
    endpoint_url: str,
    *,
    headers: dict[str, str] | None,
    configured_resource_metadata_url: str | None,
    configured_issuer: str | None,
    configured_resource: str | None,
    client: httpx.AsyncClient,
) -> MCPOAuthDiscoveryResult:
    challenge: MCPAuthorizationChallenge | None = None
    metadata_urls: tuple[str, ...]

    if configured_resource_metadata_url:
        metadata_urls = (configured_resource_metadata_url,)
    else:
        challenge = await _probe_authorization_challenge(
            endpoint_url, headers=headers, client=client
        )
        if challenge and challenge.resource_metadata_url:
            metadata_urls = (challenge.resource_metadata_url,)
        else:
            metadata_urls = protected_resource_metadata_urls(endpoint_url)

    protected_resource = await _fetch_first_protected_resource_metadata(
        metadata_urls, client=client
    )
    resource = protected_resource.resource or _canonical_resource(endpoint_url)
    if configured_resource and not _same_url(configured_resource, resource):
        raise MCPOAuthDiscoveryError(
            "resource_mismatch",
            "Configured MCP OAuth resource does not match protected resource metadata",
        )

    authorization_server_url = _select_authorization_server(
        protected_resource.authorization_servers,
        configured_issuer=configured_issuer,
    )
    authorization_server = await _fetch_authorization_server_metadata(
        authorization_server_url,
        configured_issuer=configured_issuer,
        client=client,
    )
    scopes = _select_scopes(challenge, protected_resource)

    return MCPOAuthDiscoveryResult(
        challenge=challenge,
        protected_resource=protected_resource,
        authorization_server=authorization_server,
        resource=resource,
        scopes=scopes,
    )


async def _probe_authorization_challenge(
    endpoint_url: str,
    *,
    headers: dict[str, str] | None,
    client: httpx.AsyncClient,
) -> MCPAuthorizationChallenge | None:
    try:
        response = await client.get(endpoint_url, headers=headers)
    except httpx.HTTPError as exc:
        raise MCPOAuthDiscoveryError(
            "metadata_not_found",
            f"Failed to probe MCP endpoint for OAuth challenge: {exc}",
        ) from exc

    return parse_www_authenticate_bearer(response.headers.get_list("WWW-Authenticate"))


async def _fetch_first_protected_resource_metadata(
    metadata_urls: Sequence[str], *, client: httpx.AsyncClient
) -> MCPProtectedResourceMetadata:
    last_error: Exception | None = None
    for metadata_url in metadata_urls:
        try:
            response = await client.get(metadata_url)
            if response.status_code >= 400:
                last_error = MCPOAuthDiscoveryError(
                    "metadata_not_found",
                    f"Protected resource metadata returned HTTP {response.status_code}",
                )
                continue
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("metadata response is not a JSON object")
            return _parse_protected_resource_metadata(metadata_url, payload)
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc

    raise MCPOAuthDiscoveryError(
        "metadata_not_found",
        f"Could not load MCP protected resource metadata: {last_error}",
    )


async def _fetch_authorization_server_metadata(
    authorization_server_url: str,
    *,
    configured_issuer: str | None,
    client: httpx.AsyncClient,
) -> OAuthAuthorizationServerMetadata:
    last_error: Exception | None = None
    for metadata_url in authorization_server_metadata_urls(authorization_server_url):
        try:
            response = await client.get(metadata_url)
            if response.status_code >= 400:
                last_error = MCPOAuthDiscoveryError(
                    "authorization_server_not_found",
                    f"Authorization server metadata returned HTTP {response.status_code}",
                )
                continue
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("metadata response is not a JSON object")
            metadata = _parse_authorization_server_metadata(metadata_url, payload)
            _validate_issuer(
                selected_authorization_server=authorization_server_url,
                metadata=metadata,
                configured_issuer=configured_issuer,
            )
            return metadata
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc

    raise MCPOAuthDiscoveryError(
        "authorization_server_not_found",
        f"Could not load OAuth authorization server metadata: {last_error}",
    )


def _parse_protected_resource_metadata(
    metadata_url: str, payload: dict[str, Any]
) -> MCPProtectedResourceMetadata:
    authorization_servers = _string_tuple(payload.get("authorization_servers"))
    if not authorization_servers:
        raise MCPOAuthDiscoveryError(
            "authorization_server_not_found",
            "Protected resource metadata did not include authorization_servers",
        )
    return MCPProtectedResourceMetadata(
        url=metadata_url,
        resource=_optional_string(payload.get("resource")),
        authorization_servers=authorization_servers,
        scopes_supported=_string_tuple(payload.get("scopes_supported")),
        raw=payload,
    )


def _parse_authorization_server_metadata(
    metadata_url: str, payload: dict[str, Any]
) -> OAuthAuthorizationServerMetadata:
    issuer = _required_string(payload, "issuer")
    authorization_endpoint = _required_string(payload, "authorization_endpoint")
    token_endpoint = _required_string(payload, "token_endpoint")
    return OAuthAuthorizationServerMetadata(
        url=metadata_url,
        issuer=issuer,
        authorization_endpoint=authorization_endpoint,
        token_endpoint=token_endpoint,
        registration_endpoint=_optional_string(payload.get("registration_endpoint")),
        client_id_metadata_document_supported=bool(
            payload.get("client_id_metadata_document_supported")
        ),
        raw=payload,
    )


def _parse_bearer_challenge(header_value: str) -> MCPAuthorizationChallenge | None:
    match = re.search(r"(?i)(?:^|,\s*)Bearer(?:\s+|$)", header_value)
    if not match:
        return None

    params_text = header_value[match.end() :].strip()
    params = {
        str(key).lower(): str(value)
        for key, value in parse_keqv_list(parse_http_list(params_text)).items()
        if value is not None
    }
    return MCPAuthorizationChallenge(
        resource_metadata_url=params.get("resource_metadata"),
        scope=params.get("scope"),
        params=params,
    )


def _select_authorization_server(
    authorization_servers: Sequence[str], *, configured_issuer: str | None
) -> str:
    if not authorization_servers:
        raise MCPOAuthDiscoveryError(
            "authorization_server_not_found",
            "Protected resource metadata did not include authorization servers",
        )
    if configured_issuer:
        for authorization_server in authorization_servers:
            if _same_url(authorization_server, configured_issuer):
                return authorization_server
        raise MCPOAuthDiscoveryError(
            "issuer_mismatch",
            "Configured issuer is not advertised by protected resource metadata",
        )
    return authorization_servers[0]


def _validate_issuer(
    *,
    selected_authorization_server: str,
    metadata: OAuthAuthorizationServerMetadata,
    configured_issuer: str | None,
) -> None:
    expected = configured_issuer or selected_authorization_server
    if not _same_url(metadata.issuer, expected):
        raise MCPOAuthDiscoveryError(
            "issuer_mismatch",
            "Authorization server metadata issuer did not match the selected issuer",
        )


def _select_scopes(
    challenge: MCPAuthorizationChallenge | None,
    protected_resource: MCPProtectedResourceMetadata,
) -> tuple[str, ...]:
    if challenge and challenge.scope:
        return tuple(scope for scope in challenge.scope.split() if scope)
    return protected_resource.scopes_supported


def _canonical_resource(endpoint_url: str) -> str:
    parts = urlsplit(endpoint_url)
    path = parts.path
    if path == "/":
        path = ""
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise MCPOAuthDiscoveryError(
            "unsupported_auth_server",
            f"Authorization server metadata missing required field '{key}'",
        )
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _iter_header_values(headers: str | Sequence[str] | None) -> Iterable[str]:
    if headers is None:
        return ()
    if isinstance(headers, str):
        return (headers,)
    return tuple(str(header) for header in headers)


def _dedupe_urls(urls: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return tuple(deduped)


def _same_url(left: str, right: str) -> bool:
    return _url_comparison_key(left) == _url_comparison_key(right)


def _url_comparison_key(value: str) -> str:
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return value.rstrip("/")

    path = parts.path.rstrip("/")
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, parts.query, "")
    )
