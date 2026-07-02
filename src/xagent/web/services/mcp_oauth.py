from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


class MCPOAuthRuntimeError(RuntimeError):
    """Raised when runtime cannot prepare an MCP OAuth bearer token."""

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


@dataclass(frozen=True)
class MCPOAuthRuntimeAuth:
    """Prepared MCP OAuth bearer authorization for runtime MCP connections."""

    access_token: str
    resource_owner_key: str
    issuer: str
    resource: str
    scope: str
    grant_id: int
    refreshed: bool = False


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
            resolve_dns_for_url_policy=False,
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
            resolve_dns_for_url_policy=True,
        )


async def resolve_mcp_oauth_runtime_auth(  # noqa: PLR0913
    db: Any,
    *,
    server_id: int,
    user_id: int,
    auth_config: dict[str, Any],
    resource_owner_key: str,
    resource: str | None = None,
    scope: str | None = None,
    issuer: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> MCPOAuthRuntimeAuth:
    """Resolve and refresh an MCP OAuth grant for one runtime MCP connection."""
    from ...core.utils.encryption import decrypt_value, encrypt_value
    from ..models.mcp_oauth import MCPOAuthClient

    normalized_resource = _runtime_config_value(resource, auth_config, "resource")
    selected_scope = scope if scope is not None else auth_config.get("scope")
    normalized_scope = _normalize_scope(selected_scope) if selected_scope else None
    if not normalized_resource:
        raise MCPOAuthRuntimeError(
            "authorization_required",
            "MCP OAuth runtime requires a configured resource or runtime resource",
        )

    candidate_grants = select_mcp_oauth_grants(
        db,
        server_id=server_id,
        user_id=user_id,
        auth_config=auth_config,
        resource_owner_key=resource_owner_key,
        resource=resource,
        issuer=issuer,
        scope="",
    )
    required_scopes = _scope_set(normalized_scope) if normalized_scope else set()
    grant = next(
        (
            candidate
            for candidate in candidate_grants
            if not required_scopes
            or required_scopes.issubset(_scope_set(candidate.scope))
        ),
        None,
    )
    if grant is None:
        if candidate_grants and required_scopes:
            raise MCPOAuthRuntimeError(
                "insufficient_scope",
                "No active MCP OAuth grant includes the required scope",
            )
        raise MCPOAuthRuntimeError(
            "authorization_required",
            "No active MCP OAuth grant exists for the selected resource owner",
        )

    access_token = decrypt_value(str(grant.access_token))
    refreshed = False
    if _grant_needs_refresh(grant.expires_at):
        if not grant.refresh_token:
            raise MCPOAuthRuntimeError(
                "token_refresh_failed",
                "MCP OAuth grant is expired and has no refresh token",
            )
        oauth_client = (
            db.query(MCPOAuthClient)
            .filter(MCPOAuthClient.id == grant.mcp_oauth_client_id)
            .first()
        )
        if oauth_client is None:
            raise MCPOAuthRuntimeError(
                "token_refresh_failed",
                "MCP OAuth client metadata not found for grant refresh",
            )
        token_data = await _refresh_mcp_oauth_grant(
            oauth_client,
            refresh_token=decrypt_value(str(grant.refresh_token)),
            resource=str(grant.resource),
            client=client,
        )
        grant.access_token = encrypt_value(str(token_data["access_token"]))
        access_token = str(token_data["access_token"])
        if token_data.get("refresh_token"):
            grant.refresh_token = encrypt_value(str(token_data["refresh_token"]))
        grant.token_type = str(token_data.get("token_type") or "Bearer")
        if token_data.get("scope") is not None:
            grant.scope = _normalize_scope(token_data.get("scope"))
        grant.metadata_json = {
            key: value
            for key, value in token_data.items()
            if key not in {"access_token", "refresh_token"}
        }
        if token_data.get("expires_in") is not None:
            grant.expires_at = _utc_now() + timedelta(
                seconds=int(token_data["expires_in"])
            )
        db.commit()
        refreshed = True

    return MCPOAuthRuntimeAuth(
        access_token=access_token,
        resource_owner_key=str(grant.resource_owner_key),
        issuer=str(grant.issuer),
        resource=str(grant.resource),
        scope=str(grant.scope),
        grant_id=int(grant.id),
        refreshed=refreshed,
    )


def select_mcp_oauth_grants(  # noqa: PLR0913
    db: Any,
    *,
    server_id: int,
    user_id: int,
    auth_config: dict[str, Any],
    resource_owner_key: str | None = None,
    resource: str | None = None,
    issuer: str | None = None,
    scope: str | None = None,
) -> list[Any]:
    """Return active MCP OAuth grants matching the current server auth config.

    This selector is intentionally side-effect free: it does not decrypt tokens,
    refresh grants, or mutate persistence. Runtime code can build on the selected
    grants when it needs bearer material.
    """
    from ..models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant

    normalized_resource = _runtime_config_value(resource, auth_config, "resource")
    normalized_issuer = _runtime_config_value(issuer, auth_config, "issuer")
    selected_scope = scope if scope is not None else auth_config.get("scope")
    normalized_scope = _normalize_scope(selected_scope) if selected_scope else None
    configured_client_id = _runtime_config_value(None, auth_config, "client_id")
    if not normalized_resource:
        return []

    query = (
        db.query(MCPOAuthGrant)
        .join(MCPOAuthClient, MCPOAuthGrant.mcp_oauth_client_id == MCPOAuthClient.id)
        .filter(
            MCPOAuthGrant.mcp_server_id == server_id,
            MCPOAuthGrant.user_id == user_id,
            MCPOAuthGrant.resource == normalized_resource,
            MCPOAuthGrant.status == "active",
            MCPOAuthClient.mcp_server_id == server_id,
        )
    )
    if resource_owner_key:
        query = query.filter(MCPOAuthGrant.resource_owner_key == resource_owner_key)
    if normalized_issuer:
        query = query.filter(MCPOAuthGrant.issuer == normalized_issuer)
    if configured_client_id:
        query = query.filter(MCPOAuthClient.client_id == configured_client_id)

    grants = query.order_by(MCPOAuthGrant.updated_at.desc()).all()
    required_scopes = _scope_set(normalized_scope) if normalized_scope else set()
    if not required_scopes:
        return list(grants)
    return [
        grant for grant in grants if required_scopes.issubset(_scope_set(grant.scope))
    ]


async def _discover_with_client(  # noqa: PLR0913
    endpoint_url: str,
    *,
    headers: dict[str, str] | None,
    configured_resource_metadata_url: str | None,
    configured_issuer: str | None,
    configured_resource: str | None,
    client: httpx.AsyncClient,
    resolve_dns_for_url_policy: bool,
) -> MCPOAuthDiscoveryResult:
    challenge: MCPAuthorizationChallenge | None = None
    metadata_urls: tuple[str, ...]

    if configured_resource_metadata_url:
        metadata_urls = (configured_resource_metadata_url,)
    else:
        challenge = await _probe_authorization_challenge(
            endpoint_url,
            headers=headers,
            client=client,
            resolve_dns_for_url_policy=resolve_dns_for_url_policy,
        )
        if challenge and challenge.resource_metadata_url:
            metadata_urls = (challenge.resource_metadata_url,)
        else:
            metadata_urls = protected_resource_metadata_urls(endpoint_url)

    protected_resource = await _fetch_first_protected_resource_metadata(
        metadata_urls,
        client=client,
        resolve_dns_for_url_policy=resolve_dns_for_url_policy,
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
        resolve_dns_for_url_policy=resolve_dns_for_url_policy,
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
    resolve_dns_for_url_policy: bool,
) -> MCPAuthorizationChallenge | None:
    try:
        await validate_oauth_http_url(
            endpoint_url, resolve_dns=resolve_dns_for_url_policy
        )
        response = await client.get(endpoint_url, headers=headers)
    except httpx.HTTPError as exc:
        raise MCPOAuthDiscoveryError(
            "metadata_not_found",
            f"Failed to probe MCP endpoint for OAuth challenge: {exc}",
        ) from exc

    return parse_www_authenticate_bearer(response.headers.get_list("WWW-Authenticate"))


async def _fetch_first_protected_resource_metadata(
    metadata_urls: Sequence[str],
    *,
    client: httpx.AsyncClient,
    resolve_dns_for_url_policy: bool,
) -> MCPProtectedResourceMetadata:
    last_error: Exception | None = None
    for metadata_url in metadata_urls:
        try:
            await validate_oauth_http_url(
                metadata_url, resolve_dns=resolve_dns_for_url_policy
            )
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
    resolve_dns_for_url_policy: bool,
) -> OAuthAuthorizationServerMetadata:
    last_error: Exception | None = None
    for metadata_url in authorization_server_metadata_urls(authorization_server_url):
        try:
            await validate_oauth_http_url(
                metadata_url, resolve_dns=resolve_dns_for_url_policy
            )
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
            await _validate_authorization_server_endpoints(
                metadata,
                resolve_dns_for_url_policy=resolve_dns_for_url_policy,
            )
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


async def _validate_authorization_server_endpoints(
    metadata: OAuthAuthorizationServerMetadata, *, resolve_dns_for_url_policy: bool
) -> None:
    await validate_oauth_http_url(
        metadata.authorization_endpoint, resolve_dns=resolve_dns_for_url_policy
    )
    await validate_oauth_http_url(
        metadata.token_endpoint, resolve_dns=resolve_dns_for_url_policy
    )


def _parse_bearer_challenge(header_value: str) -> MCPAuthorizationChallenge | None:
    match = re.search(r"(?i)(?:^|,\s*)Bearer(?:\s+|$)", header_value)
    if not match:
        return None

    params_text = header_value[match.end() :].strip()
    try:
        params = {
            str(key).lower(): str(value)
            for key, value in parse_keqv_list(parse_http_list(params_text)).items()
            if value is not None
        }
    except Exception:
        return None
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


async def _refresh_mcp_oauth_grant(
    oauth_client: Any,
    *,
    refresh_token: str,
    resource: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": oauth_client.client_id,
        "resource": resource,
    }
    auth: httpx.Auth | None = None
    client_secret = ""
    if oauth_client.client_secret:
        from ...core.utils.encryption import decrypt_value

        client_secret = decrypt_value(str(oauth_client.client_secret))
    auth_method = str(oauth_client.token_endpoint_auth_method or "none")
    if auth_method == "client_secret_post" and client_secret:
        data["client_secret"] = client_secret
    elif auth_method == "client_secret_basic" and client_secret:
        auth = httpx.BasicAuth(str(oauth_client.client_id), client_secret)
    elif auth_method not in {"none", "client_secret_post", "client_secret_basic"}:
        raise MCPOAuthRuntimeError(
            "token_refresh_failed",
            f"Unsupported token endpoint auth method: {auth_method}",
        )

    try:
        await validate_oauth_http_url(
            str(oauth_client.token_endpoint), resolve_dns=client is None
        )
        request_kwargs: dict[str, Any] = {
            "data": data,
            "headers": {"Content-Type": "application/x-www-form-urlencoded"},
        }
        if auth is not None:
            request_kwargs["auth"] = auth
        if client is not None:
            response = await client.post(
                str(oauth_client.token_endpoint),
                **request_kwargs,
            )
        else:
            async with httpx.AsyncClient(
                timeout=10.0, follow_redirects=True
            ) as owned_client:
                response = await owned_client.post(
                    str(oauth_client.token_endpoint),
                    **request_kwargs,
                )
        payload = response.json()
    except (MCPOAuthDiscoveryError, httpx.HTTPError, ValueError) as exc:
        raise MCPOAuthRuntimeError("token_refresh_failed", str(exc)) from exc

    if (
        response.status_code >= 400
        or not isinstance(payload, dict)
        or payload.get("error")
        or not payload.get("access_token")
    ):
        raise MCPOAuthRuntimeError(
            "token_refresh_failed",
            f"MCP OAuth refresh failed: {payload}",
        )
    return payload


def _runtime_config_value(
    request_value: str | None, auth_config: dict[str, Any], key: str
) -> str | None:
    value = request_value if request_value is not None else auth_config.get(key)
    return str(value).strip() if value else None


def _normalize_scope(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(item for item in value.split() if item)
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value if str(item))
    return ""


def _scope_set(value: Any) -> set[str]:
    return set(_normalize_scope(value).split())


def _grant_needs_refresh(expires_at: Any) -> bool:
    if not isinstance(expires_at, datetime):
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return bool(expires_at <= _utc_now() + timedelta(minutes=5))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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

    scheme = parts.scheme.lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    hostname = (parts.hostname or "").lower()
    if port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = parts.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parts.query, ""))


async def validate_oauth_http_url(value: str, *, resolve_dns: bool) -> None:
    """Reject OAuth metadata/token URLs that can target local infrastructure.

    Production discovery creates its own HTTP client, so it resolves DNS and
    blocks private, loopback, link-local, reserved, multicast, and unspecified
    addresses before connecting. Unit tests often pass a MockTransport-backed
    client for synthetic domains; those still get literal IP / localhost checks
    without turning tests into live DNS lookups.
    """
    parts = urlsplit(value)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "OAuth metadata URL must be an absolute HTTP(S) URL",
        )
    if parts.username or parts.password:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "OAuth metadata URL must not include userinfo",
        )

    hostname = parts.hostname.rstrip(".").lower()
    _reject_blocked_host(hostname)
    if not resolve_dns:
        return

    try:
        loop = asyncio.get_running_loop()
        addresses = await loop.run_in_executor(
            None,
            socket.getaddrinfo,
            hostname,
            parts.port or (443 if parts.scheme.lower() == "https" else 80),
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            f"Could not resolve OAuth metadata host: {hostname}",
        ) from exc

    for address in addresses:
        sockaddr = address[4]
        if sockaddr:
            _reject_blocked_host(str(sockaddr[0]))


def _reject_blocked_host(hostname: str) -> None:
    if hostname in {"localhost", "ip6-localhost"}:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "OAuth metadata host must not resolve to local addresses",
        )
    try:
        ip_address = ipaddress.ip_address(hostname)
    except ValueError:
        return
    if (
        ip_address.is_private
        or ip_address.is_loopback
        or ip_address.is_link_local
        or ip_address.is_multicast
        or ip_address.is_reserved
        or ip_address.is_unspecified
    ):
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "OAuth metadata host must not resolve to local addresses",
        )
