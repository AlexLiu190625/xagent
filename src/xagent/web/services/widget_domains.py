"""Widget embedding-origin normalization and allowlist enforcement."""

from urllib.parse import urlparse

from fastapi import HTTPException

__all__ = ["domain_allowed", "origin_to_domain", "require_domain_allowed"]


def origin_to_domain(origin: str) -> str:
    """Normalize an origin or referer value to a lowercased host[:port]."""
    if not origin:
        return ""
    parsed = urlparse(origin)
    return (parsed.netloc or parsed.path).lower()


def _normalize_allowed_domains(allowed_domains: object) -> list[str] | None:
    """Validate and normalize the complete persisted allowlist.

    JSON columns do not enforce their declared application-level shape. Treat
    any non-list container or non-string element as an invalid policy rather
    than coercing it into a potentially broader allowlist.
    """
    if not isinstance(allowed_domains, list):
        return None

    normalized_domains: list[str] = []
    for domain in allowed_domains:
        if type(domain) is not str:
            return None
        normalized_domains.append(domain.strip().lower())
    return normalized_domains


def domain_allowed(origin_domain: str, allowed_domains: object) -> bool:
    """Return whether a normalized ``origin_domain`` matches allowlist entries.

    ``origin_domain`` must already be normalized with :func:`origin_to_domain`.
    Allowlist entries retain their legacy surrounding-whitespace and
    case-insensitive handling inside this matcher. A malformed persisted
    allowlist denies access in full.
    """
    normalized_domains = _normalize_allowed_domains(allowed_domains)
    if normalized_domains is None:
        return False

    for normalized_domain in normalized_domains:
        if (
            normalized_domain == "*"
            or normalized_domain == origin_domain
            or (origin_domain and origin_domain.endswith("." + normalized_domain))
        ):
            return True
    return False


def require_domain_allowed(origin_domain: str, allowed_domains: object) -> None:
    """Enforce a normalized origin against stored widget allowlist entries.

    ``origin_domain`` must be the result of :func:`origin_to_domain`.
    """
    if not domain_allowed(origin_domain, allowed_domains):
        raise HTTPException(
            status_code=403, detail=f"Domain not allowed: {origin_domain}"
        )
