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


def domain_allowed(origin_domain: str, allowed_domains: list[str]) -> bool:
    """Return whether a normalized ``origin_domain`` matches allowlist entries.

    ``origin_domain`` must already be normalized with :func:`origin_to_domain`.
    Allowlist entries retain their legacy surrounding-whitespace and
    case-insensitive handling inside this matcher.
    """
    for domain in allowed_domains:
        normalized_domain = domain.strip().lower()
        if (
            normalized_domain == "*"
            or normalized_domain == origin_domain
            or (origin_domain and origin_domain.endswith("." + normalized_domain))
        ):
            return True
    return False


def require_domain_allowed(origin_domain: str, allowed_domains: list[str]) -> None:
    """Enforce a normalized origin against stored widget allowlist entries.

    ``origin_domain`` must be the result of :func:`origin_to_domain`.
    """
    if not domain_allowed(origin_domain, allowed_domains):
        raise HTTPException(
            status_code=403, detail=f"Domain not allowed: {origin_domain}"
        )
