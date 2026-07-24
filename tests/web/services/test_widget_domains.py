import pytest
from fastapi import HTTPException

from xagent.web.services.widget_domains import (
    domain_allowed,
    origin_to_domain,
    require_domain_allowed,
)


@pytest.mark.parametrize(
    ("origin", "expected_domain"),
    [
        ("https://EXAMPLE.com:8443/widget", "example.com:8443"),
        ("EXAMPLE.com", "example.com"),
        ("", ""),
    ],
)
def test_origin_to_domain_preserves_the_current_host_port_contract(
    origin: str, expected_domain: str
) -> None:
    assert origin_to_domain(origin) == expected_domain


@pytest.mark.parametrize(
    ("origin_domain", "allowed_domains", "expected"),
    [
        ("app.example.com", ["example.com"], True),
        ("evil-example.com", ["example.com"], False),
        ("example.com:443", ["example.com"], False),
        ("example.com:443", ["example.com:443"], True),
        ("example.com", [], False),
        ("", ["*"], True),
    ],
)
def test_domain_allowed_preserves_widget_allowlist_matching(
    origin_domain: str, allowed_domains: list[str], expected: bool
) -> None:
    assert domain_allowed(origin_domain, allowed_domains) is expected


@pytest.mark.parametrize(
    ("origin_domain", "allowed_domains"),
    [
        ("example.com", None),
        ("e", "example.com"),
        ("example.com", {"example.com": True}),
        ("example.com", ["example.com", None]),
        ("example.com", [None, "example.com"]),
        ("123", [123]),
        ("true", [True]),
    ],
)
def test_domain_allowed_rejects_malformed_persisted_allowlists(
    origin_domain: str, allowed_domains: object
) -> None:
    assert not domain_allowed(origin_domain, allowed_domains)


def test_normalized_domain_composition_preserves_case_insensitive_allowlists() -> None:
    raw_origin = "https://APP.EXAMPLE.COM/widget"
    raw_allowed_domain = " APP.example.COM "

    assert domain_allowed(origin_to_domain(raw_origin), [raw_allowed_domain])


def test_domain_allowed_does_not_normalize_direct_origin_input() -> None:
    assert not domain_allowed("APP.EXAMPLE.COM", ["app.example.com"])


def test_domain_allowed_does_not_parse_scheme_form_allowlist_entries() -> None:
    assert not domain_allowed("example.com", ["https://example.com"])


def test_require_domain_allowed_accepts_case_insensitive_stored_allowlist_entries() -> (
    None
):
    require_domain_allowed(
        origin_to_domain("https://APP.EXAMPLE.COM/widget"), [" APP.example.COM "]
    )


def test_require_domain_allowed_preserves_forbidden_response() -> None:
    with pytest.raises(HTTPException) as exc_info:
        require_domain_allowed("untrusted.example", ["trusted.example"])

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Domain not allowed: untrusted.example"


def test_require_domain_allowed_maps_malformed_allowlist_to_forbidden() -> None:
    with pytest.raises(HTTPException) as exc_info:
        require_domain_allowed("trusted.example", [None, "trusted.example"])

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Domain not allowed: trusted.example"
