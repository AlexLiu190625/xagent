from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from xagent.core.tools.adapters.vibe.mcp_adapter import _replace_exact_string_values
from xagent.core.tools.adapters.vibe.signed_upload import (
    SignedUploadBinding,
    SignedUploadResponseError,
    SignedUploadURLRejected,
    check_signed_upload_url,
    parse_signed_upload_response,
)

# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

SIZE_BYTES = 1234
REMOTE_FILE_ID = "rf_abcdefgh"
UPLOAD_URL = "https://uploads.example.test/staged/abc?sig=secret-signature"
EXPIRES_AT = "2030-01-01T00:00:00+00:00"


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "file_id": REMOTE_FILE_ID,
        "upload_url": UPLOAD_URL,
        "method": "PUT",
        "headers": {"Content-Type": "application/pdf"},
        "url_expires_at": EXPIRES_AT,
    }
    payload.update(overrides)
    return {k: v for k, v in payload.items() if v is not _MISSING}


_MISSING = object()


def _structured(payload: Any) -> dict[str, Any]:
    return {"content": [], "structured_content": payload, "is_error": False}


def _text(payload: Any, **extra: Any) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "content": [{"type": "text", "text": text}],
        "structured_content": None,
        "is_error": False,
        **extra,
    }


def test_parse_signed_upload_response_accepts_structured_content() -> None:
    ticket = parse_signed_upload_response(
        _structured(_payload()), size_bytes=SIZE_BYTES
    )

    assert ticket.file_id == REMOTE_FILE_ID
    assert ticket.upload_url == UPLOAD_URL
    assert dict(ticket.headers) == {"Content-Type": "application/pdf"}
    assert ticket.expires_at.utcoffset() is not None


def test_parse_signed_upload_response_falls_back_to_first_text_item() -> None:
    ticket = parse_signed_upload_response(_text(_payload()), size_bytes=SIZE_BYTES)

    assert ticket.file_id == REMOTE_FILE_ID


def test_parse_signed_upload_response_ignores_non_dict_structured_content() -> None:
    result = _text(_payload())
    result["structured_content"] = "not a dict"

    ticket = parse_signed_upload_response(result, size_bytes=SIZE_BYTES)

    assert ticket.file_id == REMOTE_FILE_ID


def test_parse_signed_upload_response_skips_non_text_items_before_the_text() -> None:
    result = _text(_payload())
    result["content"].insert(0, {"type": "image", "data": "aGk=", "mimeType": "x/y"})

    ticket = parse_signed_upload_response(result, size_bytes=SIZE_BYTES)

    assert ticket.file_id == REMOTE_FILE_ID


def test_parse_signed_upload_response_prefers_structured_content_over_text() -> None:
    result = _text(_payload(file_id="rf_from_text"))
    result["structured_content"] = _payload(file_id="rf_from_struct")

    ticket = parse_signed_upload_response(result, size_bytes=SIZE_BYTES)

    assert ticket.file_id == "rf_from_struct"


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"method": "put"}, id="method-lower-case"),
        pytest.param({"url_expires_at": "2030-01-01T00:00:00+00:00"}, id="offset"),
        pytest.param({"url_expires_at": "2030-01-01T00:00:00Z"}, id="zulu"),
        pytest.param(
            {
                "headers": {
                    "Content-Type": "application/pdf",
                    "Content-Length": str(SIZE_BYTES),
                    "x-checksum-sha256": "abc",
                }
            },
            id="allowed-headers",
        ),
    ],
)
def test_parse_signed_upload_response_accepts_variants(
    overrides: dict[str, Any],
) -> None:
    ticket = parse_signed_upload_response(
        _structured(_payload(**overrides)), size_bytes=SIZE_BYTES
    )

    assert ticket.file_id == REMOTE_FILE_ID
    assert ticket.expires_at.year == 2030
    assert ticket.expires_at.utcoffset() is not None
    if "headers" in overrides:
        assert dict(ticket.headers) == overrides["headers"]


def test_parse_signed_upload_response_headers_are_read_only() -> None:
    ticket = parse_signed_upload_response(
        _structured(_payload()), size_bytes=SIZE_BYTES
    )

    with pytest.raises(TypeError):
        ticket.headers["X-Extra"] = "1"  # type: ignore[index]


def _assert_error_hides_payload(exc: SignedUploadResponseError, *, reason: str) -> None:
    assert exc.reason == reason
    rendered = str(exc) + repr(exc)
    for needle in (
        REMOTE_FILE_ID,
        "secret-signature",
        "uploads.example.test",
        "application/pdf",
        "abcdefgh",
    ):
        assert needle not in rendered


_MALFORMED_CASES = [
    pytest.param(_structured(_payload(file_id=_MISSING)), id="file-id-missing"),
    pytest.param(_structured(_payload(file_id="abc")), id="file-id-too-short"),
    pytest.param(_structured(_payload(file_id="rf abcdefgh")), id="file-id-with-space"),
    pytest.param(_structured(_payload(file_id="a" * 129)), id="file-id-too-long"),
    pytest.param(
        _structured(_payload(file_id="rf_abcdefgh\n")), id="file-id-trailing-newline"
    ),
    pytest.param(_structured(_payload(file_id=12345678)), id="file-id-not-str"),
    pytest.param(_structured(_payload(upload_url=42)), id="upload-url-not-str"),
    pytest.param(_structured(_payload(upload_url=_MISSING)), id="upload-url-missing"),
    pytest.param(_structured(_payload(method="POST")), id="method-post"),
    pytest.param(_structured(_payload(method=_MISSING)), id="method-missing"),
    pytest.param(_structured(_payload(headers=["a: b"])), id="headers-not-dict"),
    pytest.param(_structured(_payload(headers=_MISSING)), id="headers-missing"),
    pytest.param(
        _structured(_payload(headers={"Content-Type": 5})), id="header-value-not-str"
    ),
    pytest.param(
        _structured(_payload(url_expires_at="2030-01-01T00:00:00")),
        id="expires-without-timezone",
    ),
    pytest.param(
        _structured(_payload(url_expires_at="tomorrow")), id="expires-not-iso"
    ),
    pytest.param(_structured(_payload(url_expires_at=_MISSING)), id="expires-missing"),
    pytest.param(
        _structured(_payload(url_expires_at=1893456000)), id="expires-not-str"
    ),
    pytest.param(_text("this is not json {"), id="text-not-json"),
    pytest.param(_text(json.dumps(["file_id", REMOTE_FILE_ID])), id="json-not-dict"),
    pytest.param(
        {"content": [], "structured_content": None, "is_error": False},
        id="no-payload-at-all",
    ),
    pytest.param(
        {
            "content": [{"type": "text", "text": 5}],
            "structured_content": None,
            "is_error": False,
        },
        id="text-item-without-text",
    ),
    pytest.param(
        _structured(
            _payload(headers={"Content-Length": str(SIZE_BYTES), "content-length": "1"})
        ),
        id="duplicate-content-length-different-case",
    ),
    pytest.param(
        _structured(_payload(headers={"X-Thing": "a", "x-thing": "a"})),
        id="duplicate-header-same-value",
    ),
]


@pytest.mark.parametrize("result", _MALFORMED_CASES)
def test_parse_signed_upload_response_rejects_malformed(
    result: dict[str, Any],
) -> None:
    with pytest.raises(SignedUploadResponseError) as exc_info:
        parse_signed_upload_response(result, size_bytes=SIZE_BYTES)

    _assert_error_hides_payload(exc_info.value, reason="malformed")


def test_parse_signed_upload_response_uses_only_the_first_text_item() -> None:
    result = _text("not json")
    result["content"].append({"type": "text", "text": json.dumps(_payload())})

    with pytest.raises(SignedUploadResponseError) as exc_info:
        parse_signed_upload_response(result, size_bytes=SIZE_BYTES)

    assert exc_info.value.reason == "malformed"


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({"Authorization": "Bearer planted"}, id="authorization"),
        pytest.param({"authorization": "Bearer planted"}, id="authorization-lower"),
        pytest.param({"Cookie": "a=b"}, id="cookie"),
        pytest.param({"Host": "evil.example.test"}, id="host"),
        pytest.param({"Proxy-Authorization": "Basic x"}, id="proxy-authorization"),
        pytest.param({"Proxy-Foo": "x"}, id="proxy-prefix"),
        pytest.param({"Transfer-Encoding": "chunked"}, id="transfer-encoding"),
        pytest.param({"Connection": "close"}, id="connection"),
        pytest.param({"Content-Length": "999"}, id="content-length-mismatch"),
        pytest.param({"content-length": "1234 "}, id="content-length-not-exact"),
    ],
)
def test_parse_signed_upload_response_rejects_forbidden_headers(
    headers: dict[str, str],
) -> None:
    with pytest.raises(SignedUploadResponseError) as exc_info:
        parse_signed_upload_response(
            _structured(_payload(headers=headers)), size_bytes=SIZE_BYTES
        )

    _assert_error_hides_payload(exc_info.value, reason="forbidden_header")
    for value in headers.values():
        assert value not in str(exc_info.value) + repr(exc_info.value)


# --------------------------------------------------------------------------
# URL policy
# --------------------------------------------------------------------------

BINDING = SignedUploadBinding(
    connection_host="mcp.example.test",
    tool_name="attach_document",
    file_argument="document",
    issuing_tool_name="request_upload",
    allowed_upload_hosts=frozenset({"uploads.example.test"}),
    upload_path_prefix="/staged/",
    content_types={".pdf": "application/pdf"},
)

_UPLOADS = "uploads.example.test"


@pytest.mark.parametrize(
    ("raw", "expected_host"),
    [
        pytest.param(f"http://{_UPLOADS}/staged/x", _UPLOADS, id="plain-http"),
        pytest.param(
            "https://other.example.test/staged/x",
            "other.example.test",
            id="host-not-registered",
        ),
        pytest.param(
            f"https://evil.{_UPLOADS}/staged/x",
            f"evil.{_UPLOADS}",
            id="subdomain-of-registered-host",
        ),
        pytest.param(
            f"https://{_UPLOADS}.evil/staged/x",
            f"{_UPLOADS}.evil",
            id="registered-host-as-prefix",
        ),
        pytest.param(
            f"https://{_UPLOADS}./staged/x", f"{_UPLOADS}.", id="trailing-dot"
        ),
        pytest.param(f"https://user:pw@{_UPLOADS}/staged/x", _UPLOADS, id="userinfo"),
        pytest.param(f"https://{_UPLOADS}:8443/staged/x", _UPLOADS, id="port-8443"),
        pytest.param(f"https://{_UPLOADS}/other/x", _UPLOADS, id="prefix-mismatch"),
        pytest.param(f"https://{_UPLOADS}/staged", _UPLOADS, id="prefix-without-slash"),
        pytest.param(f"https://{_UPLOADS}/staged/../x", _UPLOADS, id="dot-dot"),
        pytest.param(
            f"https://{_UPLOADS}/staged/%2e%2e/x", _UPLOADS, id="encoded-dot-dot"
        ),
        pytest.param(
            f"https://{_UPLOADS}/staged/%2E%2e/x", _UPLOADS, id="mixed-case-dot-dot"
        ),
        pytest.param(
            f"https://{_UPLOADS}/staged/a%5cb", _UPLOADS, id="encoded-backslash"
        ),
        pytest.param(f"https://{_UPLOADS}/staged/a%2fb", _UPLOADS, id="encoded-slash"),
        pytest.param(f"https://{_UPLOADS}/staged/a\\b", None, id="backslash"),
        pytest.param(f"https://{_UPLOADS}/staged/a\r\nb", None, id="crlf"),
        pytest.param(f"https://{_UPLOADS}/staged/a\x00b", None, id="nul"),
        pytest.param(f"https://{_UPLOADS}/staged/a\x7fb", None, id="del"),
        pytest.param(
            "https://xn--ls8h.example.test/staged/x", None, id="invalid-idna-host"
        ),
    ],
)
def test_signed_upload_url_policy_rejects(raw: str, expected_host: str | None) -> None:
    with pytest.raises(SignedUploadURLRejected) as exc_info:
        check_signed_upload_url(raw, BINDING)

    assert exc_info.value.host == expected_host
    assert "example.test" not in str(exc_info.value) + repr(exc_info.value)


@pytest.mark.parametrize(
    ("raw", "raw_path"),
    [
        pytest.param(
            f"https://{_UPLOADS.upper()}/staged/x", b"/staged/x", id="upper-case-host"
        ),
        pytest.param(
            f"https://{_UPLOADS}:443/staged/x?sig=1",
            b"/staged/x?sig=1",
            id="default-port-with-query",
        ),
    ],
)
def test_signed_upload_url_policy_accepts(raw: str, raw_path: bytes) -> None:
    url = check_signed_upload_url(raw, BINDING)

    assert isinstance(url, httpx.URL)
    assert url.host == _UPLOADS
    assert url.raw_path == raw_path
    assert url.scheme == "https"


def test_signed_upload_url_policy_ignores_query_when_checking_the_path() -> None:
    with pytest.raises(SignedUploadURLRejected):
        check_signed_upload_url(f"https://{_UPLOADS}/other/x?next=/staged/", BINDING)


# --------------------------------------------------------------------------
# Whole-value replacement
# --------------------------------------------------------------------------


def test_replace_exact_string_values() -> None:
    replacements = {REMOTE_FILE_ID: "file:own-id"}

    assert _replace_exact_string_values(REMOTE_FILE_ID, replacements) == "file:own-id"
    assert (
        _replace_exact_string_values(f"see {REMOTE_FILE_ID} for details", replacements)
        == f"see {REMOTE_FILE_ID} for details"
    )
    nested = {
        "a": [REMOTE_FILE_ID, {"b": (REMOTE_FILE_ID, "keep")}],
        "c": {"d": REMOTE_FILE_ID},
        "n": 7,
        "none": None,
    }
    assert _replace_exact_string_values(nested, replacements) == {
        "a": ["file:own-id", {"b": ("file:own-id", "keep")}],
        "c": {"d": "file:own-id"},
        "n": 7,
        "none": None,
    }
    assert nested["a"][0] == REMOTE_FILE_ID


def test_replace_exact_string_values_does_not_touch_mapping_keys() -> None:
    value = {REMOTE_FILE_ID: REMOTE_FILE_ID}

    assert _replace_exact_string_values(value, {REMOTE_FILE_ID: "file:own-id"}) == {
        REMOTE_FILE_ID: "file:own-id"
    }


def test_replace_exact_string_values_without_replacements_returns_same_object() -> None:
    value = {"a": [REMOTE_FILE_ID]}

    assert _replace_exact_string_values(value, {}) is value
