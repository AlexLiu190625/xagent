"""The billing boundary around an already-charged image response.

A provider that answers 200 has billed the caller. Everything from the first
read of that body onwards is therefore an *invalid response* -- never a
retryable failure -- because retrying buys another charge for an outcome that
cannot change.

The boundary was introduced for failures that happen while *walking* a decoded
body. These tests pin the earlier half that was missing: the decode itself, and
the clients that decode internally. Left outside, a body that would not parse
raised a retryable error before any metering ran, so the call was billed on
every attempt and recorded on none -- worse than the multiplied-metering bug the
boundary was built to fix, because that spend leaves no trace at all.

Each test asserts the same two numbers: how many times the provider was called,
and how many billing rows exist afterwards.
"""

import asyncio
import json

import pytest
import requests
from openai import APIResponseValidationError

from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.image.adapter import retry_image_call
from xagent.core.model.image.base import (
    InvalidImageResponseError,
    call_billed_endpoint_async,
)


def _run_with_retries(call, max_retries: int = 4):
    """Drive a call through the production retry policy, counting attempts."""
    attempts = 0
    last: Exception | None = None
    for _ in range(max_retries):
        attempts += 1
        try:
            return call(), attempts
        except Exception as error:  # noqa: BLE001
            last = error
            if not retry_image_call(error):
                break
    raise last from None


@pytest.mark.asyncio
async def test_openai_response_validation_is_treated_as_billed() -> None:
    # The OpenAI SDK reports a 200 whose body fails validation with its own
    # error type, which is not a ValueError -- so the provider names it
    # explicitly rather than the boundary guessing.
    async def call():
        raise APIResponseValidationError(
            response=requests.Response(), body=None, message="bad body"
        )

    with pytest.raises(InvalidImageResponseError):
        await call_billed_endpoint_async(
            call, "ctx", also_billed=(APIResponseValidationError,)
        )


@pytest.mark.asyncio
async def test_openai_connection_errors_stay_retryable() -> None:
    async def call():
        raise requests.exceptions.ConnectionError("reset")

    with pytest.raises(requests.exceptions.ConnectionError) as caught:
        await call_billed_endpoint_async(
            call, "ctx", also_billed=(APIResponseValidationError,)
        )
    assert retry_image_call(caught.value) is True


class _Item:
    def __init__(self, url=None, b64_json=None):
        self.url = url
        self.b64_json = b64_json


@pytest.mark.asyncio
async def test_openai_meters_a_response_the_sdk_refused_to_return() -> None:
    from xagent.core.model.image.openai import OpenAIImageModel

    model = OpenAIImageModel(api_key="k", model_id="cfg-oa")
    model._ensure_client = lambda: None  # type: ignore[method-assign]

    class _Images:
        async def generate(self, **kwargs: object) -> object:
            raise APIResponseValidationError(
                response=requests.Response(), body=None, message="bad body"
            )

    model._client = type("_C", (), {"images": _Images()})()  # type: ignore[assignment]

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await model.generate_image(prompt="p", n=2)
        usage = manager.get_usage()

    assert usage.media_calls == 1
    row = usage.details[0]
    assert row["model_id"] == "cfg-oa"
    # The requested count, and no tokens: the SDK raised instead of handing
    # over a usage payload, so the call is billed but unmeasured.
    assert row["quantity"] == 2.0
    assert row["provider_tokens"] == 0


@pytest.mark.parametrize("method", ["generate", "edit"])
@pytest.mark.asyncio
async def test_xinference_meters_a_body_the_client_could_not_decode(
    method: str,
) -> None:
    from xagent.core.model.image.xinference import XinferenceImageModel

    model = XinferenceImageModel(model_name="sd", base_url="http://x", model_id="xi-1")
    model._ensure_client = lambda: None  # type: ignore[method-assign]
    model._abilities = ["generate", "edit"]

    class _Handle:
        def text_to_image(self, **kwargs: object) -> object:
            raise json.JSONDecodeError("x", "<html>", 0)

        def image_to_image(self, **kwargs: object) -> object:
            raise json.JSONDecodeError("x", "<html>", 0)

    model._model_handle = _Handle()

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            if method == "generate":
                await model.generate_image(prompt="p", n=3)
            else:
                await model.edit_image(image_url="u", prompt="e", n=3)
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["quantity"] == 3.0
    assert usage.details[0]["model_id"] == "xi-1"


def _undecodable_gemini(monkeypatch):
    import httpx

    class _Response:
        status_code = 200
        text = "<html>502</html>"

        def json(self):
            raise json.JSONDecodeError("x", "<html>", 0)

        def raise_for_status(self):
            return None

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return _Response()

    from xagent.core.model.image.gemini import GeminiImageModel

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client())
    model = GeminiImageModel(model_name="gemini-3-pro-image-preview-2k", api_key="k")
    return model, lambda: model.generate_image(prompt="p")


def _undecodable_dashscope(monkeypatch):
    from xagent.core.model.image import dashscope as ds

    class _Response:
        status = 200

        async def json(self):
            raise ValueError("Attempt to decode JSON with unexpected mimetype")

        async def text(self):
            return "<html>"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(ds.aiohttp, "ClientSession", lambda *a, **k: _Session())
    model = ds.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"], model_id="d1"
    )
    return model, lambda: model.generate_image(prompt="p", n=2)


def _unusable_openai(monkeypatch):
    from xagent.core.model.image.openai import OpenAIImageModel

    class _Images:
        async def generate(self, **kwargs: object) -> object:
            raise APIResponseValidationError(
                response=requests.Response(), body=None, message="bad body"
            )

    model = OpenAIImageModel(api_key="k", model_id="o1")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    model._client = type("_C", (), {"images": _Images()})()
    return model, lambda: model.generate_image(prompt="p", n=2)


def _undecodable_xinference(monkeypatch):
    from xagent.core.model.image.xinference import XinferenceImageModel

    class _Handle:
        def text_to_image(self, **kwargs: object) -> object:
            raise json.JSONDecodeError("x", "<html>", 0)

    model = XinferenceImageModel(model_name="sd", base_url="http://x", model_id="x1")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    model._abilities = ["generate", "edit"]
    model._model_handle = _Handle()
    return model, lambda: model.generate_image(prompt="p", n=2)


@pytest.mark.parametrize(
    "setup",
    [
        _undecodable_gemini,
        _undecodable_dashscope,
        _unusable_openai,
        _undecodable_xinference,
    ],
    ids=["gemini", "dashscope", "openai", "xinference"],
)
@pytest.mark.asyncio
async def test_every_provider_meters_a_billed_but_unreadable_response(
    monkeypatch, setup
) -> None:
    """One charge, one row — on all four providers.

    Classifying the failure as non-retryable is only half the fix. Without the
    row, the single real charge leaves no trace at all, which is worse than the
    multiplied metering the classification prevents.
    """
    _, call = setup(monkeypatch)

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await call()
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert len(usage.details) == 1


@pytest.mark.parametrize(
    "setup",
    [
        _undecodable_gemini,
        _undecodable_dashscope,
        _unusable_openai,
        _undecodable_xinference,
    ],
    ids=["gemini", "dashscope", "openai", "xinference"],
)
@pytest.mark.asyncio
async def test_a_billed_unreadable_response_is_never_retried(
    monkeypatch, setup
) -> None:
    # The classification half, asserted on the same four so the two properties
    # cannot drift apart per provider.
    _, call = setup(monkeypatch)
    with TokenContextManager():
        with pytest.raises(InvalidImageResponseError) as caught:
            await call()
    assert retry_image_call(caught.value) is False


@pytest.mark.asyncio
async def test_openai_edit_honours_resolution(monkeypatch) -> None:
    from xagent.core.model.image.openai import OpenAIImageModel

    sent: dict = {}

    class _Images:
        async def edit(self, **kwargs: object) -> object:
            sent.update(kwargs)
            return type("R", (), {"data": [], "usage": None, "id": "x"})()

    model = OpenAIImageModel(api_key="k", model_id="o1")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    model._client = type("_C", (), {"images": _Images()})()
    monkeypatch.setattr(
        "builtins.open",
        lambda *a, **k: type(
            "F", (), {"close": lambda s: None, "read": lambda s: b""}
        )(),
    )

    with TokenContextManager() as manager:
        await model.edit_image(image_url="a.png", prompt="p", resolution="1920x1080")
        usage = manager.get_usage()

    assert sent["size"] == "1920x1080"
    assert usage.details[0]["resolution"] == "1920x1080"


@pytest.mark.asyncio
async def test_xinference_edit_honours_width_and_height(monkeypatch) -> None:
    from xagent.core.model.image.xinference import XinferenceImageModel

    sent: dict = {}

    class _Handle:
        def image_to_image(self, **kwargs: object) -> object:
            sent.update(kwargs)
            return {"data": [{"url": "https://x/y.png"}]}

    model = XinferenceImageModel(model_name="sd", base_url="http://x", model_id="x1")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    model._abilities = ["generate", "edit"]
    model._model_handle = _Handle()

    with TokenContextManager() as manager:
        await model.edit_image(image_url="u", prompt="p", width=1920, height=1080)
        usage = manager.get_usage()

    # Xinference speaks W*H, so the shared helper's WxH is normalised on the way
    # out -- and the recorded key matches what that provider is priced on.
    assert sent["size"] == "1920*1080"
    assert usage.details[0]["resolution"] == "1920*1080"


@pytest.mark.asyncio
async def test_dashscope_edit_honours_resolution(monkeypatch) -> None:
    from xagent.core.model.image import dashscope as ds

    sent: dict = {}

    class _Response:
        status = 200

        async def json(self):
            return {
                "usage": {},
                "output": {
                    "choices": [
                        {"message": {"content": [{"image": "https://x/e.png"}]}}
                    ]
                },
            }

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, headers=None, json=None, **kwargs):
            sent.update((json or {}).get("parameters", {}))
            return _Response()

    monkeypatch.setattr(ds.aiohttp, "ClientSession", lambda *a, **k: _Session())
    model = ds.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"], model_id="d1"
    )
    monkeypatch.setattr(model, "_convert_image_to_base64", lambda x: "b64")

    with TokenContextManager() as manager:
        await model.edit_image(
            image_url="https://x/s.png", prompt="p", resolution="1920x1080"
        )
        usage = manager.get_usage()

    # The two halves differ on purpose: DashScope's request takes `W*H`, the
    # recorded aggregate key takes `WxH`. Asserting both is what catches either
    # one drifting -- sending `WxH` is rejected by the endpoint, and recording
    # `W*H` splits one physical size across two billing keys.
    assert sent["size"] == "1920*1080"
    assert usage.details[0]["resolution"] == "1920x1080"


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *args, **kwargs):
        return _FakeQuery(self._rows)


class _Row:
    id = 1
    model_id = "cfg-public"
    model_name = "gpt-image-1"
    model_provider = "openai"
    api_key = "sk-test"
    base_url = "https://api.openai.com/v1"
    abilities = ["generate", "edit"]
    category = "image"
    is_active = True
    max_retries = 3


@pytest.mark.asyncio
async def test_openai_does_not_record_a_cancelled_call(monkeypatch) -> None:
    from xagent.core.model.image.openai import OpenAIImageModel

    class _Images:
        async def generate(self, **kwargs: object) -> object:
            # Stands for a cancellation anywhere in the SDK coroutine, including
            # before the request reaches OpenAI.
            raise asyncio.CancelledError()

    model = OpenAIImageModel(api_key="k", model_id="o1")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    model._client = type("_C", (), {"images": _Images()})()

    with TokenContextManager() as manager:
        with pytest.raises(asyncio.CancelledError):
            await model.generate_image(prompt="p", n=2)
        usage = manager.get_usage()

    # No row: the outcome is unknown, and inventing a charge is worse than
    # leaving an ambiguous one to reconciliation (xorbitsai/xagent#2513).
    assert usage.media_calls == 0


@pytest.mark.asyncio
async def test_dashscope_records_a_cancelled_decode(monkeypatch) -> None:
    from xagent.core.model.image import dashscope as ds

    class _Response:
        status = 200

        async def json(self):
            # Cancelled while decoding a 200 that DashScope has already billed.
            raise asyncio.CancelledError()

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(ds.aiohttp, "ClientSession", lambda *a, **k: _Session())
    model = ds.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"], model_id="d1"
    )

    with TokenContextManager() as manager:
        with pytest.raises(asyncio.CancelledError):
            await model.generate_image(prompt="p", n=2)
        usage = manager.get_usage()

    # One row: the 200 arrived, so the charge is certain even though the body
    # was never read. Re-raised unchanged so the caller's shutdown still works.
    assert usage.media_calls == 1
    assert usage.details[0]["quantity"] == 2.0


@pytest.mark.asyncio
async def test_dashscope_edit_omits_size_when_none_was_requested(monkeypatch) -> None:
    """An absent size must stay an absent key, not an empty one.

    DashScope treats `size` as optional but requires `W*H` when present, so
    `size=""` rejects an edit the base request would have accepted. The folded
    value is empty exactly when the caller passed no size, resolution, width or
    height.
    """
    from xagent.core.model.image import dashscope as ds

    sent: dict = {}

    class _Response:
        status = 200

        async def json(self):
            return {
                "usage": {},
                "output": {
                    "choices": [
                        {"message": {"content": [{"image": "https://x/e.png"}]}}
                    ]
                },
            }

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, headers=None, json=None, **kwargs):
            sent.update((json or {}).get("parameters", {}))
            return _Response()

    monkeypatch.setattr(ds.aiohttp, "ClientSession", lambda *a, **k: _Session())
    model = ds.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"], model_id="d1"
    )
    monkeypatch.setattr(model, "_convert_image_to_base64", lambda x: "b64")

    with TokenContextManager() as manager:
        await model.edit_image(image_url="https://x/s.png", prompt="p")
        usage = manager.get_usage()

    assert "size" not in sent
    # The row still records the call, with no resolution to key it on.
    assert usage.details[0]["resolution"] == ""
