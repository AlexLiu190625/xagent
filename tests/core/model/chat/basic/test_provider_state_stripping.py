"""Repository-level coverage for one shared invariant across non-OpenAI
chat clients: no ``_xagent_``-prefixed internal message key ever reaches a
provider's wire request.

``_strip_internal_message_keys`` (now on ``BaseLLM``, see basic/base.py) is
the mechanism for clients that forward a message dict's keys mostly
unchanged (Zhipu, Xinference). Claude and Gemini take a different, equally
valid path to the same result: their message-conversion functions rebuild
each provider message field-by-field and never copy a message dict
wholesale, so an internal key is simply never read in the first place. Each
check below asserts the *result* (no leaked key), not the mechanism, so a
provider that later switches mechanisms without breaking the outcome does
not need this file to change.
"""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.model.chat.basic.claude import ClaudeLLM
from xagent.core.model.chat.basic.gemini import GeminiLLM
from xagent.core.model.chat.basic.xinference import XinferenceLLM
from xagent.core.model.chat.basic.zhipu import ZhipuLLM
from xagent.core.model.chat.types import PROVIDER_STATE_METADATA_KEY

_MARKED_HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": "Search xagent"},
    {
        "role": "assistant",
        "content": "",
        PROVIDER_STATE_METADATA_KEY: {"deepseek": {"reasoning_content": "prior"}},
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "result"},
]


def _assert_no_internal_keys(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        for key in message:
            assert not key.startswith("_xagent_"), (
                f"leaked internal key {key!r} in outbound message {message!r}"
            )


def test_claude_message_conversion_drops_internal_keys(claude_llm_config):
    """Claude: safety comes from rebuilding each message field-by-field, not
    from an explicit strip call -- see the comment in
    ``_convert_messages_to_anthropic_format``.
    """
    llm = ClaudeLLM(**claude_llm_config)

    _system, anthropic_messages = llm._convert_messages_to_anthropic_format(
        _MARKED_HISTORY
    )

    _assert_no_internal_keys(anthropic_messages)


def test_gemini_message_conversion_drops_internal_keys(gemini_llm_config):
    """Gemini: same field-by-field rebuild guarantee as Claude."""
    llm = GeminiLLM(**gemini_llm_config)

    _system, gemini_messages = llm._convert_messages_to_gemini_format(_MARKED_HISTORY)

    _assert_no_internal_keys(gemini_messages)


@pytest.mark.asyncio
async def test_zhipu_chat_strips_internal_keys_before_the_sdk_call(mocker):
    """Zhipu: forwards message dicts close to unchanged, so it needs (and
    now has) an explicit ``_strip_internal_message_keys`` call.
    """
    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock()]
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "done"
    mock_response.usage = None
    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch(
        "xagent.core.model.chat.basic.zhipu.ZhipuAiClient",
        return_value=mock_client,
    )
    llm = ZhipuLLM(api_key="test-key")

    await llm.chat(_MARKED_HISTORY)

    sent_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    _assert_no_internal_keys(sent_messages)


@pytest.mark.asyncio
async def test_xinference_chat_strips_internal_keys_before_the_sdk_call(mocker):
    """Xinference: same wholesale-forwarding shape as Zhipu, same fix."""

    class _ModelHandle:
        async def chat(self, **kwargs: Any):
            self.received_messages = kwargs["messages"]
            return {
                "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]
            }

    llm = XinferenceLLM(model_name="qwen3.8")
    handle = _ModelHandle()
    llm._client = mocker.MagicMock()
    llm._model_handle = handle

    await llm.chat(_MARKED_HISTORY)

    _assert_no_internal_keys(handle.received_messages)
