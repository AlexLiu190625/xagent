"""Test cases for OpenRouter LLM provider behavior."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from xagent.core.model.chat.basic import openrouter as openrouter_module
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.openrouter import OpenRouterLLM
from xagent.core.model.chat.error import retry_on
from xagent.core.model.chat.exceptions import (
    LLMRetryableError,
    LLMToolProtocolError,
)
from xagent.core.model.chat.tool_protocol import (
    TOOL_PROTOCOL_ERROR_KEY,
    get_tool_protocol_error,
)
from xagent.core.model.chat.types import ChunkType, StreamChunk
from xagent.core.retry.strategy import FixedDelay
from xagent.core.retry.wrapper import create_retry_wrapper


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("deepseek/deepseek-v4-flash", True),
        ("openrouter/deepseek/deepseek-v4-flash", True),
        ("anthropic/claude-sonnet-4.6", False),
    ],
)
def test_openrouter_uses_deepseek_tool_protocol_only_for_deepseek_models(
    model_name, expected
):
    llm = OpenRouterLLM(model_name=model_name, api_key="test-key")

    assert llm._uses_deepseek_tool_protocol is expected


@pytest.mark.asyncio
async def test_openrouter_deepseek_marks_serialized_tool_protocol_retryable(
    mocker,
):
    message = SimpleNamespace(
        content="<｜｜DSML｜｜tool_calls>",
        tool_calls=None,
        reasoning_content=None,
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
        model_dump=lambda: {"id": "openrouter-deepseek-invalid-protocol"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    with pytest.raises(
        LLMRetryableError,
        match="serialized_tool_call_content",
    ):
        await llm.chat(
            [{"role": "user", "content": "Use a tool"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )


@pytest.mark.asyncio
async def test_openrouter_deepseek_protocol_error_uses_shared_chat_retry(mocker):
    invalid_message = SimpleNamespace(
        content="<｜｜DSML｜｜tool_calls>",
        tool_calls=None,
        reasoning_content=None,
    )
    invalid_response = SimpleNamespace(
        choices=[SimpleNamespace(message=invalid_message)],
        usage=None,
        model_dump=lambda: {"id": "openrouter-deepseek-invalid-protocol"},
    )
    tool_call = SimpleNamespace(
        id="call_route",
        type="function",
        function=SimpleNamespace(
            name="select_execution_pattern",
            arguments="{}",
        ),
    )
    valid_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tool_call]))
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-deepseek-valid-protocol"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        invalid_response,
        valid_response,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    inner = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )
    llm = create_retry_wrapper(
        inner,
        BaseLLM,  # type: ignore[type-abstract]
        retry_methods={"chat"},
        strategy=FixedDelay(delay_ms=0),
        max_retries=2,
        retry_on=retry_on,
    )

    result = await llm.chat(
        [{"role": "user", "content": "Route this request"}],
        tools=_single_tool_schema(),
        tool_choice="required",
    )

    assert result["tool_calls"][0]["function"]["name"] == ("select_execution_pattern")
    assert mock_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_openrouter_deepseek_protocol_error_uses_shared_stream_retry(mocker):
    attempts = 0

    async def invalid_stream():
        yield StreamChunk(
            type=ChunkType.TOKEN,
            delta="Let me route this. <｜｜DSML｜｜tool_calls>",
        )
        yield StreamChunk(type=ChunkType.END, finish_reason="stop")

    async def valid_stream():
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_route",
                    "type": "function",
                    "function": {
                        "name": "select_execution_pattern",
                        "arguments": "{}",
                    },
                }
            ],
        )
        yield StreamChunk(type=ChunkType.END, finish_reason="tool_calls")

    def fake_stream(**kwargs):
        nonlocal attempts
        del kwargs
        attempts += 1
        return invalid_stream() if attempts == 1 else valid_stream()

    inner = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )
    mocker.patch.object(
        inner,
        "_stream_chat_with_prefix_retry",
        side_effect=fake_stream,
    )
    llm = create_retry_wrapper(
        inner,
        BaseLLM,  # type: ignore[type-abstract]
        retry_methods={"stream_chat"},
        strategy=FixedDelay(delay_ms=0),
        max_retries=2,
        retry_on=retry_on,
    )

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "Route this request"}],
            tools=_single_tool_schema(),
            tool_choice="required",
        )
    ]

    assert attempts == 2
    assert not any(chunk.is_protocol_error() for chunk in chunks)
    assert any(
        chunk.is_tool_call()
        and chunk.tool_calls[0]["function"]["name"] == "select_execution_pattern"
        for chunk in chunks
    )


@pytest.mark.asyncio
async def test_openrouter_does_not_replay_unavailable_tool_call(mocker):
    unavailable_tool_call = SimpleNamespace(
        id="call_unavailable",
        type="function",
        function=SimpleNamespace(
            name="calculator",
            arguments='{"expression":"2+2"}',
        ),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[unavailable_tool_call],
                    reasoning_content=None,
                )
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-deepseek-unavailable-tool"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    inner = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )
    llm = create_retry_wrapper(
        inner,
        BaseLLM,  # type: ignore[type-abstract]
        retry_methods={"chat"},
        strategy=FixedDelay(delay_ms=0),
        max_retries=10,
        retry_on=retry_on,
    )

    result = await llm.chat(
        [{"role": "user", "content": "Finish this task"}],
        tools=_single_tool_schema(),
        tool_choice="required",
    )

    protocol_error = get_tool_protocol_error(result)
    assert protocol_error is not None
    assert protocol_error["code"] == "unavailable_tool_call"
    assert mock_client.chat.completions.create.await_count == 1


def test_retry_filter_defers_unavailable_tool_call_to_agent_pattern() -> None:
    error = LLMToolProtocolError(
        provider="deepseek",
        code="unavailable_tool_call",
        message="DeepSeek returned an unavailable tool call.",
    )

    assert retry_on(error) is False


def test_retry_filter_defers_malformed_tool_arguments_to_agent_pattern() -> None:
    error = LLMToolProtocolError(
        provider="deepseek",
        code="malformed_tool_arguments",
        message="DeepSeek returned malformed arguments for 'final_answer'.",
    )

    assert retry_on(error) is False


def test_deepseek_protocol_error_preserves_argument_diagnostics() -> None:
    details = {
        "original_arguments_preview": '{"answer":',
        "original_arguments_length": 10,
        "repair_status": "skipped_incomplete",
    }

    error = openrouter_module._deepseek_tool_protocol_retry_error(
        {
            TOOL_PROTOCOL_ERROR_KEY: {
                "provider": "deepseek",
                "code": "malformed_tool_arguments",
                "message": "Malformed arguments.",
                "details": details,
            }
        }
    )

    assert isinstance(error, LLMToolProtocolError)
    assert error.details == details


@pytest.mark.asyncio
async def test_openrouter_stream_defers_unavailable_tool_call_without_replay(mocker):
    attempts = 0

    async def unavailable_stream():
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_unavailable",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    },
                }
            ],
        )
        yield StreamChunk(type=ChunkType.END, finish_reason="tool_calls")

    def fake_stream(**kwargs):
        nonlocal attempts
        del kwargs
        attempts += 1
        return unavailable_stream()

    inner = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )
    mocker.patch.object(
        inner,
        "_stream_chat_with_prefix_retry",
        side_effect=fake_stream,
    )
    llm = create_retry_wrapper(
        inner,
        BaseLLM,  # type: ignore[type-abstract]
        retry_methods={"stream_chat"},
        strategy=FixedDelay(delay_ms=0),
        max_retries=10,
        retry_on=retry_on,
    )

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "Finish this task"}],
            tools=_single_tool_schema(),
            tool_choice="required",
        )
    ]

    assert attempts == 1
    protocol_errors = [chunk for chunk in chunks if chunk.is_protocol_error()]
    assert len(protocol_errors) == 1
    assert protocol_errors[0].protocol_error["code"] == "unavailable_tool_call"


def _deepseek_function_prefix_error() -> openai.BadRequestError:
    return openai.BadRequestError(
        "Error code: 400 - {'error': {'message': 'Provider returned error'}}",
        response=httpx.Response(
            400,
            request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
        ),
        body={
            "error": {
                "message": "Provider returned error",
                "code": 400,
                "metadata": {
                    "provider_name": "DeepSeek",
                    "raw": (
                        '{"error":{"message":'
                        '"Function call should not be used with prefix"}}'
                    ),
                },
            }
        },
    )


def _unrelated_bad_request() -> openai.BadRequestError:
    return openai.BadRequestError(
        "Error code: 400 - {'error': {'message': 'Unrelated invalid request'}}",
        response=httpx.Response(
            400,
            request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
        ),
        body={"error": {"message": "Unrelated invalid request", "code": 400}},
    )


def _tool_call_history() -> list[dict]:
    return [
        {"role": "user", "content": "Generate music"},
        {
            "role": "assistant",
            "content": "I will generate the music first.",
            "tool_calls": [
                {
                    "id": "call_music",
                    "type": "function",
                    "function": {
                        "name": "generate_music",
                        "arguments": '{"prompt":"intro"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_music",
            "content": '{"success":true}',
        },
    ]


def _tool_call_history_with_trailing_progress() -> list[dict]:
    messages = _tool_call_history()
    messages[1]["content"] = ""
    messages.append(
        {
            "role": "assistant",
            "content": "Still working on the generated audio.",
        }
    )
    return messages


def _single_tool_schema(name: str = "select_execution_pattern") -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.mark.parametrize(
    "tools",
    [
        [None],
        ["invalid"],
        [{}],
        [{"function": None}],
    ],
)
def test_openrouter_deepseek_preserves_required_for_malformed_single_tool(
    tools,
):
    assert (
        openrouter_module._force_single_required_deepseek_tool(tools, "required")
        == "required"
    )


@pytest.mark.asyncio
async def test_openrouter_official_provider_pinning_disabled_by_default(
    mock_chat_completion, mocker, monkeypatch
):
    """OpenRouter provider pinning is opt-in to preserve fallback behavior."""

    monkeypatch.delenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", raising=False)
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    await llm.chat([{"role": "user", "content": "Hello"}])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "provider" not in call_kwargs["extra_body"]
    assert call_kwargs["extra_body"]["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_openrouter_deepseek_uses_official_provider(
    mock_chat_completion, mocker, monkeypatch
):
    """OpenRouter DeepSeek slugs should avoid third-party host fallbacks."""

    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "true")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    await llm.chat([{"role": "user", "content": "Hello"}])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"]["provider"] == {
        "only": ["deepseek"],
        "allow_fallbacks": False,
    }


def test_openrouter_official_provider_mapping_covers_auto_router_authors(
    monkeypatch,
):
    """Auto-selected official slugs should pin to official OpenRouter providers."""

    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "true")
    cases = {
        "anthropic/claude-sonnet-4.6": ["anthropic"],
        "deepseek/deepseek-v4-flash": ["deepseek"],
        "google/gemini-3-flash-preview": ["google-ai-studio", "google-vertex"],
        "minimax/minimax-m3": ["minimax"],
        "openai/gpt-5.5": ["openai"],
        "z-ai/glm-5.2": ["z-ai"],
    }

    for model_name, expected_providers in cases.items():
        llm = OpenRouterLLM(
            model_name=model_name,
            api_key="test-key",
        )

        extra_body = llm._prepare_extra_body({})

        assert extra_body["provider"] == {
            "only": expected_providers,
            "allow_fallbacks": False,
        }


@pytest.mark.asyncio
async def test_openrouter_provider_override_is_preserved(
    mock_chat_completion, mocker, monkeypatch
):
    """Explicit provider routing should win over automatic official pinning."""

    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "true")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    await llm.chat(
        [{"role": "user", "content": "Hello"}],
        extra_body={"provider": {"only": ["deepinfra"]}, "trace_id": "manual"},
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"] == {
        "provider": {"only": ["deepinfra"]},
        "trace_id": "manual",
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }


@pytest.mark.asyncio
async def test_openrouter_deepseek_names_the_only_required_tool(mocker, monkeypatch):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    tool_call = SimpleNamespace(
        id="call_route",
        type="function",
        function=SimpleNamespace(
            name="select_execution_pattern",
            arguments="{}",
        ),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[tool_call],
                )
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-deepseek-route"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = response
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    await llm.chat(
        [{"role": "user", "content": "Route this request"}],
        tools=_single_tool_schema(),
        tool_choice="required",
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "select_execution_pattern"},
    }


@pytest.mark.asyncio
async def test_openrouter_deepseek_stream_names_the_only_required_tool(
    mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")

    async def empty_stream():
        if False:
            yield None

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = empty_stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "Route this request"}],
            tools=_single_tool_schema(),
            tool_choice="required",
        )
    ]

    assert chunks == []
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "select_execution_pattern"},
    }


@pytest.mark.asyncio
async def test_openrouter_deepseek_retries_function_call_without_assistant_prefix(
    mock_chat_completion, mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _deepseek_function_prefix_error(),
        mock_chat_completion,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )
    messages = _tool_call_history()
    strip_spy = mocker.spy(openrouter_module, "_strip_assistant_tool_call_prefixes")

    result = await llm.chat(messages)

    assert result["content"] == "Hello World"
    assert mock_client.chat.completions.create.await_count == 2
    assert strip_spy.call_count == 1
    first_messages = mock_client.chat.completions.create.call_args_list[0].kwargs[
        "messages"
    ]
    retry_messages = mock_client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert first_messages[1]["content"] == "I will generate the music first."
    assert retry_messages[1]["content"] == ""
    assert retry_messages[1]["tool_calls"] == messages[1]["tool_calls"]
    assert messages[1]["content"] == "I will generate the music first."


@pytest.mark.asyncio
async def test_openrouter_deepseek_retries_without_trailing_assistant_progress(
    mock_chat_completion, mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _deepseek_function_prefix_error(),
        mock_chat_completion,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )
    messages = _tool_call_history_with_trailing_progress()

    result = await llm.chat(messages)

    assert result["content"] == "Hello World"
    assert mock_client.chat.completions.create.await_count == 2
    retry_messages = mock_client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert retry_messages[-1]["role"] == "tool"
    assert all(
        message.get("content") != "Still working on the generated audio."
        for message in retry_messages
    )
    assert messages[-1]["content"] == "Still working on the generated audio."


@pytest.mark.asyncio
async def test_openrouter_deepseek_propagates_sanitized_retry_failure(
    mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _deepseek_function_prefix_error(),
        _deepseek_function_prefix_error(),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    with pytest.raises(RuntimeError, match="Function call should not be used"):
        await llm.chat(_tool_call_history())

    assert mock_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_openrouter_deepseek_does_not_retry_whitespace_only_prefix(
    mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = _deepseek_function_prefix_error()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )
    messages = _tool_call_history()
    messages[1]["content"] = "   "

    with pytest.raises(RuntimeError, match="Function call should not be used"):
        await llm.chat(messages)

    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_openrouter_deepseek_stream_retries_prefix_error_before_first_chunk(
    mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")

    async def empty_stream():
        if False:
            yield None

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        _deepseek_function_prefix_error(),
        empty_stream(),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    chunks = [chunk async for chunk in llm.stream_chat(_tool_call_history())]

    assert chunks == []
    assert mock_client.chat.completions.create.await_count == 2
    retry_messages = mock_client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert retry_messages[1]["content"] == ""


@pytest.mark.asyncio
async def test_openrouter_does_not_retry_unrelated_bad_request(mocker, monkeypatch):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = _unrelated_bad_request()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    with pytest.raises(RuntimeError, match="Unrelated invalid request"):
        await llm.chat(_tool_call_history())

    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_openrouter_non_deepseek_does_not_retry_function_prefix_error(
    mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = _deepseek_function_prefix_error()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="openai/gpt-5.5",
        api_key="test-key",
    )

    with pytest.raises(RuntimeError, match="Function call should not be used"):
        await llm.chat(_tool_call_history())

    assert mock_client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_openrouter_stream_deepseek_uses_official_provider(mocker, monkeypatch):
    """Streaming calls should carry the same OpenRouter provider routing."""

    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "true")

    async def empty_stream():
        if False:
            yield None

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = empty_stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    _ = [
        chunk async for chunk in llm.stream_chat([{"role": "user", "content": "Hello"}])
    ]

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"]["provider"] == {
        "only": ["deepseek"],
        "allow_fallbacks": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thinking",
    [
        {"type": "disabled", "enable": False},
        {"type": "omit"},
    ],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek/deepseek-v4-flash",
        "openrouter/deepseek/deepseek-v4-flash",
    ],
)
async def test_openrouter_deepseek_stream_uses_disabled_thinking_payload(
    mocker, monkeypatch, thinking, model_name
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")

    async def empty_stream():
        if False:
            yield None

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = empty_stream()
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    llm = OpenRouterLLM(
        model_name=model_name,
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    _ = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "Hello"}],
            tool_choice="required",
            thinking=thinking,
        )
    ]

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"] == {
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }
    assert "enable_thinking" not in call_kwargs["extra_body"]
    assert call_kwargs["tool_choice"] == "required"


def test_openrouter_reasoning_hook_enables_reasoning_payload(monkeypatch):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    extra_body = llm._prepare_provider_reasoning_extra_body(
        extra_body={"trace_id": "abc", "enable_thinking": False},
        thinking={"type": "enabled"},
        tools=None,
        response_format=None,
        output_config=None,
        is_streaming=True,
    )

    assert extra_body == {
        "trace_id": "abc",
        "reasoning": {"enabled": True},
        "thinking": {"type": "enabled"},
    }


@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek/deepseek-v4-flash",
        "openrouter/deepseek/deepseek-v4-flash",
    ],
)
@pytest.mark.parametrize("is_streaming", [True, False])
@pytest.mark.parametrize("response_format", [None, {"type": "json_object"}])
def test_openrouter_deepseek_defaults_to_disabled_thinking(
    monkeypatch, model_name, is_streaming, response_format
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    llm = OpenRouterLLM(
        model_name=model_name,
        api_key="test-key",
        abilities=["chat", "tool_calling"],
    )

    extra_body = llm._prepare_provider_reasoning_extra_body(
        extra_body={"trace_id": "abc"},
        thinking=None,
        tools=None,
        response_format=response_format,
        output_config=None,
        is_streaming=is_streaming,
    )

    assert extra_body == {
        "trace_id": "abc",
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }


def test_openrouter_deepseek_structured_streaming_disables_thinking(monkeypatch):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling"],
    )

    extra_body = llm._prepare_provider_reasoning_extra_body(
        extra_body={},
        thinking=None,
        tools=None,
        response_format={"type": "json_object"},
        output_config=None,
        is_streaming=True,
    )

    assert extra_body == {
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }


@pytest.mark.parametrize("response_format", [None, {"type": "json_object"}])
def test_openrouter_non_deepseek_defaults_leave_thinking_unset(
    monkeypatch, response_format
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    llm = OpenRouterLLM(
        model_name="openai/gpt-5",
        api_key="test-key",
        abilities=["chat", "tool_calling"],
    )

    extra_body = llm._prepare_provider_reasoning_extra_body(
        extra_body={"trace_id": "abc"},
        thinking=None,
        tools=None,
        response_format=response_format,
        output_config=None,
        is_streaming=True,
    )

    assert extra_body == {"trace_id": "abc"}


@pytest.mark.parametrize(
    ("is_streaming", "expected_extra_body"),
    [
        (True, {"reasoning": {"enabled": False}, "thinking": {"type": "disabled"}}),
        (False, {}),
    ],
)
def test_openrouter_non_deepseek_thinking_structured_output_fork(
    monkeypatch, is_streaming, expected_extra_body
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")
    llm = OpenRouterLLM(
        model_name="openai/gpt-5",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    extra_body = llm._prepare_provider_reasoning_extra_body(
        extra_body={},
        thinking=None,
        tools=None,
        response_format={"type": "json_object"},
        output_config=None,
        is_streaming=is_streaming,
    )

    assert extra_body == expected_extra_body


@pytest.mark.asyncio
async def test_structured_output_retry_disables_openrouter_reasoning(
    mocker, monkeypatch
):
    monkeypatch.setenv("XAGENT_OPENROUTER_OFFICIAL_PROVIDERS_ONLY", "false")

    first_message = SimpleNamespace(
        content="not json",
        tool_calls=None,
        reasoning_content="reasoning here",
    )
    second_message = SimpleNamespace(
        content='{"status": "ok"}',
        tool_calls=None,
        reasoning_content=None,
    )
    first_response = SimpleNamespace(
        choices=[SimpleNamespace(message=first_message)],
        usage=None,
        model_dump=lambda: {"id": "openrouter-first"},
    )
    second_response = SimpleNamespace(
        choices=[SimpleNamespace(message=second_message)],
        usage=None,
        model_dump=lambda: {"id": "openrouter-second"},
    )

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [first_response, second_response]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )

    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
        abilities=["chat", "tool_calling", "thinking_mode"],
    )

    result = await llm.chat(
        [{"role": "user", "content": "Return JSON"}],
        response_format={"type": "json_object"},
        thinking={"type": "enabled"},
    )

    assert result["type"] == "text"
    assert result["content"] == '{"status": "ok"}'
    second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert second_call["extra_body"]["reasoning"] == {"enabled": False}
    assert second_call["extra_body"]["thinking"] == {"type": "disabled"}


# ==========================================================================
# Provider-compatibility retries owned by the client (direct-slug entrypoints)
# ==========================================================================

_MANDATORY_REASONING_ERROR = (
    "Reasoning is mandatory for this endpoint and cannot be disabled."
)
_THINKING_TOOL_CHOICE_ERROR = "Thinking mode does not support this tool_choice"
_OPENROUTER_TOOL_CHOICE_ERROR = (
    "No endpoints found that support the provided 'tool_choice' value."
)


def _two_tool_schema() -> list[dict]:
    return [
        _single_tool_schema("answer")[0],
        _single_tool_schema("skip")[0],
    ]


@pytest.mark.asyncio
async def test_openrouter_direct_relaxes_tool_choice_on_endpoint_404(
    mock_chat_completion, mocker
):
    """A direct (non-auto) OpenRouter slug retries a rejected tool_choice itself."""
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(_OPENROUTER_TOOL_CHOICE_ERROR),
        mock_chat_completion,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    result = await llm.chat(
        [{"role": "user", "content": "score?"}],
        tools=_two_tool_schema(),
        tool_choice="required",
    )

    assert result["content"] == "Hello World"
    assert mock_client.chat.completions.create.await_count == 2
    second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert second_call["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_openrouter_direct_does_not_repeat_mandatory_reasoning_retry(mocker):
    """Each compat action fires at most once per call, even across a 3-error run."""
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(_MANDATORY_REASONING_ERROR),
        RuntimeError(_THINKING_TOOL_CHOICE_ERROR),
        RuntimeError(_MANDATORY_REASONING_ERROR),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    with pytest.raises(RuntimeError, match="Reasoning is mandatory"):
        await llm.chat(
            [{"role": "user", "content": "score?"}],
            tools=_two_tool_schema(),
            tool_choice="required",
            thinking={"type": "disabled", "enable": False},
        )

    assert mock_client.chat.completions.create.await_count == 3
    thinking_values = [
        call.kwargs["extra_body"].get("thinking")
        for call in mock_client.chat.completions.create.call_args_list
    ]
    assert thinking_values == [
        {"type": "disabled"},
        {"type": "enabled"},
        {"type": "disabled"},
    ]


@pytest.mark.asyncio
async def test_openrouter_direct_chains_thinking_and_tool_choice_retries(mocker):
    """A thinking-conflict 400 followed by a tool_choice 404 chains two adjustments."""
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(_THINKING_TOOL_CHOICE_ERROR),
        RuntimeError(_OPENROUTER_TOOL_CHOICE_ERROR),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok", tool_calls=None, reasoning_content=None
                    )
                )
            ],
            usage=None,
            model_dump=lambda: {"id": "openrouter-chain"},
        ),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    result = await llm.chat(
        [{"role": "user", "content": "score?"}],
        tools=_two_tool_schema(),
        tool_choice="required",
        thinking={"type": "enabled", "enable": True},
    )

    assert result["content"] == "ok"
    tool_choices = [
        call.kwargs["tool_choice"]
        for call in mock_client.chat.completions.create.call_args_list
    ]
    assert tool_choices == ["required", "required", "auto"]


@pytest.mark.asyncio
async def test_openrouter_stream_relaxes_tool_choice_before_first_chunk(mocker):
    """Streaming retries the same compat rules while nothing has been yielded yet."""
    attempts = 0

    async def rejects_tool_choice():
        if False:
            yield None
        raise RuntimeError(_OPENROUTER_TOOL_CHOICE_ERROR)

    async def succeeds():
        yield StreamChunk(type=ChunkType.TOKEN, content="ok", delta="ok")

    def fake_stream(*_args, **kwargs):
        nonlocal attempts
        del kwargs
        attempts += 1
        return rejects_tool_choice() if attempts == 1 else succeeds()

    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")
    mocker.patch.object(llm, "_stream_chat_inner", side_effect=fake_stream)

    chunks = [
        chunk
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "score?"}],
            tools=_two_tool_schema(),
            tool_choice="required",
        )
    ]

    assert [chunk.delta for chunk in chunks] == ["ok"]
    assert attempts == 2


@pytest.mark.asyncio
async def test_openrouter_stream_error_after_first_chunk_not_retried(mocker):
    """Once a chunk has actually reached the caller, a later error is never retried."""
    attempts = 0

    async def yields_then_fails():
        nonlocal attempts
        attempts += 1
        yield StreamChunk(type=ChunkType.TOKEN, content="partial", delta="partial")
        raise RuntimeError(_OPENROUTER_TOOL_CHOICE_ERROR)

    def fake_stream(*_args, **kwargs):
        del kwargs
        return yields_then_fails()

    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")
    mocker.patch.object(llm, "_stream_chat_inner", side_effect=fake_stream)

    received = []
    with pytest.raises(RuntimeError, match="No endpoints found"):
        async for chunk in llm.stream_chat(
            [{"role": "user", "content": "score?"}],
            tools=_two_tool_schema(),
            tool_choice="required",
        ):
            received.append(chunk)

    assert [chunk.delta for chunk in received] == ["partial"]
    assert attempts == 1


@pytest.mark.asyncio
async def test_openrouter_compat_retry_does_not_catch_llm_retryable_error(mocker):
    """A retryable protocol error from the inner call is never a compat adjustment."""
    protocol_error = LLMToolProtocolError(
        provider="deepseek",
        code="malformed_tool_arguments",
        message="DeepSeek returned malformed arguments.",
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")
    prefix_retry_mock = mocker.patch.object(
        llm, "_chat_with_prefix_retry", side_effect=protocol_error
    )

    with pytest.raises(LLMToolProtocolError, match="malformed arguments"):
        await llm.chat([{"role": "user", "content": "score?"}])

    assert prefix_retry_mock.call_count == 1


@pytest.mark.asyncio
async def test_openrouter_direct_thinking_retry_changes_rendered_extra_body(mocker):
    """A thinking-rule retry (rules 1/2) must change what is actually sent."""
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(_MANDATORY_REASONING_ERROR),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok", tool_calls=None, reasoning_content=None
                    )
                )
            ],
            usage=None,
            model_dump=lambda: {"id": "openrouter-thinking"},
        ),
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(model_name="z-ai/glm-5.2", api_key="test-key")

    await llm.chat(
        [{"role": "user", "content": "score?"}],
        thinking={"type": "disabled", "enable": False},
    )

    extra_bodies = [
        call.kwargs["extra_body"]
        for call in mock_client.chat.completions.create.call_args_list
    ]
    assert extra_bodies[0] != extra_bodies[1]
    assert extra_bodies[0]["thinking"] == {"type": "disabled"}
    assert extra_bodies[1]["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_openrouter_vision_chat_retries_mandatory_reasoning(mocker):
    """vision_chat shares the same compat retry as chat, with no prefix retry."""
    success_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello World", tool_calls=None, reasoning_content=None
                )
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-vision"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(_MANDATORY_REASONING_ERROR),
        success_response,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="z-ai/glm-5.2",
        api_key="test-key",
        abilities=["chat", "tool_calling", "vision"],
    )

    result = await llm.vision_chat(
        [{"role": "user", "content": "describe this image"}],
        thinking={"type": "disabled", "enable": False},
    )

    assert result["content"] == "Hello World"
    assert mock_client.chat.completions.create.await_count == 2
    second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert second_call["extra_body"]["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_openrouter_deepseek_no_op_thinking_retry_falls_through_to_next_rule(
    mocker,
):
    """A no-op disable-thinking match is skipped so a real fix (enable) still fires."""
    combined_error = (
        "Reasoning is mandatory for this endpoint and cannot be disabled, "
        "and thinking conflicts with tool_choice here."
    )
    success_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ok", tool_calls=None, reasoning_content=None
                )
            )
        ],
        usage=None,
        model_dump=lambda: {"id": "openrouter-noop"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError(combined_error),
        success_response,
    ]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    llm = OpenRouterLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_key="test-key",
    )

    # deepseek's default (thinking=None) already renders as disabled, so the
    # "disable thinking" rule (2) would be a no-op against this error; the
    # aggregator must fall through to rule 1 (enable thinking) instead of
    # wasting the retry budget replaying an unchanged request.
    result = await llm.chat(
        [{"role": "user", "content": "score?"}],
        tools=_two_tool_schema(),
        tool_choice="required",
    )

    assert result["content"] == "ok"
    assert mock_client.chat.completions.create.await_count == 2
    second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert second_call["extra_body"]["thinking"] == {"type": "enabled"}
