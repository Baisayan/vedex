from collections.abc import AsyncIterator, Mapping
from json import loads

import httpx
import pytest

from agent import AgentTool, AgentToolResult, SimpleCancellationToken, ToolCall, UserMessage
from agent.types import JSONValue
from ai import (
    FakeProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    ProviderErrorEvent,
    ProviderResponseEndEvent,
    ProviderRetryEvent,
    ProviderThinkingDeltaEvent,
    ProviderToolCallEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
    openai_compatible_config_from_env,
)


async def _collect(stream: AsyncIterator[object]) -> list[object]:
    return [event async for event in stream]


@pytest.mark.anyio
async def test_fake_provider_replays_scripted_events() -> None:
    scripted = [
        ProviderResponseStartEvent(model="fake-model"),
        ProviderTextDeltaEvent(delta="hello"),
        ProviderResponseEndEvent(message={"role": "assistant", "content": "hello"}),
    ]
    provider = FakeProvider([scripted])

    events = await _collect(
        provider.stream_response(
            model="fake-model",
            system="system prompt",
            messages=[UserMessage(content="hi")],
            tools=[],
        )
    )

    assert events == scripted
    assert provider.calls[0][0] == "fake-model"
    assert provider.calls[0][1] == "system prompt"


def test_openai_compatible_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "2")
    monkeypatch.setenv("OPENAI_MAX_RETRY_DELAY_SECONDS", "0.25")

    config = openai_compatible_config_from_env()

    assert config.api_key == "test-key"
    assert config.base_url == "https://example.test/v1"
    assert config.timeout_seconds == 12.5
    assert config.max_retries == 2
    assert config.max_retry_delay_seconds == 0.25


def test_openai_compatible_config_from_env_rejects_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "0")

    with pytest.raises(RuntimeError, match="greater than 0"):
        openai_compatible_config_from_env()


def test_openai_compatible_config_from_env_rejects_invalid_retry_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "-1")

    with pytest.raises(RuntimeError, match="0 or greater"):
        openai_compatible_config_from_env()


@pytest.mark.anyio
async def test_openai_compatible_provider_uses_configured_timeout() -> None:
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            api_key="test-key",
            base_url="https://example.test/v1",
            timeout_seconds=7.5,
        )
    )
    try:
        client = provider._get_client()

        assert client.timeout.connect == 7.5
        assert client.timeout.read == 7.5
    finally:
        await provider.aclose()


@pytest.mark.anyio
async def test_openai_compatible_provider_formats_request_and_streams_text() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                headers={"X-HF-Bill-To": "my-org"},
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Vedex.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    assert [event.type for event in events] == [
        "response_start",
        "text_delta",
        "text_delta",
        "response_end",
    ]
    assert isinstance(events[-1], ProviderResponseEndEvent)
    assert events[-1].message.content == "Hello"
    assert events[-1].finish_reason == "stop"

    request = requests[0]
    assert request.url == "https://example.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-hf-bill-to"] == "my-org"

    payload = loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["stream"] is True
    assert "reasoning_effort" not in payload
    assert payload["messages"] == [
        {"role": "system", "content": "You are Vedex."},
        {"role": "user", "content": "Say hello"},
    ]


@pytest.mark.anyio
async def test_openai_compatible_provider_includes_configured_reasoning_effort() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                reasoning_effort="high",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Vedex.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert isinstance(events[-1], ProviderResponseEndEvent)
    assert loads(requests[0].content)["reasoning_effort"] == "high"


@pytest.mark.anyio
async def test_openai_compatible_provider_can_send_responses_reasoning_effort() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                reasoning_effort="high",
                reasoning_effort_parameter="reasoning.effort",
            ),
            client=client,
        )

        await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Vedex.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert loads(requests[0].content)["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in loads(requests[0].content)


@pytest.mark.anyio
async def test_openai_compatible_provider_streams_reasoning_content() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"reasoning_content":"plan "}}]}\n\n'
                'data: {"choices":[{"delta":{"reasoning_content":"steps"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", base_url="https://example.test/v1"),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Vedex.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert [event.type for event in events] == [
        "response_start",
        "thinking_delta",
        "thinking_delta",
        "text_delta",
        "response_end",
    ]
    thinking_events = [event for event in events if isinstance(event, ProviderThinkingDeltaEvent)]
    assert [event.delta for event in thinking_events] == ["plan ", "steps"]
    assert isinstance(events[-1], ProviderResponseEndEvent)
    assert events[-1].message.content == "done"


@pytest.mark.anyio
async def test_openai_compatible_provider_streams_tool_calls() -> None:
    async def executor(
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
    ) -> AgentToolResult:
        del signal
        return AgentToolResult(
            tool_call_id="call-1",
            name="read",
            ok=True,
            content=str(arguments),
        )

    tool = AgentTool(
        name="read",
        description="Read a file.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        executor=executor,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = loads(request.content)
        assert payload["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file.",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1",'
                '"function":{"name":"read","arguments":"{\\"path\\":"}}]}}]}\n\n'
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"\\"README.md\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", base_url="https://example.test/v1"),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Vedex.",
                messages=[UserMessage(content="Read README.md")],
                tools=[tool],
            )
        )

    tool_call_events = [event for event in events if isinstance(event, ProviderToolCallEvent)]

    assert tool_call_events == [
        ProviderToolCallEvent(
            tool_call=ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
        )
    ]
    assert isinstance(events[-1], ProviderResponseEndEvent)
    assert events[-1].message.tool_calls == [
        ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    ]
    assert events[-1].finish_reason == "tool_calls"


@pytest.mark.anyio
async def test_openai_compatible_provider_retries_transient_status() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(500, text="try again")
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                max_retries=1,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Vedex.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert len(requests) == 2
    assert isinstance(events[0], ProviderRetryEvent)
    assert events[0].attempt == 2
    assert events[0].max_attempts == 2
    assert events[0].delay_seconds == 0
    assert events[0].data == {"status_code": 500, "body": "try again"}
    assert [event.type for event in events] == [
        "retry",
        "response_start",
        "text_delta",
        "response_end",
    ]


@pytest.mark.anyio
async def test_openai_compatible_provider_cancellation_stops_retry_backoff() -> None:
    requests: list[httpx.Request] = []
    signal = SimpleCancellationToken()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, text="try later")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                max_retries=2,
                max_retry_delay_seconds=1,
            ),
            client=client,
        )

        events: list[object] = []
        async for event in provider.stream_response(
            model="test-model",
            system="You are Vedex.",
            messages=[UserMessage(content="Say ok")],
            tools=[],
            signal=signal,
        ):
            events.append(event)
            if isinstance(event, ProviderRetryEvent):
                signal.cancel()

    assert len(requests) == 1
    assert [event.type for event in events] == ["retry"]


@pytest.mark.anyio
async def test_openai_compatible_provider_does_not_retry_non_transient_status() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(400, text="bad request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                max_retries=3,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Vedex.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert len(requests) == 1
    assert isinstance(events[-1], ProviderErrorEvent)
    assert events[-1].data == {"body": "bad request", "attempts": 1}
