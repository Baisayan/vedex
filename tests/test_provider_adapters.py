from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterable
from typing import cast

import pytest
import vedex.models.openai as openai_module
from google.genai import interactions
from google.genai import types as gemini_types
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
)
from pydantic import SecretStr
from vedex.cli import load_adapter
from vedex.models import (
    ModelAdapter,
    ModelCancelledEvent,
    ModelCompletedEvent,
    ModelEvent,
    ModelFailureEvent,
    ModelRequest,
    ModelSettings,
    ModelStartEvent,
    ModelTextDeltaEvent,
    ModelThinkingDeltaEvent,
    ModelToolDefinition,
)
from vedex.models.gemini import GeminiAdapter, GeminiAdapterConfig
from vedex.models.openai import OpenAIAdapter, OpenAIAdapterConfig
from vedex.schema import (
    AgentMessage,
    CancellationToken,
    JSONValue,
    ToolResultMessage,
    UserMessage,
)

from .conftest import run_async


class _FakeStream[Event]:
    def __init__(self, events: Iterable[Event]) -> None:
        self._events = deque(events)
        self.closed = False

    def __aiter__(self) -> _FakeStream[Event]:
        return self

    async def __anext__(self) -> Event:
        if not self._events:
            raise StopAsyncIteration
        return self._events.popleft()

    async def close(self) -> None:
        self.closed = True


class _BlockingStream:
    def __init__(self) -> None:
        self.closed = False
        self._never = asyncio.Event()

    def __aiter__(self) -> _BlockingStream:
        return self

    async def __anext__(self) -> object:
        await self._never.wait()
        raise StopAsyncIteration

    async def close(self) -> None:
        self.closed = True


class _FakeEndpoint:
    def __init__(self, result: object | Exception, calls: list[dict[str, object]]) -> None:
        self._result = result
        self._calls = calls

    async def create(self, **kwargs: object) -> object:
        self._calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeOpenAIClient:
    def __init__(self, result: object | Exception, calls: list[dict[str, object]]) -> None:
        self.responses = _FakeEndpoint(result, calls)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _OpenAIHarness:
    def __init__(self, results: Iterable[object | Exception]) -> None:
        self._results = deque(results)
        self.calls: list[dict[str, object]] = []
        self.client_kwargs: list[dict[str, object]] = []
        self.clients: list[_FakeOpenAIClient] = []

    def factory(self, **kwargs: object) -> _FakeOpenAIClient:
        self.client_kwargs.append(kwargs)
        client = _FakeOpenAIClient(self._results.popleft(), self.calls)
        self.clients.append(client)
        return client


class _FakeGeminiAsyncClient:
    def __init__(self, result: object | Exception, calls: list[dict[str, object]]) -> None:
        self.interactions = _FakeEndpoint(result, calls)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FakeGeminiClient:
    def __init__(self, result: object | Exception, calls: list[dict[str, object]]) -> None:
        self.aio = _FakeGeminiAsyncClient(result, calls)


class _GeminiHarness:
    def __init__(self, results: Iterable[object | Exception]) -> None:
        self._results = deque(results)
        self.calls: list[dict[str, object]] = []
        self.client_kwargs: list[dict[str, object]] = []
        self.clients: list[_FakeGeminiClient] = []

    def factory(self, **kwargs: object) -> _FakeGeminiClient:
        self.client_kwargs.append(kwargs)
        client = _FakeGeminiClient(self._results.popleft(), self.calls)
        self.clients.append(client)
        return client


class _DumpedEvent:
    def __init__(self, value: dict[str, JSONValue]) -> None:
        self._value = value

    def model_dump(self, **_kwargs: object) -> dict[str, JSONValue]:
        return self._value


class _MutableCancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


class _ProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: int | None = None,
        status: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.status = status


async def _collect(
    adapter: ModelAdapter,
    request: ModelRequest,
    *,
    signal: CancellationToken | None = None,
) -> list[ModelEvent]:
    return [event async for event in adapter.stream(request, signal=signal)]


def _request(
    *,
    model: str,
    messages: list[AgentMessage] | None = None,
    options: dict[str, JSONValue] | None = None,
) -> ModelRequest:
    return ModelRequest(
        system="You are a coding agent.",
        messages=messages or [UserMessage(content="Read README.md")],
        tools=[
            ModelToolDefinition(
                name="read",
                description="Read a workspace file",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            )
        ],
        settings=ModelSettings(model=model, options=options or {}),
    )


def _openai_completed(
    *,
    content: str,
    include_tool_call: bool,
) -> ResponseCompletedEvent:
    output: list[dict[str, object]] = [
        {
            "id": "message-1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        }
    ]
    if include_tool_call:
        output.append(
            {
                "id": "item-call-1",
                "type": "function_call",
                "call_id": "call-1",
                "name": "read",
                "arguments": '{"path":"README.md"}',
                "status": "completed",
            }
        )
    response = Response.model_validate(
        {
            "id": "response-1",
            "created_at": 1.0,
            "model": "gpt-5.4-mini",
            "object": "response",
            "output": output,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "usage": {
                "input_tokens": 12,
                "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 0},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 2},
                "total_tokens": 17,
            },
        }
    )
    return ResponseCompletedEvent(
        response=response,
        sequence_number=3,
        type="response.completed",
    )


def _gemini_completed(
    *,
    content: str,
    include_tool_call: bool,
) -> interactions.InteractionCompletedEvent:
    steps: list[dict[str, object]] = [
        {
            "type": "thought",
            "signature": "c2lnbmF0dXJl",
            "summary": [{"type": "text", "text": "Inspect the file."}],
        },
        {
            "type": "model_output",
            "content": [{"type": "text", "text": content}],
        },
    ]
    if include_tool_call:
        steps.append(
            {
                "type": "function_call",
                "id": "call-1",
                "name": "read",
                "arguments": {"path": "README.md"},
            }
        )
    interaction = interactions.InteractionSseEventInteraction.model_validate(
        {
            "id": "interaction-1",
            "status": "completed",
            "model": "gemini-3.7-flash",
            "steps": steps,
            "usage": {
                "total_input_tokens": 20,
                "total_output_tokens": 7,
                "total_cached_tokens": 4,
                "total_thought_tokens": 3,
                "total_tokens": 30,
            },
        }
    )
    return interactions.InteractionCompletedEvent(
        event_type="interaction.completed",
        interaction=interaction,
    )


def test_openai_adapter_streams_and_replays_stateless_tool_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_stream = _FakeStream(
        [
            ResponseTextDeltaEvent(
                content_index=0,
                delta="Reading.",
                item_id="message-1",
                logprobs=[],
                output_index=0,
                sequence_number=1,
                type="response.output_text.delta",
            ),
            ResponseReasoningSummaryTextDeltaEvent(
                delta="Need the file.",
                item_id="reasoning-1",
                output_index=0,
                sequence_number=2,
                summary_index=0,
                type="response.reasoning_summary_text.delta",
            ),
            _openai_completed(content="Reading.", include_tool_call=True),
        ]
    )
    second_stream = _FakeStream([_openai_completed(content="Done.", include_tool_call=False)])
    harness = _OpenAIHarness([first_stream, second_stream])
    monkeypatch.setattr(openai_module, "AsyncOpenAI", harness.factory)
    adapter = OpenAIAdapter(
        OpenAIAdapterConfig(
            api_key=SecretStr("test-key"),
            cancellation_poll_seconds=0.001,
            default_options={"temperature": 1},
        )
    )

    first_events = run_async(
        _collect(
            adapter,
            _request(model="gpt-5.4-mini", options={"temperature": 0}),
        )
    )

    assert first_events[:3] == [
        ModelStartEvent(),
        ModelTextDeltaEvent(delta="Reading."),
        ModelThinkingDeltaEvent(delta="Need the file."),
    ]
    completed = first_events[-1]
    assert isinstance(completed, ModelCompletedEvent)
    assert completed.message.content == "Reading."
    assert completed.message.tool_calls[0].model_dump() == {
        "id": "call-1",
        "name": "read",
        "arguments": {"path": "README.md"},
    }
    assert completed.usage.model_dump() == {
        "input_tokens": 12,
        "output_tokens": 5,
        "cached_tokens": 3,
        "thinking_tokens": 2,
    }
    openai_metadata = cast(dict[str, JSONValue], completed.message.metadata["openai"])
    assert openai_metadata["response_id"] == "response-1"
    assert openai_metadata["model"] == "gpt-5.4-mini"
    assert isinstance(openai_metadata["output"], list)

    follow_up = _request(
        model="gpt-5.4-mini",
        messages=[
            UserMessage(content="Read README.md"),
            completed.message,
            ToolResultMessage(
                tool_call_id="call-1",
                name="read",
                content="# Vedex",
            ),
        ],
    )
    second_events = run_async(_collect(adapter, follow_up))

    assert isinstance(second_events[-1], ModelCompletedEvent)
    first_call = harness.calls[0]
    assert first_call["instructions"] == "You are a coding agent."
    assert first_call["model"] == "gpt-5.4-mini"
    assert first_call["stream"] is True
    assert first_call["store"] is False
    assert first_call["parallel_tool_calls"] is False
    assert first_call["temperature"] == 0
    assert first_call["include"] == ["reasoning.encrypted_content"]
    assert first_call["tools"] == [
        {
            "type": "function",
            "name": "read",
            "description": "Read a workspace file",
            "parameters": _request(model="x").tools[0].input_schema,
            "strict": False,
        }
    ]
    second_input = cast(list[dict[str, object]], harness.calls[1]["input"])
    assert [item["type"] for item in second_input[1:]] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    assert second_input[-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "# Vedex",
    }
    assert harness.client_kwargs[0]["max_retries"] == 2
    assert first_stream.closed and second_stream.closed
    assert all(client.closed for client in harness.clients)


def test_gemini_adapter_streams_and_replays_stateless_tool_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_stream = _FakeStream(
        [
            interactions.StepDelta.model_validate(
                {
                    "event_type": "step.delta",
                    "index": 0,
                    "delta": {"type": "text", "text": "Reading."},
                }
            ),
            interactions.StepDelta.model_validate(
                {
                    "event_type": "step.delta",
                    "index": 0,
                    "delta": {
                        "type": "thought_summary",
                        "content": {"type": "text", "text": "Need the file."},
                    },
                }
            ),
            interactions.StepStart.model_validate(
                {
                    "event_type": "step.start",
                    "index": 1,
                    "step": {
                        "type": "function_call",
                        "id": "call-1",
                        "name": "read",
                        "arguments": {},
                    },
                }
            ),
            interactions.StepDelta.model_validate(
                {
                    "event_type": "step.delta",
                    "index": 1,
                    "delta": {
                        "type": "arguments_delta",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ),
            _gemini_completed(content="Reading.", include_tool_call=False),
        ]
    )
    second_stream = _FakeStream([_gemini_completed(content="Done.", include_tool_call=False)])
    harness = _GeminiHarness([first_stream, second_stream])
    monkeypatch.setattr("vedex.models.gemini.genai.Client", harness.factory)
    adapter = GeminiAdapter(
        GeminiAdapterConfig(
            api_key=SecretStr("test-key"),
            cancellation_poll_seconds=0.001,
            default_options={"max_output_tokens": 100},
        )
    )

    first_events = run_async(
        _collect(
            adapter,
            _request(model="gemini-3.7-flash", options={"max_output_tokens": 50}),
        )
    )

    assert first_events[:3] == [
        ModelStartEvent(),
        ModelTextDeltaEvent(delta="Reading."),
        ModelThinkingDeltaEvent(delta="Need the file."),
    ]
    completed = first_events[-1]
    assert isinstance(completed, ModelCompletedEvent)
    assert completed.message.content == "Reading."
    assert completed.message.tool_calls[0].model_dump() == {
        "id": "call-1",
        "name": "read",
        "arguments": {"path": "README.md"},
    }
    assert completed.usage.model_dump() == {
        "input_tokens": 20,
        "output_tokens": 7,
        "cached_tokens": 4,
        "thinking_tokens": 3,
    }

    follow_up = _request(
        model="gemini-3.7-flash",
        messages=[
            UserMessage(content="Read README.md"),
            completed.message,
            ToolResultMessage(
                tool_call_id="call-1",
                name="read",
                content="# Vedex",
            ),
        ],
    )
    second_events = run_async(_collect(adapter, follow_up))

    assert isinstance(second_events[-1], ModelCompletedEvent)
    first_call = harness.calls[0]
    assert first_call["system_instruction"] == "You are a coding agent."
    assert first_call["model"] == "gemini-3.7-flash"
    assert first_call["stream"] is True
    assert first_call["store"] is False
    assert first_call["generation_config"] == {"max_output_tokens": 50}
    assert first_call["tools"] == [
        {
            "type": "function",
            "name": "read",
            "description": "Read a workspace file",
            "parameters": _request(model="x").tools[0].input_schema,
        }
    ]
    second_input = cast(list[dict[str, object]], harness.calls[1]["input"])
    assert [item["type"] for item in second_input] == [
        "user_input",
        "thought",
        "model_output",
        "function_call",
        "function_result",
    ]
    assert second_input[-1] == {
        "type": "function_result",
        "call_id": "call-1",
        "name": "read",
        "is_error": False,
        "result": [{"type": "text", "text": "# Vedex"}],
    }
    http_options = cast(gemini_types.HttpOptions, harness.client_kwargs[0]["http_options"])
    assert http_options.api_version == "v1beta"
    assert http_options.timeout == 120_000
    assert http_options.retry_options is not None
    assert http_options.retry_options.attempts == 5
    assert first_stream.closed and second_stream.closed
    assert all(client.aio.closed for client in harness.clients)


@pytest.mark.parametrize(
    ("adapter_name", "event", "expected_kind", "retryable"),
    [
        (
            "openai",
            _DumpedEvent(
                {
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "code": "context_length_exceeded",
                            "message": "Context window exceeded",
                        }
                    },
                }
            ),
            "context_overflow",
            False,
        ),
        (
            "gemini",
            _DumpedEvent(
                {
                    "event_type": "error",
                    "error": {"code": "RESOURCE_EXHAUSTED", "message": "Quota exhausted"},
                }
            ),
            "rate_limit",
            True,
        ),
    ],
)
def test_provider_stream_errors_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
    adapter_name: str,
    event: object,
    expected_kind: str,
    retryable: bool,
) -> None:
    provider_stream = _FakeStream([event])
    if adapter_name == "openai":
        harness = _OpenAIHarness([provider_stream])
        monkeypatch.setattr(openai_module, "AsyncOpenAI", harness.factory)
        adapter: ModelAdapter = OpenAIAdapter(OpenAIAdapterConfig(api_key=SecretStr("test")))
        request = _request(model="gpt-5.4-mini")
    else:
        gemini_harness = _GeminiHarness([provider_stream])
        monkeypatch.setattr("vedex.models.gemini.genai.Client", gemini_harness.factory)
        adapter = GeminiAdapter(GeminiAdapterConfig(api_key=SecretStr("test")))
        request = _request(model="gemini-3.7-flash")

    events = run_async(_collect(adapter, request))

    assert events[0] == ModelStartEvent()
    failure = events[-1]
    assert isinstance(failure, ModelFailureEvent)
    assert failure.kind == expected_kind
    assert failure.retryable is retryable
    assert provider_stream.closed


@pytest.mark.parametrize("adapter_name", ["openai", "gemini"])
def test_provider_transport_errors_and_invalid_options_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
    adapter_name: str,
) -> None:
    if adapter_name == "openai":
        harness = _OpenAIHarness([_ProviderError("slow down", status_code=429)])
        monkeypatch.setattr(openai_module, "AsyncOpenAI", harness.factory)
        adapter: ModelAdapter = OpenAIAdapter(OpenAIAdapterConfig(api_key=SecretStr("test")))
        model = "gpt-5.4-mini"
        expected_kind = "rate_limit"
    else:
        gemini_harness = _GeminiHarness(
            [_ProviderError("invalid key", code=401, status="UNAUTHENTICATED")]
        )
        monkeypatch.setattr("vedex.models.gemini.genai.Client", gemini_harness.factory)
        adapter = GeminiAdapter(GeminiAdapterConfig(api_key=SecretStr("test")))
        model = "gemini-3.7-flash"
        expected_kind = "authentication"

    transport_events = run_async(_collect(adapter, _request(model=model)))
    invalid_events = run_async(
        _collect(adapter, _request(model=model, options={"unsupported_option": True}))
    )

    transport_failure = transport_events[-1]
    assert isinstance(transport_failure, ModelFailureEvent)
    assert transport_failure.kind == expected_kind
    invalid_failure = invalid_events[-1]
    assert isinstance(invalid_failure, ModelFailureEvent)
    assert invalid_failure.kind == "invalid_request"


@pytest.mark.parametrize("adapter_name", ["openai", "gemini"])
def test_provider_adapter_cancels_blocked_stream_and_closes_clients(
    monkeypatch: pytest.MonkeyPatch,
    adapter_name: str,
) -> None:
    provider_stream = _BlockingStream()
    token = _MutableCancellationToken()
    if adapter_name == "openai":
        harness = _OpenAIHarness([provider_stream])
        monkeypatch.setattr(openai_module, "AsyncOpenAI", harness.factory)
        adapter: ModelAdapter = OpenAIAdapter(
            OpenAIAdapterConfig(
                api_key=SecretStr("test"),
                cancellation_poll_seconds=0.001,
            )
        )
        request = _request(model="gpt-5.4-mini")
    else:
        gemini_harness = _GeminiHarness([provider_stream])
        monkeypatch.setattr("vedex.models.gemini.genai.Client", gemini_harness.factory)
        adapter = GeminiAdapter(
            GeminiAdapterConfig(
                api_key=SecretStr("test"),
                cancellation_poll_seconds=0.001,
            )
        )
        request = _request(model="gemini-3.7-flash")

    async def cancel_blocked_stream() -> list[ModelEvent]:
        stream = adapter.stream(request, signal=token)
        events = [await anext(stream)]
        pending: asyncio.Task[ModelEvent] = asyncio.create_task(_next_model_event(stream))
        await asyncio.sleep(0.01)
        token.cancel()
        events.append(await asyncio.wait_for(pending, timeout=1))
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        return events

    events = run_async(cancel_blocked_stream())

    assert events == [ModelStartEvent(), ModelCancelledEvent()]
    assert provider_stream.closed


async def _next_model_event(stream: AsyncIterator[ModelEvent]) -> ModelEvent:
    return await anext(stream)


def test_provider_factories_load_through_the_existing_cli_boundary() -> None:
    assert isinstance(load_adapter("vedex.models.openai:create_adapter"), OpenAIAdapter)
    assert isinstance(load_adapter("vedex.models.gemini:create_adapter"), GeminiAdapter)
