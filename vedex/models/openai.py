from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from typing import Protocol, cast

from openai import AsyncOpenAI
from openai.types.responses import ResponseInputParam, ToolParam
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError

from ..schema import (
    AssistantMessage,
    CancellationToken,
    JSONValue,
    ToolCall,
    UserMessage,
)
from .base import (
    ModelCancelledEvent,
    ModelCompletedEvent,
    ModelEvent,
    ModelFailureEvent,
    ModelFailureKind,
    ModelRequest,
    ModelStartEvent,
    ModelTextDeltaEvent,
    ModelThinkingDeltaEvent,
    Usage,
)

_JSON_VALUE_ADAPTER: TypeAdapter[JSONValue] = TypeAdapter(JSONValue)
_CONTEXT_CODES = {
    "context_length_exceeded",
    "context_window_exceeded",
    "input_too_large",
}


class OpenAIAdapterConfig(BaseModel):
    """OpenAI credentials, transport policy, and default request settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr | None = None
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    cancellation_poll_seconds: float = Field(default=0.1, gt=0)
    store: bool = False
    parallel_tool_calls: bool = False
    strict_tools: bool = False
    include_reasoning_metadata: bool = True
    default_options: dict[str, JSONValue] = Field(default_factory=dict)


class _OpenAIOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_output_tokens: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=1)
    prompt_cache_key: str | None = None
    reasoning: dict[str, JSONValue] | None = None
    safety_identifier: str | None = None
    service_tier: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    text: dict[str, JSONValue] | None = None
    tool_choice: str | dict[str, JSONValue] | None = None
    top_p: float | None = Field(default=None, ge=0, le=1)
    truncation: str | None = None


class _ClosableAsyncStream(Protocol):
    def __aiter__(self) -> AsyncIterator[object]: ...

    async def close(self) -> None: ...


class _StreamCancelled(Exception):
    pass


class OpenAIAdapter:
    """OpenAI Responses API implementation of Vedex's normalized model boundary."""

    def __init__(self, config: OpenAIAdapterConfig | None = None) -> None:
        self._config = config or OpenAIAdapterConfig()

    @property
    def config(self) -> OpenAIAdapterConfig:
        return self._config.model_copy(deep=True)

    def stream(
        self,
        request: ModelRequest,
        *,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ModelEvent]:
        async def run() -> AsyncIterator[ModelEvent]:
            if _is_cancelled(signal):
                yield ModelCancelledEvent()
                return

            client: AsyncOpenAI | None = None
            provider_stream: _ClosableAsyncStream | None = None
            terminal: ModelCompletedEvent | ModelFailureEvent | ModelCancelledEvent | None = None
            cleanup_error: Exception | None = None

            try:
                params = self._request_params(request)
                client = AsyncOpenAI(
                    api_key=(
                        self._config.api_key.get_secret_value()
                        if self._config.api_key is not None
                        else None
                    ),
                    base_url=self._config.base_url,
                    organization=self._config.organization,
                    project=self._config.project,
                    timeout=self._config.timeout_seconds,
                    max_retries=self._config.max_retries,
                )
                raw_stream = await client.responses.create(**params)
                provider_stream = cast(_ClosableAsyncStream, raw_stream)
                yield ModelStartEvent()

                async for raw_event in _iter_with_cancellation(
                    provider_stream,
                    signal=signal,
                    poll_seconds=self._config.cancellation_poll_seconds,
                ):
                    event = _object(raw_event)
                    event_type = _optional_string(event.get("type"))

                    if event_type == "response.output_text.delta":
                        yield ModelTextDeltaEvent(delta=_required_string(event, "delta"))
                    elif event_type in {
                        "response.reasoning_summary_text.delta",
                        "response.reasoning_text.delta",
                    }:
                        yield ModelThinkingDeltaEvent(delta=_required_string(event, "delta"))
                    elif event_type == "response.completed":
                        terminal = _completed_event(event)
                        break
                    elif event_type == "response.failed":
                        terminal = _failure_from_response_event(event)
                        break
                    elif event_type == "response.incomplete":
                        terminal = _incomplete_response_event(event)
                        break
                    elif event_type == "response.cancelled":
                        terminal = ModelCancelledEvent(message="OpenAI response cancelled")
                        break
                    elif event_type == "error":
                        terminal = _failure_from_error_object(event.get("error", event))
                        break
            except _StreamCancelled:
                terminal = ModelCancelledEvent()
            except Exception as exc:
                terminal = _failure_from_exception(exc)
            finally:
                if provider_stream is not None:
                    try:
                        await provider_stream.close()
                    except Exception as exc:
                        cleanup_error = exc
                if client is not None:
                    try:
                        await client.close()
                    except Exception as exc:
                        cleanup_error = cleanup_error or exc

            if cleanup_error is not None and not isinstance(terminal, ModelCancelledEvent):
                terminal = ModelFailureEvent(
                    kind="unavailable",
                    message=f"OpenAI client cleanup failed: {_error_message(cleanup_error)}",
                    retryable=True,
                )
            if terminal is None:
                terminal = ModelFailureEvent(
                    message="OpenAI stream ended without a terminal response",
                )
            yield terminal

        return run()

    def _request_params(self, request: ModelRequest) -> ResponseCreateParamsStreaming:
        merged_options = dict(self._config.default_options)
        merged_options.update(request.settings.options)
        options = _OpenAIOptions.model_validate(merged_options).model_dump(exclude_none=True)

        values: dict[str, object] = {
            "model": request.settings.model,
            "input": cast(ResponseInputParam, _request_input(request)),
            "instructions": request.system,
            "parallel_tool_calls": self._config.parallel_tool_calls,
            "store": self._config.store,
            "stream": True,
        }
        if request.tools:
            values["tools"] = cast(list[ToolParam], _request_tools(request, self._config))
        if self._config.include_reasoning_metadata:
            values["include"] = ["reasoning.encrypted_content"]
        values.update(options)
        return cast(ResponseCreateParamsStreaming, cast(object, values))


def create_adapter() -> OpenAIAdapter:
    """Create an environment-configured adapter for CLI factory loading."""

    return OpenAIAdapter()


def _request_input(request: ModelRequest) -> list[dict[str, JSONValue]]:
    provider_input: list[dict[str, JSONValue]] = []
    for message in request.messages:
        if isinstance(message, UserMessage):
            provider_input.append({"role": "user", "content": message.content})
            continue

        if isinstance(message, AssistantMessage):
            metadata = message.metadata.get("openai")
            if isinstance(metadata, dict):
                output = metadata.get("output")
                if isinstance(output, list) and output:
                    provider_input.extend(_object(item) for item in output)
                    continue

            if message.content:
                provider_input.append({"role": "assistant", "content": message.content})
            provider_input.extend(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, separators=(",", ":")),
                }
                for call in message.tool_calls
            )
            continue

        provider_input.append(
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id,
                "output": message.content,
            }
        )

    return provider_input


def _request_tools(
    request: ModelRequest,
    config: OpenAIAdapterConfig,
) -> list[dict[str, JSONValue]]:
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": config.strict_tools,
        }
        for tool in request.tools
    ]


def _completed_event(event: Mapping[str, JSONValue]) -> ModelCompletedEvent | ModelFailureEvent:
    response_value = event.get("response")
    try:
        response = _object(response_value)
        output = _object_list(response.get("output", []))
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for item in output:
            item_type = _optional_string(item.get("type"))
            if item_type == "message":
                for part in _object_list(item.get("content", [])):
                    part_type = _optional_string(part.get("type"))
                    if part_type in {"output_text", "refusal"}:
                        text = _optional_string(part.get("text")) or _optional_string(
                            part.get("refusal")
                        )
                        if text:
                            content_parts.append(text)
            elif item_type == "function_call":
                arguments_text = _optional_string(item.get("arguments")) or "{}"
                arguments = _object(_json_loads(arguments_text))
                call_id = _optional_string(item.get("call_id")) or _required_string(item, "id")
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        name=_required_string(item, "name"),
                        arguments=arguments,
                    )
                )

        metadata: dict[str, JSONValue] = {"output": cast(JSONValue, output)}
        response_id = _optional_string(response.get("id"))
        model = _optional_string(response.get("model"))
        if response_id is not None:
            metadata["response_id"] = response_id
        if model is not None:
            metadata["model"] = model

        return ModelCompletedEvent(
            message=AssistantMessage(
                content="".join(content_parts),
                tool_calls=tool_calls,
                metadata={"openai": metadata},
            ),
            usage=_usage(response.get("usage")),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return ModelFailureEvent(
            kind="invalid_request",
            message=f"Malformed OpenAI completed response: {_error_message(exc)}",
        )


def _usage(value: JSONValue) -> Usage:
    if value is None:
        return Usage()
    usage = _object(value)
    input_details = _optional_object(usage.get("input_tokens_details"))
    output_details = _optional_object(usage.get("output_tokens_details"))
    return Usage(
        input_tokens=_nonnegative_int(usage.get("input_tokens")),
        output_tokens=_nonnegative_int(usage.get("output_tokens")),
        cached_tokens=_nonnegative_int(input_details.get("cached_tokens")),
        thinking_tokens=_optional_nonnegative_int(output_details.get("reasoning_tokens")),
    )


def _failure_from_response_event(event: Mapping[str, JSONValue]) -> ModelFailureEvent:
    response = _optional_object(event.get("response"))
    return _failure_from_error_object(response.get("error"))


def _incomplete_response_event(event: Mapping[str, JSONValue]) -> ModelFailureEvent:
    response = _optional_object(event.get("response"))
    details = _optional_object(response.get("incomplete_details"))
    reason = _optional_string(details.get("reason")) or "unknown reason"
    return ModelFailureEvent(
        kind="invalid_request",
        message=f"OpenAI response was incomplete: {reason}",
    )


def _failure_from_error_object(value: JSONValue) -> ModelFailureEvent:
    error = _optional_object(value)
    code = _optional_string(error.get("code"))
    message = _optional_string(error.get("message")) or "OpenAI request failed"
    return _failure(code=code, message=message, status_code=None)


def _failure_from_exception(error: Exception) -> ModelFailureEvent:
    if isinstance(error, (TypeError, ValueError, ValidationError)):
        return ModelFailureEvent(
            kind="invalid_request",
            message=_error_message(error),
        )

    status_code = getattr(error, "status_code", None)
    if not isinstance(status_code, int):
        status_code = None

    body_candidate: object = getattr(error, "body", None)
    code = _error_code_from_body(body_candidate)

    message = _error_message(error)
    if code is None and "api key" in message.lower():
        code = "authentication_error"
    if isinstance(error, TimeoutError):
        status_code = 408
    return _failure(code=code, message=message, status_code=status_code)


def _error_code_from_body(body: object) -> str | None:
    entries = _json_object_or_none(body)
    if entries is None:
        return None
    code = _optional_string(entries.get("code"))
    if code is not None:
        return code
    nested = _json_object_or_none(entries.get("error"))
    if nested is None:
        return None
    return _optional_string(nested.get("code"))


def _json_object_or_none(value: object) -> dict[str, JSONValue] | None:
    try:
        parsed = _JSON_VALUE_ADAPTER.validate_python(value)
    except ValidationError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _failure(*, code: str | None, message: str, status_code: int | None) -> ModelFailureEvent:
    normalized_code = (code or "").lower()
    kind: ModelFailureKind = "unknown"
    retryable = False

    if normalized_code in _CONTEXT_CODES or "context window" in message.lower():
        kind = "context_overflow"
    elif status_code in {401, 403} or normalized_code in {
        "authentication_error",
        "invalid_api_key",
    }:
        kind = "authentication"
    elif status_code == 408 or normalized_code in {"request_timeout", "timeout"}:
        kind = "timeout"
        retryable = True
    elif status_code == 429 or normalized_code in {"rate_limit", "rate_limit_exceeded"}:
        kind = "rate_limit"
        retryable = True
    elif status_code is not None and (status_code >= 500 or status_code == 409):
        kind = "unavailable"
        retryable = True
    elif status_code is not None and 400 <= status_code < 500:
        kind = "invalid_request"
    elif normalized_code in {"server_error", "service_unavailable"}:
        kind = "unavailable"
        retryable = True

    return ModelFailureEvent(kind=kind, message=message, retryable=retryable)


async def _iter_with_cancellation(
    stream: _ClosableAsyncStream,
    *,
    signal: CancellationToken | None,
    poll_seconds: float,
) -> AsyncIterator[object]:
    iterator = stream.__aiter__()
    while True:
        if _is_cancelled(signal):
            raise _StreamCancelled

        next_event: asyncio.Task[object] = asyncio.create_task(_next_stream_event(iterator))
        try:
            while True:
                done, _ = await asyncio.wait((next_event,), timeout=poll_seconds)
                if done:
                    break
                if _is_cancelled(signal):
                    next_event.cancel()
                    with suppress(asyncio.CancelledError):
                        await next_event
                    raise _StreamCancelled
            try:
                yield next_event.result()
            except StopAsyncIteration:
                return
        finally:
            if not next_event.done():
                next_event.cancel()
                with suppress(asyncio.CancelledError):
                    await next_event


async def _next_stream_event(iterator: AsyncIterator[object]) -> object:
    return await anext(iterator)


def _json_loads(text: str) -> object:
    return cast(object, json.loads(text))


def _object(value: object) -> dict[str, JSONValue]:
    raw: object = value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        raw = dump(mode="json", by_alias=True, exclude_none=True)
    parsed = _JSON_VALUE_ADAPTER.validate_python(raw)
    if not isinstance(parsed, dict):
        raise TypeError("Expected a JSON object")
    return parsed


def _object_list(value: JSONValue) -> list[dict[str, JSONValue]]:
    if not isinstance(value, list):
        raise TypeError("Expected a list of JSON objects")
    return [_object(item) for item in value]


def _optional_object(value: JSONValue) -> dict[str, JSONValue]:
    return _object(value) if value is not None else {}


def _required_string(value: Mapping[str, JSONValue], key: str) -> str:
    result = _optional_string(value.get(key))
    if not result:
        raise ValueError(f"Missing string field {key!r}")
    return result


def _optional_string(value: JSONValue) -> str | None:
    return value if isinstance(value, str) else None


def _nonnegative_int(value: JSONValue) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _optional_nonnegative_int(value: JSONValue) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value)


def _is_cancelled(signal: CancellationToken | None) -> bool:
    return signal is not None and signal.is_cancelled()


def _error_message(error: BaseException) -> str:
    return str(error) or type(error).__name__


__all__ = ["OpenAIAdapter", "OpenAIAdapterConfig", "create_adapter"]
