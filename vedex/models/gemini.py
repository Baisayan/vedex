from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol, cast

from google import genai
from google.genai import interactions, types
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError

from ..schema import (
    AssistantMessage,
    CancellationToken,
    JSONValue,
    ToolCall,
    ToolResultMessage,
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
_RETRYABLE_STATUS_CODES = [408, 409, 429, 500, 502, 503, 504]


class GeminiAdapterConfig(BaseModel):
    """Gemini credentials, transport policy, and default request settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr | None = None
    base_url: str | None = None
    api_version: str = "v1beta"
    timeout_seconds: float = Field(default=120.0, gt=0)
    retry_attempts: int = Field(default=5, ge=1)
    retry_initial_delay_seconds: float = Field(default=1.0, ge=0)
    retry_max_delay_seconds: float = Field(default=60.0, ge=0)
    retry_multiplier: float = Field(default=2.0, ge=1)
    retry_jitter: float = Field(default=1.0, ge=0)
    cancellation_poll_seconds: float = Field(default=0.1, gt=0)
    store: bool = False
    default_options: dict[str, JSONValue] = Field(default_factory=dict)


class _GeminiOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_output_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None
    service_tier: str | None = None
    stop_sequences: list[str] | None = None
    thinking_level: str | None = None
    thinking_summaries: str | None = None
    tool_choice: str | dict[str, JSONValue] | None = None


class _ClosableAsyncStream(Protocol):
    def __aiter__(self) -> AsyncIterator[object]: ...

    async def close(self) -> None: ...


class _StreamCancelled(Exception):
    pass


@dataclass(slots=True)
class _PendingCall:
    call_id: str
    name: str
    arguments: str = ""
    received_delta: bool = False


@dataclass(slots=True)
class _GeminiStreamState:
    text_parts: list[str] = field(default_factory=list)
    calls: dict[int, _PendingCall] = field(default_factory=dict)
    interaction_id: str | None = None
    usage: Usage = field(default_factory=Usage)


class GeminiAdapter:
    """Gemini Interactions API implementation of Vedex's normalized model boundary."""

    def __init__(self, config: GeminiAdapterConfig | None = None) -> None:
        self._config = config or GeminiAdapterConfig()

    @property
    def config(self) -> GeminiAdapterConfig:
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

            client: genai.Client | None = None
            provider_stream: _ClosableAsyncStream | None = None
            terminal: ModelCompletedEvent | ModelFailureEvent | ModelCancelledEvent | None = None
            cleanup_error: Exception | None = None
            state = _GeminiStreamState()

            try:
                params = self._request_params(request)
                retry_options = types.HttpRetryOptions(
                    attempts=self._config.retry_attempts,
                    initial_delay=self._config.retry_initial_delay_seconds,
                    max_delay=self._config.retry_max_delay_seconds,
                    exp_base=self._config.retry_multiplier,
                    jitter=self._config.retry_jitter,
                    http_status_codes=_RETRYABLE_STATUS_CODES,
                )
                http_options = types.HttpOptions(
                    base_url=self._config.base_url,
                    api_version=self._config.api_version,
                    timeout=round(self._config.timeout_seconds * 1000),
                    retry_options=retry_options,
                )
                client = genai.Client(
                    api_key=(
                        self._config.api_key.get_secret_value()
                        if self._config.api_key is not None
                        else None
                    ),
                    http_options=http_options,
                )
                raw_stream = await client.aio.interactions.create(**params)
                provider_stream = cast(_ClosableAsyncStream, raw_stream)
                yield ModelStartEvent()

                async for raw_event in _iter_with_cancellation(
                    provider_stream,
                    signal=signal,
                    poll_seconds=self._config.cancellation_poll_seconds,
                ):
                    event = _object(raw_event)
                    event_type = _optional_string(event.get("event_type")) or _optional_string(
                        event.get("type")
                    )

                    if event_type == "interaction.created":
                        interaction = _optional_object(event.get("interaction"))
                        state.interaction_id = _optional_string(interaction.get("id"))
                    elif event_type == "step.start":
                        _record_step_start(event, state)
                    elif event_type == "step.delta":
                        async for delta_event in _record_step_delta(event, state):
                            yield delta_event
                    elif event_type == "step.stop":
                        raw_usage = event.get("usage") or event.get("step_usage")
                        if raw_usage is not None:
                            state.usage = _usage(raw_usage)
                    elif event_type in {"interaction.completed", "interaction.complete"}:
                        terminal = _completed_event(event, state)
                        break
                    elif event_type == "interaction.status_update":
                        status = _optional_string(event.get("status"))
                        if status == "cancelled":
                            terminal = ModelCancelledEvent(message="Gemini interaction cancelled")
                            break
                        if status in {"failed", "incomplete", "budget_exceeded"}:
                            terminal = ModelFailureEvent(
                                kind="invalid_request" if status == "incomplete" else "unknown",
                                message=f"Gemini interaction ended with status {status}",
                            )
                            break
                    elif event_type == "error":
                        terminal = _failure_from_error_object(event.get("error"))
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
                        await client.aio.aclose()
                    except Exception as exc:
                        cleanup_error = cleanup_error or exc

            if cleanup_error is not None and not isinstance(terminal, ModelCancelledEvent):
                terminal = ModelFailureEvent(
                    kind="unavailable",
                    message=f"Gemini client cleanup failed: {_error_message(cleanup_error)}",
                    retryable=True,
                )
            if terminal is None:
                terminal = ModelFailureEvent(
                    message="Gemini stream ended without a terminal interaction",
                )
            yield terminal

        return run()

    def _request_params(
        self,
        request: ModelRequest,
    ) -> interactions.CreateModelInteractionParamsStreaming:
        merged_options = dict(self._config.default_options)
        merged_options.update(request.settings.options)
        options = _GeminiOptions.model_validate(merged_options)
        generation_config = options.model_dump(
            exclude={"service_tier"},
            exclude_none=True,
        )

        values: dict[str, object] = {
            "model": request.settings.model,
            "input": cast(interactions.InteractionsInputParam, _request_input(request)),
            "store": self._config.store,
            "stream": True,
            "system_instruction": request.system,
        }
        if request.tools:
            values["tools"] = cast(list[interactions.ToolParam], _request_tools(request))
        if generation_config:
            values["generation_config"] = generation_config
        if options.service_tier is not None:
            values["service_tier"] = options.service_tier
        return cast(interactions.CreateModelInteractionParamsStreaming, values)


def create_adapter() -> GeminiAdapter:
    """Create an environment-configured adapter for CLI factory loading."""

    return GeminiAdapter()


def _request_input(request: ModelRequest) -> list[dict[str, JSONValue]]:
    provider_input: list[dict[str, JSONValue]] = []
    for message in request.messages:
        if isinstance(message, UserMessage):
            provider_input.append(
                {
                    "type": "user_input",
                    "content": [{"type": "text", "text": message.content}],
                }
            )
            continue

        if isinstance(message, AssistantMessage):
            metadata = message.metadata.get("gemini")
            if isinstance(metadata, dict):
                steps = metadata.get("steps")
                if isinstance(steps, list) and steps:
                    provider_input.extend(_object(step) for step in steps)
                    continue

            if message.content:
                provider_input.append(
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": message.content}],
                    }
                )
            provider_input.extend(
                {
                    "type": "function_call",
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in message.tool_calls
            )
            continue

        if isinstance(message, ToolResultMessage):
            provider_input.append(
                {
                    "type": "function_result",
                    "call_id": message.tool_call_id,
                    "name": message.name,
                    "is_error": not message.ok,
                    "result": [{"type": "text", "text": message.content}],
                }
            )

    return provider_input


def _request_tools(request: ModelRequest) -> list[dict[str, JSONValue]]:
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
        for tool in request.tools
    ]


def _record_step_start(event: Mapping[str, JSONValue], state: _GeminiStreamState) -> None:
    step = _optional_object(event.get("step"))
    if _optional_string(step.get("type")) != "function_call":
        return

    index = _nonnegative_int(event.get("index"))
    arguments = step.get("arguments")
    if isinstance(arguments, dict):
        argument_text = json.dumps(arguments, separators=(",", ":")) if arguments else ""
    else:
        argument_text = _optional_string(arguments) or ""
    state.calls[index] = _PendingCall(
        call_id=_required_string(step, "id"),
        name=_required_string(step, "name"),
        arguments=argument_text,
    )


async def _record_step_delta(
    event: Mapping[str, JSONValue],
    state: _GeminiStreamState,
) -> AsyncIterator[ModelEvent]:
    delta = _optional_object(event.get("delta"))
    delta_type = _optional_string(delta.get("type"))

    if delta_type == "text":
        text = _required_string(delta, "text")
        state.text_parts.append(text)
        yield ModelTextDeltaEvent(delta=text)
        return

    if delta_type in {"thought", "thought_summary"}:
        text = _optional_string(delta.get("text")) or _content_text(delta.get("content"))
        if text:
            yield ModelThinkingDeltaEvent(delta=text)
        return

    if delta_type in {"arguments", "arguments_delta"}:
        index = _nonnegative_int(event.get("index"))
        call = state.calls.get(index)
        if call is None:
            raise ValueError(f"Gemini streamed arguments for unknown step index {index}")
        if not call.received_delta:
            call.arguments = ""
            call.received_delta = True
        call.arguments += (
            _optional_string(delta.get("partial_arguments"))
            or _optional_string(delta.get("arguments"))
            or ""
        )


def _completed_event(
    event: Mapping[str, JSONValue],
    state: _GeminiStreamState,
) -> ModelCompletedEvent | ModelFailureEvent:
    try:
        interaction = _object(event.get("interaction"))
        steps = _object_list(interaction.get("steps", []))
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for step in steps:
            step_type = _optional_string(step.get("type"))
            if step_type == "model_output":
                content_parts.extend(_content_texts(step.get("content")))
            elif step_type == "function_call":
                arguments = step.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = _object(json.loads(arguments or "{}"))
                else:
                    arguments = _object(arguments)
                tool_calls.append(
                    ToolCall(
                        id=_required_string(step, "id"),
                        name=_required_string(step, "name"),
                        arguments=arguments,
                    )
                )

        has_model_output = any(step.get("type") == "model_output" for step in steps)
        if not content_parts:
            content_parts = list(state.text_parts)
            if content_parts and not has_model_output:
                steps.append(
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "".join(content_parts)}],
                    }
                )

        existing_call_ids = {call.id for call in tool_calls}
        for _, pending_call in sorted(state.calls.items()):
            if pending_call.call_id in existing_call_ids:
                continue
            call = _pending_tool_call(pending_call)
            tool_calls.append(call)
            steps.append(
                {
                    "type": "function_call",
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
            )

        metadata: dict[str, JSONValue] = {"steps": cast(JSONValue, steps)}
        interaction_id = _optional_string(interaction.get("id")) or state.interaction_id
        model = _optional_string(interaction.get("model"))
        if interaction_id is not None:
            metadata["interaction_id"] = interaction_id
        if model is not None:
            metadata["model"] = model

        raw_usage = interaction.get("usage")
        usage = _usage(raw_usage) if raw_usage is not None else state.usage
        return ModelCompletedEvent(
            message=AssistantMessage(
                content="".join(content_parts),
                tool_calls=tool_calls,
                metadata={"gemini": metadata},
            ),
            usage=usage,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return ModelFailureEvent(
            kind="invalid_request",
            message=f"Malformed Gemini completed interaction: {_error_message(exc)}",
        )


def _pending_tool_call(call: _PendingCall) -> ToolCall:
    arguments = _object(json.loads(call.arguments or "{}"))
    return ToolCall(id=call.call_id, name=call.name, arguments=arguments)


def _content_texts(value: JSONValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for content in value:
        if isinstance(content, str):
            result.append(content)
            continue
        item = _object(content)
        if _optional_string(item.get("type")) == "text":
            text = _optional_string(item.get("text"))
            if text:
                result.append(text)
    return result


def _content_text(value: JSONValue) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _optional_string(value.get("text")) or ""
    return "".join(_content_texts(value))


def _usage(value: JSONValue) -> Usage:
    usage = _optional_object(value)
    return Usage(
        input_tokens=_first_nonnegative_int(
            usage,
            "total_input_tokens",
            "prompt_tokens",
            "prompt_token_count",
        ),
        output_tokens=_first_nonnegative_int(
            usage,
            "total_output_tokens",
            "completion_tokens",
            "candidates_token_count",
        ),
        cached_tokens=_first_nonnegative_int(
            usage,
            "total_cached_tokens",
            "cached_tokens",
            "cached_content_token_count",
        ),
        thinking_tokens=_first_optional_nonnegative_int(
            usage,
            "total_thought_tokens",
            "thought_tokens",
            "thoughts_token_count",
        ),
    )


def _failure_from_error_object(value: JSONValue) -> ModelFailureEvent:
    error = _optional_object(value)
    code_value = error.get("code")
    status_code = (
        code_value if isinstance(code_value, int) and not isinstance(code_value, bool) else None
    )
    code = code_value if isinstance(code_value, str) else _optional_string(error.get("status"))
    message = _optional_string(error.get("message")) or "Gemini request failed"
    return _failure(code=code, message=message, status_code=status_code)


def _failure_from_exception(error: Exception) -> ModelFailureEvent:
    if isinstance(error, (TypeError, ValueError, ValidationError)):
        return ModelFailureEvent(
            kind="invalid_request",
            message=_error_message(error),
        )

    raw_status = getattr(error, "code", None)
    status_code = raw_status if isinstance(raw_status, int) else None
    raw_code = getattr(error, "status", None)
    code = raw_code if isinstance(raw_code, str) else None
    message = _error_message(error)
    if code is None and "api key" in message.lower():
        code = "UNAUTHENTICATED"
    if isinstance(error, TimeoutError):
        status_code = 408
    return _failure(code=code, message=message, status_code=status_code)


def _failure(*, code: str | None, message: str, status_code: int | None) -> ModelFailureEvent:
    normalized_code = (code or "").upper()
    lowered_message = message.lower()
    kind: ModelFailureKind = "unknown"
    retryable = False

    if "context" in lowered_message and any(
        term in lowered_message for term in ("length", "window", "token")
    ):
        kind = "context_overflow"
    elif status_code in {401, 403} or normalized_code in {
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
    }:
        kind = "authentication"
    elif status_code == 408 or normalized_code == "DEADLINE_EXCEEDED":
        kind = "timeout"
        retryable = True
    elif status_code == 429 or normalized_code == "RESOURCE_EXHAUSTED":
        kind = "rate_limit"
        retryable = True
    elif (
        status_code is not None and (status_code >= 500 or status_code == 409)
    ) or normalized_code == "UNAVAILABLE":
        kind = "unavailable"
        retryable = True
    elif (
        status_code is not None
        and 400 <= status_code < 500
        or normalized_code in {"INVALID_ARGUMENT", "FAILED_PRECONDITION", "NOT_FOUND"}
    ):
        kind = "invalid_request"

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


def _first_nonnegative_int(value: Mapping[str, JSONValue], *keys: str) -> int:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            return candidate
    return 0


def _first_optional_nonnegative_int(
    value: Mapping[str, JSONValue],
    *keys: str,
) -> int | None:
    if not any(key in value for key in keys):
        return None
    return _first_nonnegative_int(value, *keys)


def _is_cancelled(signal: CancellationToken | None) -> bool:
    return signal is not None and signal.is_cancelled()


def _error_message(error: BaseException) -> str:
    return str(error) or type(error).__name__


__all__ = ["GeminiAdapter", "GeminiAdapterConfig", "create_adapter"]
