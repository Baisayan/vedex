from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from agent.messages import AgentMessage, AssistantMessage, ToolResultMessage, UserMessage
from agent.tools import AgentTool
from agent.tools import ToolCall as AgentToolCall
from agent.types import JSONValue
from ollama.client import CancellationToken, OllamaClient
from ollama.models import get_model_info
from ollama.types import ChatChunk, ChatRequest, OllamaMessage, OllamaTool, OllamaToolCall


# ── Protocol ──────────────────────────────────────────────────────────────────

class OllamaProvider(Protocol):
    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        ...


# ── Provider events ───────────────────────────────────────────────────────────

class ProviderResponseStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["response_start"] = "response_start"
    model: str


class ProviderRetryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["retry"] = "retry"
    attempt: int
    max_attempts: int
    delay_seconds: float
    message: str
    data: dict[str, JSONValue] | None = None


class ProviderTextDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text_delta"] = "text_delta"
    delta: str


class ProviderThinkingDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class ProviderToolCallEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_call"] = "tool_call"
    tool_call: AgentToolCall


class ProviderResponseEndEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["response_end"] = "response_end"
    message: AssistantMessage
    finish_reason: str | None = None


class ProviderErrorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["error"] = "error"
    message: str
    data: dict[str, JSONValue] | None = None


type ProviderEvent = (
    ProviderResponseStartEvent
    | ProviderRetryEvent
    | ProviderTextDeltaEvent
    | ProviderThinkingDeltaEvent
    | ProviderToolCallEvent
    | ProviderResponseEndEvent
    | ProviderErrorEvent
)


# ── OllamaChat ────────────────────────────────────────────────────────────────

class OllamaChat:
    def __init__(self, client: OllamaClient) -> None:
        self._client = client

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        async def _iter() -> AsyncIterator[ProviderEvent]:
            try:
                model_info = await get_model_info(model, client=self._client)
                if tools and not model_info.supports_tools:
                    yield ProviderErrorEvent(
                        message=f"Ollama model does not support tools: {model}"
                    )
                    return

                request = _build_request(
                    model=model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    think=True if model_info.supports_thinking else None,
                )
                yield ProviderResponseStartEvent(model=model)

                content_parts: list[str] = []
                tool_calls: list[AgentToolCall] = []
                finish_reason: str | None = None
                completed = False

                async for raw in self._client.stream(
                    "/api/chat", body=request.to_dict(), signal=signal
                ):
                    error = raw.get("error")
                    if error is not None:
                        yield ProviderErrorEvent(
                            message=str(error),
                            data={"raw": raw},
                        )
                        return

                    try:
                        chunk = ChatChunk.model_validate(raw)
                    except ValidationError as exc:
                        yield ProviderErrorEvent(
                            message="Failed to parse Ollama response",
                            data={"error": str(exc)},
                        )
                        return

                    if chunk.message.thinking:
                        yield ProviderThinkingDeltaEvent(delta=chunk.message.thinking)

                    if chunk.message.content:
                        content_parts.append(chunk.message.content)
                        yield ProviderTextDeltaEvent(delta=chunk.message.content)

                    if chunk.message.tool_calls:
                        for raw_call in chunk.message.tool_calls:
                            try:
                                call = _parse_tool_call(raw_call)
                            except ValueError as exc:
                                yield ProviderErrorEvent(message=str(exc))
                                return
                            tool_calls.append(call)

                    if chunk.done:
                        completed = True
                        finish_reason = chunk.done_reason
                        break

                if signal is not None and signal.is_cancelled():
                    return
                if not completed:
                    yield ProviderErrorEvent(
                        message="Ollama stream ended before the final response"
                    )
                    return

                for call in tool_calls:
                    yield ProviderToolCallEvent(tool_call=call)

                yield ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content="".join(content_parts),
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            except (httpx.HTTPError, LookupError, ValueError) as exc:
                yield ProviderErrorEvent(
                    message=str(exc),
                    data={"error_type": type(exc).__name__},
                )

        return _iter()


# ── Conversion helpers ────────────────────────────────────────────────────────

def _build_request(
    *,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    think: bool | None = None,
) -> ChatRequest:
    ollama_messages = [OllamaMessage(role="system", content=system)]
    for msg in messages:
        ollama_messages.append(_to_ollama_message(msg))

    ollama_tools = [
        OllamaTool(
            name=t.name,
            description=t.description,
            parameters=dict(t.input_schema),
        )
        for t in tools
    ]

    return ChatRequest(
        model=model,
        messages=ollama_messages,
        tools=ollama_tools,
        think=think,
    )


def _to_ollama_message(message: AgentMessage) -> OllamaMessage:
    if isinstance(message, UserMessage):
        return OllamaMessage(role="user", content=message.content)

    if isinstance(message, AssistantMessage):
        tool_calls = (
            [OllamaToolCall(name=c.name, arguments=dict(c.arguments))
             for c in message.tool_calls]
            if message.tool_calls else None
        )
        return OllamaMessage(role="assistant", content=message.content, tool_calls=tool_calls)

    if isinstance(message, ToolResultMessage):
        return OllamaMessage(role="tool", content=message.content)

    return OllamaMessage(role="user", content=str(message))


def _parse_tool_call(raw: dict[str, object]) -> AgentToolCall:
    func = raw.get("function")
    if not isinstance(func, dict):
        raise ValueError("Ollama returned a malformed tool call")
    name = func.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Ollama returned a tool call without a name")
    arguments = func.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError(f"Ollama returned invalid arguments for tool: {name}")
    return AgentToolCall(
        id=f"call-{uuid4().hex[:8]}",
        name=name,
        arguments=arguments,
    )
