from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, assert_never
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.client import OllamaClient
from agent.events import (
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
)
from agent.messages import AgentMessage, AssistantMessage, ToolResultMessage, UserMessage
from agent.models import get_model_info
from agent.tools import AgentTool, ToolCall
from agent.types import CancellationToken


# ── Wire types (Ollama protocol) ───────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class OllamaMessage:
    role: str
    content: str
    tool_calls: list[OllamaToolCall] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d


@dataclass(frozen=True, slots=True)
class OllamaToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"function": {"name": self.name, "arguments": self.arguments}}


@dataclass(frozen=True, slots=True)
class OllamaTool:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ChatRequest:
    model: str
    messages: list[OllamaMessage]
    tools: list[OllamaTool] = field(default_factory=list)
    stream: bool = True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "stream": self.stream,
        }
        if self.tools:
            d["tools"] = [t.to_dict() for t in self.tools]
        return d


class ChatChunkMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = "assistant"
    content: str = ""
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    message: ChatChunkMessage = Field(default_factory=ChatChunkMessage)
    done: bool = False
    done_reason: str | None = None


# ── Ollama provider ────────────────────────────────────────────────────────────

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
                    tools=tools
                )
                yield ProviderResponseStartEvent(model=model)

                content_parts: list[str] = []
                tool_calls: list[ToolCall] = []
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
        tools=ollama_tools
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

    assert_never(message)


def _parse_tool_call(raw: dict[str, object]) -> ToolCall:
    func = raw.get("function")
    if not isinstance(func, dict):
        raise ValueError("Ollama returned a malformed tool call")
    name = func.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Ollama returned a tool call without a name")
    arguments = func.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError(f"Ollama returned invalid arguments for tool: {name}")
    return ToolCall(
        id=f"call-{uuid4().hex[:8]}",
        name=name,
        arguments=arguments,
    )
