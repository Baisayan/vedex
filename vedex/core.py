from __future__ import annotations

from asyncio import sleep
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from json import JSONDecodeError, loads
from typing import Any, assert_never
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from schema import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    CancellationToken,
    ErrorEvent,
    JSONValue,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResultMessage,
    TurnEndEvent,
    TurnStartEvent,
    UserMessage,
)


# HTTP client

OLLAMA_HOST = "http://localhost:11434"

_RETRY_BASE_DELAY = 0.5
_RETRY_MAX_DELAY = 8.0
_RETRY_POLL = 0.05
_TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}


class OllamaClient:
    def __init__(
        self,
        host: str = OLLAMA_HOST,
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._http_client = http_client
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._http_client is not None and self._owns_client:
            await self._http_client.aclose()
            self._http_client = None

    async def get(self, path: str) -> dict[str, Any]:
        client = self._client()
        response = await client.get(f"{self._host}{path}")
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def stream(
        self,
        path: str,
        *,
        body: dict[str, Any],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        async def _iter() -> AsyncIterator[dict[str, Any]]:
            client = self._client()
            url = f"{self._host}{path}"

            emitted_chunk = False
            for attempt in range(self._max_retries + 1):
                if attempt > 0 and not await _backoff(attempt - 1, signal):
                    return

                if signal is not None and signal.is_cancelled():
                    return

                try:
                    async with client.stream("POST", url, json=body) as response:
                        if response.status_code >= 400:
                            await response.aread()
                            if attempt < self._max_retries and response.status_code in _TRANSIENT_STATUSES:
                                continue
                            response.raise_for_status()
                            return

                        async for line in response.aiter_lines():
                            if signal is not None and signal.is_cancelled():
                                return
                            chunk = _parse_object(line)
                            if chunk is not None:
                                emitted_chunk = True
                                yield chunk
                        return

                except httpx.HTTPStatusError:
                    raise
                except httpx.HTTPError:
                    if not emitted_chunk and attempt < self._max_retries:
                        continue
                    raise

        return _iter()

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self._timeout)
        return self._http_client


async def _backoff(attempt: int, signal: CancellationToken | None) -> bool:
    delay = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
    remaining = delay
    while remaining > 0:
        if signal is not None and signal.is_cancelled():
            return False
        interval = min(_RETRY_POLL, remaining)
        await sleep(interval)
        remaining -= interval
    return signal is None or not signal.is_cancelled()


def _parse_object(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        value = loads(line)
    except JSONDecodeError as exc:
        raise ValueError("Ollama returned malformed NDJSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Ollama returned a non-object NDJSON value")
    return value


# Wire types (Ollama protocol)

@dataclass(frozen=True, slots=True)
class _OllamaMessage:
    role: str
    content: str
    tool_calls: list[_OllamaToolCall] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d


@dataclass(frozen=True, slots=True)
class _OllamaToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"function": {"name": self.name, "arguments": self.arguments}}


@dataclass(frozen=True, slots=True)
class _OllamaTool:
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
class _ChatRequest:
    model: str
    messages: list[_OllamaMessage]
    tools: list[_OllamaTool] = field(default_factory=list)
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


class _ChatChunkMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = "assistant"
    content: str = ""
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class _ChatChunk(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = ""
    message: _ChatChunkMessage = Field(default_factory=_ChatChunkMessage)
    done: bool = False
    done_reason: str | None = None


# Model discovery

@dataclass(frozen=True, slots=True)
class OllamaModelInfo:
    name: str
    context_length: int | None
    supports_tools: bool


async def list_model_info(
    host: str = OLLAMA_HOST,
    *,
    client: OllamaClient | None = None,
) -> list[OllamaModelInfo]:

    owns_client = client is None
    client = client or OllamaClient(host)
    try:
        data = await client.get("/api/tags")
    finally:
        if owns_client:
            await client.aclose()

    models = data.get("models")
    if not isinstance(models, list):
        return []

    result: list[OllamaModelInfo] = []
    for model in models:
        info = _parse_model_info(model)
        if info is not None:
            result.append(info)
    return result


async def get_model_info(
    model: str,
    host: str = OLLAMA_HOST,
    *,
    client: OllamaClient | None = None,
) -> OllamaModelInfo:
    target = model if ":" in model else f"{model}:latest"
    for info in await list_model_info(host, client=client):
        if info.name == model or info.name == target:
            return info
    raise LookupError(f"Ollama model is not available locally: {model}")


def _parse_model_info(value: Any) -> OllamaModelInfo | None:
    if not isinstance(value, dict):
        return None

    name = value.get("name") or value.get("model")
    if not isinstance(name, str) or not name:
        return None

    context_length: int | None = None
    details = value.get("details")
    if isinstance(details, dict):
        raw = details.get("context_length")
        if isinstance(raw, int) and raw > 0:
            context_length = raw

    raw_capabilities = value.get("capabilities")
    capabilities = (
        {item for item in raw_capabilities if isinstance(item, str)}
        if isinstance(raw_capabilities, list)
        else set()
    )

    return OllamaModelInfo(
        name=name,
        context_length=context_length,
        supports_tools="tools" in capabilities
    )
    

def _build_request(
    *,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
) -> _ChatRequest:
    ollama_messages = [_OllamaMessage(role="system", content=system)] if system.strip() else []
    for msg in messages:
        ollama_messages.append(_to_ollama_message(msg))

    ollama_tools = [
        _OllamaTool(
            name=t.name,
            description=t.description,
            parameters=dict(t.input_schema),
        )
        for t in tools
    ]

    return _ChatRequest(
        model=model,
        messages=ollama_messages,
        tools=ollama_tools
    )


def _to_ollama_message(message: AgentMessage) -> _OllamaMessage:
    if isinstance(message, UserMessage):
        return _OllamaMessage(role="user", content=message.content)

    if isinstance(message, AssistantMessage):
        tool_calls = (
            [_OllamaToolCall(name=c.name, arguments=dict(c.arguments))
             for c in message.tool_calls]
            if message.tool_calls else None
        )
        return _OllamaMessage(role="assistant", content=message.content, tool_calls=tool_calls)

    if isinstance(message, ToolResultMessage):
        return _OllamaMessage(role="tool", content=message.content)

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


# Tool execution

async def _execute_tool_calls(
    tool_calls: list[ToolCall],
    tool_by_name: Mapping[str, AgentTool],
    messages: list[AgentMessage],
    signal: CancellationToken | None,
) -> AsyncIterator[AgentEvent]:
    for index, tool_call in enumerate(tool_calls):
        if signal is not None and signal.is_cancelled():
            for cancelled_tool_call in tool_calls[index:]:
                result = _cancelled_tool_result(cancelled_tool_call)
                messages.append(_tool_result_message(result))
                yield ToolExecutionEndEvent(result=result)
            yield ErrorEvent(message="Agent run cancelled", recoverable=True)
            return

        yield ToolExecutionStartEvent(tool_call=tool_call)

        tool = tool_by_name.get(tool_call.name)
        if tool is None:
            result = _unknown_tool_result(tool_call)
        else:
            result = await _execute_tool(tool, tool_call, signal)

        messages.append(_tool_result_message(result))
        yield ToolExecutionEndEvent(result=result)


async def _execute_tool(
    tool: AgentTool,
    tool_call: ToolCall,
    signal: CancellationToken | None,
) -> AgentToolResult:
    try:
        result = await tool.execute(tool_call.arguments, signal=signal)
    except Exception as exc:
        return AgentToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            ok=False,
            content=str(exc),
            error=str(exc),
        )

    if result.tool_call_id != tool_call.id:
        return result.model_copy(update={"tool_call_id": tool_call.id})
    return result


def _unknown_tool_result(tool_call: ToolCall) -> AgentToolResult:
    message = f"Unknown tool: {tool_call.name}"
    return AgentToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        ok=False,
        content=message,
        error=message,
    )


def _cancelled_tool_result(tool_call: ToolCall) -> AgentToolResult:
    message = "Tool call cancelled"
    return AgentToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        ok=False,
        content=message,
        error=message,
    )


def _tool_result_message(result: AgentToolResult) -> ToolResultMessage:
    data: dict[str, JSONValue] | None = result.data
    content = result.content
    if not result.ok and result.error and result.error not in content:
        content = f"{content}\n\nError: {result.error}"
    if data is not None and not content:
        content = str(data)

    return ToolResultMessage(
        tool_call_id=result.tool_call_id,
        name=result.name,
        content=content,
        ok=result.ok,
        data=result.data,
        details=result.details,
        error=result.error,
    )


# Agent entry point

async def run_agent_loop(
    *,
    client: OllamaClient,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    max_turns: int | None = None,
    signal: CancellationToken | None = None,
) -> AsyncIterator[AgentEvent]:

    yield AgentStartEvent()

    if max_turns is not None and max_turns < 1:
        yield ErrorEvent(message="max_turns must be at least 1", recoverable=False)
        yield AgentEndEvent()
        return

    try:
        model_info = await get_model_info(model, client=client)
    except LookupError as exc:
        yield ErrorEvent(message=str(exc))
        yield AgentEndEvent()
        return

    if tools and not model_info.supports_tools:
        yield ErrorEvent(message=f"Ollama model does not support tools: {model}")
        yield AgentEndEvent()
        return

    tool_by_name = {tool.name: tool for tool in tools}
    turn = 1

    while max_turns is None or turn <= max_turns:
        if signal is not None and signal.is_cancelled():
            yield ErrorEvent(message="Agent run cancelled", recoverable=True)
            break

        yield TurnStartEvent(turn=turn)

        request = _build_request(model=model, system=system, messages=messages, tools=tools)
        yield MessageStartEvent()

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        completed = False
        stream_failed = False

        try:
            async for raw in client.stream("/api/chat", body=request.to_dict(), signal=signal):
                if raw.get("error") is not None:
                    yield ErrorEvent(message=str(raw["error"]), data={"raw": raw})
                    stream_failed = True
                    break

                try:
                    chunk = _ChatChunk.model_validate(raw)
                except ValidationError as exc:
                    yield ErrorEvent(
                        message="Failed to parse Ollama response",
                        data={"error": str(exc)},
                    )
                    stream_failed = True
                    break

                if chunk.message.thinking:
                    yield ThinkingDeltaEvent(delta=chunk.message.thinking)

                if chunk.message.content:
                    content_parts.append(chunk.message.content)
                    yield MessageDeltaEvent(delta=chunk.message.content)

                if chunk.message.tool_calls:
                    for raw_call in chunk.message.tool_calls:
                        try:
                            call = _parse_tool_call(raw_call)
                        except ValueError as exc:
                            yield ErrorEvent(message=str(exc))
                            stream_failed = True
                            break
                        tool_calls.append(call)
                    if stream_failed:
                        break

                if chunk.done:
                    completed = True
                    break
        except (httpx.HTTPError, LookupError, ValueError) as exc:
            yield ErrorEvent(message=str(exc), data={"error_type": type(exc).__name__})
            stream_failed = True

        if signal is not None and signal.is_cancelled():
            yield TurnEndEvent(turn=turn)
            break

        if not completed and not stream_failed:
            yield ErrorEvent(message="Ollama stream ended before the final response")
            stream_failed = True

        if stream_failed:
            partial = AssistantMessage(content="".join(content_parts))
            yield MessageEndEvent(message=partial)
            yield TurnEndEvent(turn=turn)
            break

        assistant_message = AssistantMessage(
            content="".join(content_parts),
            tool_calls=tool_calls,
        )
        messages.append(assistant_message)
        yield MessageEndEvent(message=assistant_message)

        if not assistant_message.tool_calls:
            yield TurnEndEvent(turn=turn)
            break

        async for tool_event in _execute_tool_calls(
            assistant_message.tool_calls,
            tool_by_name,
            messages,
            signal,
        ):
            yield tool_event

        yield TurnEndEvent(turn=turn)
        turn += 1
    else:
        yield ErrorEvent(
            message=f"Agent loop stopped after reaching max_turns={max_turns}",
            recoverable=True,
        )

    yield AgentEndEvent()
