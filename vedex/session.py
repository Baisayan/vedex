from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .core import OllamaClient, run_agent_loop, tool_result_message
from .schema import (
    AgentEvent,
    AgentMessage,
    AgentTool,
    ErrorEvent,
    MessageEndEvent,
    ToolExecutionEndEvent,
    UserMessage,
)

_MESSAGE_ADAPTER: TypeAdapter[AgentMessage] = TypeAdapter(AgentMessage)

_CHARS_PER_TOKEN = 4
_MESSAGE_OVERHEAD_TOKENS = 4
_TOOL_OVERHEAD_TOKENS = 16
_DEFAULT_CONTEXT_RESERVE_TOKENS = 4_096


@dataclass(frozen=True, slots=True)
class ContextUsage:
    """A small, deliberately approximate context-use estimate."""

    total_tokens: int
    system_tokens: int
    message_tokens: int
    tool_tokens: int
    message_count: int
    tool_count: int


class SessionError(ValueError):
    """Raised when a message-only session file cannot be read safely."""


class SessionStore:
    """Synchronous JSONL storage for validated conversation messages."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[AgentMessage]:
        if not self.path.exists():
            return []

        messages: list[AgentMessage] = []
        with self.path.open("rb") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    messages.append(_MESSAGE_ADAPTER.validate_json(line))
                except ValidationError as exc:
                    raise SessionError(
                        f"Invalid session message in {self.path} at line {line_number}"
                    ) from exc
        return messages

    def append(self, message: AgentMessage) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        validated_message = _MESSAGE_ADAPTER.validate_python(message)
        with self.path.open("ab") as file:
            file.write(_MESSAGE_ADAPTER.dump_json(validated_message))
            file.write(b"\n")

    def rewrite(self, messages: list[AgentMessage]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                for message in messages:
                    validated_message = _MESSAGE_ADAPTER.validate_python(message)
                    file.write(_MESSAGE_ADAPTER.dump_json(validated_message))
                    file.write(b"\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


class Session:
    """The owner of one conversation's history, execution, and persistence."""

    def __init__(
        self,
        *,
        cwd: Path,
        model: str,
        system_prompt: str,
        tools: list[AgentTool],
        client: OllamaClient,
        store: SessionStore,
        context_window_tokens: int | None,
    ) -> None:
        self.cwd = cwd
        self._model = model
        self._system_prompt = system_prompt
        self._tools = tools
        self._client = client
        self.store = store
        self._context_window_tokens = context_window_tokens
        self.messages = store.load()
        self._context_usage_cache: ContextUsage | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        return tuple(self._tools)

    @property
    def context_window_tokens(self) -> int | None:
        return self._context_window_tokens

    @property
    def context_usage(self) -> ContextUsage:
        if self._context_usage_cache is None:
            self._context_usage_cache = _estimate_context_usage(
                system=self._system_prompt,
                messages=tuple(self.messages),
                tools=self.tools,
            )
        return self._context_usage_cache

    @property
    def context_token_estimate(self) -> int:
        return self.context_usage.total_tokens

    @property
    def context_token_limit(self) -> int | None:
        if self._context_window_tokens is None:
            return None
        return max(1, self._context_window_tokens - self._context_reserve_tokens())

    def set_model(self, model: str, context_window_tokens: int | None) -> None:
        self._model = model
        self._context_window_tokens = context_window_tokens

    def set_system_prompt(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self._invalidate_context_usage()

    def add_user_message(self, content: str) -> None:
        self._accept_message(UserMessage(content=content))

    async def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        try:
            self.add_user_message(content)
            if not self._prepare_context():
                yield ErrorEvent(
                    message=(
                        "The system prompt, tools, and newest message exceed "
                        "the model context window."
                    )
                )
                return

            overflow = False
            async for event in self._run_once():
                if isinstance(event, ErrorEvent) and not event.recoverable:
                    overflow = overflow or _is_context_overflow(event)
                yield event

            if overflow and self._drop_oldest_turn():
                async for event in self._run_once():
                    yield event
            else:
                self._truncate_to_context_limit()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            yield ErrorEvent(message=f"Agent run failed: {exc}")

    async def close(self) -> None:
        await self._client.aclose()

    async def _run_once(self) -> AsyncIterator[AgentEvent]:
        async for event in run_agent_loop(
            client=self._client,
            model=self._model,
            system=self._system_prompt,
            messages=self.messages,
            tools=self._tools,
        ):
            if isinstance(event, MessageEndEvent):
                self._accept_message(event.message)
            elif isinstance(event, ToolExecutionEndEvent):
                self._accept_message(tool_result_message(event.result))
            yield event

    def _accept_message(self, message: AgentMessage) -> None:
        self.store.append(message)
        self.messages.append(message)
        self._invalidate_context_usage()

    def _prepare_context(self) -> bool:
        self._truncate_to_context_limit()
        limit = self.context_token_limit
        return limit is None or self.context_token_estimate <= limit

    def _truncate_to_context_limit(self) -> bool:
        limit = self.context_token_limit
        if limit is None or self.context_token_estimate <= limit:
            return False

        first_kept_index = self._first_fitting_turn_index()
        return self._replace_history_from(first_kept_index)

    def _drop_oldest_turn(self) -> bool:
        """Drop one complete oldest user-led turn after an Ollama overflow."""

        return self._replace_history_from(_next_user_message_index(self.messages, start=1))

    def _replace_history_from(self, first_kept_index: int | None) -> bool:
        if first_kept_index is None or first_kept_index <= 0:
            return False

        retained_messages = self.messages[first_kept_index:]
        self.store.rewrite(retained_messages)
        self.messages[:] = retained_messages
        self._invalidate_context_usage()
        return True

    def _first_fitting_turn_index(self) -> int | None:
        context_window = self._context_window_tokens
        if context_window is None:
            return None

        base_tokens = self.context_usage.system_tokens + self.context_usage.tool_tokens
        message_budget = context_window - self._context_reserve_tokens() - base_tokens
        candidate = len(self.messages)
        used_tokens = 0
        for index in range(len(self.messages) - 1, -1, -1):
            message_tokens = _estimate_message_tokens(self.messages[index])
            if used_tokens + message_tokens > message_budget:
                break
            used_tokens += message_tokens
            candidate = index

        if candidate == 0:
            return 0
        next_user_index = _next_user_message_index(self.messages, start=candidate)
        if next_user_index is not None:
            return next_user_index
        return _current_turn_start(self.messages, candidate)

    def _invalidate_context_usage(self) -> None:
        self._context_usage_cache = None

    def _context_reserve_tokens(self) -> int:
        context_window = self._context_window_tokens
        if context_window is None:
            return 0
        return min(_DEFAULT_CONTEXT_RESERVE_TOKENS, max(1, context_window // 8))


def _next_user_message_index(messages: list[AgentMessage], *, start: int) -> int | None:
    for index in range(start, len(messages)):
        if messages[index].role == "user":
            return index
    return None


def _current_turn_start(messages: list[AgentMessage], candidate: int) -> int | None:
    for index in range(min(candidate, len(messages) - 1), -1, -1):
        if messages[index].role == "user":
            return index
    return None


def _is_context_overflow(event: ErrorEvent) -> bool:
    text = f"{event.message} {event.data or ''}".lower()
    markers = (
        "context length",
        "context window",
        "context limit",
        "maximum context",
        "max context",
        "input is too long",
        "input length",
        "prompt is too long",
        "too many tokens",
        "token limit",
        "exceeds the limit",
        "exceeded the limit",
    )
    return any(marker in text for marker in markers)


def _estimate_context_usage(
    *,
    system: str,
    messages: tuple[AgentMessage, ...],
    tools: tuple[AgentTool, ...],
) -> ContextUsage:
    system_tokens = _estimate_text_tokens(system)
    message_tokens = sum(_estimate_message_tokens(message) for message in messages)
    tool_tokens = sum(_estimate_tool_tokens(tool) for tool in tools)
    return ContextUsage(
        total_tokens=system_tokens + message_tokens + tool_tokens,
        system_tokens=system_tokens,
        message_tokens=message_tokens,
        tool_tokens=tool_tokens,
        message_count=len(messages),
        tool_count=len(tools),
    )


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _estimate_message_tokens(message: AgentMessage) -> int:
    if message.role == "user":
        return _MESSAGE_OVERHEAD_TOKENS + _estimate_text_tokens(message.content)
    if message.role == "assistant":
        tool_call_tokens = sum(
            _estimate_text_tokens(call.name) + _estimate_text_tokens(str(call.arguments))
            for call in message.tool_calls
        )
        return _MESSAGE_OVERHEAD_TOKENS + _estimate_text_tokens(message.content) + tool_call_tokens
    return (
        _MESSAGE_OVERHEAD_TOKENS
        + _estimate_text_tokens(message.name)
        + _estimate_text_tokens(message.content)
    )


def _estimate_tool_tokens(tool: AgentTool) -> int:
    return (
        _TOOL_OVERHEAD_TOKENS
        + _estimate_text_tokens(tool.name)
        + _estimate_text_tokens(tool.description)
        + _estimate_text_tokens(str(tool.input_schema))
    )
