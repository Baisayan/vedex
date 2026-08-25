from __future__ import annotations

import sys
from collections.abc import Mapping

from rich.console import Console
from rich.text import Text

from .schema import (
    AgentEndEvent,
    AgentEvent,
    ErrorEvent,
    JSONValue,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)

TOOL_RESULT_PREVIEW_LINES = 12
TOOL_RESULT_PREVIEW_CHARS = 1_500


class CommandLineRenderer:
    """Render normalized Agent events without knowing model-provider formats."""

    def __init__(self) -> None:
        self._assistant_started = False
        self._assistant_ended = False
        self._thinking_started = False
        self._thinking_ended = False
        self._failed = False
        self._output = Console(file=sys.stdout, highlight=False)
        self._diagnostics = Console(file=sys.stderr, highlight=False)

    def render(self, event: AgentEvent) -> None:
        if isinstance(event, MessageStartEvent):
            if event.message_role == "assistant":
                self._assistant_started = False
                self._assistant_ended = False
                self._thinking_started = False
                self._thinking_ended = False
            return

        if isinstance(event, MessageDeltaEvent):
            self._ensure_thinking_newline()
            self._assistant_started = True
            self._output.print(Text(event.delta), end="", soft_wrap=True)
            return

        if isinstance(event, ThinkingDeltaEvent):
            if not self._thinking_started:
                self._diagnostics.print(Text("thinking: ", style="dim"), end="")
                self._thinking_started = True
            self._diagnostics.print(Text(event.delta, style="dim"), end="", soft_wrap=True)
            return

        if isinstance(event, ToolExecutionStartEvent):
            self._ensure_thinking_newline()
            self._ensure_assistant_newline()
            self._diagnostics.print(Text(format_tool_call_block(event.tool_call), style="cyan"))
            return

        if isinstance(event, ToolExecutionEndEvent):
            self._ensure_thinking_newline()
            self._ensure_assistant_newline()
            status = "✓" if event.result.ok else "✗"
            style = "green" if event.result.ok else "red"

            line = Text()
            line.append(f"{status} completed: {event.result.name}", style=style)
            self._diagnostics.print(line)

            if event.result.content:
                preview = _preview_text(event.result.content, max_lines=TOOL_RESULT_PREVIEW_LINES)
                for line_text in preview.splitlines():
                    self._diagnostics.print(Text(f"  {line_text}", style="white"))
            return

        if isinstance(event, ErrorEvent):
            if not event.recoverable:
                self._failed = True
            self._ensure_thinking_newline()
            self._ensure_assistant_newline()
            self._diagnostics.print(Text(f"Error: {event.message}", style="red"))
            return

        if isinstance(event, MessageEndEvent):
            if event.message.role == "assistant":
                self._ensure_thinking_newline(final=True)
                if not self._assistant_started and event.message.content:
                    self._assistant_started = True
                    self._output.print(Text(event.message.content), end="", soft_wrap=True)
                self._ensure_assistant_newline(final=True)
            return

        if isinstance(event, AgentEndEvent):
            self._ensure_thinking_newline(final=True)
            self._ensure_assistant_newline(final=True)

    def finish(self) -> bool:
        self._ensure_thinking_newline(final=True)
        self._ensure_assistant_newline(final=True)
        return not self._failed

    def _ensure_assistant_newline(self, *, final: bool = False) -> None:
        if self._assistant_started and not self._assistant_ended:
            self._output.print()
            self._assistant_ended = True
        elif final and not self._assistant_started:
            self._assistant_ended = True

    def _ensure_thinking_newline(self, *, final: bool = False) -> None:
        if self._thinking_started and not self._thinking_ended:
            self._diagnostics.print()
            self._thinking_ended = True
        elif final and not self._thinking_started:
            self._thinking_ended = True


def format_tool_call_block(tool_call: ToolCall) -> str:
    arguments = tool_call.arguments or {}
    name = tool_call.name

    if name == "read":
        path = arguments.get("path", "unknown")
        return f"→ read {path}{_read_line_suffix(arguments)}"

    if name in ("edit", "write"):
        path = arguments.get("path", "unknown")
        return f"→ {name} {path}"

    if name == "bash":
        command = arguments.get("command", "")
        timeout = arguments.get("timeout")
        suffix = f" (timeout {timeout}s)" if timeout is not None else ""
        return f"$ {command}{suffix}"

    if arguments:
        return f"→ {name} {arguments}"
    return f"→ {name}"


def _read_line_suffix(arguments: Mapping[str, JSONValue]) -> str:
    offset = _positive_int(arguments.get("offset"))
    limit = _positive_int(arguments.get("limit"))
    if offset is None and limit is None:
        return ""
    start = 1 if offset is None else max(1, offset)
    if limit is None:
        return f":{start}-"
    return f":{start}-{start + max(1, limit) - 1}"


def _positive_int(value: JSONValue | None) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _preview_text(text: str, *, max_lines: int) -> str:
    lines = text.splitlines()
    if not lines:
        return text[:TOOL_RESULT_PREVIEW_CHARS]

    preview_lines = lines[:max_lines]
    preview = "\n".join(preview_lines)
    hidden_lines = max(0, len(lines) - len(preview_lines))

    truncated_by_chars = len(preview) > TOOL_RESULT_PREVIEW_CHARS
    if truncated_by_chars:
        preview = preview[:TOOL_RESULT_PREVIEW_CHARS].rstrip()

    if hidden_lines or truncated_by_chars:
        details: list[str] = []
        if hidden_lines:
            details.append(f"{hidden_lines} more lines")
        if truncated_by_chars:
            details.append("additional characters")
        preview = (
            f"{preview}\n\n  [Output truncated for terminal safety: {', '.join(details)} hidden]"
        )
    return preview
