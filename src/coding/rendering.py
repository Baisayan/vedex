from __future__ import annotations

import typer
from typing import Any
from rich.console import Console
from rich.text import Text

from agent import (
    AgentEndEvent,
    AgentEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)

TOOL_RESULT_PREVIEW_LINES = 12
TOOL_RESULT_PREVIEW_CHARS = 1_500


class CommandLineRenderer:

    def __init__(self) -> None:
        self._assistant_started = False
        self._assistant_ended = False
        self._failed = False
        self._console = Console(stderr=True, highlight=False)

    def render(self, event: AgentEvent) -> None:
        if isinstance(event, MessageStartEvent):
            self._assistant_started = False
            self._assistant_ended = False
            return

        if isinstance(event, MessageDeltaEvent):
            self._assistant_started = True
            typer.echo(event.delta, nl=False)
            return

        if isinstance(event, ToolExecutionStartEvent):
            self._ensure_assistant_newline()
            self._console.print(Text(format_tool_call_block(event.tool_call), style="cyan"))
            return

        if isinstance(event, ToolExecutionEndEvent):
            self._ensure_assistant_newline()
            status = "✓" if event.result.ok else "✗"
            style = "green" if event.result.ok else "red"

            line = Text()
            line.append(f"{status} completed: {event.result.name}", style=style)
            self._console.print(line)

            if event.result.content:
                preview = _preview_text(event.result.content, max_lines=TOOL_RESULT_PREVIEW_LINES)
                for i in preview.splitlines():
                    self._console.print(Text(f"  {i}", style="white"))
            return

        if isinstance(event, ErrorEvent):
            if not event.recoverable:
                self._failed = True
            self._ensure_assistant_newline()
            self._console.print(Text(f"Error: {event.message}", style="red"))
            return

        if isinstance(event, MessageEndEvent | AgentEndEvent):
            self._ensure_assistant_newline(final=True)

    def finish(self) -> bool:
        return not self._failed

    def _ensure_assistant_newline(self, *, final: bool = False) -> None:
        if self._assistant_started and not self._assistant_ended:
            typer.echo()
            self._assistant_ended = True
        elif final and not self._assistant_started:
            self._assistant_ended = True

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


def _read_line_suffix(arguments: dict[str, Any]) -> str:
    offset = arguments.get("offset")
    limit = arguments.get("limit")
    if offset is None and limit is None:
        return ""
    start = 1 if offset is None else max(1, int(offset))
    if limit is None:
        return f":{start}-"
    return f":{start}-{start + max(1, int(limit)) - 1}"


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
        preview = f"{preview}\n\n  [Output truncated for terminal safety: {', '.join(details)} hidden]"
    return preview
            