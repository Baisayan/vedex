from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agent.messages import AgentMessage
from agent.tools import AgentTool
from agent.types import JSONValue

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_OVERHEAD_TOKENS = 16
DEFAULT_CONTEXT_WINDOW_TOKENS = 32_000
DEFAULT_COMPACTION_RESERVE_TOKENS = 4_096
DEFAULT_COMPACTION_KEEP_RECENT_TOKENS = 4_096
COMPACTION_SUMMARY_PREFIX = "Previous conversation summary:\n"


@dataclass(frozen=True, slots=True)
class ContextUsageEstimate:
    total_tokens: int
    system_tokens: int
    message_tokens: int
    tool_tokens: int
    message_count: int
    tool_count: int


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: AgentMessage) -> int:
    match message.role:
        case "user":
            return MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(message.content)
        case "assistant":
            tool_call_tokens = sum(
                estimate_text_tokens(call.name) + estimate_text_tokens(str(call.arguments))
                for call in message.tool_calls
            )
            return (
                MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(message.content) + tool_call_tokens
            )
        case "tool":
            return (
                MESSAGE_OVERHEAD_TOKENS
                + estimate_text_tokens(message.name)
                + estimate_text_tokens(message.content)
            )


def estimate_tool_tokens(tool: AgentTool) -> int:
    return (
        TOOL_OVERHEAD_TOKENS
        + estimate_text_tokens(tool.name)
        + estimate_text_tokens(tool.description)
        + estimate_text_tokens(str(tool.input_schema))
    )


def auto_compaction_threshold_for_context_window(context_window_tokens: int) -> int | None:
    if context_window_tokens <= 0:
        return None
    return max(1, context_window_tokens - DEFAULT_COMPACTION_RESERVE_TOKENS)


def estimate_context_usage(
    *,
    system: str,
    messages: tuple[AgentMessage, ...],
    tools: tuple[AgentTool, ...],
) -> ContextUsageEstimate:
    system_tokens = estimate_text_tokens(system)
    message_tokens = sum(estimate_message_tokens(message) for message in messages)
    tool_tokens = sum(estimate_tool_tokens(tool) for tool in tools)
    return ContextUsageEstimate(
        total_tokens=system_tokens + message_tokens + tool_tokens,
        system_tokens=system_tokens,
        message_tokens=message_tokens,
        tool_tokens=tool_tokens,
        message_count=len(messages),
        tool_count=len(tools),
    )


def build_truncation_summary(messages: tuple[AgentMessage, ...]) -> str:
    """Build a deterministic summary from old messages without an LLM call."""
    if not messages:
        return "No prior messages."

    files_read: list[str] = []
    files_written: list[str] = []
    commands: list[str] = []
    errors: list[str] = []
    first_user_content: str | None = None
    last_assistant_content: str | None = None

    for message in messages:
        if (
            message.role == "user"
            and message.content.startswith(COMPACTION_SUMMARY_PREFIX)
        ):
            continue

        if message.role == "user" and first_user_content is None:
            first_user_content = message.content

        if message.role == "assistant":
            if message.content:
                last_assistant_content = message.content
            for call in message.tool_calls:
                _extract_tool_signal(
                    call.name,
                    call.arguments,
                    files_read=files_read,
                    files_written=files_written,
                    commands=commands,
                )

        if message.role == "tool" and not message.ok:
            error_text = _truncate_text(message.content, limit=150)
            errors.append(f"{message.name}: {error_text}")

    parts: list[str] = [f"Context compacted ({len(messages)} messages):"]

    if first_user_content:
        parts.append(f"Goal: {_truncate_text(first_user_content, limit=200)}")

    if files_read:
        parts.append(f"Files read: {', '.join(files_read)}")

    if files_written:
        parts.append(f"Files modified: {', '.join(files_written)}")

    if commands:
        parts.append(f"Commands: {'; '.join(commands)}")

    if errors:
        parts.append(f"Errors: {'; '.join(errors[:5])}")

    if last_assistant_content:
        parts.append(f"Last: {_truncate_text(last_assistant_content, limit=200)}")

    return "\n".join(parts)


def _extract_tool_signal(
    name: str,
    arguments: Mapping[str, JSONValue],
    *,
    files_read: list[str],
    files_written: list[str],
    commands: list[str],
) -> None:
    match name:
        case "read":
            path = arguments.get("path")
            if isinstance(path, str) and path not in files_read:
                files_read.append(path)
        case "write" | "edit":
            path = arguments.get("path")
            if isinstance(path, str) and path not in files_written:
                files_written.append(path)
        case "bash":
            command = arguments.get("command")
            if isinstance(command, str):
                truncated = _truncate_text(command, limit=100)
                if truncated not in commands:
                    commands.append(truncated)


def _truncate_text(text: str, *, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."
