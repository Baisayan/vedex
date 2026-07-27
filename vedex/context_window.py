from __future__ import annotations

from dataclasses import dataclass

from .schema import AgentMessage, AgentTool

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_OVERHEAD_TOKENS = 16
DEFAULT_CONTEXT_WINDOW_TOKENS = 32_000
DEFAULT_COMPACTION_RESERVE_TOKENS = 4_096
DEFAULT_COMPACTION_KEEP_RECENT_TOKENS = 4_096


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
