from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from json import dumps

from .schema import AgentMessage, AgentTool

_CHARS_PER_TOKEN = 4
_MESSAGE_OVERHEAD_TOKENS = 4
_TOOL_OVERHEAD_TOKENS = 16


@dataclass(frozen=True, slots=True)
class ContextEstimate:
    """A provider-neutral, deliberately approximate context-use estimate."""

    total_tokens: int
    system_tokens: int
    message_tokens: int
    tool_tokens: int
    message_count: int
    tool_count: int


@dataclass(frozen=True, slots=True)
class ContextDecision:
    """The immutable result of applying a context limit to a transcript."""

    messages: tuple[AgentMessage, ...]
    estimate: ContextEstimate
    dropped_turns: int
    fits: bool


@dataclass(frozen=True, slots=True)
class ContextTrim:
    """The result of removing one complete oldest user-led turn."""

    messages: tuple[AgentMessage, ...]
    dropped: bool


def apply_context_policy(
    *,
    system: str,
    messages: Sequence[AgentMessage],
    tools: Sequence[AgentTool],
    max_tokens: int,
    reserve_tokens: int = 0,
) -> ContextDecision:
    """Drop complete oldest turns until the normalized request fits.

    The newest user-led turn is always retained. The function never mutates the
    input sequence, and every retained history begins at a user-message boundary.
    """

    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if reserve_tokens < 0:
        raise ValueError("reserve_tokens must not be negative")
    if reserve_tokens >= max_tokens:
        raise ValueError("reserve_tokens must be smaller than max_tokens")

    retained = tuple(messages)
    dropped_turns = 0
    budget = max_tokens - reserve_tokens
    estimate = estimate_context_usage(system=system, messages=retained, tools=tools)

    while estimate.total_tokens > budget:
        trim = drop_oldest_user_turn(retained)
        if not trim.dropped:
            return ContextDecision(
                messages=retained,
                estimate=estimate,
                dropped_turns=dropped_turns,
                fits=False,
            )
        retained = trim.messages
        dropped_turns += 1
        estimate = estimate_context_usage(system=system, messages=retained, tools=tools)

    return ContextDecision(
        messages=retained,
        estimate=estimate,
        dropped_turns=dropped_turns,
        fits=True,
    )


def drop_oldest_user_turn(messages: Sequence[AgentMessage]) -> ContextTrim:
    """Remove exactly one complete old turn while preserving the newest turn."""

    user_indexes = [index for index, message in enumerate(messages) if message.role == "user"]
    if len(user_indexes) < 2:
        return ContextTrim(messages=tuple(messages), dropped=False)

    return ContextTrim(messages=tuple(messages[user_indexes[1] :]), dropped=True)


def estimate_context_usage(
    *,
    system: str,
    messages: Sequence[AgentMessage],
    tools: Sequence[AgentTool],
) -> ContextEstimate:
    system_tokens = estimate_text_tokens(system)
    message_tokens = sum(estimate_message_tokens(message) for message in messages)
    tool_tokens = sum(estimate_tool_tokens(tool) for tool in tools)
    return ContextEstimate(
        total_tokens=system_tokens + message_tokens + tool_tokens,
        system_tokens=system_tokens,
        message_tokens=message_tokens,
        tool_tokens=tool_tokens,
        message_count=len(messages),
        tool_count=len(tools),
    )


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def estimate_message_tokens(message: AgentMessage) -> int:
    if message.role == "user":
        return _MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(message.content)

    if message.role == "assistant":
        tool_call_tokens = sum(
            estimate_text_tokens(call.name) + estimate_text_tokens(_json_text(dict(call.arguments)))
            for call in message.tool_calls
        )
        metadata_tokens = estimate_text_tokens(_json_text(message.metadata))
        return (
            _MESSAGE_OVERHEAD_TOKENS
            + estimate_text_tokens(message.content)
            + tool_call_tokens
            + metadata_tokens
        )

    structured_tokens = estimate_text_tokens(
        _json_text(
            {
                "data": message.data,
                "details": message.details,
                "error": message.error,
            }
        )
    )
    return (
        _MESSAGE_OVERHEAD_TOKENS
        + estimate_text_tokens(message.name)
        + estimate_text_tokens(message.content)
        + structured_tokens
    )


def estimate_tool_tokens(tool: AgentTool) -> int:
    return (
        _TOOL_OVERHEAD_TOKENS
        + estimate_text_tokens(tool.name)
        + estimate_text_tokens(tool.description)
        + estimate_text_tokens(_json_text(dict(tool.input_schema)))
    )


def _json_text(value: object) -> str:
    return dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = [
    "ContextDecision",
    "ContextEstimate",
    "ContextTrim",
    "apply_context_policy",
    "drop_oldest_user_turn",
    "estimate_context_usage",
    "estimate_message_tokens",
    "estimate_text_tokens",
    "estimate_tool_tokens",
]
