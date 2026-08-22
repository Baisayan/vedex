from __future__ import annotations

from vedex.context import (
    apply_context_policy,
    estimate_context_usage,
)
from vedex.schema import (
    AgentMessage,
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def test_context_policy_removes_complete_oldest_user_led_turn() -> None:
    messages: list[AgentMessage] = [
        UserMessage(content="old request"),
        AssistantMessage(tool_calls=[ToolCall(id="call-old", name="read", arguments={})]),
        ToolResultMessage(
            tool_call_id="call-old",
            name="read",
            content="old result" * 20,
        ),
        AssistantMessage(content="old answer"),
        UserMessage(content="new request"),
    ]
    newest_only = messages[-1:]
    newest_estimate = estimate_context_usage(
        system="system",
        messages=newest_only,
        tools=(),
    )

    decision = apply_context_policy(
        system="system",
        messages=messages,
        tools=(),
        max_tokens=newest_estimate.total_tokens,
    )

    assert decision.fits is True
    assert decision.dropped_turns == 1
    assert decision.messages == (UserMessage(content="new request"),)
    assert not any(isinstance(message, ToolResultMessage) for message in decision.messages)
    assert messages[0] == UserMessage(content="old request")


def test_context_policy_keeps_newest_turn_intact_when_it_cannot_fit() -> None:
    messages: list[AgentMessage] = [
        UserMessage(content="old"),
        AssistantMessage(content="old answer"),
        UserMessage(content="newest request is deliberately large" * 10),
    ]
    newest_estimate = estimate_context_usage(
        system="",
        messages=messages[-1:],
        tools=(),
    )

    decision = apply_context_policy(
        system="",
        messages=messages,
        tools=(),
        max_tokens=newest_estimate.total_tokens - 1,
    )

    assert decision.fits is False
    assert decision.dropped_turns == 1
    assert decision.messages == (messages[-1],)


def test_context_policy_validates_limit_configuration() -> None:
    try:
        apply_context_policy(
            system="",
            messages=(),
            tools=(),
            max_tokens=10,
            reserve_tokens=10,
        )
    except ValueError as exc:
        assert "reserve_tokens" in str(exc)
    else:
        raise AssertionError("Expected an invalid context reserve to fail")
