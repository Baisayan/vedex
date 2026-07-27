from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import TypeAdapter, ValidationError
from vedex.schema import (
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    CancellationToken,
    ErrorEvent,
    JSONValue,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

from .conftest import run_async


def test_message_contracts_round_trip_through_discriminated_union() -> None:
    adapter: TypeAdapter[UserMessage | AssistantMessage | ToolResultMessage] = TypeAdapter(
        UserMessage | AssistantMessage | ToolResultMessage
    )
    messages: list[UserMessage | AssistantMessage | ToolResultMessage] = [
        UserMessage(content="hello"),
        AssistantMessage(content="working", tool_calls=[ToolCall(id="call-1", name="read")]),
        ToolResultMessage(tool_call_id="call-1", name="read", content="contents"),
    ]

    assert [adapter.validate_json(adapter.dump_json(message)) for message in messages] == messages


@pytest.mark.parametrize(
    ("model", "value"),
    [
        (UserMessage, {"content": "ok", "unexpected": True}),
        (ToolCall, {"id": "call", "name": "read", "arguments": {"bad": object()}}),
        (ErrorEvent, {"message": "no", "extra": "field"}),
    ],
)
def test_pydantic_contracts_reject_unknown_or_non_json_values(
    model: type[UserMessage] | type[ToolCall] | type[ErrorEvent],
    value: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(value)


def test_agent_tool_forwards_arguments_and_cancellation_token() -> None:
    token = _Token()
    received: dict[str, object] = {}

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        received["arguments"] = dict(arguments)
        received["signal"] = signal
        return AgentToolResult(tool_call_id="call", name="check", ok=True, content="ok")

    tool = AgentTool(
        name="check",
        description="Check input forwarding.",
        input_schema={"type": "object"},
        executor=execute,
    )

    result = run_async(tool.execute({"path": "file.txt"}, signal=token))

    assert result.ok is True
    assert received == {"arguments": {"path": "file.txt"}, "signal": token}


class _Token:
    def is_cancelled(self) -> bool:
        return False
