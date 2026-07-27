from __future__ import annotations

from typing import Any

from vedex.core import run_agent_loop, tool_result_message
from vedex.schema import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AgentToolResult,
    AssistantMessage,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    ThinkingDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
    UserMessage,
)

from .conftest import (
    make_tool,
    native_ollama_client,
    native_tags_response,
    ndjson_response,
    run_async,
)


async def _events(**kwargs: Any) -> list[AgentEvent]:
    return [event async for event in run_agent_loop(**kwargs)]


def test_agent_loop_emits_stream_events_without_mutating_history() -> None:
    messages = [UserMessage(content="hello")]

    def handler(request: Any) -> Any:
        if request.url.path == "/api/tags":
            return native_tags_response()
        return ndjson_response(
            {"message": {"thinking": "plan", "content": "Hel"}},
            {"message": {"content": "lo"}, "done": True},
        )

    client = native_ollama_client(handler)
    try:
        events = run_async(
            _events(
                client=client, model="local:latest", system="system", messages=messages, tools=[]
            )
        )
    finally:
        run_async(client.aclose())

    assert isinstance(events[0], AgentStartEvent)
    assert any(isinstance(event, TurnStartEvent) for event in events)
    assert any(isinstance(event, ThinkingDeltaEvent) and event.delta == "plan" for event in events)
    assert [event.delta for event in events if isinstance(event, MessageDeltaEvent)] == [
        "Hel",
        "lo",
    ]
    completed = next(event.message for event in events if isinstance(event, MessageEndEvent))
    assert isinstance(completed, AssistantMessage)
    assert completed.content == "Hello"
    assert isinstance(events[-1], AgentEndEvent)
    assert messages == [UserMessage(content="hello")]


def test_agent_loop_rejects_missing_or_non_tool_model() -> None:
    missing_client = native_ollama_client(
        lambda _request: native_tags_response(name="other:latest")
    )
    unsupported_client = native_ollama_client(
        lambda _request: native_tags_response(supports_tools=False)
    )
    tool = make_tool()
    try:
        missing = run_async(
            _events(client=missing_client, model="missing", system="", messages=[], tools=[])
        )
        unsupported = run_async(
            _events(
                client=unsupported_client,
                model="local:latest",
                system="",
                messages=[],
                tools=[tool],
            )
        )
    finally:
        run_async(missing_client.aclose())
        run_async(unsupported_client.aclose())

    assert any(
        isinstance(event, ErrorEvent) and "not available" in event.message for event in missing
    )
    assert any(
        isinstance(event, ErrorEvent) and "does not support tools" in event.message
        for event in unsupported
    )


def test_agent_loop_emits_tool_events_and_turn_limit_error() -> None:
    def handler(request: Any) -> Any:
        if request.url.path == "/api/tags":
            return native_tags_response()
        return ndjson_response(
            {
                "message": {
                    "tool_calls": [{"function": {"name": "test_tool", "arguments": {}}}],
                },
                "done": True,
            }
        )

    client = native_ollama_client(handler)
    try:
        events = run_async(
            _events(
                client=client,
                model="local:latest",
                system="",
                messages=[],
                tools=[make_tool()],
                max_turns=1,
            )
        )
    finally:
        run_async(client.aclose())

    assert any(isinstance(event, ToolExecutionStartEvent) for event in events)
    assert any(isinstance(event, ToolExecutionEndEvent) and event.result.ok for event in events)
    assert any(isinstance(event, TurnEndEvent) for event in events)
    assert any(isinstance(event, ErrorEvent) and "max_turns" in event.message for event in events)


def test_agent_loop_turns_tool_failures_and_bad_streams_into_events() -> None:
    responses = [
        ndjson_response(
            {
                "message": {
                    "tool_calls": [{"function": {"name": "unknown", "arguments": {}}}],
                },
                "done": True,
            }
        ),
        ndjson_response({"message": {"content": "partial"}}),
    ]

    def handler(request: Any) -> Any:
        if request.url.path == "/api/tags":
            return native_tags_response()
        return responses.pop(0)

    client = native_ollama_client(handler)
    try:
        unknown_tool_events = run_async(
            _events(
                client=client,
                model="local:latest",
                system="",
                messages=[],
                tools=[],
                max_turns=1,
            )
        )
        incomplete_events = run_async(
            _events(client=client, model="local:latest", system="", messages=[], tools=[])
        )
    finally:
        run_async(client.aclose())

    failed_result = next(
        event.result for event in unknown_tool_events if isinstance(event, ToolExecutionEndEvent)
    )
    assert failed_result.ok is False
    assert failed_result.name == "unknown"
    assert any(
        isinstance(event, ErrorEvent) and "ended before" in event.message
        for event in incomplete_events
    )
    assert not any(isinstance(event, MessageEndEvent) for event in incomplete_events)


def test_tool_result_message_preserves_error_and_data_context() -> None:
    result = tool_result_message(
        AgentToolResult(
            tool_call_id="call-1",
            name="read",
            ok=False,
            content="failed",
            error="permission denied",
            data={"path": "secret"},
        )
    )

    assert result.ok is False
    assert "permission denied" in result.content
    assert result.data == {"path": "secret"}


def test_agent_loop_handles_invalid_turn_limits_bad_tool_payloads_and_cancellation() -> None:
    class Cancelled:
        def is_cancelled(self) -> bool:
            return True

    bad_tool_client = native_ollama_client(
        lambda request: (
            native_tags_response()
            if request.url.path == "/api/tags"
            else ndjson_response(
                {"message": {"tool_calls": [{"function": {"arguments": {}}}]}, "done": True}
            )
        )
    )
    idle_client = native_ollama_client(lambda _request: native_tags_response())
    try:
        invalid_limit = run_async(
            _events(
                client=idle_client,
                model="local:latest",
                system="",
                messages=[],
                tools=[],
                max_turns=0,
            )
        )
        malformed_tool = run_async(
            _events(client=bad_tool_client, model="local:latest", system="", messages=[], tools=[])
        )
        cancelled = run_async(
            _events(
                client=idle_client,
                model="local:latest",
                system="",
                messages=[],
                tools=[],
                signal=Cancelled(),
            )
        )
    finally:
        run_async(bad_tool_client.aclose())
        run_async(idle_client.aclose())

    assert any(
        isinstance(event, ErrorEvent) and "max_turns" in event.message for event in invalid_limit
    )
    assert any(
        isinstance(event, ErrorEvent) and "without a name" in event.message
        for event in malformed_tool
    )
    assert any(isinstance(event, ErrorEvent) and event.recoverable for event in cancelled)


def test_agent_loop_corrects_tool_result_ids_and_catches_executor_exceptions() -> None:
    responses = [
        ndjson_response(
            {
                "message": {
                    "tool_calls": [{"function": {"name": "mismatch", "arguments": {}}}],
                },
                "done": True,
            }
        ),
        ndjson_response(
            {
                "message": {
                    "tool_calls": [{"function": {"name": "explode", "arguments": {}}}],
                },
                "done": True,
            }
        ),
    ]

    def handler(request: Any) -> Any:
        if request.url.path == "/api/tags":
            return native_tags_response()
        return responses.pop(0)

    client = native_ollama_client(handler)
    mismatch = make_tool(
        name="mismatch",
        result=AgentToolResult(tool_call_id="wrong", name="mismatch", ok=True, content="ok"),
    )
    explode = make_tool(name="explode", raises=RuntimeError("boom"))
    try:
        mismatch_events = run_async(
            _events(
                client=client,
                model="local:latest",
                system="",
                messages=[],
                tools=[mismatch],
                max_turns=1,
            )
        )
        exploded_events = run_async(
            _events(
                client=client,
                model="local:latest",
                system="",
                messages=[],
                tools=[explode],
                max_turns=1,
            )
        )
    finally:
        run_async(client.aclose())

    mismatch_result = next(
        event.result for event in mismatch_events if isinstance(event, ToolExecutionEndEvent)
    )
    exploded_result = next(
        event.result for event in exploded_events if isinstance(event, ToolExecutionEndEvent)
    )
    assert mismatch_result.tool_call_id.startswith("call-")
    assert exploded_result.ok is False
    assert exploded_result.error == "boom"
