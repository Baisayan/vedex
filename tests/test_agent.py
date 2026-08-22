from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import cast

import pytest
from vedex.agent import Agent, AgentLimits
from vedex.models import (
    FakeAdapter,
    ModelAdapter,
    ModelCancelledEvent,
    ModelCompletedEvent,
    ModelEvent,
    ModelFailureEvent,
    ModelRequest,
    ModelSettings,
    ModelStartEvent,
    ModelTextDeltaEvent,
    ModelThinkingDeltaEvent,
    Usage,
)
from vedex.schema import (
    AgentEndEvent,
    AgentEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    CancellationToken,
    ErrorEvent,
    FatalEnvironmentError,
    JSONValue,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResultMessage,
    UserMessage,
)

from .conftest import make_tool, run_async


async def _collect(agent: Agent, content: str) -> list[AgentEvent]:
    return [event async for event in agent.run(content)]


def _agent(
    adapter: ModelAdapter,
    *,
    tools: list[AgentTool] | None = None,
    limits: AgentLimits | None = None,
    system_prompt: str = "You are a coding agent.",
) -> Agent:
    return Agent(
        adapter=adapter,
        settings=ModelSettings(model="fake-model", options={"temperature": 0}),
        system_prompt=system_prompt,
        tools=() if tools is None else tools,
        limits=limits,
    )


def _end_event(events: list[AgentEvent]) -> AgentEndEvent:
    end_events = [event for event in events if isinstance(event, AgentEndEvent)]
    assert len(end_events) == 1
    return end_events[0]


def _completed_stream(
    message: AssistantMessage,
    *,
    usage: Usage | None = None,
) -> list[ModelEvent]:
    return [
        ModelStartEvent(),
        ModelCompletedEvent(message=message, usage=usage or Usage()),
    ]


def test_agent_streams_normalized_events_and_completes_with_usage() -> None:
    assistant = AssistantMessage(
        content="Done.",
        metadata={"continuation": {"id": "response-1"}},
    )
    usage = Usage(
        input_tokens=10,
        output_tokens=3,
        cached_tokens=2,
        thinking_tokens=1,
    )
    fake = FakeAdapter(
        [
            [
                ModelStartEvent(),
                ModelThinkingDeltaEvent(delta="Checking."),
                ModelTextDeltaEvent(delta="Done."),
                ModelCompletedEvent(message=assistant, usage=usage),
            ]
        ]
    )
    agent = _agent(fake)

    events = run_async(_collect(agent, "Fix it"))

    assert any(
        isinstance(event, ThinkingDeltaEvent) and event.delta == "Checking." for event in events
    )
    assert any(isinstance(event, MessageDeltaEvent) and event.delta == "Done." for event in events)
    assert agent.messages == (UserMessage(content="Fix it"), assistant)
    assert fake.requests[0].messages == [UserMessage(content="Fix it")]
    assert _end_event(events).status == "completed"
    assert agent.last_result is not None
    assert agent.last_result.status == "completed"
    assert agent.last_result.turns == 1
    assert agent.last_result.tool_calls == 0
    assert agent.usage == usage


def test_agent_replaces_system_prompt_without_changing_history() -> None:
    adapter = FakeAdapter(
        [
            _completed_stream(AssistantMessage(content="first")),
            _completed_stream(AssistantMessage(content="second")),
        ]
    )
    agent = _agent(adapter, system_prompt="old prompt")

    run_async(_collect(agent, "one"))
    messages_before = agent.messages
    agent.set_system_prompt("new prompt")
    run_async(_collect(agent, "two"))

    assert agent.system_prompt == "new prompt"
    assert agent.messages[: len(messages_before)] == messages_before
    assert adapter.requests[0].system == "old prompt"
    assert adapter.requests[1].system == "new prompt"


def test_agent_rejects_prompt_changes_and_reset_during_an_active_run() -> None:
    agent = _agent(_BlockingAdapter())

    async def exercise() -> None:
        events = cast(AsyncGenerator[AgentEvent, None], agent.run("wait"))
        while True:
            event = await anext(events)
            if isinstance(event, MessageStartEvent) and event.message_role == "assistant":
                break

        with pytest.raises(RuntimeError, match="change the system prompt"):
            agent.set_system_prompt("new")
        with pytest.raises(RuntimeError, match="reset"):
            agent.reset()
        await events.aclose()

    run_async(exercise())

    assert agent.is_running is False


def test_agent_executes_tools_sequentially_and_keeps_ordinary_failures() -> None:
    execution_order: list[str] = []

    async def first_executor(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        execution_order.append("first")
        return AgentToolResult(
            tool_call_id="adapter-must-correct-this",
            name="wrong-name",
            ok=True,
            content="first output",
        )

    async def failing_executor(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        execution_order.append("failing")
        raise ValueError("ordinary tool failure")

    tools = [
        AgentTool(
            name="first",
            description="First tool",
            input_schema={"type": "object"},
            executor=first_executor,
        ),
        AgentTool(
            name="failing",
            description="Failing tool",
            input_schema={"type": "object"},
            executor=failing_executor,
        ),
    ]
    fake = FakeAdapter(
        [
            _completed_stream(
                AssistantMessage(
                    tool_calls=[
                        ToolCall(id="call-1", name="first"),
                        ToolCall(id="call-2", name="failing"),
                    ]
                ),
                usage=Usage(input_tokens=10, output_tokens=2),
            ),
            _completed_stream(
                AssistantMessage(content="Finished after observing both results."),
                usage=Usage(
                    input_tokens=20,
                    output_tokens=5,
                    cached_tokens=4,
                    thinking_tokens=3,
                ),
            ),
        ]
    )
    agent = _agent(fake, tools=tools)

    events = run_async(_collect(agent, "Use both tools"))

    assert execution_order == ["first", "failing"]
    results = [message for message in agent.messages if isinstance(message, ToolResultMessage)]
    assert [(result.tool_call_id, result.name, result.ok) for result in results] == [
        ("call-1", "first", True),
        ("call-2", "failing", False),
    ]
    assert "ordinary tool failure" in results[1].content
    assert [message.role for message in fake.requests[1].messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert _end_event(events).status == "completed"
    assert agent.last_result is not None
    assert agent.last_result.turns == 2
    assert agent.last_result.tool_calls == 2
    assert agent.usage == Usage(
        input_tokens=30,
        output_tokens=7,
        cached_tokens=4,
        thinking_tokens=3,
    )


def test_agent_returns_normalized_model_failure() -> None:
    fake = FakeAdapter(
        [[ModelFailureEvent(kind="unavailable", message="synthetic outage", retryable=True)]]
    )
    agent = _agent(fake)

    events = run_async(_collect(agent, "Try"))

    assert _end_event(events).status == "model_failure"
    assert any(
        isinstance(event, ErrorEvent)
        and event.recoverable
        and event.data
        == {
            "status": "model_failure",
            "model_failure_kind": "unavailable",
        }
        for event in events
    )
    assert agent.messages == (UserMessage(content="Try"),)


@pytest.mark.parametrize(
    "stream",
    [
        [],
        [ModelStartEvent()],
        [ModelTextDeltaEvent(delta="early")],
        [ModelCompletedEvent(message=AssistantMessage(content="early"))],
        [ModelStartEvent(), ModelStartEvent()],
        [
            ModelStartEvent(),
            ModelCompletedEvent(message=AssistantMessage(content="one")),
            ModelCompletedEvent(message=AssistantMessage(content="two")),
        ],
    ],
)
def test_agent_rejects_malformed_model_streams(stream: list[ModelEvent]) -> None:
    agent = _agent(FakeAdapter([stream]))

    events = run_async(_collect(agent, "Malformed please"))

    assert _end_event(events).status == "malformed_model_stream"
    assert agent.messages == (UserMessage(content="Malformed please"),)


def test_agent_normalizes_unexpected_adapter_exception_as_model_failure() -> None:
    agent = _agent(_ExplodingAdapter())

    events = run_async(_collect(agent, "Explode"))

    assert _end_event(events).status == "model_failure"
    assert any(
        isinstance(event, ErrorEvent) and "adapter exploded" in event.message for event in events
    )


def test_agent_rejects_unsupported_runtime_model_event() -> None:
    agent = _agent(_UnknownEventAdapter())

    events = run_async(_collect(agent, "Unknown event"))

    assert _end_event(events).status == "malformed_model_stream"


def test_agent_cancel_stops_active_model_stream() -> None:
    fake = FakeAdapter(
        [
            [
                ModelStartEvent(),
                ModelTextDeltaEvent(delta="must not arrive"),
                ModelCompletedEvent(message=AssistantMessage(content="must not complete")),
            ]
        ]
    )
    agent = _agent(fake)

    async def cancel_after_response_start() -> list[AgentEvent]:
        stream = agent.run("Cancel me")
        events: list[AgentEvent] = []
        while True:
            event = await anext(stream)
            events.append(event)
            if isinstance(event, MessageStartEvent) and event.message_role == "assistant":
                break
        agent.cancel()
        events.extend([event async for event in stream])
        return events

    events = run_async(cancel_after_response_start())

    assert _end_event(events).status == "cancelled"
    assert not any(isinstance(event, MessageDeltaEvent) for event in events)
    assert agent.messages == (UserMessage(content="Cancel me"),)


def test_agent_time_limit_interrupts_model_stream() -> None:
    agent = _agent(
        _BlockingAdapter(),
        limits=AgentLimits(time_limit_seconds=0.01),
    )

    events = run_async(_collect(agent, "Wait forever"))

    assert _end_event(events).status == "timed_out"
    assert agent.last_result is not None
    assert agent.last_result.turns == 1


def test_agent_time_limit_interrupts_tool_and_records_failure() -> None:
    async def block(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    tool = AgentTool(
        name="block",
        description="Never finishes",
        input_schema={"type": "object"},
        executor=block,
    )
    fake = FakeAdapter(
        [_completed_stream(AssistantMessage(tool_calls=[ToolCall(id="call-1", name="block")]))]
    )
    agent = _agent(
        fake,
        tools=[tool],
        limits=AgentLimits(time_limit_seconds=0.01),
    )

    events = run_async(_collect(agent, "Block"))

    assert _end_event(events).status == "timed_out"
    assert isinstance(agent.messages[-1], ToolResultMessage)
    assert agent.messages[-1].ok is False


def test_agent_cancellation_after_tool_start_prevents_execution() -> None:
    executed = False

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        nonlocal executed
        del arguments, signal
        executed = True
        return AgentToolResult(tool_call_id="", name="tool", ok=True, content="ran")

    tool = AgentTool(
        name="tool",
        description="Tool",
        input_schema={"type": "object"},
        executor=execute,
    )
    fake = FakeAdapter(
        [_completed_stream(AssistantMessage(tool_calls=[ToolCall(id="call-1", name="tool")]))]
    )
    agent = _agent(fake, tools=[tool])

    async def cancel_at_tool_start() -> list[AgentEvent]:
        stream = agent.run("Cancel before tool")
        events: list[AgentEvent] = []
        while True:
            event = await anext(stream)
            events.append(event)
            if isinstance(event, ToolExecutionStartEvent):
                break
        agent.cancel()
        events.extend([event async for event in stream])
        return events

    events = run_async(cancel_at_tool_start())

    assert _end_event(events).status == "cancelled"
    assert executed is False
    assert isinstance(agent.messages[-1], ToolResultMessage)
    assert agent.messages[-1].ok is False


def test_agent_turn_limit_stops_after_tool_results_without_orphans() -> None:
    fake = FakeAdapter(
        [_completed_stream(AssistantMessage(tool_calls=[ToolCall(id="call-1", name="test_tool")]))]
    )
    agent = _agent(
        fake,
        tools=[make_tool()],
        limits=AgentLimits(max_turns=1),
    )

    events = run_async(_collect(agent, "One turn only"))

    assert _end_event(events).status == "turn_limit"
    assert [message.role for message in agent.messages] == ["user", "assistant", "tool"]
    assert len(fake.requests) == 1


def test_agent_tool_call_limit_records_unexecuted_results() -> None:
    executed: list[str] = []

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        executed.append("tool")
        return AgentToolResult(tool_call_id="", name="tool", ok=True, content="ok")

    tool = AgentTool(
        name="tool",
        description="Tool",
        input_schema={"type": "object"},
        executor=execute,
    )
    fake = FakeAdapter(
        [
            _completed_stream(
                AssistantMessage(
                    tool_calls=[
                        ToolCall(id="call-1", name="tool"),
                        ToolCall(id="call-2", name="tool"),
                    ]
                )
            )
        ]
    )
    agent = _agent(
        fake,
        tools=[tool],
        limits=AgentLimits(max_tool_calls=1),
    )

    events = run_async(_collect(agent, "Use twice"))

    assert _end_event(events).status == "tool_call_limit"
    assert executed == ["tool"]
    results = [message for message in agent.messages if isinstance(message, ToolResultMessage)]
    assert len(results) == 2
    assert results[0].ok is True
    assert results[1].ok is False
    assert "max_tool_calls=1" in results[1].content
    assert agent.last_result is not None
    assert agent.last_result.tool_calls == 1


def test_agent_fatal_environment_failure_is_terminal_but_recorded() -> None:
    fatal_tool = make_tool(raises=FatalEnvironmentError("container disappeared"))
    fake = FakeAdapter(
        [_completed_stream(AssistantMessage(tool_calls=[ToolCall(id="call-1", name="test_tool")]))]
    )
    agent = _agent(fake, tools=[fatal_tool])

    events = run_async(_collect(agent, "Run in container"))

    assert _end_event(events).status == "fatal_environment_failure"
    assert isinstance(agent.messages[-1], ToolResultMessage)
    assert agent.messages[-1].ok is False
    assert "container disappeared" in agent.messages[-1].content


def test_agent_preflight_context_limit_avoids_model_request() -> None:
    fake = FakeAdapter([_completed_stream(AssistantMessage(content="unused"))])
    agent = _agent(
        fake,
        limits=AgentLimits(max_context_tokens=10),
        system_prompt="system prompt that is already much too large" * 10,
    )

    events = run_async(_collect(agent, "newest request"))

    assert _end_event(events).status == "context_limit"
    assert fake.requests == []
    assert agent.messages == (UserMessage(content="newest request"),)


def test_agent_retries_one_normalized_context_overflow_after_coherent_trim() -> None:
    fake = FakeAdapter(
        [
            _completed_stream(AssistantMessage(content="old answer")),
            [ModelFailureEvent(kind="context_overflow", message="too many tokens")],
            _completed_stream(AssistantMessage(content="retried answer")),
        ]
    )
    agent = _agent(fake)
    run_async(_collect(agent, "old request"))

    events = run_async(_collect(agent, "new request"))

    assert _end_event(events).status == "completed"
    assert len(fake.requests) == 3
    assert [message.role for message in fake.requests[1].messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert fake.requests[2].messages == [UserMessage(content="new request")]
    assert agent.messages == (
        UserMessage(content="new request"),
        AssistantMessage(content="retried answer"),
    )


def test_agent_retries_context_overflow_only_once_per_run() -> None:
    fake = FakeAdapter(
        [
            _completed_stream(AssistantMessage(content="first answer")),
            _completed_stream(AssistantMessage(content="second answer")),
            [ModelFailureEvent(kind="context_overflow", message="first overflow")],
            [ModelFailureEvent(kind="context_overflow", message="second overflow")],
        ]
    )
    agent = _agent(fake)
    run_async(_collect(agent, "first request"))
    run_async(_collect(agent, "second request"))

    events = run_async(_collect(agent, "third request"))

    assert _end_event(events).status == "context_limit"
    assert len(fake.requests) == 4
    assert [message.content for message in agent.messages if isinstance(message, UserMessage)] == [
        "second request",
        "third request",
    ]


def test_agent_accepts_explicit_model_cancellation_without_response_start() -> None:
    agent = _agent(FakeAdapter([[ModelCancelledEvent(message="adapter cancelled")]]))

    events = run_async(_collect(agent, "Cancel"))

    assert _end_event(events).status == "cancelled"
    assert agent.last_result is not None
    assert agent.last_result.message == "adapter cancelled"


def test_agent_emits_tool_result_events_for_every_appended_result() -> None:
    fake = FakeAdapter(
        [
            _completed_stream(AssistantMessage(tool_calls=[ToolCall(id="call-1", name="unknown")])),
            _completed_stream(AssistantMessage(content="Observed the failure")),
        ]
    )
    agent = _agent(fake)

    events = run_async(_collect(agent, "Call unknown"))

    execution_results = [
        event.result for event in events if isinstance(event, ToolExecutionEndEvent)
    ]
    tool_messages = [
        event.message
        for event in events
        if isinstance(event, MessageEndEvent) and isinstance(event.message, ToolResultMessage)
    ]
    assert len(execution_results) == len(tool_messages) == 1
    assert execution_results[0].ok is False
    assert tool_messages[0].tool_call_id == execution_results[0].tool_call_id
    assert _end_event(events).status == "completed"


def test_agent_limits_and_duplicate_tools_are_validated() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        AgentLimits(max_turns=0)
    with pytest.raises(ValueError, match="context_reserve_tokens"):
        AgentLimits(max_context_tokens=10, context_reserve_tokens=10)
    duplicate = make_tool(name="same")
    with pytest.raises(ValueError, match="unique"):
        _agent(FakeAdapter(), tools=[duplicate, duplicate])


class _ExplodingAdapter:
    def stream(
        self,
        request: ModelRequest,
        *,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ModelEvent]:
        del request, signal

        async def events() -> AsyncIterator[ModelEvent]:
            yield ModelStartEvent()
            raise RuntimeError("adapter exploded")

        return events()


class _BlockingAdapter:
    def stream(
        self,
        request: ModelRequest,
        *,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ModelEvent]:
        del request, signal

        async def events() -> AsyncIterator[ModelEvent]:
            yield ModelStartEvent()
            await asyncio.Event().wait()

        return events()


class _UnknownEventAdapter:
    def stream(
        self,
        request: ModelRequest,
        *,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ModelEvent]:
        del request, signal

        async def events() -> AsyncIterator[ModelEvent]:
            yield cast(ModelEvent, object())

        return events()
