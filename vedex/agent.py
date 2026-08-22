from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from time import monotonic

from .context import apply_context_policy, drop_oldest_user_turn
from .models import (
    ModelAdapter,
    ModelCancelledEvent,
    ModelCompletedEvent,
    ModelFailureEvent,
    ModelRequest,
    ModelSettings,
    ModelStartEvent,
    ModelTextDeltaEvent,
    ModelThinkingDeltaEvent,
    Usage,
)
from .schema import (
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    AgentStatus,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
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
    TurnEndEvent,
    TurnStartEvent,
    UserMessage,
)


@dataclass(frozen=True, slots=True)
class AgentLimits:
    max_turns: int | None = None
    max_tool_calls: int | None = None
    time_limit_seconds: float | None = None
    max_context_tokens: int | None = None
    context_reserve_tokens: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("max_turns", self.max_turns),
            ("max_tool_calls", self.max_tool_calls),
            ("max_context_tokens", self.max_context_tokens),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be greater than 0")
        if self.context_reserve_tokens < 0:
            raise ValueError("context_reserve_tokens must not be negative")
        if (
            self.max_context_tokens is not None
            and self.context_reserve_tokens >= self.max_context_tokens
        ):
            raise ValueError("context_reserve_tokens must be smaller than max_context_tokens")


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    status: AgentStatus
    usage: Usage
    turns: int
    tool_calls: int
    message: str | None = None


class _CancellationState:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


@dataclass(slots=True)
class _RunState:
    started_at: float
    cancellation: _CancellationState
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    tool_calls: int = 0


@dataclass(slots=True)
class _ModelTurnState:
    started: bool = False
    terminal: bool = False
    message: AssistantMessage | None = None
    usage: Usage | None = None
    failure: ModelFailureEvent | None = None
    cancellation_message: str | None = None
    malformed_message: str | None = None


@dataclass(slots=True)
class _ToolBatchState:
    status: AgentStatus | None = None
    message: str | None = None


class Agent:
    """Provider-neutral in-memory coordinator for one conversation."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        settings: ModelSettings,
        system_prompt: str,
        tools: Sequence[AgentTool] = (),
        limits: AgentLimits | None = None,
    ) -> None:
        tool_names = [tool.name for tool in tools]
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("Tool names must be unique")

        self._adapter = adapter
        self._settings = settings.model_copy(deep=True)
        self._system_prompt = system_prompt
        self._tools = tuple(tools)
        self._tool_by_name = {tool.name: tool for tool in self._tools}
        self._limits = limits or AgentLimits()
        self._messages: list[AgentMessage] = []
        self._active_run: _RunState | None = None
        self._last_result: AgentRunResult | None = None

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return tuple(self._messages)

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        return self._tools

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def is_running(self) -> bool:
        return self._active_run is not None

    @property
    def usage(self) -> Usage:
        if self._active_run is not None:
            return self._active_run.usage.model_copy(deep=True)
        if self._last_result is not None:
            return self._last_result.usage.model_copy(deep=True)
        return Usage()

    @property
    def last_result(self) -> AgentRunResult | None:
        return self._last_result

    def cancel(self) -> None:
        if self._active_run is not None:
            self._active_run.cancellation.cancel()

    def reset(self) -> None:
        if self.is_running:
            raise RuntimeError("Cannot reset an Agent while it is running")
        self._messages.clear()
        self._last_result = None

    def set_system_prompt(self, system_prompt: str) -> None:
        """Replace resource-derived prompt state without changing conversation history."""
        if self.is_running:
            raise RuntimeError("Cannot change the system prompt while an Agent is running")
        self._system_prompt = system_prompt

    async def run(self, content: str) -> AsyncIterator[AgentEvent]:
        if self.is_running:
            raise RuntimeError("Agent already has an active run")

        state = _RunState(started_at=monotonic(), cancellation=_CancellationState())
        self._active_run = state
        self._last_result = None
        overflow_retry_used = False

        try:
            yield AgentStartEvent()
            user_message = UserMessage(content=content)
            self._messages.append(user_message)
            yield MessageStartEvent(message_role="user")
            yield MessageEndEvent(message=user_message)

            while True:
                terminal = self._run_condition(state)
                if terminal is not None:
                    status, message = terminal
                    yield self._error(status, message, recoverable=True)
                    yield self._finish(state, status, message)
                    return

                if self._limits.max_turns is not None and state.turns >= self._limits.max_turns:
                    message = f"Agent reached max_turns={self._limits.max_turns}"
                    yield self._error("turn_limit", message, recoverable=True)
                    yield self._finish(state, "turn_limit", message)
                    return

                context_error = self._apply_context_limit()
                if context_error is not None:
                    yield self._error("context_limit", context_error, recoverable=True)
                    yield self._finish(state, "context_limit", context_error)
                    return

                state.turns += 1
                turn = state.turns
                yield TurnStartEvent(turn=turn)

                model_state = _ModelTurnState()
                while True:
                    request = ModelRequest.from_agent_inputs(
                        system=self._system_prompt,
                        messages=self._messages,
                        tools=self._tools,
                        settings=self._settings,
                    )
                    model_state = _ModelTurnState()
                    try:
                        async with asyncio.timeout(self._remaining_seconds(state)):
                            async for event in self._stream_model(
                                request=request,
                                state=model_state,
                                cancellation=state.cancellation,
                            ):
                                yield event
                    except TimeoutError:
                        message = "Agent run exceeded its time limit during a model response"
                        yield TurnEndEvent(turn=turn)
                        yield self._error("timed_out", message, recoverable=True)
                        yield self._finish(state, "timed_out", message)
                        return
                    except Exception as exc:
                        message = f"Model adapter failed: {exc}"
                        yield TurnEndEvent(turn=turn)
                        yield self._error("model_failure", message)
                        yield self._finish(state, "model_failure", message)
                        return

                    if model_state.malformed_message is not None:
                        message = model_state.malformed_message
                        yield TurnEndEvent(turn=turn)
                        yield self._error("malformed_model_stream", message)
                        yield self._finish(state, "malformed_model_stream", message)
                        return

                    if model_state.cancellation_message is not None:
                        message = model_state.cancellation_message
                        yield TurnEndEvent(turn=turn)
                        yield self._error("cancelled", message, recoverable=True)
                        yield self._finish(state, "cancelled", message)
                        return

                    if model_state.failure is not None:
                        failure = model_state.failure
                        if failure.kind == "context_overflow":
                            trim = drop_oldest_user_turn(self._messages)
                            if not overflow_retry_used and trim.dropped:
                                self._messages[:] = trim.messages
                                overflow_retry_used = True
                                continue

                            message = failure.message
                            yield TurnEndEvent(turn=turn)
                            yield self._error("context_limit", message, recoverable=True)
                            yield self._finish(state, "context_limit", message)
                            return

                        message = failure.message
                        yield TurnEndEvent(turn=turn)
                        yield self._error(
                            "model_failure",
                            message,
                            recoverable=failure.retryable,
                            data={"model_failure_kind": failure.kind},
                        )
                        yield self._finish(state, "model_failure", message)
                        return

                    assistant_message = model_state.message
                    response_usage = model_state.usage
                    if assistant_message is None or response_usage is None:
                        message = "Model stream ended without a completed assistant message"
                        yield TurnEndEvent(turn=turn)
                        yield self._error("malformed_model_stream", message)
                        yield self._finish(state, "malformed_model_stream", message)
                        return

                    self._messages.append(assistant_message)
                    state.usage = _add_usage(state.usage, response_usage)
                    break

                if not assistant_message.tool_calls:
                    yield TurnEndEvent(turn=turn)
                    yield self._finish(state, "completed", None)
                    return

                tool_state = _ToolBatchState()
                async for event in self._execute_tools(
                    assistant_message.tool_calls,
                    run_state=state,
                    batch_state=tool_state,
                ):
                    yield event

                yield TurnEndEvent(turn=turn)
                if tool_state.status is not None:
                    message = tool_state.message or "Agent tool execution stopped"
                    yield self._error(
                        tool_state.status,
                        message,
                        recoverable=tool_state.status != "fatal_environment_failure",
                    )
                    yield self._finish(state, tool_state.status, message)
                    return
        finally:
            self._active_run = None

    async def _stream_model(
        self,
        *,
        request: ModelRequest,
        state: _ModelTurnState,
        cancellation: _CancellationState,
    ) -> AsyncIterator[AgentEvent]:
        async for event in self._adapter.stream(request, signal=cancellation):
            if state.terminal:
                state.malformed_message = "Model stream emitted data after a terminal event"
                return
            if cancellation.is_cancelled():
                state.terminal = True
                state.cancellation_message = "Agent run cancelled"
                return

            if isinstance(event, ModelStartEvent):
                if state.started:
                    state.malformed_message = "Model stream emitted more than one response start"
                    return
                state.started = True
                yield MessageStartEvent(message_role="assistant")
            elif isinstance(event, ModelTextDeltaEvent):
                if not state.started:
                    state.malformed_message = "Model stream emitted text before response start"
                    return
                yield MessageDeltaEvent(delta=event.delta)
            elif isinstance(event, ModelThinkingDeltaEvent):
                if not state.started:
                    state.malformed_message = "Model stream emitted thinking before response start"
                    return
                yield ThinkingDeltaEvent(delta=event.delta)
            elif isinstance(event, ModelCompletedEvent):
                if not state.started:
                    state.malformed_message = (
                        "Model stream completed before emitting response start"
                    )
                    return
                state.terminal = True
                state.message = event.message
                state.usage = event.usage
            elif isinstance(event, ModelFailureEvent):
                state.terminal = True
                state.failure = event
            elif isinstance(event, ModelCancelledEvent):
                state.terminal = True
                state.cancellation_message = event.message
            else:
                state.malformed_message = (
                    f"Model stream emitted an unsupported event: {type(event).__name__}"
                )
                return

        if not state.terminal:
            if cancellation.is_cancelled():
                state.terminal = True
                state.cancellation_message = "Agent run cancelled"
            else:
                state.malformed_message = "Model stream ended without a terminal event"
        elif state.message is not None and state.malformed_message is None:
            yield MessageEndEvent(message=state.message)

    async def _execute_tools(
        self,
        tool_calls: Sequence[ToolCall],
        *,
        run_state: _RunState,
        batch_state: _ToolBatchState,
    ) -> AsyncIterator[AgentEvent]:
        for index, tool_call in enumerate(tool_calls):
            terminal = self._run_condition(run_state)
            if terminal is not None:
                status, message = terminal
                async for event in self._skip_tools(tool_calls[index:], message):
                    yield event
                batch_state.status = status
                batch_state.message = message
                return

            if (
                self._limits.max_tool_calls is not None
                and run_state.tool_calls >= self._limits.max_tool_calls
            ):
                message = f"Agent reached max_tool_calls={self._limits.max_tool_calls}"
                async for event in self._skip_tools(tool_calls[index:], message):
                    yield event
                batch_state.status = "tool_call_limit"
                batch_state.message = message
                return

            yield ToolExecutionStartEvent(tool_call=tool_call)
            terminal = self._run_condition(run_state)
            if terminal is not None:
                status, message = terminal
                for event in self._record_tool_result(_failed_tool_result(tool_call, message)):
                    yield event
                async for event in self._skip_tools(tool_calls[index + 1 :], message):
                    yield event
                batch_state.status = status
                batch_state.message = message
                return

            run_state.tool_calls += 1
            tool = self._tool_by_name.get(tool_call.name)

            try:
                if tool is None:
                    result = _failed_tool_result(
                        tool_call,
                        f"Unknown tool: {tool_call.name}",
                    )
                else:
                    async with asyncio.timeout(self._remaining_seconds(run_state)):
                        result = await tool.execute(
                            tool_call.arguments,
                            signal=run_state.cancellation,
                        )
                    if result.tool_call_id != tool_call.id or result.name != tool_call.name:
                        result = result.model_copy(
                            update={
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                            }
                        )
            except TimeoutError:
                message = "Agent run exceeded its time limit during tool execution"
                result = _failed_tool_result(tool_call, message)
                for event in self._record_tool_result(result):
                    yield event
                async for event in self._skip_tools(tool_calls[index + 1 :], message):
                    yield event
                batch_state.status = "timed_out"
                batch_state.message = message
                return
            except FatalEnvironmentError as exc:
                message = str(exc) or "Execution environment failed"
                result = _failed_tool_result(tool_call, message)
                for event in self._record_tool_result(result):
                    yield event
                async for event in self._skip_tools(tool_calls[index + 1 :], message):
                    yield event
                batch_state.status = "fatal_environment_failure"
                batch_state.message = message
                return
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                result = _failed_tool_result(tool_call, message)

            for event in self._record_tool_result(result):
                yield event

    async def _skip_tools(
        self,
        tool_calls: Sequence[ToolCall],
        message: str,
    ) -> AsyncIterator[AgentEvent]:
        for tool_call in tool_calls:
            yield ToolExecutionStartEvent(tool_call=tool_call)
            for event in self._record_tool_result(_failed_tool_result(tool_call, message)):
                yield event

    def _record_tool_result(self, result: AgentToolResult) -> tuple[AgentEvent, ...]:
        message = _tool_result_message(result)
        self._messages.append(message)
        return (
            ToolExecutionEndEvent(result=result),
            MessageStartEvent(message_role="tool"),
            MessageEndEvent(message=message),
        )

    def _apply_context_limit(self) -> str | None:
        max_tokens = self._limits.max_context_tokens
        if max_tokens is None:
            return None

        decision = apply_context_policy(
            system=self._system_prompt,
            messages=self._messages,
            tools=self._tools,
            max_tokens=max_tokens,
            reserve_tokens=self._limits.context_reserve_tokens,
        )
        self._messages[:] = decision.messages
        if decision.fits:
            return None
        return "The system prompt, tools, and newest user-led turn exceed the context limit"

    def _run_condition(
        self,
        state: _RunState,
    ) -> tuple[AgentStatus, str] | None:
        if state.cancellation.is_cancelled():
            return "cancelled", "Agent run cancelled"
        remaining = self._remaining_seconds(state)
        if remaining is not None and remaining <= 0:
            return "timed_out", "Agent run exceeded its time limit"
        return None

    def _remaining_seconds(self, state: _RunState) -> float | None:
        limit = self._limits.time_limit_seconds
        if limit is None:
            return None
        return max(0.0, limit - (monotonic() - state.started_at))

    def _error(
        self,
        status: AgentStatus,
        message: str,
        *,
        recoverable: bool = False,
        data: dict[str, JSONValue] | None = None,
    ) -> ErrorEvent:
        event_data: dict[str, JSONValue] = {"status": status}
        if data is not None:
            event_data.update(data)
        return ErrorEvent(message=message, recoverable=recoverable, data=event_data)

    def _finish(
        self,
        state: _RunState,
        status: AgentStatus,
        message: str | None,
    ) -> AgentEndEvent:
        self._last_result = AgentRunResult(
            status=status,
            usage=state.usage.model_copy(deep=True),
            turns=state.turns,
            tool_calls=state.tool_calls,
            message=message,
        )
        return AgentEndEvent(status=status, message=message)


def _failed_tool_result(tool_call: ToolCall, message: str) -> AgentToolResult:
    return AgentToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        ok=False,
        content=message,
        error=message,
    )


def _tool_result_message(result: AgentToolResult) -> ToolResultMessage:
    content = result.content
    if not result.ok and result.error and result.error not in content:
        content = f"{content}\n\nError: {result.error}"
    if result.data is not None and not content:
        content = str(result.data)

    return ToolResultMessage(
        tool_call_id=result.tool_call_id,
        name=result.name,
        content=content,
        ok=result.ok,
        data=result.data,
        details=result.details,
        error=result.error,
    )


def _add_usage(total: Usage, response: Usage) -> Usage:
    thinking_tokens: int | None = None
    if total.thinking_tokens is not None or response.thinking_tokens is not None:
        thinking_tokens = (total.thinking_tokens or 0) + (response.thinking_tokens or 0)

    return Usage(
        input_tokens=total.input_tokens + response.input_tokens,
        output_tokens=total.output_tokens + response.output_tokens,
        cached_tokens=total.cached_tokens + response.cached_tokens,
        thinking_tokens=thinking_tokens,
    )


__all__ = ["Agent", "AgentLimits", "AgentRunResult"]
