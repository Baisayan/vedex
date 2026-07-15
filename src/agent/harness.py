from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Literal

from agent.events import AgentEvent, MessageEndEvent, MessageStartEvent, QueueUpdateEvent
from agent.loop import run_agent_loop
from agent.messages import AgentMessage, AssistantMessage, ToolResultMessage, UserMessage
from agent.tools import AgentTool
from ollama.chat import OllamaProvider

EventListener = Callable[[AgentEvent], Awaitable[None] | None]
QueueMode = Literal["one_at_a_time", "all"]


@dataclass(frozen=True, slots=True)
class QueuedMessages:
    steering: tuple[AgentMessage, ...] = ()
    follow_up: tuple[AgentMessage, ...] = ()

    @property
    def count(self) -> int:
        return len(self.steering) + len(self.follow_up)


@dataclass(slots=True)
class AgentHarnessConfig:
    provider: OllamaProvider
    model: str
    system: str
    tools: list[AgentTool] = field(default_factory=list)
    max_turns: int | None = None
    queue_mode: QueueMode = "one_at_a_time"


class SimpleCancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


class AgentHarness:
    def __init__(
        self,
        config: AgentHarnessConfig,
        *,
        messages: Sequence[AgentMessage] = (),
    ) -> None:
        self._config = config
        self._messages = list(messages)
        self._listeners: list[EventListener] = []
        self._current_signal: SimpleCancellationToken | None = None
        self._running = False
        self._steering_queue: deque[AgentMessage] = deque()
        self._follow_up_queue: deque[AgentMessage] = deque()

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return tuple(self._messages)

    @property
    def config(self) -> AgentHarnessConfig:
        return self._config

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queued_messages(self) -> QueuedMessages:
        return QueuedMessages(
            steering=tuple(self._steering_queue),
            follow_up=tuple(self._follow_up_queue),
        )

    @property
    def pending_message_count(self) -> int:
        return self.queued_messages.count

    def has_queued_messages(self) -> bool:
        return bool(self._steering_queue or self._follow_up_queue)

    def append_message(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def replace_messages(self, messages: Sequence[AgentMessage]) -> None:
        self._messages = list(messages)

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._listeners.remove(listener)

        return unsubscribe

    def cancel(self) -> None:
        if self._current_signal is not None:
            self._current_signal.cancel()

    def steer(self, content: str) -> QueueUpdateEvent:
        return self.steer_message(UserMessage(content=content))

    def steer_message(self, message: AgentMessage) -> QueueUpdateEvent:
        self._steering_queue.append(message)
        return self.queue_update_event()

    def follow_up(self, content: str) -> QueueUpdateEvent:
        return self.follow_up_message(UserMessage(content=content))

    def follow_up_message(self, message: AgentMessage) -> QueueUpdateEvent:
        self._follow_up_queue.append(message)
        return self.queue_update_event()

    def clear_queues(self) -> QueuedMessages:
        snapshot = self.queued_messages
        self._steering_queue.clear()
        self._follow_up_queue.clear()
        return snapshot

    def pop_latest_follow_up(self) -> AgentMessage | None:
        if not self._follow_up_queue:
            return None
        return self._follow_up_queue.pop()

    def queue_update_event(self) -> QueueUpdateEvent:
        return QueueUpdateEvent(
            steering=tuple(message.content for message in self._steering_queue),
            follow_up=tuple(message.content for message in self._follow_up_queue),
        )

    def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        self._ensure_not_running()
        self._append_interrupted_tool_results()
        self._running = True
        message = UserMessage(content=content)
        self._messages.append(message)
        return self._run(prompt_message=message)

    def continue_(self) -> AsyncIterator[AgentEvent]:
        self._ensure_not_running()
        self._append_interrupted_tool_results()
        self._running = True
        return self._run()

    async def _run(self, *, prompt_message: UserMessage | None = None) -> AsyncIterator[AgentEvent]:
        signal = SimpleCancellationToken()
        self._current_signal = signal
        pending_prompt_event = prompt_message
        try:
            async for event in run_agent_loop(
                provider=self._config.provider,
                model=self._config.model,
                system=self._config.system,
                messages=self._messages,
                tools=self._config.tools,
                max_turns=self._config.max_turns,
                signal=signal,
                get_steering_messages=self._drain_steering_messages,
                get_follow_up_messages=self._drain_follow_up_messages,
                get_queue_update=self.queue_update_event,
            ):
                await self._notify(event)
                yield event
                if pending_prompt_event is not None and event.type == "turn_start":
                    start = MessageStartEvent(message_role="user")
                    end = MessageEndEvent(message=pending_prompt_event)
                    for prompt_event in (start, end):
                        await self._notify(prompt_event)
                        yield prompt_event
                    pending_prompt_event = None
        finally:
            if signal.is_cancelled():
                self._append_interrupted_tool_results()
            if self._current_signal is signal:
                self._current_signal = None
            self._running = False

    async def _notify(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            result = listener(event)
            if isawaitable(result):
                await result

    def _ensure_not_running(self) -> None:
        if self._running:
            raise RuntimeError("AgentHarness is already running; use steer() or follow_up() to queue messages.")

    def _drain_steering_messages(self) -> tuple[AgentMessage, ...]:
        return self._drain_queue(self._steering_queue)

    def _drain_follow_up_messages(self) -> tuple[AgentMessage, ...]:
        return self._drain_queue(self._follow_up_queue)

    def _drain_queue(self, queue: deque[AgentMessage]) -> tuple[AgentMessage, ...]:
        if not queue:
            return ()
        if self._config.queue_mode == "all":
            messages = tuple(queue)
            queue.clear()
            return messages
        return (queue.popleft(),)

    def _append_interrupted_tool_results(self) -> None:
        assistant_index = _latest_open_tool_call_assistant_index(self._messages)
        if assistant_index is None:
            return

        assistant = self._messages[assistant_index]
        if not isinstance(assistant, AssistantMessage):
            return

        returned_ids = {
            message.tool_call_id
            for message in self._messages[assistant_index + 1 :]
            if isinstance(message, ToolResultMessage)
        }
        for tool_call in assistant.tool_calls:
            if tool_call.id in returned_ids:
                continue
            message = "Tool call interrupted by user"
            self._messages.append(
                ToolResultMessage(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=message,
                    ok=False,
                    error=message,
                )
            )


def _latest_open_tool_call_assistant_index(messages: Sequence[AgentMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, UserMessage):
            return None
        if isinstance(message, AssistantMessage):
            if message.tool_calls:
                return index
            return None
    return None