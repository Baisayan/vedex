from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


# ── Primitives ─────────────────────────────────────────────────────────────────

class CancellationToken(Protocol):
    def is_cancelled(self) -> bool: ...


type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | list[JSONValue] | dict[str, JSONValue]


# ── Tools ──────────────────────────────────────────────────────────────────────

class ToolExecutor(Protocol):
    def __call__(
        self,
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> Awaitable[AgentToolResult]: ...


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    arguments: dict[str, JSONValue] = Field(default_factory=dict)


class AgentToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_call_id: str
    name: str
    ok: bool
    content: str
    data: dict[str, JSONValue] | None = None
    details: dict[str, JSONValue] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentTool:
    name: str
    description: str
    input_schema: Mapping[str, JSONValue]
    executor: ToolExecutor
    prompt_snippet: str | None = None
    prompt_guidelines: tuple[str, ...] = ()

    async def execute(
        self,
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        return await self.executor(arguments, signal=signal)


# ── Messages ───────────────────────────────────────────────────────────────────

class UserMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["assistant"] = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolResultMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    content: str
    ok: bool = True
    data: dict[str, JSONValue] | None = None
    details: dict[str, JSONValue] | None = None
    error: str | None = None


type AgentMessage = UserMessage | AssistantMessage | ToolResultMessage


# ── Public events (AgentEvent = what the loop yields to the CLI) ───────────────

class AgentStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["agent_end"] = "agent_end"


class TurnStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["turn_start"] = "turn_start"
    turn: int


class TurnEndEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["turn_end"] = "turn_end"
    turn: int


class MessageStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["message_start"] = "message_start"
    message_role: Literal["user", "assistant", "tool"] = "assistant"


class MessageDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["message_delta"] = "message_delta"
    delta: str


class ThinkingDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class MessageEndEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call: ToolCall


class ToolExecutionEndEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_execution_end"] = "tool_execution_end"
    result: AgentToolResult


class ErrorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["error"] = "error"
    message: str
    recoverable: bool = False
    data: dict[str, JSONValue] | None = None


type AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageDeltaEvent
    | ThinkingDeltaEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionEndEvent
    | ErrorEvent
)
