from typing import Literal
from pydantic import BaseModel, ConfigDict

from agent.messages import AgentMessage, AssistantMessage
from agent.tools import AgentToolResult, ToolCall
from agent.types import JSONValue


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


class QueueUpdateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["queue_update"] = "queue_update"
    steering: tuple[str, ...] = ()
    follow_up: tuple[str, ...] = ()


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


class ToolExecutionUpdateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    message: str
    data: dict[str, JSONValue] | None = None


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


class ProviderResponseStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["response_start"] = "response_start"
    model: str


class ProviderTextDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text_delta"] = "text_delta"
    delta: str


class ProviderThinkingDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class ProviderResponseEndEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["response_end"] = "response_end"
    message: AssistantMessage
    finish_reason: str | None = None


class ProviderErrorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["error"] = "error"
    message: str
    data: dict[str, JSONValue] | None = None


type ProviderEvent = (
    ProviderResponseStartEvent
    | ProviderTextDeltaEvent
    | ProviderThinkingDeltaEvent
    | ProviderResponseEndEvent
    | ProviderErrorEvent
)


type AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | QueueUpdateEvent
    | MessageStartEvent
    | MessageDeltaEvent
    | ThinkingDeltaEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | ErrorEvent
)