from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field

from ..schema import (
    AgentMessage,
    AgentTool,
    AssistantMessage,
    CancellationToken,
    JSONValue,
)


class _ModelContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Usage(_ModelContract):
    """Normalized token usage reported for one completed model response."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    thinking_tokens: int | None = Field(default=None, ge=0)


class ModelSettings(_ModelContract):
    """Model selection plus adapter-owned, JSON-serializable options."""

    model: str = Field(min_length=1)
    options: dict[str, JSONValue] = Field(default_factory=dict)


class ModelToolDefinition(_ModelContract):
    """The provider-neutral part of a tool exposed to a model."""

    name: str = Field(min_length=1)
    description: str
    input_schema: dict[str, JSONValue]

    @classmethod
    def from_agent_tool(cls, tool: AgentTool) -> Self:
        return cls(
            name=tool.name,
            description=tool.description,
            input_schema=dict(tool.input_schema),
        )


class ModelRequest(_ModelContract):
    """Complete normalized input for one model response."""

    system: str
    messages: list[AgentMessage]
    tools: list[ModelToolDefinition]
    settings: ModelSettings

    @classmethod
    def from_agent_inputs(
        cls,
        *,
        system: str,
        messages: Sequence[AgentMessage],
        tools: Sequence[AgentTool],
        settings: ModelSettings,
    ) -> Self:
        return cls(
            system=system,
            messages=list(messages),
            tools=[ModelToolDefinition.from_agent_tool(tool) for tool in tools],
            settings=settings,
        )


class ModelStartEvent(_ModelContract):
    type: Literal["response_start"] = "response_start"


class ModelTextDeltaEvent(_ModelContract):
    type: Literal["text_delta"] = "text_delta"
    delta: str


class ModelThinkingDeltaEvent(_ModelContract):
    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class ModelCompletedEvent(_ModelContract):
    type: Literal["response_completed"] = "response_completed"
    message: AssistantMessage
    usage: Usage = Field(default_factory=Usage)


type ModelFailureKind = Literal[
    "authentication",
    "context_overflow",
    "invalid_request",
    "rate_limit",
    "timeout",
    "unavailable",
    "unknown",
]


class ModelFailureEvent(_ModelContract):
    type: Literal["response_failure"] = "response_failure"
    kind: ModelFailureKind = "unknown"
    message: str
    retryable: bool = False


class ModelCancelledEvent(_ModelContract):
    type: Literal["response_cancelled"] = "response_cancelled"
    message: str = "Model request cancelled"


type ModelEvent = Annotated[
    ModelStartEvent
    | ModelTextDeltaEvent
    | ModelThinkingDeltaEvent
    | ModelCompletedEvent
    | ModelFailureEvent
    | ModelCancelledEvent,
    Field(discriminator="type"),
]


class ModelAdapter(Protocol):
    """Provider-neutral asynchronous model boundary."""

    def stream(
        self,
        request: ModelRequest,
        *,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ModelEvent]: ...


__all__ = [
    "ModelAdapter",
    "ModelCancelledEvent",
    "ModelCompletedEvent",
    "ModelEvent",
    "ModelFailureEvent",
    "ModelFailureKind",
    "ModelRequest",
    "ModelSettings",
    "ModelStartEvent",
    "ModelTextDeltaEvent",
    "ModelThinkingDeltaEvent",
    "ModelToolDefinition",
    "Usage",
]
