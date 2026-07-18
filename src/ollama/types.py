from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── Outgoing (what we send to Ollama) ─────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class OllamaMessage:
    role: str
    content: str
    tool_calls: list[OllamaToolCall] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d


@dataclass(frozen=True, slots=True)
class OllamaToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"function": {"name": self.name, "arguments": self.arguments}}


@dataclass(frozen=True, slots=True)
class OllamaTool:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ChatRequest:
    model: str
    messages: list[OllamaMessage]
    tools: list[OllamaTool] = field(default_factory=list)
    stream: bool = True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "stream": self.stream,
        }
        if self.tools:
            d["tools"] = [t.to_dict() for t in self.tools]
        return d


# ── Incoming (what Ollama streams back) ───────────────────────────────────────

class ChatChunkMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = "assistant"
    content: str = ""
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    message: ChatChunkMessage = Field(default_factory=ChatChunkMessage)
    done: bool = False
    done_reason: str | None = None
