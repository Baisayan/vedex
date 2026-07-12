from collections.abc import AsyncIterator
from typing import Protocol

from agent.messages import AgentMessage
from agent.tools import AgentTool
from ollama.events import ProviderEvent


class CancellationToken(Protocol):
    def is_cancelled(self) -> bool:
        ...


class ModelProvider(Protocol):
    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        ...