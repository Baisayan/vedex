"""A deterministic model for local development and integration checks."""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Iterable
from typing import Any

from . import (
    Message,
    Observation,
    format_message,
    format_observation_messages,
    parse_native_tool_calls,
)


class FakeModel:
    def __init__(
        self,
        responses: Iterable[Message],
        *,
        context_window_tokens: int | None = None,
        model_name: str = "fake",
    ) -> None:
        self._responses = deque(copy.deepcopy(list(responses)))
        self.requests: list[list[Message]] = []
        self.context_window_tokens = context_window_tokens
        self.model_name = model_name

    async def query(self, messages: list[Message]) -> Message:
        self.requests.append(copy.deepcopy(messages))
        if not self._responses:
            raise RuntimeError("FakeModel has no response left")
        response = copy.deepcopy(self._responses.popleft())
        extra = response.setdefault("extra", {})
        if "actions" not in extra:
            extra["actions"] = parse_native_tool_calls(response.get("tool_calls", []))
        return response

    def format_message(self, role: str, content: str, **kwargs: Any) -> Message:
        return format_message(role, content, **kwargs)

    def format_observation_messages(
        self,
        message: Message,
        outputs: list[Observation],
        template_vars: Any = None,
        **kwargs: Any,
    ) -> list[Message]:
        return format_observation_messages(message, outputs, template_vars, **kwargs)

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {"model": self.model_name, "model_type": "fake"},
            }
        }


__all__ = ["FakeModel"]
