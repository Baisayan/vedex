"""Async LiteLLM model adapter."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import litellm

from . import (
    BASH_TOOL,
    Message,
    Observation,
    format_message,
    format_observation_messages,
    parse_native_tool_calls,
    to_jsonable,
    value_of,
)


@dataclass(slots=True)
class LiteLLMConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    context_window_tokens: int | None = None
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class LiteLLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        kind: str,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status_code = status_code


class LiteLLMModel:
    def __init__(
        self,
        model_name: str,
        *,
        model_kwargs: dict[str, Any] | None = None,
        context_window_tokens: int | None = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self.config = LiteLLMConfig(
            model_name=model_name,
            model_kwargs=dict(model_kwargs or {}),
            context_window_tokens=context_window_tokens,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
        self.context_window_tokens = context_window_tokens

    async def query(self, messages: list[Message]) -> Message:
        response = await self._complete(messages)
        raw_message = self._response_message(response)
        actions = parse_native_tool_calls(value_of(raw_message, "tool_calls", []))
        raw_message_json = to_jsonable(raw_message)
        message: dict[str, Any]
        if isinstance(raw_message_json, dict):
            message = raw_message_json
        else:
            message = {"role": "assistant", "content": str(raw_message_json)}
        message.setdefault("role", "assistant")
        message.setdefault("content", "")
        message["extra"] = {
            "actions": actions,
            "response": to_jsonable(response),
            "usage": to_jsonable(value_of(response, "usage", {})),
            "cost": self._cost(response),
            "timestamp": time.time(),
        }
        return message

    async def _complete(self, messages: list[Message]) -> Any:
        request = [self._request_message(message) for message in messages]
        for attempt in range(self.config.max_retries):
            try:
                return await litellm.acompletion(
                    model=self.config.model_name,
                    messages=request,
                    tools=[BASH_TOOL],
                    **self.config.model_kwargs,
                )
            except Exception as exc:
                if attempt + 1 >= self.config.max_retries or not self._retryable(exc):
                    raise self._normalize_error(exc) from exc
                await asyncio.sleep(self.config.retry_delay_seconds * (attempt + 1))
        raise RuntimeError("LiteLLM request failed")

    @staticmethod
    def _request_message(message: Message) -> Message:
        return {key: value for key, value in message.items() if key != "extra"}

    @staticmethod
    def _response_message(response: Any) -> Any:
        choices = value_of(response, "choices", [])
        if not choices:
            raise ValueError("LiteLLM response did not contain a choice")
        message = value_of(choices[0], "message")
        if message is None:
            raise ValueError("LiteLLM response did not contain a message")
        return message

    @staticmethod
    def _retryable(error: Exception) -> bool:
        name = type(error).__name__.lower()
        status_code = value_of(error, "status_code")
        if status_code in {408, 409, 429} or isinstance(status_code, int) and status_code >= 500:
            return True
        return any(
            marker in name
            for marker in (
                "timeout",
                "ratelimit",
                "connection",
                "serviceunavailable",
                "internalserver",
            )
        )

    @classmethod
    def _normalize_error(cls, error: Exception) -> LiteLLMError:
        status_code = value_of(error, "status_code")
        status_code = status_code if isinstance(status_code, int) else None
        name = type(error).__name__.lower()
        if status_code == 429 or "ratelimit" in name:
            kind = "rate_limit"
        elif "timeout" in name:
            kind = "timeout"
        elif "connection" in name:
            kind = "connection"
        elif status_code is not None and status_code >= 500:
            kind = "server"
        else:
            kind = "provider"
        return LiteLLMError(
            f"LiteLLM {kind} error: {error}",
            kind=kind,
            retryable=cls._retryable(error),
            status_code=status_code,
        )

    def _cost(self, response: Any) -> float:
        try:
            cost = litellm.cost_calculator.completion_cost(
                completion_response=response,
                model=self.config.model_name,
            )
            return float(cost or 0.0)
        except Exception:
            return 0.0

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
        return {"info": {"config": asdict(self.config), "model_type": "litellm"}}


__all__ = ["LiteLLMConfig", "LiteLLMError", "LiteLLMModel"]
