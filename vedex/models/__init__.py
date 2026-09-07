"""Shared model types and helpers."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .fake import FakeModel
    from .litellm import LiteLLMConfig, LiteLLMError, LiteLLMModel

Message = dict[str, Any]
Action = dict[str, Any]
Observation = dict[str, Any]

BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the repository environment.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


class Model(Protocol):
    context_window_tokens: int | None

    async def query(self, messages: list[Message]) -> Message: ...

    def format_message(self, role: str, content: str, **kwargs: Any) -> Message: ...

    def format_observation_messages(
        self,
        message: Message,
        outputs: list[Observation],
        **kwargs: Any,
    ) -> list[Message]: ...

    def serialize(self) -> dict[str, Any]: ...


def format_message(role: str, content: str, **kwargs: Any) -> Message:
    message: Message = {"role": role, "content": content}
    message.update(kwargs)
    return message


def format_observation_messages(
    message: Message,
    outputs: list[Observation],
    template_vars: Mapping[str, Any] | None = None,
    **_: Any,
) -> list[Message]:
    del template_vars
    actions = message.get("extra", {}).get("actions", [])
    result: list[Message] = []
    for index, action in enumerate(actions):
        output = outputs[index] if index < len(outputs) else {}
        returncode = output.get("returncode", 0)
        text = str(output.get("output", ""))
        content = f"<returncode>{returncode}</returncode>\n<output>\n{text}\n</output>"
        extra: dict[str, Any] = {
            "raw_output": text,
            "returncode": returncode,
            "timestamp": time.time(),
        }
        if output.get("exception_info"):
            extra["exception_info"] = output["exception_info"]
        observation = format_message("tool", content, name="bash", extra=extra)
        if action.get("tool_call_id"):
            observation["tool_call_id"] = action["tool_call_id"]
        result.append(observation)
    return result


def parse_native_tool_calls(tool_calls: Any) -> list[Action]:
    actions: list[Action] = []
    for call in tool_calls or []:
        function = value_of(call, "function", {})
        name = value_of(function, "name")
        if name != "bash":
            raise ValueError(f"Unsupported tool call: {name!r}")
        arguments = value_of(function, "arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, Mapping) or not isinstance(arguments.get("command"), str):
            raise ValueError("Bash tool call must contain a string command")
        action: Action = {"command": arguments["command"]}
        call_id = value_of(call, "id")
        if call_id:
            action["tool_call_id"] = call_id
        actions.append(action)
    return actions


def value_of(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    for method in ("model_dump", "to_dict"):
        converter = getattr(value, method, None)
        if converter:
            return to_jsonable(converter())
    return repr(value)


def __getattr__(name: str) -> Any:
    if name == "FakeModel":
        from .fake import FakeModel

        return FakeModel
    if name == "LiteLLMConfig":
        from .litellm import LiteLLMConfig

        return LiteLLMConfig
    if name == "LiteLLMError":
        from .litellm import LiteLLMError

        return LiteLLMError
    if name == "LiteLLMModel":
        from .litellm import LiteLLMModel

        return LiteLLMModel
    raise AttributeError(name)


__all__ = [
    "Action",
    "BASH_TOOL",
    "Message",
    "Model",
    "Observation",
    "format_message",
    "format_observation_messages",
    "parse_native_tool_calls",
    "to_jsonable",
    "FakeModel",
    "LiteLLMConfig",
    "LiteLLMError",
    "LiteLLMModel",
]
