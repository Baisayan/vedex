from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

Message = dict[str, Any]
Action = dict[str, Any]
Observation = dict[str, Any]

DEFAULT_SYSTEM_PROMPT = """You are a software engineering agent working in a terminal.
Use the Bash tool to inspect the repository, edit files, and run tests.
Continue working until the requested task is complete. When no further Bash
command is needed, respond with a concise final answer."""


class AsyncModel(Protocol):
    context_window_tokens: int | None

    async def query(self, messages: list[Message]) -> Message: ...

    def format_message(
        self,
        *,
        role: str,
        content: str,
        extra: Mapping[str, Any] | None = None,
    ) -> Message: ...

    def format_observation_messages(
        self,
        message: Message,
        outputs: list[Observation],
        template_vars: Mapping[str, Any] | None = None,
    ) -> list[Message]: ...

    def serialize(self) -> dict[str, Any]: ...


class AsyncEnvironment(Protocol):
    async def execute(self, action: Action) -> Observation: ...

    def serialize(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class AgentConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    output_path: Path | None = None


class DefaultAgent:
    def __init__(
        self,
        model: AsyncModel,
        env: AsyncEnvironment,
        *,
        config: AgentConfig | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.model = model
        self.env = env
        self.messages: list[Message] = []
        self.cost = 0.0
        self.n_calls = 0
        self.status = "created"
        self.submission = ""
        self._started_at = 0.0

    def add_messages(self, *messages: Message) -> list[Message]:
        self.messages.extend(messages)
        return list(messages)

    async def run(self, task: str = "") -> dict[str, Any]:
        self.messages = [
            self.model.format_message(role="system", content=self.config.system_prompt),
            self.model.format_message(role="user", content=task),
        ]
        self.cost = 0.0
        self.n_calls = 0
        self.status = "running"
        self.submission = ""
        self._started_at = time.time()

        try:
            while self.status == "running":
                await self.step()
                self.save(self.config.output_path)
        except Exception as exc:
            self.status = type(exc).__name__
            self.save(self.config.output_path)
            raise

        return self.save(self.config.output_path)

    async def step(self) -> list[Message]:
        message = await self.query()
        actions = self._actions(message)
        if not actions:
            self.status = "completed"
            self.submission = str(message.get("content") or "")
            return []
        return await self.execute_actions(message)

    async def query(self) -> Message:
        self._trim_history()
        self.n_calls += 1
        message = await self.model.query(self.messages)
        self.cost += float(message.get("extra", {}).get("cost", 0.0))
        self.add_messages(message)
        return message

    async def execute_actions(self, message: Message) -> list[Message]:
        outputs: list[Observation] = []
        for action in self._actions(message):
            outputs.append(await self.env.execute(action))
        observations = self.model.format_observation_messages(
            message,
            outputs,
            self.template_vars(),
        )
        return self.add_messages(*observations)

    def template_vars(self) -> dict[str, Any]:
        return {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "elapsed_seconds": int(time.time() - self._started_at) if self._started_at else 0,
        }

    def _trim_history(self) -> None:
        limit = getattr(self.model, "context_window_tokens", None)
        if not isinstance(limit, int) or limit <= 0:
            return
        while _estimate_tokens(self.messages) > limit:
            turn = _oldest_complete_turn(self.messages)
            if turn is None:
                return
            del self.messages[turn[0] : turn[1]]

    def serialize(self, *extra: Mapping[str, Any]) -> dict[str, Any]:
        data: dict[str, Any] = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": _jsonable(asdict(self.config)),
                    "agent_type": f"{type(self).__module__}.{type(self).__name__}",
                },
                "exit_status": self.status,
                "submission": self.submission,
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        for addition in (self.model.serialize(), self.env.serialize(), *extra):
            _merge(data, addition)
        return _jsonable(data)

    def save(self, path: Path | None, *extra: Mapping[str, Any]) -> dict[str, Any]:
        data = self.serialize(*extra)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    @staticmethod
    def _actions(message: Message) -> list[Action]:
        actions = message.get("extra", {}).get("actions", [])
        return [action for action in actions if isinstance(action, dict)]


def _merge(target: dict[str, Any], addition: Mapping[str, Any]) -> None:
    for key, value in addition.items():
        if isinstance(target.get(key), dict) and isinstance(value, Mapping):
            _merge(target[key], value)
        else:
            target[key] = value


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return str(value)
    return value


def _estimate_tokens(messages: list[Message]) -> int:
    return len(json.dumps(messages, default=str)) // 4


def _oldest_complete_turn(messages: list[Message]) -> tuple[int, int] | None:
    index = 1
    while index < len(messages):
        if messages[index].get("role") != "assistant":
            index += 1
            continue
        end = index + 1
        while end < len(messages) and messages[end].get("role") == "tool":
            end += 1
        return (index, end) if end > index + 1 else None
    return None


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "AgentConfig",
    "AsyncEnvironment",
    "AsyncModel",
    "DefaultAgent",
]
