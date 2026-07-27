from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .schema import AgentMessage

_MESSAGE_ADAPTER: TypeAdapter[AgentMessage] = TypeAdapter(AgentMessage)


class SessionError(ValueError):
    """Raised when a message-only session file cannot be read safely."""


class SessionStore:
    """Synchronous JSONL storage for validated conversation messages."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[AgentMessage]:
        if not self.path.exists():
            return []

        messages: list[AgentMessage] = []
        with self.path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    messages.append(_MESSAGE_ADAPTER.validate_json(line))
                except ValidationError as exc:
                    raise SessionError(
                        f"Invalid session message in {self.path} at line {line_number}"
                    ) from exc
        return messages

    def append(self, message: AgentMessage) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as file:
            file.write(_MESSAGE_ADAPTER.dump_json(message))
            file.write(b"\n")

    def rewrite(self, messages: list[AgentMessage]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                for message in messages:
                    file.write(_MESSAGE_ADAPTER.dump_json(message))
                    file.write(b"\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
