from __future__ import annotations

from time import time
from pathlib import Path
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agent.schema import AgentMessage, UserMessage


def new_entry_id() -> str:
    return uuid4().hex


def current_timestamp() -> float:
    return time()


class BaseSessionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=new_entry_id)
    timestamp: float = Field(default_factory=current_timestamp)


class MessageEntry(BaseSessionEntry):
    type: Literal["message"] = "message"
    message: AgentMessage


class ModelChangeEntry(BaseSessionEntry):
    type: Literal["model_change"] = "model_change"
    model: str


class CompactionEntry(BaseSessionEntry):
    type: Literal["compaction"] = "compaction"
    summary: str
    replaces_entry_ids: list[str] = Field(default_factory=list)


class SessionInfoEntry(BaseSessionEntry):
    type: Literal["session_info"] = "session_info"
    created_at: float = Field(default_factory=current_timestamp)
    cwd: str | None = None
    title: str | None = None


type SessionEntry = Annotated[
    MessageEntry | ModelChangeEntry | CompactionEntry | SessionInfoEntry, Field(discriminator="type"),
]


@dataclass(frozen=True, slots=True)
class SessionState:
    messages: tuple[AgentMessage, ...]
    model: str | None
    session_info: SessionInfoEntry | None
    compaction_entries: tuple[CompactionEntry, ...]
    context_entry_ids: tuple[str, ...]
    entries: tuple[SessionEntry, ...]

    @classmethod
    def from_entries(cls, entries: list[SessionEntry]) -> SessionState:
        message_rows: list[tuple[str, AgentMessage]] = []
        model: str | None = None
        session_info: SessionInfoEntry | None = None
        compaction_entries: list[CompactionEntry] = []

        for entry in entries:
            match entry.type:
                case "message":
                    message_rows.append((entry.id, entry.message))
                case "model_change":
                    model = entry.model
                case "session_info":
                    session_info = entry
                case "compaction":
                    compaction_entries.append(entry)
                    message_rows = _apply_compaction(message_rows, entry)

        return cls(
            messages=tuple(message for _entry_id, message in message_rows),
            model=model,
            session_info=session_info,
            compaction_entries=tuple(compaction_entries),
            context_entry_ids=tuple(entry_id for entry_id, _message in message_rows),
            entries=tuple(entries),
        )


def _apply_compaction(
    message_rows: list[tuple[str, AgentMessage]],
    entry: CompactionEntry,
) -> list[tuple[str, AgentMessage]]:
    replaced_ids = set(entry.replaces_entry_ids)
    retained: list[tuple[str, AgentMessage]] = []
    inserted_summary = False
    for entry_id, message in message_rows:
        if entry_id not in replaced_ids:
            retained.append((entry_id, message))
            continue
        if not inserted_summary:
            retained.append(
                (entry.id, UserMessage(content=_format_compaction_summary(entry.summary)))
            )
            inserted_summary = True

    if not inserted_summary:
        retained.append((entry.id, UserMessage(content=_format_compaction_summary(entry.summary))))
    return retained


def _format_compaction_summary(summary: str) -> str:
    return f"Previous conversation summary:\n{summary}"


_SESSION_ENTRY_ADAPTER: TypeAdapter[SessionEntry] = TypeAdapter(SessionEntry)


class SessionJsonlError(ValueError):
    """Raised when a session JSONL line cannot be decoded."""


def entry_to_json_line(entry: SessionEntry) -> str:
    return _SESSION_ENTRY_ADAPTER.dump_json(entry).decode() + "\n"


def entry_from_json_line(line: str, *, line_number: int | None = None) -> SessionEntry:
    try:
        return _SESSION_ENTRY_ADAPTER.validate_json(line)
    except ValidationError as exc:
        location = f" on line {line_number}" if line_number is not None else ""
        raise SessionJsonlError(f"Invalid session entry{location}: {exc}") from exc


def entries_from_json_lines(lines: list[str]) -> list[SessionEntry]:
    entries: list[SessionEntry] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        entries.append(entry_from_json_line(line, line_number=index))
    return entries


class SessionStorage(Protocol):
    async def append(self, entry: SessionEntry) -> None:
        ...

    async def read_all(self) -> list[SessionEntry]:
        ...


class JsonlSessionStorage:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def append(self, entry: SessionEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(entry_to_json_line(entry))

    async def read_all(self) -> list[SessionEntry]:
        if not self.path.exists():
            return []
        return entries_from_json_lines(self.path.read_text(encoding="utf-8").splitlines())