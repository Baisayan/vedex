from __future__ import annotations

from dataclasses import dataclass

from agent.messages import AgentMessage, UserMessage
from agent.session.entries import (
    CompactionEntry,
    SessionEntry,
    SessionInfoEntry,
)


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
