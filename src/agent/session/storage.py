from pathlib import Path
from typing import Protocol

from agent.session.entries import SessionEntry
from agent.session.jsonl import entries_from_json_lines, entry_to_json_line


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