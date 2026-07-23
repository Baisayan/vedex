from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from schema import ErrorEvent
from paths import VedexPaths


@dataclass(frozen=True, slots=True)
class AgentCallDiagnosticContext:
    model: str
    cwd: Path
    session_id: str | None
    run_id: str


class AgentCallDiagnosticLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_paths(cls, paths: VedexPaths | None = None) -> AgentCallDiagnosticLogger:
        return cls((paths or VedexPaths()).agent_calls_log_path)

    def log_exception(
        self,
        *,
        context: AgentCallDiagnosticContext,
        phase: str,
        exc: BaseException,
    ) -> Path:
        entry = _base_entry(context, phase=phase, kind="exception")
        entry["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        self._append(entry)
        return self.path

    def log_error_event(
        self,
        *,
        context: AgentCallDiagnosticContext,
        phase: str,
        event: ErrorEvent,
    ) -> Path:
        entry = _base_entry(context, phase=phase, kind="error_event")
        entry["error"] = {
            "message": event.message,
            "recoverable": event.recoverable,
        }
        if event.data is not None:
            entry["error"]["data"] = event.data
        self._append(entry)
        return self.path

    def _append(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, sort_keys=True) + "\n")


def new_agent_call_run_id() -> str:
    return uuid4().hex


def _base_entry(
    context: AgentCallDiagnosticContext,
    *,
    phase: str,
    kind: str,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "kind": kind,
        "phase": phase,
        "run_id": context.run_id,
        "session_id": context.session_id,
        "model": context.model,
        "cwd": str(context.cwd),
    }