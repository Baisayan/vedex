from __future__ import annotations

import os
from pathlib import Path

import pytest
from vedex.schema import AgentMessage, AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from vedex.session import SessionError, SessionStore


def test_load_missing_file_and_append_messages_creates_parent_directories(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "nested" / "session.jsonl")
    assert store.load() == []

    messages: list[AgentMessage] = [
        UserMessage(content="hello"),
        AssistantMessage(content="checking", tool_calls=[ToolCall(id="call-1", name="read")]),
        ToolResultMessage(tool_call_id="call-1", name="read", content="result"),
    ]
    for message in messages:
        store.append(message)

    assert store.load() == messages
    assert len(store.path.read_text(encoding="utf-8").splitlines()) == 3


def test_load_ignores_blank_rows_and_rejects_invalid_or_legacy_rows(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(b'\n{"role":"user","content":"ok"}\n\n{"type":"message","entry":{}}\n')
    store = SessionStore(path)

    with pytest.raises(SessionError, match=r"Invalid session message .* at line 4"):
        store.load()


def test_rewrite_replaces_history_and_removes_temporary_sibling(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    store.append(UserMessage(content="old"))
    retained: list[AgentMessage] = [UserMessage(content="new"), AssistantMessage(content="answer")]

    store.rewrite(retained)

    assert store.load() == retained
    assert list(tmp_path.glob(".session.jsonl.*.tmp")) == []


def test_failed_atomic_replace_leaves_original_file_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    store.append(UserMessage(content="old"))

    def fail_replace(
        _source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        _target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.rewrite([UserMessage(content="new")])

    assert store.load() == [UserMessage(content="old")]
    assert list(tmp_path.glob(".session.jsonl.*.tmp")) == []
