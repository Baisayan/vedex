from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from vedex.core import OllamaClient
from vedex.schema import (
    AgentEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    ErrorEvent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from vedex.session import Session, SessionStore

from .conftest import (
    make_tool,
    native_ollama_client,
    native_tags_response,
    ndjson_response,
    run_async,
)


async def _prompt_events(session: Session, content: str) -> list[AgentEvent]:
    return [event async for event in session.prompt(content)]


def _session(
    tmp_path: Path,
    client: OllamaClient,
    *,
    tools: list[AgentTool] | None = None,
    context_window_tokens: int | None = 4096,
) -> Session:
    return Session(
        cwd=tmp_path,
        model="local:latest",
        system_prompt="system prompt",
        tools=[] if tools is None else tools,
        client=client,
        store=SessionStore(tmp_path / "session.jsonl"),
        context_window_tokens=context_window_tokens,
    )


def test_session_persists_user_assistant_and_tool_result_messages(tmp_path: Path) -> None:
    chat_calls = 0

    def handler(request: Any) -> Any:
        nonlocal chat_calls
        if request.url.path == "/api/tags":
            return native_tags_response()
        chat_calls += 1
        if chat_calls == 1:
            return ndjson_response(
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "test_tool", "arguments": {}}},
                        ]
                    },
                    "done": True,
                }
            )
        return ndjson_response({"message": {"content": "finished"}, "done": True})

    client = native_ollama_client(handler)
    tool = make_tool(
        result=AgentToolResult(tool_call_id="", name="test_tool", ok=True, content="tool output")
    )
    session = _session(tmp_path, client, tools=[tool])
    try:
        run_async(_prompt_events(session, "do work"))
        persisted = session.store.load()
    finally:
        run_async(session.close())

    assert [message.role for message in persisted] == ["user", "assistant", "tool", "assistant"]
    assert isinstance(persisted[2], ToolResultMessage)
    assert persisted == session.messages


def test_failed_tool_results_and_partial_streams_are_persisted_correctly(tmp_path: Path) -> None:
    responses = [
        ndjson_response(
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "test_tool", "arguments": {}}},
                    ]
                },
                "done": True,
            }
        ),
        ndjson_response({"message": {"content": "partial"}}),
    ]

    def handler(request: Any) -> Any:
        if request.url.path == "/api/tags":
            return native_tags_response()
        return responses.pop(0)

    client = native_ollama_client(handler)
    failed_tool = make_tool(raises=RuntimeError("tool failed"))
    session = _session(tmp_path, client, tools=[failed_tool])
    try:
        run_async(_prompt_events(session, "run tool"))
        tool_message = session.messages[-1]
        events = run_async(_prompt_events(session, "stream breaks"))
    finally:
        run_async(session.close())

    assert isinstance(tool_message, ToolResultMessage)
    assert tool_message.ok is False
    assert "tool failed" in tool_message.content
    assert any(isinstance(event, ErrorEvent) for event in events)
    assert not any(
        isinstance(message, AssistantMessage) and message.content == "partial"
        for message in session.messages
    )
    assert session.store.load() == session.messages


def test_storage_failure_does_not_diverge_memory_from_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = native_ollama_client(lambda _request: native_tags_response())
    session = _session(tmp_path, client)

    def fail_append(_message: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(session.store, "append", fail_append)
    events = run_async(_prompt_events(session, "cannot save"))
    run_async(session.close())

    assert session.messages == []
    assert any(isinstance(event, ErrorEvent) and "disk full" in event.message for event in events)


def test_context_truncation_keeps_complete_newest_user_led_turns(tmp_path: Path) -> None:
    client = native_ollama_client(lambda _request: native_tags_response())
    session = _session(tmp_path, client, context_window_tokens=120)
    try:
        session.add_user_message("a" * 120)
        session._accept_message(
            AssistantMessage(tool_calls=[ToolCall(id="call-1", name="read", arguments={})])
        )
        session._accept_message(
            ToolResultMessage(tool_call_id="call-1", name="read", content="b" * 120)
        )
        session.add_user_message("c" * 120)
        session.add_user_message("d" * 120)

        assert session._prepare_context() is True
        assert [message.role for message in session.messages] == ["user", "user"]
        assert session.store.load() == session.messages
    finally:
        run_async(session.close())


def test_truncation_never_keeps_an_orphan_tool_result(tmp_path: Path) -> None:
    client = native_ollama_client(lambda _request: native_tags_response())
    session = _session(tmp_path, client, context_window_tokens=160)
    try:
        for number in (1, 2):
            session.add_user_message("u" * 120)
            session._accept_message(
                AssistantMessage(tool_calls=[ToolCall(id=f"call-{number}", name="read")])
            )
            session._accept_message(
                ToolResultMessage(tool_call_id=f"call-{number}", name="read", content="t" * 120)
            )

        assert session._prepare_context() is True
        assert [message.role for message in session.messages] == ["user", "assistant", "tool"]
        assistant = session.messages[1]
        result = session.messages[2]
        assert isinstance(assistant, AssistantMessage)
        assert isinstance(result, ToolResultMessage)
        assert result.tool_call_id == assistant.tool_calls[0].id
    finally:
        run_async(session.close())


def test_context_rewrite_failure_leaves_memory_history_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = native_ollama_client(lambda _request: native_tags_response())
    session = _session(tmp_path, client, context_window_tokens=60)
    try:
        session.add_user_message("a" * 100)
        session.add_user_message("b" * 100)
        before = list(session.messages)

        def fail_rewrite(_messages: list[object]) -> None:
            raise OSError("rewrite failed")

        monkeypatch.setattr(session.store, "rewrite", fail_rewrite)
        with pytest.raises(OSError, match="rewrite failed"):
            session._truncate_to_context_limit()
        assert session.messages == before
    finally:
        run_async(session.close())


def test_too_large_newest_input_returns_error_without_chat_request(tmp_path: Path) -> None:
    chat_called = False

    def handler(request: Any) -> Any:
        nonlocal chat_called
        if request.url.path == "/api/tags":
            return native_tags_response(context_length=30)
        chat_called = True
        return ndjson_response({"message": {"content": "should not run"}, "done": True})

    client = native_ollama_client(handler)
    session = _session(tmp_path, client, context_window_tokens=30)
    try:
        events = run_async(_prompt_events(session, "x" * 400))
    finally:
        run_async(session.close())

    assert chat_called is False
    assert len(session.messages) == 1
    assert isinstance(events[0], ErrorEvent)
    assert "newest message exceed" in events[0].message


def test_context_overflow_drops_one_old_turn_and_retries_once(tmp_path: Path) -> None:
    chat_calls = 0

    def handler(request: Any) -> Any:
        nonlocal chat_calls
        if request.url.path == "/api/tags":
            return native_tags_response()
        chat_calls += 1
        if chat_calls == 1:
            return ndjson_response({"error": "context window exceeded"})
        return ndjson_response({"message": {"content": "retried"}, "done": True})

    client = native_ollama_client(handler)
    session = _session(tmp_path, client, context_window_tokens=None)
    try:
        session.add_user_message("old turn")
        events = run_async(_prompt_events(session, "new turn"))
    finally:
        run_async(session.close())

    assert chat_calls == 2
    assert not any(
        isinstance(event, ErrorEvent) and "Agent run failed" in event.message for event in events
    )
    assert [
        message.content for message in session.messages if isinstance(message, UserMessage)
    ] == ["new turn"]
    assert isinstance(session.messages[-1], AssistantMessage)


def test_session_updates_runtime_properties_and_loads_existing_history(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    store.append(UserMessage(content="saved"))
    client = native_ollama_client(lambda _request: native_tags_response())
    session = Session(
        cwd=tmp_path,
        model="old:latest",
        system_prompt="old prompt",
        tools=[],
        client=client,
        store=store,
        context_window_tokens=100,
    )
    try:
        assert session.messages == [UserMessage(content="saved")]
        session.set_model("new:latest", 200)
        session.set_system_prompt("new prompt")
        assert session.model == "new:latest"
        assert session.context_window_tokens == 200
        assert session.system_prompt == "new prompt"
        assert session.context_token_estimate > 0
    finally:
        run_async(session.close())
