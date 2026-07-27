from __future__ import annotations

import pytest
from vedex.rendering import CommandLineRenderer, _preview_text, format_tool_call_block
from vedex.schema import (
    AgentEndEvent,
    AgentToolResult,
    ErrorEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)


def test_tool_call_blocks_and_previews_cover_known_and_unknown_tools() -> None:
    assert ":2-4" in format_tool_call_block(
        ToolCall(id="1", name="read", arguments={"path": "a", "offset": 2, "limit": 3})
    )
    assert format_tool_call_block(ToolCall(id="2", name="write", arguments={"path": "a"})).endswith(
        "write a"
    )
    assert "timeout 3s" in format_tool_call_block(
        ToolCall(id="3", name="bash", arguments={"command": "ls", "timeout": 3})
    )
    assert "custom" in format_tool_call_block(ToolCall(id="4", name="custom", arguments={"x": 1}))

    preview = _preview_text("one\ntwo\nthree", max_lines=2)
    assert "one\ntwo" in preview
    assert "1 more lines" in preview
    assert "additional characters" in _preview_text("x" * 2_000, max_lines=1)


def test_renderer_streams_messages_tools_and_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = CommandLineRenderer()
    renderer.render(MessageStartEvent())
    renderer.render(MessageDeltaEvent(delta="Hello"))
    renderer.render(
        ToolExecutionStartEvent(tool_call=ToolCall(id="1", name="read", arguments={"path": "file"}))
    )
    renderer.render(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="1", name="read", ok=True, content="line one\nline two"
            )
        )
    )
    renderer.render(ErrorEvent(message="recoverable", recoverable=True))
    renderer.render(ErrorEvent(message="fatal"))
    renderer.render(AgentEndEvent())

    captured = capsys.readouterr()
    assert "Hello" in captured.out
    assert "completed: read" in captured.err
    assert "Error: recoverable" in captured.err
    assert "Error: fatal" in captured.err
    assert renderer.finish() is False
