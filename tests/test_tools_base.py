from __future__ import annotations

import pytest
from vedex.environments import LocalEnvironment
from vedex.schema import AgentTool, JSONValue
from vedex.tools import create_coding_tools
from vedex.tools.base import (
    ToolInputError,
    _optional_int_arg,
    _str_arg,
    _workspace_path_arg,
    append_status_block,
    format_size,
    truncate_head,
    truncate_tail,
)

from .conftest import run_async


def test_public_tools_package_and_factory_expose_the_four_coding_tools(
    local_environment: LocalEnvironment,
) -> None:
    tools = create_coding_tools(environment=local_environment)

    assert [tool.name for tool in tools] == ["read", "write", "edit", "bash"]
    assert all(isinstance(tool, AgentTool) for tool in tools)


def test_coding_tools_reject_arguments_outside_their_published_schemas(
    local_environment: LocalEnvironment,
) -> None:
    tools = {tool.name: tool for tool in create_coding_tools(environment=local_environment)}
    arguments: dict[str, dict[str, JSONValue]] = {
        "read": {"path": "file.txt", "unexpected": True},
        "write": {"path": "file.txt", "content": "text", "unexpected": True},
        "edit": {"path": "file.txt", "edits": [], "unexpected": True},
        "bash": {"command": "echo test", "unexpected": True},
    }

    for name, tool_arguments in arguments.items():
        with pytest.raises(ToolInputError, match="Unexpected argument"):
            run_async(tools[name].execute(tool_arguments))


@pytest.mark.parametrize(
    ("bytes_count", "expected"),
    [(1, "1B"), (1024, "1.0KB"), (1024 * 1024, "1.0MB")],
)
def test_format_size(bytes_count: int, expected: str) -> None:
    assert format_size(bytes_count) == expected


def test_status_and_head_tail_truncation_cover_lines_bytes_and_large_single_line() -> None:
    assert append_status_block("", "failed") == "failed"
    assert append_status_block("output", "failed") == "output\n\nfailed"

    head = truncate_head("one\ntwo\nthree", max_lines=2, max_bytes=100)
    tail = truncate_tail("one\ntwo\nthree", max_lines=2, max_bytes=100)
    bytes_limited = truncate_head("x" * 20, max_lines=10, max_bytes=10)
    tail_bytes = truncate_tail("x" * 20, max_lines=10, max_bytes=10)

    assert head.content == "one\ntwo"
    assert head.truncated_by == "lines"
    assert tail.content == "two\nthree"
    assert tail.truncated_by == "lines"
    assert bytes_limited.first_line_exceeds_limit is True
    assert tail_bytes.last_line_partial is True
    assert tail_bytes.content == "x" * 10
    assert tail_bytes.to_json()["max_bytes"] == 50 * 1024


def test_tool_argument_helpers_validate_types_and_resolve_paths(
    local_environment: LocalEnvironment,
) -> None:
    assert _str_arg({"name": "value"}, "name") == "value"
    assert _optional_int_arg({"value": 2}, "value") == 2
    assert (
        _workspace_path_arg(
            {"path": "nested/../child.txt"},
            "path",
            environment=local_environment,
        )
        == "child.txt"
    )

    for callback in (
        lambda: _str_arg({}, "name"),
        lambda: _optional_int_arg({"value": "two"}, "value"),
        lambda: _optional_int_arg({"value": True}, "value"),
    ):
        with pytest.raises(ToolInputError):
            callback()
