from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from vedex.tools import create_coding_tools
from vedex.tools.base import (
    ToolDefinition,
    ToolInputError,
    _file_lock,
    _optional_int_arg,
    _path_arg,
    _str_arg,
    append_status_block,
    format_size,
    truncate_head,
    truncate_tail,
)

from .conftest import make_tool, run_async


def test_public_tools_package_and_factory_expose_the_four_coding_tools(tmp_path: Path) -> None:
    tools = create_coding_tools(cwd=tmp_path)
    definition = ToolDefinition(
        name="check",
        description="Check.",
        prompt_snippet="Check things",
        prompt_guidelines=("Check first",),
        input_schema={"type": "object"},
        executor=make_tool().executor,
    )

    assert [tool.name for tool in tools] == ["read", "write", "edit", "bash"]
    assert definition.to_agent_tool().prompt_guidelines == ("Check first",)


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


def test_tool_argument_helpers_validate_types_and_resolve_paths(tmp_path: Path) -> None:
    assert _str_arg({"name": "value"}, "name") == "value"
    assert _optional_int_arg({"value": 2}, "value") == 2
    assert _path_arg({"path": "child.txt"}, "path", cwd=tmp_path) == tmp_path / "child.txt"

    for callback in (
        lambda: _str_arg({}, "name"),
        lambda: _optional_int_arg({"value": "two"}, "value"),
    ):
        with pytest.raises(ToolInputError):
            callback()


def test_file_lock_serializes_same_path_access(tmp_path: Path) -> None:
    async def run() -> list[str]:
        order: list[str] = []
        release_first = asyncio.Event()

        async def first() -> None:
            async with _file_lock(tmp_path / "same.txt"):
                order.append("first")
                await release_first.wait()

        async def second() -> None:
            async with _file_lock(tmp_path / "same.txt"):
                order.append("second")

        first_task = asyncio.create_task(first())
        await asyncio.sleep(0)
        second_task = asyncio.create_task(second())
        await asyncio.sleep(0)
        assert order == ["first"]
        release_first.set()
        await asyncio.gather(first_task, second_task)
        return order

    assert run_async(run()) == ["first", "second"]
