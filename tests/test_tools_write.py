from __future__ import annotations

from pathlib import Path

import pytest
from vedex.schema import JSONValue
from vedex.tools import ToolInputError, create_write_tool

from .conftest import run_async


def test_write_creates_parents_and_overwrites_utf8_content(tmp_path: Path) -> None:
    tool = create_write_tool(cwd=tmp_path)

    first = run_async(tool.execute({"path": "nested/file.txt", "content": "héllo"}))
    second = run_async(tool.execute({"path": "nested/file.txt", "content": "replacement"}))

    assert (tmp_path / "nested" / "file.txt").read_text(encoding="utf-8") == "replacement"
    assert first.data == {"path": str(tmp_path / "nested" / "file.txt"), "characters": 5}
    assert second.content.startswith("Successfully wrote")


@pytest.mark.parametrize(
    "arguments",
    [{}, {"path": "file.txt"}, {"content": "text"}, {"path": 1, "content": "text"}],
)
def test_write_rejects_invalid_arguments(arguments: dict[str, JSONValue], tmp_path: Path) -> None:
    tool = create_write_tool(cwd=tmp_path)
    with pytest.raises(ToolInputError):
        run_async(tool.execute(arguments))
