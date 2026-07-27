from __future__ import annotations

import base64
from pathlib import Path

import pytest
from vedex.schema import JSONValue
from vedex.tools import ToolInputError, create_read_tool

from .conftest import run_async


def test_read_text_offsets_limits_and_continuation_hints(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("one\ntwo\nthree\nfour", encoding="utf-8")
    tool = create_read_tool(cwd=tmp_path)

    first = run_async(tool.execute({"path": "notes.txt", "limit": 2}))
    second = run_async(tool.execute({"path": "notes.txt", "offset": 3, "limit": 1}))

    assert first.content.startswith("one\ntwo")
    assert "Use offset=3" in first.content
    assert second.content.startswith("three")
    assert "Use offset=4" in second.content


def test_read_rejects_invalid_paths_offsets_and_limits(tmp_path: Path) -> None:
    (tmp_path / "directory").mkdir()
    (tmp_path / "file.txt").write_text("line", encoding="utf-8")
    tool = create_read_tool(cwd=tmp_path)

    invalid_arguments: list[dict[str, JSONValue]] = [
        {"path": "missing.txt"},
        {"path": "directory"},
        {"path": "file.txt", "offset": -1},
        {"path": "file.txt", "limit": 0},
        {"path": "file.txt", "offset": 2},
        {"path": 3},
    ]
    for arguments in invalid_arguments:
        with pytest.raises(ToolInputError):
            run_async(tool.execute(arguments))


def test_read_truncates_many_lines_and_large_single_line(tmp_path: Path) -> None:
    many_lines = tmp_path / "many.txt"
    many_lines.write_text("\n".join(str(index) for index in range(2_001)), encoding="utf-8")
    huge_line = tmp_path / "huge.txt"
    huge_line.write_text("x" * (51 * 1024), encoding="utf-8")
    tool = create_read_tool(cwd=tmp_path)

    line_result = run_async(tool.execute({"path": "many.txt"}))
    huge_result = run_async(tool.execute({"path": "huge.txt"}))

    assert "Showing lines 1-2000 of 2001" in line_result.content
    assert "exceeds 50.0KB limit" in huge_result.content
    assert "Use bash:" in huge_result.content


def test_read_supported_image_returns_base64_metadata(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    payload = b"not-a-real-png-but-metadata-is-supported"
    path.write_bytes(payload)
    tool = create_read_tool(cwd=tmp_path)

    result = run_async(tool.execute({"path": "image.png"}))

    assert result.content == "Read image file [image/png]"
    assert result.data is not None
    assert result.data["mime_type"] == "image/png"
    assert result.data["image_base64"] == base64.b64encode(payload).decode("ascii")


def test_read_accepts_absolute_paths_and_treats_unsupported_images_as_text(tmp_path: Path) -> None:
    path = tmp_path / "image.bmp"
    path.write_text("ordinary text", encoding="utf-8")
    tool = create_read_tool(cwd=tmp_path / "other")

    result = run_async(tool.execute({"path": str(path)}))

    assert result.content == "ordinary text"
