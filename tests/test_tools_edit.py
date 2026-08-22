from __future__ import annotations

from pathlib import Path

import pytest
from vedex.environments import LocalEnvironment
from vedex.schema import JSONValue
from vedex.tools import (
    UTF8_BOM,
    ToolInputError,
    apply_edits_to_normalized_content,
    create_edit_tool,
    detect_line_ending,
    normalize_to_lf,
    restore_line_endings,
)

from .conftest import run_async


def test_edit_applies_disjoint_original_matches(
    tmp_path: Path,
    local_environment: LocalEnvironment,
) -> None:
    path = tmp_path / "file.txt"
    path.write_text("alpha\nbeta\ngamma", encoding="utf-8")
    tool = create_edit_tool(environment=local_environment)

    result = run_async(
        tool.execute(
            {
                "path": "file.txt",
                "edits": [
                    {"oldText": "alpha", "newText": "one"},
                    {"oldText": "gamma", "newText": "three"},
                ],
            }
        )
    )

    assert path.read_text(encoding="utf-8") == "one\nbeta\nthree"
    assert result.content == "Edited file.txt: 2 replacement(s)."


def test_edit_accepts_json_and_legacy_edit_arguments_and_preserves_bom_crlf(
    tmp_path: Path,
    local_environment: LocalEnvironment,
) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(f"{UTF8_BOM}old\r\nkeep\r\n".encode())
    tool = create_edit_tool(environment=local_environment)

    run_async(
        tool.execute(
            {
                "path": "file.txt",
                "edits": '[{"oldText": "old", "newText": "new"}]',
                "oldText": "keep",
                "newText": "stays changed",
            }
        )
    )

    assert path.read_bytes() == f"{UTF8_BOM}new\r\nstays changed\r\n".encode()


@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "missing.txt", "edits": [{"oldText": "a", "newText": "b"}]},
        {"path": "file.txt", "edits": []},
        {"path": "file.txt", "edits": [{"oldText": "", "newText": "b"}]},
        {"path": "file.txt", "edits": [{"oldText": "missing", "newText": "b"}]},
        {"path": "file.txt", "edits": [{"oldText": "same", "newText": "same"}]},
    ],
)
def test_edit_rejects_invalid_or_unapplicable_edits(
    arguments: dict[str, JSONValue],
    tmp_path: Path,
    local_environment: LocalEnvironment,
) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("same\nsame", encoding="utf-8")
    original = file_path.read_bytes()
    tool = create_edit_tool(environment=local_environment)
    with pytest.raises(ToolInputError):
        run_async(tool.execute(arguments))
    assert file_path.read_bytes() == original


def test_edit_rejects_duplicate_and_overlapping_matches() -> None:
    with pytest.raises(ToolInputError, match="unique"):
        apply_edits_to_normalized_content(
            "same\nsame", [{"oldText": "same", "newText": "x"}], "file"
        )


def test_edit_rejects_directory_and_non_object_edit_entries(
    tmp_path: Path,
    local_environment: LocalEnvironment,
) -> None:
    (tmp_path / "directory").mkdir()
    (tmp_path / "file.txt").write_text("text", encoding="utf-8")
    tool = create_edit_tool(environment=local_environment)

    with pytest.raises(ToolInputError, match="Path is a directory"):
        run_async(tool.execute({"path": "directory", "edits": [{"oldText": "a", "newText": "b"}]}))
    with pytest.raises(ToolInputError, match="must be an object"):
        run_async(tool.execute({"path": "file.txt", "edits": ["not an object"]}))
    with pytest.raises(ToolInputError, match="overlap"):
        apply_edits_to_normalized_content(
            "abcdef",
            [{"oldText": "abc", "newText": "x"}, {"oldText": "bcd", "newText": "y"}],
            "file",
        )


def test_edit_formatting_helpers_cover_lf_and_crlf() -> None:
    assert detect_line_ending("one\r\ntwo\n") == "\r\n"
    assert normalize_to_lf("one\r\ntwo\rthree") == "one\ntwo\nthree"
    assert restore_line_endings("one\ntwo", "\r\n") == "one\r\ntwo"
