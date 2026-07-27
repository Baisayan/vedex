from __future__ import annotations

import sys
from pathlib import Path

import pytest
from vedex.resources import VedexPaths
from vedex.tools import (
    ShellConfigError,
    ToolInputError,
    create_bash_tool,
    load_shell_settings,
    shell_settings_from_json,
    shell_settings_path,
)
from vedex.tools.bash import _prefixed_shell_command

from .conftest import run_async


def _python_command(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


def test_shell_settings_loading_and_validation(tmp_path: Path) -> None:
    paths = VedexPaths(home=tmp_path / ".vedex")
    settings_path = shell_settings_path(paths)
    assert load_shell_settings(paths).shell_command_prefix is None
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text('{"shellCommandPrefix": "  export X=1  "}', encoding="utf-8")
    assert load_shell_settings(paths).shell_command_prefix == "export X=1"
    assert (
        shell_settings_from_json({"shell_command_prefix": "prefix"}).shell_command_prefix
        == "prefix"
    )

    for value in (
        {"bad": "field"},
        {"shellCommandPrefix": "a", "shell_command_prefix": "b"},
        {"shellCommandPrefix": 3},
    ):
        with pytest.raises(ShellConfigError):
            shell_settings_from_json(value)

    settings_path.write_text("not json", encoding="utf-8")
    with pytest.raises(ShellConfigError, match="not valid JSON"):
        load_shell_settings(paths)
    settings_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ShellConfigError, match="must be a JSON object"):
        load_shell_settings(paths)
    assert _prefixed_shell_command("command", "prefix") == "prefix\ncommand"


def test_bash_runs_in_cwd_and_reports_nonzero_and_timeout(tmp_path: Path) -> None:
    tool = create_bash_tool(cwd=tmp_path)
    success = run_async(tool.execute({"command": _python_command("import os; print(os.getcwd())")}))
    failed = run_async(
        tool.execute({"command": _python_command("import sys; print('bad'); sys.exit(3)")})
    )
    timed_out = run_async(
        tool.execute({"command": _python_command("import time; time.sleep(1)"), "timeout": 0.01})
    )

    assert success.ok is True
    assert str(tmp_path) in success.content
    assert failed.ok is False
    assert "Command exited with code 3" in failed.content
    assert timed_out.ok is False
    assert "timed out" in timed_out.content.lower()


def test_bash_cancellation_prefix_and_truncated_output(tmp_path: Path) -> None:
    class Cancelled:
        def is_cancelled(self) -> bool:
            return True

    tool = create_bash_tool(cwd=tmp_path, shell_command_prefix="prefix")
    with pytest.raises(ToolInputError, match="cancelled"):
        run_async(tool.execute({"command": "ignored"}, signal=Cancelled()))

    output_tool = create_bash_tool(cwd=tmp_path)
    result = run_async(output_tool.execute({"command": _python_command("print('x' * 60000)")}))
    assert result.data is not None
    truncation = result.data["truncation"]
    assert isinstance(truncation, dict)
    assert truncation["truncated"] is True
    output_path = result.data["full_output_path"]
    assert isinstance(output_path, str)
    try:
        assert Path(output_path).suffix == ".txt"
        assert Path(output_path).read_text(encoding="utf-8").startswith("x")
    finally:
        Path(output_path).unlink(missing_ok=True)
