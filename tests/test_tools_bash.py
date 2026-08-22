from __future__ import annotations

import sys
from pathlib import Path

import pytest
from vedex.environments import LocalEnvironment
from vedex.tools import ToolInputError, create_bash_tool

from .conftest import run_async


def _python_command(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


def test_bash_runs_in_cwd_and_reports_nonzero_and_timeout(
    tmp_path: Path,
    local_environment: LocalEnvironment,
) -> None:
    tool = create_bash_tool(environment=local_environment)
    success = run_async(tool.execute({"command": _python_command("import os; print(os.getcwd())")}))
    failed = run_async(
        tool.execute({"command": _python_command("import sys; print('bad'); sys.exit(3)")})
    )
    timed_out = run_async(
        tool.execute({"command": _python_command("import time; time.sleep(2)"), "timeout": 1})
    )

    assert success.ok is True
    assert str(tmp_path) in success.content
    assert failed.ok is False
    assert "Command exited with code 3" in failed.content
    assert timed_out.ok is False
    assert "timed out" in timed_out.content.lower()
    for timeout in (0, 601):
        with pytest.raises(ToolInputError, match="between 1 and 600"):
            run_async(tool.execute({"command": "ignored", "timeout": timeout}))


def test_bash_cancellation_and_truncated_output(
    local_environment: LocalEnvironment,
) -> None:
    class Cancelled:
        def is_cancelled(self) -> bool:
            return True

    tool = create_bash_tool(environment=local_environment)
    cancelled = run_async(tool.execute({"command": "ignored"}, signal=Cancelled()))
    assert cancelled.ok is False
    assert "cancelled" in cancelled.content.lower()

    output_tool = create_bash_tool(environment=local_environment)
    result = run_async(output_tool.execute({"command": _python_command("print('x' * 60000)")}))
    assert result.data is not None
    truncation = result.data["truncation"]
    assert isinstance(truncation, dict)
    assert truncation["truncated"] is True
    assert "Output truncated" in result.content
    assert "full_output_path" not in result.data
