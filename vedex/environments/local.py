"""Run Bash commands on the local workspace."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from typing import Any

from pydantic import BaseModel, Field


class LocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    timeout: float = 30.0


class LocalEnvironment:
    def __init__(
        self,
        *,
        config_class: type[LocalEnvironmentConfig] = LocalEnvironmentConfig,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)

    async def execute(
        self,
        action: dict[str, Any],
        cwd: str = "",
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            command = _command(action)
            workdir = cwd or self.config.cwd or os.getcwd()
            limit = timeout if timeout is not None else self.config.timeout
            result = await asyncio.to_thread(
                _run,
                command,
                workdir,
                {**os.environ, **self.config.env},
                limit,
            )
            return _result(result)
        except Exception as exc:
            return _error(exc)

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{type(self).__module__}.{type(self).__name__}",
                }
            }
        }


def _command(action: dict[str, Any]) -> str:
    command = action.get("command", "")
    if not isinstance(command, str):
        raise ValueError("Bash action command must be a string")
    return command


def _run(
    command: str,
    cwd: str,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        shell=True,
        text=True,
        cwd=cwd,
        env=env,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout) from exc
    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout)


def _result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "output": result.stdout or "",
        "returncode": result.returncode,
        "exception_info": "",
    }


def _error(error: Exception) -> dict[str, Any]:
    raw_output = getattr(error, "output", "")
    if isinstance(raw_output, bytes):
        raw_output = raw_output.decode("utf-8", errors="replace")
    return {
        "output": raw_output or "",
        "returncode": -1,
        "exception_info": f"An error occurred while executing the command: {error}",
        "extra": {"exception_type": type(error).__name__, "exception": str(error)},
    }


__all__ = ["LocalEnvironment", "LocalEnvironmentConfig"]
