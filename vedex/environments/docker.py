"""Run Bash commands in a Docker or Podman container."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import uuid
from typing import Any

from pydantic import BaseModel, Field


class DockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    env: dict[str, str] = Field(default_factory=dict)
    forward_env: list[str] = Field(default_factory=list)
    timeout: float = 30.0
    executable: str = Field(default_factory=lambda: os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker"))
    run_args: list[str] = Field(default_factory=lambda: ["--rm"])
    container_timeout: str = "2h"
    pull_timeout: float = 120.0
    interpreter: list[str] = Field(default_factory=lambda: ["bash", "-lc"])


class DockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type[DockerEnvironmentConfig] = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> None:
        self.logger = logger or logging.getLogger("vedex.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    async def execute(
        self,
        action: dict[str, Any],
        cwd: str = "",
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            command = action.get("command", "")
            if not isinstance(command, str):
                raise ValueError("Bash action command must be a string")
            if self.container_id is None:
                raise RuntimeError("Docker container is not running")
            workdir = cwd or self.config.cwd
            limit = timeout if timeout is not None else self.config.timeout
            result = await asyncio.to_thread(self._run_command, command, workdir, limit)
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

    def _start_container(self) -> None:
        name = f"vedex-{uuid.uuid4().hex[:8]}"
        command = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug("Starting container with command: %s", shlex.join(command))
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,
            check=True,
        )
        self.container_id = result.stdout.strip()
        self.logger.info("Started container %s", name)

    def _run_command(
        self,
        command: str,
        cwd: str,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if self.container_id is None:
            raise RuntimeError("Docker container is not running")
        docker_command = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                docker_command.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            docker_command.extend(["-e", f"{key}={value}"])
        docker_command.extend([self.container_id, *self.config.interpreter, command])
        return subprocess.run(
            docker_command,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def cleanup(self) -> None:
        if self.container_id is None:
            return
        container_id = self.container_id
        self.container_id = None
        subprocess.Popen(
            [self.config.executable, "rm", "--force", container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def __del__(self) -> None:
        self.cleanup()


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


__all__ = ["DockerEnvironment", "DockerEnvironmentConfig"]
