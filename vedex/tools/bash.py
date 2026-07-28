from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Mapping
from pathlib import Path
from time import monotonic
from typing import Any

from ..schema import AgentTool, AgentToolResult, CancellationToken, JSONValue
from .base import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_OUTPUT_LINES,
    ToolDefinition,
    ToolInputError,
    _optional_int_arg,
    _str_arg,
    append_status_block,
    format_size,
    truncate_tail,
)

DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600


def create_bash_tool_definition(
    *,
    cwd: str | Path | None = None,
) -> ToolDefinition:
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        command = _str_arg(arguments, "command")
        timeout = _optional_int_arg(arguments, "timeout")
        if timeout is None:
            timeout = DEFAULT_TIMEOUT_SECONDS
        if not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
            raise ToolInputError(f"timeout must be between 1 and {MAX_TIMEOUT_SECONDS} seconds")
        if signal is not None and signal.is_cancelled():
            raise ToolInputError("Command cancelled")

        start = monotonic()
        if os.name == "posix":
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        output_bytes, timed_out, cancelled = await _communicate_with_cancellation(
            process,
            timeout=timeout,
            signal=signal,
        )

        output = output_bytes.decode(errors="replace")
        truncation = truncate_tail(output)
        output_text = truncation.content or "(no output)"
        if truncation.truncated:
            output_text = append_status_block(
                output_text,
                f"Output truncated to the last {format_size(truncation.output_bytes)}.",
            )

        exit_code = process.returncode
        status: str | None = None
        if timed_out:
            status = f"Command timed out after {timeout} seconds"
        elif cancelled:
            status = "Command cancelled"
        elif exit_code not in (0, None):
            status = f"Command exited with code {exit_code}"
        if status:
            output_text = append_status_block(output_text, status)

        ok = exit_code == 0 and not timed_out and not cancelled
        return AgentToolResult(
            tool_call_id="",
            name="bash",
            ok=ok,
            content=output_text,
            error=None if ok else status,
            data={
                "command": command,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "cancelled": cancelled,
                "duration_seconds": round(monotonic() - start, 3),
                "truncation": truncation.to_json(),
            },
        )

    return ToolDefinition(
        name="bash",
        description=(
            "Execute a shell command in the current working directory. Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_OUTPUT_LINES} lines or "
            f"{DEFAULT_MAX_OUTPUT_BYTES // 1024}KB (whichever is hit first). Commands time out "
            f"after {DEFAULT_TIMEOUT_SECONDS} seconds by default; the maximum is "
            f"{MAX_TIMEOUT_SECONDS} seconds."
        ),
        prompt_snippet="Execute shell commands (ls, grep, find, etc.)",
        prompt_guidelines=(),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_TIMEOUT_SECONDS,
                    "default": DEFAULT_TIMEOUT_SECONDS,
                    "description": "Timeout in seconds.",
                },
            },
            "required": ["command"],
        },
        executor=execute,
    )


def create_bash_tool(
    *,
    cwd: str | Path | None = None,
) -> AgentTool:
    return create_bash_tool_definition(cwd=cwd).to_agent_tool()


async def _communicate_with_cancellation(
    process: asyncio.subprocess.Process,
    *,
    timeout: float,
    signal: CancellationToken | None,
) -> tuple[bytes, bool, bool]:
    communicate = asyncio.create_task(process.communicate())
    cancel_watch: asyncio.Task[None] | None = None
    try:
        wait_for: set[asyncio.Task[Any]] = {communicate}
        if signal is not None:
            cancel_watch = asyncio.create_task(_wait_for_cancel(signal))
            wait_for.add(cancel_watch)

        done, _pending = await asyncio.wait(
            wait_for,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if communicate in done:
            output_bytes, _stderr = communicate.result()
            return output_bytes, False, False

        cancelled = cancel_watch is not None and cancel_watch in done
        _kill_process_tree(process)
        output_bytes, _stderr = await communicate
        return output_bytes, not cancelled, cancelled
    except asyncio.CancelledError:
        _kill_process_tree(process)
        if not communicate.done():
            communicate.cancel()
        raise
    finally:
        if cancel_watch is not None:
            cancel_watch.cancel()


async def _wait_for_cancel(signal: CancellationToken) -> None:
    while not signal.is_cancelled():
        await asyncio.sleep(0.05)


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        kill_process_group = getattr(os, "killpg", None)
        kill_signal = getattr(signal, "SIGKILL", None)
        if kill_process_group is None or kill_signal is None:
            return
        try:
            kill_process_group(process.pid, kill_signal)
        except ProcessLookupError:
            return
    else:
        try:
            process.kill()
        except ProcessLookupError:
            return
