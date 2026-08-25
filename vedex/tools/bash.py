from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..schema import AgentTool, AgentToolResult, CancellationToken, JSONValue
from .base import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_OUTPUT_LINES,
    ToolInputError,
    append_status_block,
    format_size,
    optional_int_argument,
    reject_unknown_arguments,
    str_argument,
    truncate_tail,
)

if TYPE_CHECKING:
    from ..environments.base import Environment

DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600


def create_bash_tool(
    *,
    environment: Environment,
) -> AgentTool:
    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        reject_unknown_arguments(arguments, {"command", "timeout"})
        command = str_argument(arguments, "command")
        timeout = optional_int_argument(arguments, "timeout")
        if timeout is None:
            timeout = DEFAULT_TIMEOUT_SECONDS
        if not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
            raise ToolInputError(f"timeout must be between 1 and {MAX_TIMEOUT_SECONDS} seconds")
        result = await environment.run_command(
            command,
            timeout_seconds=timeout,
            signal=signal,
        )

        output = result.stdout.decode(errors="replace")
        truncation = truncate_tail(output)
        output_text = truncation.content or "(no output)"
        if truncation.truncated:
            output_text = append_status_block(
                output_text,
                f"Output truncated to the last {format_size(truncation.output_bytes)}.",
            )

        exit_code = result.exit_code
        status: str | None = None
        if result.error is not None:
            status = f"Command failed to start: {result.error}"
        elif result.timed_out:
            status = f"Command timed out after {timeout} seconds"
        elif result.cancelled:
            status = "Command cancelled"
        elif exit_code not in (0, None):
            status = f"Command exited with code {exit_code}"
        elif exit_code is None:
            status = "Command ended without an exit code"
        if status:
            output_text = append_status_block(output_text, status)

        ok = exit_code == 0 and not result.timed_out and not result.cancelled
        return AgentToolResult(
            tool_call_id="",
            name="bash",
            ok=ok,
            content=output_text,
            error=None if ok else status,
            data={
                "command": command,
                "exit_code": exit_code,
                "timed_out": result.timed_out,
                "cancelled": result.cancelled,
                "duration_seconds": round(result.duration_seconds, 3),
                "truncation": truncation.to_json(),
            },
        )

    return AgentTool(
        name="bash",
        description=(
            "Execute a shell command in the workspace root. Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_OUTPUT_LINES} lines or "
            f"{DEFAULT_MAX_OUTPUT_BYTES // 1024}KB (whichever is hit first). Commands time out "
            f"after {DEFAULT_TIMEOUT_SECONDS} seconds by default; the maximum is "
            f"{MAX_TIMEOUT_SECONDS} seconds."
        ),
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
        prompt_snippet="Execute shell commands (ls, grep, find, etc.)",
        prompt_guidelines=(),
    )
