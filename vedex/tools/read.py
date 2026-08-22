from __future__ import annotations

from collections.abc import Mapping

from ..environments import Environment, EnvironmentFileError, WorkspacePathError
from ..schema import AgentTool, AgentToolResult, CancellationToken, JSONValue
from .base import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_OUTPUT_LINES,
    ToolInputError,
    _optional_int_arg,
    _reject_unknown_args,
    _workspace_path_arg,
    format_size,
    truncate_head,
)


def create_read_tool(*, environment: Environment) -> AgentTool:
    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        _reject_unknown_args(arguments, {"path", "offset", "limit"})
        path = _workspace_path_arg(arguments, "path", environment=environment)
        offset = _optional_int_arg(arguments, "offset")
        limit = _optional_int_arg(arguments, "limit")

        if offset is None:
            offset = 1
        if limit is None:
            limit = DEFAULT_MAX_OUTPUT_LINES
        if offset < 1:
            raise ToolInputError("offset must be at least 1")
        if not 1 <= limit <= DEFAULT_MAX_OUTPUT_LINES:
            raise ToolInputError(f"limit must be between 1 and {DEFAULT_MAX_OUTPUT_LINES}")
        try:
            data = await environment.read_bytes(path, signal=signal)
        except WorkspacePathError as exc:
            raise ToolInputError(f"Invalid workspace path {path!r}: {exc.reason}") from exc
        except EnvironmentFileError as exc:
            if exc.kind == "not_found":
                raise ToolInputError(f"File not found: {path}") from exc
            if exc.kind == "is_directory":
                raise ToolInputError(f"Path is a directory: {path}") from exc
            detail = f": {exc.detail}" if exc.detail else ""
            raise ToolInputError(f"Could not read file {path}{detail}") from exc
        if b"\0" in data:
            raise ToolInputError(f"File is not valid UTF-8 text: {path}")
        try:
            all_lines = data.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ToolInputError(f"File is not valid UTF-8 text: {path}") from exc
        if not all_lines:
            return AgentToolResult(
                tool_call_id="",
                name="read",
                ok=True,
                content="(empty file)",
            )

        start_line = offset - 1
        if start_line >= len(all_lines):
            raise ToolInputError(
                f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
            )

        end_line = min(start_line + limit, len(all_lines))
        selected = "\n".join(
            f"{line_number:>6}  {line}"
            for line_number, line in enumerate(all_lines[start_line:end_line], start=offset)
        )

        truncation = truncate_head(selected)
        start_display = start_line + 1

        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(all_lines[start_line].encode()))
            output = (
                f"[Line {start_display} is {first_line_size}, exceeds "
                f"{format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit. Use bash to inspect it.]"
            )
        elif truncation.truncated:
            end_display = start_display + truncation.output_lines - 1
            next_offset = end_display + 1
            output = truncation.content
            if truncation.truncated_by == "lines":
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)} "
                    f"({format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit). "
                    f"Use offset={next_offset} to continue.]"
                )
        elif end_line < len(all_lines):
            remaining = len(all_lines) - end_line
            next_offset = end_line + 1
            output = (
                f"{truncation.content}\n\n[{remaining} more lines in file. "
                f"Use offset={next_offset} to continue.]"
            )
        else:
            output = truncation.content

        return AgentToolResult(
            tool_call_id="",
            name="read",
            ok=True,
            content=output,
        )

    return AgentTool(
        name="read",
        description=(
            "Read the contents of a UTF-8 text file. Output is truncated to "
            f"{DEFAULT_MAX_OUTPUT_LINES} lines or {DEFAULT_MAX_OUTPUT_BYTES // 1024}KB "
            "(whichever is hit first). Use offset/limit for large files. When you need the "
            "full file, continue with offset until complete. Paths must be workspace-relative."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "First one-based line to read",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": DEFAULT_MAX_OUTPUT_LINES,
                    "default": DEFAULT_MAX_OUTPUT_LINES,
                    "description": "Maximum number of lines to read",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        executor=execute,
        prompt_snippet="Read file contents",
        prompt_guidelines=("Use read to examine files instead of cat or sed.",),
    )
