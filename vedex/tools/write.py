from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..schema import AgentTool, AgentToolResult, CancellationToken, JSONValue
from .base import ToolDefinition, _file_lock, _path_arg, _str_arg


def create_write_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        del signal
        path = _path_arg(arguments, "path", cwd=root)
        content = _str_arg(arguments, "content")

        async with _file_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as file:
                file.write(content)

        return AgentToolResult(
            tool_call_id="",
            name="write",
            ok=True,
            content=f"Successfully wrote to {path}.",
            data={"path": str(path), "characters": len(content)},
        )

    return ToolDefinition(
        name="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        prompt_snippet="Create or overwrite files",
        prompt_guidelines=("Use write only for new files or complete rewrites.",),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
        executor=execute,
    )


def create_write_tool(*, cwd: str | Path | None = None) -> AgentTool:
    return create_write_tool_definition(cwd=cwd).to_agent_tool()
