from __future__ import annotations

from collections.abc import Mapping

from ..environments import Environment, EnvironmentFileError, WorkspacePathError
from ..schema import AgentTool, AgentToolResult, CancellationToken, JSONValue
from .base import (
    ToolInputError,
    _reject_unknown_args,
    _str_arg,
    _workspace_path_arg,
)


def create_write_tool(*, environment: Environment) -> AgentTool:
    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        _reject_unknown_args(arguments, {"path", "content"})
        path = _workspace_path_arg(arguments, "path", environment=environment)
        content = _str_arg(arguments, "content")
        encoded = content.encode("utf-8")

        try:
            await environment.write_bytes(path, encoded, signal=signal)
        except WorkspacePathError as exc:
            raise ToolInputError(f"Invalid workspace path {path!r}: {exc.reason}") from exc
        except EnvironmentFileError as exc:
            if exc.kind == "is_directory":
                raise ToolInputError(f"Could not write file {path}: path is a directory") from exc
            detail = exc.detail or exc.kind.replace("_", " ")
            raise ToolInputError(f"Could not write file {path}: {detail}") from exc

        return AgentToolResult(
            tool_call_id="",
            name="write",
            ok=True,
            content=f"Successfully wrote to {path}.",
            data={"path": path, "characters": len(content), "bytes": len(encoded)},
        )

    return AgentTool(
        name="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories. Paths must be workspace-relative."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        executor=execute,
        prompt_snippet="Create or overwrite files",
        prompt_guidelines=("Use write only for new files or complete rewrites.",),
    )
