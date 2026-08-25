from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..schema import (
    AgentTool,
    AgentToolResult,
    CancellationToken,
    EnvironmentFileError,
    JSONValue,
    WorkspacePathError,
)
from .base import (
    ToolInputError,
    reject_unknown_arguments,
    str_argument,
    workspace_path_argument,
)

if TYPE_CHECKING:
    from ..environments.base import Environment


def create_write_tool(*, environment: Environment) -> AgentTool:
    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        reject_unknown_arguments(arguments, {"path", "content"})
        path = workspace_path_argument(arguments, "path", environment=environment)
        content = str_argument(arguments, "content")
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
