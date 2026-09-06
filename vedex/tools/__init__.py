"""Built-in coding tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..schema import AgentTool
from .base import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_OUTPUT_LINES,
    ToolInputError,
    TruncationResult,
    append_status_block,
    format_size,
    optional_int_argument,
    reject_unknown_arguments,
    str_argument,
    truncate_head,
    truncate_tail,
    workspace_path_argument,
)
from .bash import create_bash_tool

if TYPE_CHECKING:
    from ..environments.base import Environment


def create_coding_tools(
    *,
    environment: Environment,
) -> list[AgentTool]:
    return [
        create_bash_tool(environment=environment),
    ]


__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MAX_OUTPUT_LINES",
    "AgentTool",
    "ToolInputError",
    "TruncationResult",
    "append_status_block",
    "create_bash_tool",
    "create_coding_tools",
    "format_size",
    "optional_int_argument",
    "reject_unknown_arguments",
    "str_argument",
    "truncate_head",
    "truncate_tail",
    "workspace_path_argument",
]
