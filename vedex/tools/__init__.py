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
from .edit import (
    UTF8_BOM,
    apply_edits_to_normalized_content,
    create_edit_tool,
    detect_line_ending,
    normalize_to_lf,
    restore_line_endings,
)
from .read import create_read_tool
from .write import create_write_tool

if TYPE_CHECKING:
    from ..environments.base import Environment


def create_coding_tools(
    *,
    environment: Environment,
) -> list[AgentTool]:
    return [
        create_read_tool(environment=environment),
        create_write_tool(environment=environment),
        create_edit_tool(environment=environment),
        create_bash_tool(environment=environment),
    ]


__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MAX_OUTPUT_LINES",
    "AgentTool",
    "ToolInputError",
    "TruncationResult",
    "UTF8_BOM",
    "append_status_block",
    "apply_edits_to_normalized_content",
    "create_bash_tool",
    "create_coding_tools",
    "create_edit_tool",
    "create_read_tool",
    "create_write_tool",
    "detect_line_ending",
    "format_size",
    "normalize_to_lf",
    "optional_int_argument",
    "reject_unknown_arguments",
    "restore_line_endings",
    "str_argument",
    "truncate_head",
    "truncate_tail",
    "workspace_path_argument",
]
