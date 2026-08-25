from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..schema import (
    AgentTool,
    CancellationToken,
    EnvironmentCancelledError,
    EnvironmentFileError,
    EnvironmentFileErrorKind,
    FatalEnvironmentError,
    JSONValue,
    WorkspacePathError,
)


class _EnvironmentContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


type EnvironmentState = Literal["created", "started", "stopping", "stopped"]


class EnvironmentLimits(_EnvironmentContract):
    max_command_seconds: float = Field(default=600.0, gt=0)


class WorkspaceIdentity(_EnvironmentContract):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    model_root: str = "."


class CommandResult(_EnvironmentContract):
    command: str
    stdout: bytes = b""
    exit_code: int | None = None
    timed_out: bool = False
    cancelled: bool = False
    duration_seconds: float = Field(default=0.0, ge=0)
    error: str | None = None


class WorkspacePatch(_EnvironmentContract):
    text: str
    base_revision: str | None = None
    is_git_repository: bool
    changed: bool


class WorkspaceExport(_EnvironmentContract):
    path: str
    format: Literal["tar.gz"] = "tar.gz"
    file_count: int = Field(ge=0)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EnvironmentMetadata(_EnvironmentContract):
    environment_type: str
    workspace: WorkspaceIdentity
    operating_system: str
    operating_system_release: str
    architecture: str
    python_version: str
    shell: str
    git_revision: str | None = None
    git_dirty: bool | None = None
    details: dict[str, JSONValue] = Field(default_factory=dict)


class EnvironmentStateError(FatalEnvironmentError):
    """Raised when an operation is invalid for the environment lifecycle state."""


class EnvironmentExportError(RuntimeError):
    """Raised when the workspace cannot be exported safely."""


@runtime_checkable
class Environment(Protocol):
    @property
    def state(self) -> EnvironmentState: ...

    @property
    def workspace(self) -> WorkspaceIdentity: ...

    @property
    def limits(self) -> EnvironmentLimits: ...

    @property
    def tools(self) -> Sequence[AgentTool]: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def normalize_path(self, path: str) -> str: ...

    async def read_bytes(
        self,
        path: str,
        *,
        signal: CancellationToken | None = None,
    ) -> bytes: ...

    async def write_bytes(
        self,
        path: str,
        data: bytes,
        *,
        signal: CancellationToken | None = None,
    ) -> None: ...

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float,
        signal: CancellationToken | None = None,
    ) -> CommandResult: ...

    async def collect_patch(
        self,
        *,
        signal: CancellationToken | None = None,
    ) -> WorkspacePatch: ...

    async def export_workspace(
        self,
        destination: Path,
        *,
        signal: CancellationToken | None = None,
    ) -> WorkspaceExport: ...

    async def get_metadata(self) -> EnvironmentMetadata: ...


def normalize_workspace_path(path: str) -> str:
    """Normalize a model-visible path without consulting a host filesystem."""

    if not path:
        raise WorkspacePathError(path, "path must not be empty")
    if "\x00" in path:
        raise WorkspacePathError(path, "path must not contain a null byte")

    windows_path = PureWindowsPath(path)
    portable_path = PurePosixPath(path.replace("\\", "/"))
    if portable_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise WorkspacePathError(path, "path must be workspace-relative")

    parts: list[str] = []
    for part in portable_path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise WorkspacePathError(path, "path escapes the workspace")
            parts.pop()
            continue
        parts.append(part)

    return "/".join(parts) or "."


__all__ = [
    "CommandResult",
    "Environment",
    "EnvironmentCancelledError",
    "EnvironmentExportError",
    "EnvironmentFileError",
    "EnvironmentFileErrorKind",
    "EnvironmentLimits",
    "EnvironmentMetadata",
    "EnvironmentState",
    "EnvironmentStateError",
    "WorkspaceExport",
    "WorkspaceIdentity",
    "WorkspacePatch",
    "WorkspacePathError",
    "normalize_workspace_path",
]
