from __future__ import annotations

import asyncio
import contextlib
import copy
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Literal, Protocol, Self, cast
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from ..schema import AgentTool, CancellationToken, JSONValue
from .base import (
    CommandResult,
    EnvironmentCancelledError,
    EnvironmentExportError,
    EnvironmentFileError,
    EnvironmentFileErrorKind,
    EnvironmentLimits,
    EnvironmentMetadata,
    EnvironmentState,
    EnvironmentStateError,
    WorkspaceExport,
    WorkspaceIdentity,
    WorkspacePatch,
    WorkspacePathError,
    normalize_workspace_path,
)

type DockerWorkspaceMode = Literal["image", "copy", "mount"]
type DockerPullPolicy = Literal["always", "missing", "never"]

_CONTAINER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_JSON_OBJECT_ADAPTER: TypeAdapter[dict[str, JSONValue]] = TypeAdapter(dict[str, JSONValue])
_GIT_TIMEOUT_SECONDS = 30.0
_FILE_NOT_FOUND = 44
_FILE_IS_DIRECTORY = 45
_FILE_PERMISSION_DENIED = 46
_FILE_IO_ERROR = 47
_FILE_OUTSIDE_WORKSPACE = 48

_READ_FILE_SCRIPT = r"""
root=$1
target=$2
if [ ! -e "$target" ] && [ ! -L "$target" ]; then
    exit 44
fi
if [ -L "$target" ]; then
    exit 48
fi
parent=${target%/*}
if [ -z "$parent" ]; then
    parent=/
fi
resolved_parent=$(CDPATH= cd -- "$parent" 2>/dev/null && pwd -P) || exit 47
case "$resolved_parent" in
    "$root"|"$root"/*) ;;
    *) exit 48 ;;
esac
if [ -d "$target" ]; then
    exit 45
fi
if [ ! -r "$target" ]; then
    exit 46
fi
cat -- "$target" || exit 47
""".strip()

_WRITE_FILE_SCRIPT = r"""
root=$1
target=$2
parent=${target%/*}
if [ -z "$parent" ]; then
    parent=/
fi
name=${target##*/}
if [ -d "$target" ]; then
    exit 45
fi
if [ -L "$target" ]; then
    exit 48
fi
probe=$parent
while [ ! -e "$probe" ] && [ "$probe" != "/" ]; do
    probe=${probe%/*}
    if [ -z "$probe" ]; then
        probe=/
    fi
done
resolved_probe=$(CDPATH= cd -- "$probe" 2>/dev/null && pwd -P) || exit 47
case "$resolved_probe" in
    "$root"|"$root"/*) ;;
    *) exit 48 ;;
esac
mkdir -p -- "$parent" || exit 47
resolved_parent=$(CDPATH= cd -- "$parent" 2>/dev/null && pwd -P) || exit 47
case "$resolved_parent" in
    "$root"|"$root"/*) ;;
    *) exit 48 ;;
esac
temporary="$resolved_parent/.vedex-write-$$"
trap 'rm -f -- "$temporary"' EXIT HUP INT TERM
cat > "$temporary" || exit 47
mv -f -- "$temporary" "$resolved_parent/$name" || exit 47
trap - EXIT HUP INT TERM
""".strip()

_RESOLVE_WORKDIR_SCRIPT = 'CDPATH= cd -- "$1" 2>/dev/null && pwd -P'


class DockerEnvironmentConfig(BaseModel):
    """Configuration owned by a single ephemeral Docker environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image: str = Field(min_length=1)
    workspace_mode: DockerWorkspaceMode = "image"
    workspace: Path | None = None
    workdir: str = "/workspace"
    environment: dict[str, str] = Field(default_factory=dict)
    cpus: float | None = Field(default=None, gt=0)
    memory: str | None = None
    network: str | None = "none"
    engine: str = "docker"
    pull: DockerPullPolicy = "missing"
    interpreter: tuple[str, ...] = ("sh", "-lc")
    keepalive_command: tuple[str, ...] = (
        "sh",
        "-c",
        "while :; do sleep 3600; done",
    )
    container_name_prefix: str = "vedex"
    startup_timeout_seconds: float = Field(default=120.0, gt=0)
    operation_timeout_seconds: float = Field(default=30.0, gt=0)
    cleanup_timeout_seconds: float = Field(default=30.0, gt=0)

    @field_validator("image", "engine", "memory", "network", mode="before")
    @classmethod
    def _reject_blank_optional_strings(cls, value: object) -> object:
        if isinstance(value, str) and (not value.strip() or "\x00" in value):
            raise ValueError("value must be non-empty and contain no null bytes")
        return value

    @field_validator("workdir")
    @classmethod
    def _validate_workdir(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("workdir must not contain a null byte")
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("workdir must be an absolute container path without '..'")
        return path.as_posix()

    @field_validator("container_name_prefix")
    @classmethod
    def _validate_container_name_prefix(cls, value: str) -> str:
        if len(value) > 40 or _CONTAINER_NAME_PATTERN.fullmatch(value) is None:
            raise ValueError("container_name_prefix must be 1-40 Docker name characters")
        return value

    @field_validator("interpreter", "keepalive_command")
    @classmethod
    def _validate_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not part or "\x00" in part for part in value):
            raise ValueError("container commands must contain non-empty arguments")
        return value

    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        for name, item in value.items():
            if not name or "=" in name or "\x00" in name or "\x00" in item:
                raise ValueError("environment entries must be valid NAME=value pairs")
        return value

    @model_validator(mode="after")
    def _validate_workspace_mode(self) -> Self:
        if self.workspace_mode == "image" and self.workspace is not None:
            raise ValueError("workspace must be omitted when workspace_mode='image'")
        if self.workspace_mode != "image" and self.workspace is None:
            raise ValueError(f"workspace is required when workspace_mode={self.workspace_mode!r}")
        return self


@dataclass(frozen=True, slots=True)
class DockerCLIResult:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


class DockerCLI(Protocol):
    async def run(
        self,
        arguments: Sequence[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: float | None = None,
        merge_stderr: bool = False,
    ) -> DockerCLIResult: ...


class _SubprocessDockerCLI:
    def __init__(self, executable: str) -> None:
        self._executable = executable

    async def run(
        self,
        arguments: Sequence[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: float | None = None,
        merge_stderr: bool = False,
    ) -> DockerCLIResult:
        process = await asyncio.create_subprocess_exec(
            self._executable,
            *arguments,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=(asyncio.subprocess.STDOUT if merge_stderr else asyncio.subprocess.PIPE),
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        )
        communication = asyncio.create_task(process.communicate(stdin))
        try:
            if timeout_seconds is None:
                stdout, stderr = await communication
            else:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communication),
                    timeout=timeout_seconds,
                )
        except TimeoutError:
            await _terminate_process(process)
            with contextlib.suppress(Exception):
                await communication
            raise
        except asyncio.CancelledError:
            await _terminate_process(process)
            with contextlib.suppress(Exception):
                await asyncio.shield(communication)
            raise
        return DockerCLIResult(
            returncode=process.returncode or 0,
            stdout=stdout or b"",
            stderr=stderr or b"",
        )


@dataclass(frozen=True, slots=True)
class _ControlledResult:
    result: DockerCLIResult | None = None
    timed_out: bool = False
    cancelled: bool = False


class DockerEnvironment:
    """Execute the shared coding tools in one explicitly managed container."""

    def __init__(
        self,
        config: DockerEnvironmentConfig,
        *,
        limits: EnvironmentLimits | None = None,
        cli: DockerCLI | None = None,
    ) -> None:
        self._config = config.model_copy(deep=True)
        self._limits = limits or EnvironmentLimits()
        suffix = uuid4().hex[:12]
        self._container_name = f"{self._config.container_name_prefix}-{suffix}"
        source = self._config.workspace
        workspace_name = (
            source.expanduser().name
            if source is not None
            else self._config.image.rsplit("/", maxsplit=1)[-1]
        )
        self._workspace = WorkspaceIdentity(
            id=f"docker-{suffix}",
            name=workspace_name or "workspace",
        )
        self._state: EnvironmentState = "created"
        self._container_id: str | None = None
        self._creation_attempted = False
        self._resolved_workdir = self._config.workdir
        self._shutdown_requested = False
        self._operation_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._cli = cli or _SubprocessDockerCLI(self._config.engine)
        self._tools: tuple[AgentTool, ...] | None = None
        self._engine_version = "unknown"
        self._image_id = "unknown"
        self._image_digest = "unknown"
        self._image_reference = self._config.image
        self._resolved_image_reference = "unknown"
        self._image_architecture = "unknown"
        self._image_os = "unknown"

    @property
    def state(self) -> EnvironmentState:
        return self._state

    @property
    def config(self) -> DockerEnvironmentConfig:
        return self._config.model_copy(deep=True)

    @property
    def container_name(self) -> str:
        return self._container_name

    @property
    def workspace(self) -> WorkspaceIdentity:
        return self._workspace

    @property
    def limits(self) -> EnvironmentLimits:
        return self._limits

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        if self._tools is None:
            from ..tools import create_coding_tools

            self._tools = tuple(create_coding_tools(environment=self))
        return self._tools

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        await self.stop()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._state == "started":
                return
            if self._state == "stopping":
                raise EnvironmentStateError(
                    "Cannot start the Docker environment while it is stopping"
                )
            if self._creation_attempted:
                raise EnvironmentStateError(
                    "A previous container cleanup failed; call stop() again before restarting"
                )

            source = self._resolved_source()
            self._shutdown_requested = False
            self._creation_attempted = False
            try:
                await self._capture_engine_version()
                self._creation_attempted = True
                create_result = await self._run_startup_cli(self._create_arguments(source))
                self._require_cli_success(create_result, "container creation")
                container_id = create_result.stdout.decode(errors="replace").strip()
                if not container_id:
                    raise EnvironmentStateError(
                        "Container engine returned no ID after container creation"
                    )
                self._container_id = container_id

                start_result = await self._run_startup_cli(
                    ("container", "start", self._container_name)
                )
                self._require_cli_success(start_result, "container startup")
                if self._config.workspace_mode == "copy":
                    if source is None:
                        raise AssertionError("Copy workspace source unexpectedly missing")
                    copy_result = await self._run_startup_cli(
                        (
                            "container",
                            "cp",
                            f"{source.as_posix().rstrip('/')}/.",
                            f"{self._container_name}:{self._config.workdir}",
                        )
                    )
                    self._require_cli_success(copy_result, "workspace copy")

                root_result = await self._run_startup_cli(
                    self._exec_arguments(
                        (
                            "sh",
                            "-c",
                            _RESOLVE_WORKDIR_SCRIPT,
                            "vedex-resolve-workdir",
                            self._config.workdir,
                        )
                    )
                )
                self._require_cli_success(root_result, "workspace resolution")
                resolved_workdir = root_result.stdout.decode(errors="replace").strip()
                if not resolved_workdir.startswith("/"):
                    raise EnvironmentStateError(
                        "Container workdir did not resolve to an absolute path"
                    )
                self._resolved_workdir = resolved_workdir
                await self._capture_image_metadata()
            except BaseException as start_error:
                cleanup_error = await self._cleanup_after_failed_start()
                self._state = "stopped"
                if cleanup_error is not None:
                    raise BaseExceptionGroup(
                        "Docker environment startup and cleanup both failed",
                        [start_error, cleanup_error],
                    ) from None
                raise

            self._state = "started"

    async def stop(self) -> None:
        if self._state == "stopping":
            while self._state == "stopping":
                await asyncio.sleep(0)
            return

        async with self._lifecycle_lock:
            if self._state != "started" and not self._creation_attempted:
                self._state = "stopped"
                return

            self._state = "stopping"
            self._shutdown_requested = True
            cleanup_error: BaseException | None = None
            cancellation: asyncio.CancelledError | None = None
            cleanup_task = asyncio.create_task(self._remove_container())
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as exc:
                cancellation = exc
                try:
                    await asyncio.shield(cleanup_task)
                except BaseException as cleanup_exc:
                    cleanup_error = cleanup_exc
            except BaseException as exc:
                cleanup_error = exc
            finally:
                self._state = "stopped"

            if cleanup_error is not None:
                if isinstance(cleanup_error, asyncio.CancelledError):
                    raise cleanup_error
                raise EnvironmentStateError(
                    f"Docker environment cleanup failed: {_error_detail(cleanup_error)}"
                ) from cleanup_error
            if cancellation is not None:
                raise cancellation

    def normalize_path(self, path: str) -> str:
        return normalize_workspace_path(path)

    async def read_bytes(
        self,
        path: str,
        *,
        signal: CancellationToken | None = None,
    ) -> bytes:
        self._require_started()
        normalized = self.normalize_path(path)
        self._raise_if_cancelled(signal)
        target = self._container_path(normalized)
        async with self._operation_lock:
            self._require_started()
            result = await self._run_exec_operation(
                (
                    "sh",
                    "-c",
                    _READ_FILE_SCRIPT,
                    "vedex-read",
                    self._resolved_workdir,
                    target,
                ),
                signal=signal,
            )
        if result.returncode != 0:
            self._raise_file_error(result, path=normalized, operation="read")
        self._raise_if_cancelled(signal)
        return result.stdout

    async def write_bytes(
        self,
        path: str,
        data: bytes,
        *,
        signal: CancellationToken | None = None,
    ) -> None:
        self._require_started()
        normalized = self.normalize_path(path)
        self._raise_if_cancelled(signal)
        target = self._container_path(normalized)
        async with self._operation_lock:
            self._require_started()
            result = await self._run_exec_operation(
                (
                    "sh",
                    "-c",
                    _WRITE_FILE_SCRIPT,
                    "vedex-write",
                    self._resolved_workdir,
                    target,
                ),
                signal=signal,
                stdin=data,
            )
        if result.returncode != 0:
            self._raise_file_error(result, path=normalized, operation="write")
        self._raise_if_cancelled(signal)

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float,
        signal: CancellationToken | None = None,
    ) -> CommandResult:
        self._require_started()
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if timeout_seconds > self._limits.max_command_seconds:
            raise ValueError(
                f"timeout_seconds must not exceed {self._limits.max_command_seconds:g}"
            )
        if self._is_cancelled(signal):
            return CommandResult(command=command, cancelled=True)

        started_at = monotonic()
        async with self._operation_lock:
            self._require_started()
            arguments = self._exec_arguments((*self._config.interpreter, command))
            try:
                controlled = await self._run_controlled(
                    arguments,
                    timeout_seconds=timeout_seconds,
                    signal=signal,
                    merge_stderr=True,
                )
            except OSError as exc:
                return CommandResult(
                    command=command,
                    duration_seconds=monotonic() - started_at,
                    error=exc.strerror or type(exc).__name__,
                )
            except asyncio.CancelledError:
                await self._recover_after_interrupted_exec()
                raise

            if controlled.result is not None:
                return CommandResult(
                    command=command,
                    stdout=controlled.result.stdout,
                    exit_code=controlled.result.returncode,
                    duration_seconds=monotonic() - started_at,
                )

            if not self._shutdown_requested:
                await self._recover_after_interrupted_exec()
            return CommandResult(
                command=command,
                timed_out=controlled.timed_out,
                cancelled=controlled.cancelled,
                duration_seconds=monotonic() - started_at,
            )

    async def collect_patch(
        self,
        *,
        signal: CancellationToken | None = None,
    ) -> WorkspacePatch:
        self._require_started()
        self._raise_if_cancelled(signal)
        async with self._operation_lock:
            self._require_started()
            return await self._collect_patch_locked(signal)

    async def export_workspace(
        self,
        destination: Path,
        *,
        signal: CancellationToken | None = None,
    ) -> WorkspaceExport:
        self._require_started()
        self._raise_if_cancelled(signal)
        output_path = destination.expanduser().resolve(strict=False)
        source = (
            self._config.workspace.expanduser().resolve(strict=False)
            if self._config.workspace_mode == "mount" and self._config.workspace is not None
            else None
        )
        if (
            self._config.workspace_mode == "mount"
            and source is not None
            and (output_path == source or output_path.is_relative_to(source))
        ):
            raise EnvironmentExportError(
                "Workspace exports must be written outside a mounted workspace"
            )
        if output_path.exists() and output_path.is_dir():
            raise EnvironmentExportError("Workspace export destination must be a file")

        async with self._operation_lock:
            self._require_started()
            controlled = await self._run_controlled(
                (
                    "container",
                    "cp",
                    f"{self._container_name}:{_container_workspace_contents(self._config.workdir)}",
                    "-",
                ),
                timeout_seconds=self._config.operation_timeout_seconds,
                signal=signal,
            )
        if controlled.result is None:
            if controlled.cancelled:
                raise EnvironmentCancelledError("Workspace export cancelled")
            raise EnvironmentExportError("Container workspace export timed out")
        if controlled.result.returncode != 0:
            raise EnvironmentExportError(
                f"Could not copy the container workspace: {_cli_error_detail(controlled.result)}"
            )

        try:
            file_count = await asyncio.to_thread(
                _write_workspace_archive,
                controlled.result.stdout,
                output_path,
                signal,
            )
            size_bytes = output_path.stat().st_size
            digest = await asyncio.to_thread(_sha256_file, output_path)
        except EnvironmentCancelledError:
            raise
        except (OSError, tarfile.TarError) as exc:
            raise EnvironmentExportError(f"Could not export workspace: {exc}") from exc
        return WorkspaceExport(
            path=str(output_path),
            file_count=file_count,
            size_bytes=size_bytes,
            sha256=digest,
        )

    async def get_metadata(self) -> EnvironmentMetadata:
        self._require_started()
        async with self._operation_lock:
            self._require_started()
            git_revision: str | None = None
            git_dirty: bool | None = None
            revision = await self._exec_command(("git", "rev-parse", "HEAD"))
            if revision.returncode == 0:
                git_revision = revision.stdout.decode("ascii", errors="replace").strip()
                status = await self._exec_command(
                    ("git", "status", "--porcelain=v1", "--untracked-files=normal")
                )
                if status.returncode == 0:
                    git_dirty = bool(status.stdout.strip())

            release_result = await self._exec_command(("uname", "-r"))
            operating_system_release = (
                release_result.stdout.decode(errors="replace").strip()
                if release_result.returncode == 0
                else "unknown"
            )
            python_version = await self._python_version()

        details: dict[str, JSONValue] = {
            "engine": self._config.engine,
            "engine_version": self._engine_version,
            "container_name": self._container_name,
            "image_reference": self._image_reference,
            "resolved_image_reference": self._resolved_image_reference,
            "image_id": self._image_id,
            "resolved_image_digest": self._image_digest,
            "workspace_mode": self._config.workspace_mode,
            "workdir": self._config.workdir,
            "network": self._config.network or "default",
            "resource_limits": {
                "cpus": self._config.cpus,
                "memory": self._config.memory,
            },
            "environment_variable_names": [
                cast(JSONValue, name) for name in sorted(self._config.environment)
            ],
            "max_command_seconds": self._limits.max_command_seconds,
            "lifecycle_timeouts_seconds": {
                "startup": self._config.startup_timeout_seconds,
                "operation": self._config.operation_timeout_seconds,
                "cleanup": self._config.cleanup_timeout_seconds,
            },
        }
        return EnvironmentMetadata(
            environment_type="docker",
            workspace=self._workspace,
            operating_system=self._image_os,
            operating_system_release=operating_system_release,
            architecture=self._image_architecture,
            python_version=python_version,
            shell=self._config.interpreter[0],
            git_revision=git_revision,
            git_dirty=git_dirty,
            details=details,
        )

    def _resolved_source(self) -> Path | None:
        source = self._config.workspace
        if source is None:
            return None
        resolved = source.expanduser().resolve(strict=False)
        if not resolved.exists():
            raise EnvironmentStateError(f"Workspace does not exist: {resolved}")
        if not resolved.is_dir():
            raise EnvironmentStateError(f"Workspace is not a directory: {resolved}")
        return resolved

    def _create_arguments(self, source: Path | None) -> tuple[str, ...]:
        arguments = [
            "container",
            "create",
            "--name",
            self._container_name,
            "--init",
            "--workdir",
            self._config.workdir,
            "--pull",
            self._config.pull,
            "--label",
            "dev.vedex.managed=true",
        ]
        if self._config.network is not None:
            arguments.extend(("--network", self._config.network))
        if self._config.cpus is not None:
            arguments.extend(("--cpus", str(self._config.cpus)))
        if self._config.memory is not None:
            arguments.extend(("--memory", self._config.memory))
        for name, value in sorted(self._config.environment.items()):
            arguments.extend(("--env", f"{name}={value}"))
        if self._config.workspace_mode == "mount":
            if source is None:
                raise AssertionError("Mounted workspace source unexpectedly missing")
            arguments.extend(
                (
                    "--mount",
                    f"type=bind,source={source},target={self._config.workdir}",
                )
            )
        arguments.extend((self._config.image, *self._config.keepalive_command))
        return tuple(arguments)

    def _exec_arguments(
        self,
        command: Sequence[str],
        *,
        interactive: bool = False,
    ) -> tuple[str, ...]:
        arguments = ["container", "exec"]
        if interactive:
            arguments.append("--interactive")
        arguments.extend(
            (
                "--workdir",
                self._config.workdir,
                self._container_name,
                *command,
            )
        )
        return tuple(arguments)

    def _container_path(self, normalized_path: str) -> str:
        if normalized_path == ".":
            return self._resolved_workdir
        return f"{self._resolved_workdir.rstrip('/')}/{normalized_path}"

    def _require_started(self) -> None:
        if self._state != "started":
            raise EnvironmentStateError(
                f"Environment operation requires state 'started'; current state is '{self._state}'"
            )

    def _is_cancelled(self, signal: CancellationToken | None) -> bool:
        return self._shutdown_requested or (signal is not None and signal.is_cancelled())

    def _raise_if_cancelled(self, signal: CancellationToken | None) -> None:
        if self._is_cancelled(signal):
            raise EnvironmentCancelledError("Environment operation cancelled")

    async def _run_startup_cli(self, arguments: Sequence[str]) -> DockerCLIResult:
        try:
            return await self._cli.run(
                arguments,
                timeout_seconds=self._config.startup_timeout_seconds,
            )
        except TimeoutError as exc:
            raise EnvironmentStateError("Container engine startup command timed out") from exc
        except OSError as exc:
            detail = exc.strerror or type(exc).__name__
            raise EnvironmentStateError(
                f"Could not start container engine {self._config.engine!r}: {detail}"
            ) from exc

    async def _run_controlled(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float,
        signal: CancellationToken | None,
        stdin: bytes | None = None,
        merge_stderr: bool = False,
    ) -> _ControlledResult:
        command_task = asyncio.create_task(
            self._cli.run(
                arguments,
                stdin=stdin,
                merge_stderr=merge_stderr,
            )
        )
        shutdown_watch = asyncio.create_task(self._wait_for_shutdown())
        cancellation_watch: asyncio.Task[None] | None = None
        watchers: set[asyncio.Task[object]] = {
            cast(asyncio.Task[object], command_task),
            cast(asyncio.Task[object], shutdown_watch),
        }
        if signal is not None:
            cancellation_watch = asyncio.create_task(_wait_for_cancellation(signal))
            watchers.add(cast(asyncio.Task[object], cancellation_watch))

        try:
            done, _pending = await asyncio.wait(
                watchers,
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if command_task in done:
                return _ControlledResult(result=command_task.result())

            cancelled = shutdown_watch in done or (
                cancellation_watch is not None and cancellation_watch in done
            )
            command_task.cancel()
            await asyncio.gather(command_task, return_exceptions=True)
            return _ControlledResult(timed_out=not cancelled, cancelled=cancelled)
        except asyncio.CancelledError:
            command_task.cancel()
            await asyncio.gather(command_task, return_exceptions=True)
            raise
        finally:
            pending_watchers: list[asyncio.Task[object]] = []
            for watcher in (shutdown_watch, cancellation_watch):
                if watcher is not None and not watcher.done():
                    watcher.cancel()
                    pending_watchers.append(cast(asyncio.Task[object], watcher))
            if pending_watchers:
                await asyncio.gather(*pending_watchers, return_exceptions=True)

    async def _run_exec_operation(
        self,
        command: Sequence[str],
        *,
        signal: CancellationToken | None,
        stdin: bytes | None = None,
    ) -> DockerCLIResult:
        try:
            controlled = await self._run_controlled(
                self._exec_arguments(command, interactive=stdin is not None),
                timeout_seconds=self._config.operation_timeout_seconds,
                signal=signal,
                stdin=stdin,
            )
        except OSError as exc:
            raise EnvironmentStateError(
                f"Container engine command could not start: {_error_detail(exc)}"
            ) from exc
        except asyncio.CancelledError:
            await self._recover_after_interrupted_exec()
            raise
        if controlled.result is not None:
            return controlled.result
        if not self._shutdown_requested:
            await self._recover_after_interrupted_exec()
        if controlled.cancelled:
            raise EnvironmentCancelledError("Environment operation cancelled")
        raise EnvironmentStateError("Container environment operation timed out")

    async def _exec_command(
        self,
        command: Sequence[str],
        *,
        signal: CancellationToken | None = None,
        timeout_seconds: float | None = None,
    ) -> DockerCLIResult:
        timeout = timeout_seconds or self._config.operation_timeout_seconds
        try:
            controlled = await self._run_controlled(
                self._exec_arguments(command),
                timeout_seconds=timeout,
                signal=signal,
            )
        except OSError as exc:
            raise EnvironmentStateError(
                f"Container engine command could not start: {_error_detail(exc)}"
            ) from exc
        except asyncio.CancelledError:
            await self._recover_after_interrupted_exec()
            raise
        if controlled.result is not None:
            return controlled.result
        if not self._shutdown_requested:
            await self._recover_after_interrupted_exec()
        if controlled.cancelled:
            raise EnvironmentCancelledError("Environment operation cancelled")
        raise EnvironmentStateError("Container command timed out")

    async def _recover_after_interrupted_exec(self) -> None:
        if self._state != "started" or self._container_id is None:
            return
        try:
            kill_result = await self._cli.run(
                ("container", "kill", self._container_name),
                timeout_seconds=self._config.cleanup_timeout_seconds,
            )
            if kill_result.returncode != 0 and not _is_stopped_container(kill_result):
                raise EnvironmentStateError(
                    f"Could not stop interrupted container: {_cli_error_detail(kill_result)}"
                )
            start_result = await self._cli.run(
                ("container", "start", self._container_name),
                timeout_seconds=self._config.startup_timeout_seconds,
            )
            self._require_cli_success(start_result, "container recovery")
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            raise EnvironmentStateError(
                f"Container could not recover after interruption: {_error_detail(exc)}"
            ) from exc

    def _raise_file_error(
        self,
        result: DockerCLIResult,
        *,
        path: str,
        operation: Literal["read", "write"],
    ) -> None:
        if result.returncode == _FILE_OUTSIDE_WORKSPACE:
            raise WorkspacePathError(path, "path resolves outside the workspace")
        kinds: dict[int, EnvironmentFileErrorKind] = {
            _FILE_NOT_FOUND: "not_found",
            _FILE_IS_DIRECTORY: "is_directory",
            _FILE_PERMISSION_DENIED: "permission_denied",
            _FILE_IO_ERROR: "io_error",
        }
        kind = kinds.get(result.returncode)
        if kind is None:
            detail = _cli_error_detail(result)
            raise EnvironmentStateError(
                f"Container file helper failed with exit code {result.returncode}: {detail}"
            )
        detail = _cli_error_detail(result) if result.stderr else None
        raise EnvironmentFileError(
            path=path,
            operation=operation,
            kind=kind,
            detail=detail,
        )

    async def _collect_patch_locked(
        self,
        signal: CancellationToken | None,
    ) -> WorkspacePatch:
        inside = await self._exec_command(
            ("git", "rev-parse", "--is-inside-work-tree"),
            signal=signal,
            timeout_seconds=_GIT_TIMEOUT_SECONDS,
        )
        if inside.returncode != 0 or inside.stdout.strip() != b"true":
            return WorkspacePatch(text="", is_git_repository=False, changed=False)

        revision = await self._exec_command(
            ("git", "rev-parse", "HEAD"),
            signal=signal,
            timeout_seconds=_GIT_TIMEOUT_SECONDS,
        )
        base_revision = (
            revision.stdout.decode("ascii", errors="replace").strip()
            if revision.returncode == 0
            else None
        )
        patch_parts: list[bytes] = []
        if base_revision is not None:
            diff = await self._exec_command(
                ("git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", "."),
                signal=signal,
                timeout_seconds=_GIT_TIMEOUT_SECONDS,
            )
            if diff.returncode not in (0, 1):
                raise EnvironmentExportError(
                    "Git failed while collecting the container workspace patch"
                )
            if diff.stdout:
                patch_parts.append(diff.stdout)
        else:
            for command in (
                ("git", "diff", "--binary", "--no-ext-diff", "--cached", "--", "."),
                ("git", "diff", "--binary", "--no-ext-diff", "--", "."),
            ):
                diff = await self._exec_command(
                    command,
                    signal=signal,
                    timeout_seconds=_GIT_TIMEOUT_SECONDS,
                )
                if diff.returncode not in (0, 1):
                    raise EnvironmentExportError(
                        "Git failed while collecting the container workspace patch"
                    )
                if diff.stdout:
                    patch_parts.append(diff.stdout)

        untracked = await self._exec_command(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            signal=signal,
            timeout_seconds=_GIT_TIMEOUT_SECONDS,
        )
        if untracked.returncode != 0:
            raise EnvironmentExportError(
                "Git failed while listing untracked container workspace files"
            )
        for raw_path in untracked.stdout.split(b"\0"):
            if not raw_path:
                continue
            self._raise_if_cancelled(signal)
            relative_path = raw_path.decode("utf-8", errors="replace")
            diff = await self._exec_command(
                (
                    "git",
                    "diff",
                    "--binary",
                    "--no-index",
                    "--",
                    "/dev/null",
                    relative_path,
                ),
                signal=signal,
                timeout_seconds=_GIT_TIMEOUT_SECONDS,
            )
            if diff.returncode not in (0, 1):
                raise EnvironmentExportError(
                    f"Git failed while collecting untracked file: {relative_path}"
                )
            if diff.stdout:
                patch_parts.append(diff.stdout)

        patch_bytes = b"\n".join(part.rstrip(b"\n") for part in patch_parts)
        if patch_bytes:
            patch_bytes += b"\n"
        return WorkspacePatch(
            text=patch_bytes.decode("utf-8", errors="replace"),
            base_revision=base_revision,
            is_git_repository=True,
            changed=bool(patch_bytes),
        )

    async def _capture_engine_version(self) -> None:
        result = await self._run_startup_cli(("version", "--format", "{{json .}}"))
        self._require_cli_success(result, "engine version inspection")
        data = _json_object(result.stdout, "container engine version")
        server = data.get("Server")
        client = data.get("Client")
        version = None
        if isinstance(server, Mapping):
            version = server.get("Version")
        if not isinstance(version, str) and isinstance(client, Mapping):
            version = client.get("Version")
        if not isinstance(version, str) or not version:
            raise EnvironmentStateError(
                "Container engine version response did not contain a version"
            )
        self._engine_version = version

    async def _capture_image_metadata(self) -> None:
        container_result = await self._run_startup_cli(
            (
                "container",
                "inspect",
                "--format",
                "{{json .}}",
                self._container_name,
            )
        )
        self._require_cli_success(container_result, "container inspection")
        container_data = _json_object(container_result.stdout, "container inspection")
        image_id = container_data.get("Image")
        if not isinstance(image_id, str) or not image_id:
            raise EnvironmentStateError("Container inspection did not contain an image ID")

        image_result = await self._run_startup_cli(
            ("image", "inspect", "--format", "{{json .}}", image_id)
        )
        self._require_cli_success(image_result, "image inspection")
        image_data = _json_object(image_result.stdout, "image inspection")
        architecture = image_data.get("Architecture")
        image_os = image_data.get("Os")
        if not isinstance(architecture, str) or not architecture:
            raise EnvironmentStateError("Image inspection did not contain an architecture")
        variant = image_data.get("Variant")
        self._image_architecture = (
            f"{architecture}/{variant}" if isinstance(variant, str) and variant else architecture
        )
        self._image_os = image_os if isinstance(image_os, str) and image_os else "unknown"
        self._image_id = image_id

        repo_digests = image_data.get("RepoDigests")
        resolved_reference: str | None = None
        if isinstance(repo_digests, list):
            resolved_reference = next(
                (item for item in repo_digests if isinstance(item, str) and "@" in item),
                None,
            )
        if resolved_reference is not None:
            self._resolved_image_reference = resolved_reference
            self._image_digest = resolved_reference.rpartition("@")[2]
        else:
            self._resolved_image_reference = image_id
            self._image_digest = image_id

    async def _python_version(self) -> str:
        for executable in ("python3", "python"):
            result = await self._exec_command((executable, "--version"))
            if result.returncode == 0:
                output = result.stdout or result.stderr
                return output.decode(errors="replace").strip() or "unknown"
        return "unavailable"

    async def _cleanup_after_failed_start(self) -> BaseException | None:
        if not self._creation_attempted:
            return None
        try:
            result = await self._cli.run(
                ("container", "rm", "--force", self._container_name),
                timeout_seconds=self._config.cleanup_timeout_seconds,
            )
            if result.returncode != 0 and not _is_missing_container(result):
                return EnvironmentStateError(_cli_error_detail(result))
        except BaseException as exc:
            return exc
        self._container_id = None
        self._creation_attempted = False
        return None

    async def _remove_container(self) -> None:
        async with self._operation_lock:
            result = await self._cli.run(
                ("container", "rm", "--force", self._container_name),
                timeout_seconds=self._config.cleanup_timeout_seconds,
            )
            if result.returncode != 0 and not _is_missing_container(result):
                raise EnvironmentStateError(_cli_error_detail(result))
            self._container_id = None
            self._creation_attempted = False

    async def _wait_for_shutdown(self) -> None:
        while not self._shutdown_requested:
            await asyncio.sleep(0.05)

    @staticmethod
    def _require_cli_success(result: DockerCLIResult, action: str) -> None:
        if result.returncode != 0:
            raise EnvironmentStateError(f"Container {action} failed: {_cli_error_detail(result)}")


async def _wait_for_cancellation(signal: CancellationToken) -> None:
    while not signal.is_cancelled():
        await asyncio.sleep(0.05)


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    with contextlib.suppress(TimeoutError, ProcessLookupError):
        await asyncio.wait_for(process.wait(), timeout=5.0)


def _json_object(raw: bytes, description: str) -> dict[str, JSONValue]:
    try:
        return _JSON_OBJECT_ADAPTER.validate_python(_json_loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise EnvironmentStateError(f"Invalid JSON returned by {description}") from exc


def _json_loads(raw: bytes) -> object:
    return cast(object, json.loads(raw))


def _cli_error_detail(result: DockerCLIResult) -> str:
    raw = result.stderr.strip() or result.stdout.strip()
    return raw.decode(errors="replace") or f"exit code {result.returncode}"


def _error_detail(error: BaseException) -> str:
    return str(error) or type(error).__name__


def _is_missing_container(result: DockerCLIResult) -> bool:
    detail = _cli_error_detail(result).casefold()
    return "no such container" in detail or "container does not exist" in detail


def _is_stopped_container(result: DockerCLIResult) -> bool:
    detail = _cli_error_detail(result).casefold()
    return "is not running" in detail or "already stopped" in detail


def _container_workspace_contents(workdir: str) -> str:
    return f"{workdir.rstrip('/')}/." if workdir != "/" else "/."


def _write_workspace_archive(
    source_bytes: bytes,
    destination: Path,
    signal: CancellationToken | None,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    file_count = 0
    seen: set[str] = set()
    try:
        with (
            tarfile.open(fileobj=io.BytesIO(source_bytes), mode="r:*") as source,
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file,
        ):
            temporary_path = Path(temporary_file.name)
            with (
                gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=temporary_file,
                    mtime=0,
                ) as compressed,
                tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as output,
            ):
                members: list[tuple[str, tarfile.TarInfo]] = []
                for member in source.getmembers():
                    normalized = _safe_archive_name(member.name)
                    if normalized == "." or _is_git_archive_path(normalized):
                        continue
                    if normalized in seen:
                        raise EnvironmentExportError(
                            f"Container workspace archive contains duplicate path: {normalized}"
                        )
                    seen.add(normalized)
                    _validate_archive_member(member, normalized)
                    members.append((normalized, member))

                for normalized, member in sorted(members, key=lambda item: item[0]):
                    if signal is not None and signal.is_cancelled():
                        raise EnvironmentCancelledError("Workspace export cancelled")
                    info = copy.copy(member)
                    info.name = normalized
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.pax_headers = {}
                    if member.isfile():
                        source_file = source.extractfile(member)
                        if source_file is None:
                            raise EnvironmentExportError(
                                f"Container archive file has no data: {normalized}"
                            )
                        with source_file:
                            output.addfile(info, source_file)
                        file_count += 1
                    else:
                        output.addfile(info)
                        if member.issym() or member.islnk():
                            file_count += 1
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
        return file_count
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _safe_archive_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\x00" in name:
        raise EnvironmentExportError(f"Container workspace archive contains unsafe path: {name}")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    return "/".join(parts) or "."


def _is_git_archive_path(path: str) -> bool:
    return path == ".git" or path.startswith(".git/")


def _validate_archive_member(member: tarfile.TarInfo, normalized: str) -> None:
    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
        raise EnvironmentExportError(
            f"Container workspace archive contains unsupported entry: {normalized}"
        )
    if member.issym():
        target = PurePosixPath(normalized).parent / member.linkname
        safe_target = _safe_archive_name(target.as_posix())
        if _is_git_archive_path(safe_target):
            raise EnvironmentExportError(
                f"Container workspace symlink targets excluded data: {normalized}"
            )
    elif member.islnk():
        safe_target = _safe_archive_name(member.linkname)
        if _is_git_archive_path(safe_target):
            raise EnvironmentExportError(
                f"Container workspace hard link targets excluded data: {normalized}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DockerEnvironment",
    "DockerEnvironmentConfig",
    "DockerPullPolicy",
    "DockerWorkspaceMode",
]
