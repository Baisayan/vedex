from __future__ import annotations

import asyncio
import contextlib
import gzip
import hashlib
import os
import platform
import shutil
import signal as process_signal
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Self

from ..schema import AgentTool, CancellationToken, JSONValue
from .base import (
    CommandResult,
    EnvironmentCancelledError,
    EnvironmentExportError,
    EnvironmentFileError,
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

_GIT_TIMEOUT_SECONDS = 30.0
_PROCESS_STOP_TIMEOUT_SECONDS = 5.0


class LocalEnvironment:
    """Execute workspace operations directly on the local host."""

    def __init__(
        self,
        root: str | Path,
        *,
        limits: EnvironmentLimits | None = None,
    ) -> None:
        self._root = Path(root).expanduser().resolve(strict=False)
        self._limits = limits or EnvironmentLimits()
        self._state: EnvironmentState = "created"
        self._shutdown_requested = False
        self._processes: set[asyncio.subprocess.Process] = set()
        self._tools: tuple[AgentTool, ...] | None = None
        digest_input = str(self._root).casefold() if os.name == "nt" else str(self._root)
        workspace_id = hashlib.sha256(digest_input.encode()).hexdigest()[:16]
        self._workspace = WorkspaceIdentity(
            id=f"local-{workspace_id}",
            name=self._root.name or self._root.anchor,
        )

    @property
    def state(self) -> EnvironmentState:
        return self._state

    @property
    def root(self) -> Path:
        """Host path for application code and diagnostics, never for model tool input."""

        return self._root

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
        if self._state == "started":
            return
        if self._state == "stopping":
            raise EnvironmentStateError("Cannot start the environment while it is stopping")
        if not self._root.exists():
            raise EnvironmentStateError(f"Workspace does not exist: {self._root}")
        if not self._root.is_dir():
            raise EnvironmentStateError(f"Workspace is not a directory: {self._root}")

        self._root = self._root.resolve()
        self._shutdown_requested = False
        self._state = "started"

    async def stop(self) -> None:
        if self._state == "stopping":
            while self._state == "stopping":
                await asyncio.sleep(0)
            return
        if self._state != "started":
            self._state = "stopped"
            return

        self._state = "stopping"
        self._shutdown_requested = True
        cleanup_errors: list[BaseException] = []
        try:
            cleanup_results = await asyncio.gather(
                *(self._terminate_process_tree(process) for process in tuple(self._processes)),
                return_exceptions=True,
            )
            cleanup_errors = [
                result for result in cleanup_results if isinstance(result, BaseException)
            ]
        finally:
            self._state = "stopped"
        if cleanup_errors:
            raise EnvironmentStateError(
                f"Environment cleanup failed: {cleanup_errors[0]}"
            ) from cleanup_errors[0]

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
        target = self._host_path(normalized)
        try:
            data = await asyncio.to_thread(_read_bytes, target)
        except FileNotFoundError as exc:
            raise EnvironmentFileError(
                path=normalized,
                operation="read",
                kind="not_found",
            ) from exc
        except IsADirectoryError as exc:
            raise EnvironmentFileError(
                path=normalized,
                operation="read",
                kind="is_directory",
            ) from exc
        except PermissionError as exc:
            raise EnvironmentFileError(
                path=normalized,
                operation="read",
                kind="permission_denied",
                detail=_os_error_detail(exc),
            ) from exc
        except OSError as exc:
            raise EnvironmentFileError(
                path=normalized,
                operation="read",
                kind="io_error",
                detail=_os_error_detail(exc),
            ) from exc
        self._raise_if_cancelled(signal)
        return data

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
        target = self._host_path(normalized)
        try:
            await asyncio.to_thread(_atomic_write, target, data)
        except IsADirectoryError as exc:
            raise EnvironmentFileError(
                path=normalized,
                operation="write",
                kind="is_directory",
            ) from exc
        except PermissionError as exc:
            raise EnvironmentFileError(
                path=normalized,
                operation="write",
                kind="permission_denied",
                detail=_os_error_detail(exc),
            ) from exc
        except OSError as exc:
            raise EnvironmentFileError(
                path=normalized,
                operation="write",
                kind="io_error",
                detail=_os_error_detail(exc),
            ) from exc

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
        try:
            process = await self._start_process(command)
        except OSError as exc:
            return CommandResult(
                command=command,
                duration_seconds=monotonic() - started_at,
                error=exc.strerror or type(exc).__name__,
            )

        self._processes.add(process)
        try:
            stdout, timed_out, cancelled = await self._communicate(
                process,
                timeout_seconds=timeout_seconds,
                signal=signal,
            )
            return CommandResult(
                command=command,
                stdout=stdout,
                exit_code=process.returncode,
                timed_out=timed_out,
                cancelled=cancelled,
                duration_seconds=monotonic() - started_at,
            )
        finally:
            self._processes.discard(process)

    async def collect_patch(
        self,
        *,
        signal: CancellationToken | None = None,
    ) -> WorkspacePatch:
        self._require_started()
        self._raise_if_cancelled(signal)
        if shutil.which("git") is None:
            return WorkspacePatch(
                text="",
                is_git_repository=False,
                changed=False,
            )

        inside_code, inside_output = await self._run_git("rev-parse", "--is-inside-work-tree")
        if inside_code != 0 or inside_output.strip() != b"true":
            return WorkspacePatch(
                text="",
                is_git_repository=False,
                changed=False,
            )

        revision_code, revision_output = await self._run_git("rev-parse", "HEAD")
        base_revision = (
            revision_output.decode("ascii", errors="replace").strip()
            if revision_code == 0
            else None
        )
        patch_parts: list[bytes] = []
        if base_revision is not None:
            diff_code, diff_output = await self._run_git(
                "diff",
                "--binary",
                "--no-ext-diff",
                "HEAD",
                "--",
                ".",
            )
            if diff_code not in (0, 1):
                raise EnvironmentExportError("Git failed while collecting the workspace patch")
            if diff_output:
                patch_parts.append(diff_output)
        else:
            for arguments in (
                ("diff", "--binary", "--no-ext-diff", "--cached", "--", "."),
                ("diff", "--binary", "--no-ext-diff", "--", "."),
            ):
                diff_code, diff_output = await self._run_git(*arguments)
                if diff_code not in (0, 1):
                    raise EnvironmentExportError("Git failed while collecting the workspace patch")
                if diff_output:
                    patch_parts.append(diff_output)

        untracked_code, untracked_output = await self._run_git(
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
        if untracked_code != 0:
            raise EnvironmentExportError("Git failed while listing untracked workspace files")
        for raw_path in untracked_output.split(b"\0"):
            if not raw_path:
                continue
            self._raise_if_cancelled(signal)
            relative_path = raw_path.decode("utf-8", errors="surrogateescape")
            diff_code, diff_output = await self._run_git(
                "diff",
                "--binary",
                "--no-index",
                "--",
                "/dev/null",
                relative_path,
            )
            if diff_code not in (0, 1):
                raise EnvironmentExportError(
                    f"Git failed while collecting untracked file: {relative_path}"
                )
            if diff_output:
                patch_parts.append(diff_output)

        patch_bytes = b"\n".join(part.rstrip(b"\n") for part in patch_parts)
        if patch_bytes:
            patch_bytes += b"\n"
        return WorkspacePatch(
            text=patch_bytes.decode("utf-8", errors="replace"),
            base_revision=base_revision,
            is_git_repository=True,
            changed=bool(patch_bytes),
        )

    async def export_workspace(
        self,
        destination: Path,
        *,
        signal: CancellationToken | None = None,
    ) -> WorkspaceExport:
        self._require_started()
        self._raise_if_cancelled(signal)
        output_path = destination.expanduser().resolve(strict=False)
        if output_path == self._root or output_path.is_relative_to(self._root):
            raise EnvironmentExportError("Workspace exports must be written outside the workspace")
        if output_path.exists() and output_path.is_dir():
            raise EnvironmentExportError("Workspace export destination must be a file")

        try:
            file_count = await asyncio.to_thread(
                _export_tar_gz,
                self._root,
                output_path,
                signal,
            )
        except EnvironmentCancelledError:
            raise
        except (OSError, tarfile.TarError) as exc:
            raise EnvironmentExportError(f"Could not export workspace: {exc}") from exc

        archive_bytes = await asyncio.to_thread(output_path.read_bytes)
        return WorkspaceExport(
            path=str(output_path),
            file_count=file_count,
            size_bytes=len(archive_bytes),
            sha256=hashlib.sha256(archive_bytes).hexdigest(),
        )

    async def get_metadata(self) -> EnvironmentMetadata:
        self._require_started()
        git_revision: str | None = None
        git_dirty: bool | None = None
        if shutil.which("git") is not None:
            revision_code, revision_output = await self._run_git("rev-parse", "HEAD")
            if revision_code == 0:
                git_revision = revision_output.decode("ascii", errors="replace").strip()
                status_code, status_output = await self._run_git(
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=normal",
                )
                if status_code == 0:
                    git_dirty = bool(status_output.strip())

        shell = os.environ.get("COMSPEC" if os.name == "nt" else "SHELL")
        details: dict[str, JSONValue] = {
            "os_name": os.name,
            "cpu_count": os.cpu_count(),
            "max_command_seconds": self._limits.max_command_seconds,
        }
        return EnvironmentMetadata(
            environment_type="local",
            workspace=self._workspace,
            operating_system=platform.system(),
            operating_system_release=platform.release(),
            architecture=platform.machine(),
            python_version=platform.python_version(),
            shell=Path(shell).name if shell else ("cmd.exe" if os.name == "nt" else "sh"),
            git_revision=git_revision,
            git_dirty=git_dirty,
            details=details,
        )

    def _require_started(self) -> None:
        if self._state != "started":
            raise EnvironmentStateError(
                f"Environment operation requires state 'started'; current state is '{self._state}'"
            )

    def _host_path(self, normalized_path: str) -> Path:
        relative = PurePosixPath(normalized_path)
        candidate = self._root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as exc:
            raise WorkspacePathError(normalized_path, "path could not be resolved") from exc
        if not resolved.is_relative_to(self._root):
            raise WorkspacePathError(normalized_path, "path resolves outside the workspace")
        return resolved

    def _is_cancelled(self, signal: CancellationToken | None) -> bool:
        return self._shutdown_requested or (signal is not None and signal.is_cancelled())

    def _raise_if_cancelled(self, signal: CancellationToken | None) -> None:
        if self._is_cancelled(signal):
            raise EnvironmentCancelledError("Environment operation cancelled")

    async def _start_process(self, command: str) -> asyncio.subprocess.Process:
        if os.name == "posix":
            return await asyncio.create_subprocess_shell(
                command,
                cwd=self._root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        return await asyncio.create_subprocess_shell(
            command,
            cwd=self._root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    async def _communicate(
        self,
        process: asyncio.subprocess.Process,
        *,
        timeout_seconds: float,
        signal: CancellationToken | None,
    ) -> tuple[bytes, bool, bool]:
        communicate = asyncio.create_task(process.communicate())
        shutdown_watch = asyncio.create_task(self._wait_for_shutdown())
        cancellation_watch: asyncio.Task[None] | None = None
        watchers: set[asyncio.Task[object]] = {communicate, shutdown_watch}
        if signal is not None:
            cancellation_watch = asyncio.create_task(_wait_for_cancellation(signal))
            watchers.add(cancellation_watch)

        try:
            done, _pending = await asyncio.wait(
                watchers,
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if communicate in done:
                stdout, _stderr = communicate.result()
                return stdout or b"", False, self._is_cancelled(signal)

            cancelled = shutdown_watch in done or (
                cancellation_watch is not None and cancellation_watch in done
            )
            await self._terminate_process_tree(process)
            stdout, _stderr = await communicate
            return stdout or b"", not cancelled, cancelled
        except asyncio.CancelledError:
            await self._terminate_process_tree(process)
            with contextlib.suppress(Exception):
                await communicate
            raise
        finally:
            cancelled_watchers: list[asyncio.Task[object]] = []
            for watcher in (shutdown_watch, cancellation_watch):
                if watcher is not None and not watcher.done():
                    watcher.cancel()
                    cancelled_watchers.append(watcher)
            if cancelled_watchers:
                await asyncio.gather(*cancelled_watchers, return_exceptions=True)

    async def _wait_for_shutdown(self) -> None:
        while not self._shutdown_requested:
            await asyncio.sleep(0.05)

    async def _terminate_process_tree(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return

        if os.name == "posix":
            kill_process_group = getattr(os, "killpg", None)
            kill_signal = getattr(process_signal, "SIGKILL", None)
            if kill_process_group is None or kill_signal is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            else:
                try:
                    kill_process_group(process.pid, kill_signal)
                except ProcessLookupError:
                    return
                except OSError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
        else:
            try:
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await killer.communicate()
                if killer.returncode != 0:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
            except OSError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()

        with contextlib.suppress(TimeoutError, ProcessLookupError):
            await asyncio.wait_for(process.wait(), timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
        if process.returncode is None:
            raise EnvironmentStateError(
                f"Failed to terminate process tree for process {process.pid}"
            )

    async def _run_git(self, *arguments: str) -> tuple[int, bytes]:
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *arguments,
                cwd=self._root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise EnvironmentExportError("Git could not be started") from exc
        try:
            stdout, _stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=_GIT_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise EnvironmentExportError("Git timed out while inspecting the workspace") from exc
        return process.returncode or 0, stdout or b""


def _atomic_write(path: Path, data: bytes) -> None:
    if path.exists() and path.is_dir():
        raise IsADirectoryError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        if existing_mode is not None:
            temporary_path.chmod(existing_mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _read_bytes(path: Path) -> bytes:
    if path.is_dir():
        raise IsADirectoryError(path)
    return path.read_bytes()


def _os_error_detail(error: OSError) -> str:
    return error.strerror or str(error) or type(error).__name__


async def _wait_for_cancellation(signal: CancellationToken) -> None:
    while not signal.is_cancelled():
        await asyncio.sleep(0.05)


def _export_tar_gz(
    root: Path,
    destination: Path,
    signal: CancellationToken | None,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    file_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
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
                ) as archive,
            ):
                for path in _workspace_entries(root):
                    if signal is not None and signal.is_cancelled():
                        raise EnvironmentCancelledError("Workspace export cancelled")
                    if path.is_symlink() and not path.resolve(strict=False).is_relative_to(root):
                        outside = path.relative_to(root).as_posix()
                        raise EnvironmentExportError(
                            f"Workspace contains a symlink outside its root: {outside}"
                        )
                    relative = path.relative_to(root).as_posix()
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.pax_headers = {}
                    if path.is_file() and not path.is_symlink():
                        with path.open("rb") as source:
                            archive.addfile(info, source)
                        file_count += 1
                    else:
                        archive.addfile(info)
                        if path.is_symlink():
                            file_count += 1
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
        return file_count
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _workspace_entries(root: Path) -> Iterable[Path]:
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = sorted(name for name in directory_names if name != ".git")
        current = Path(current_root)
        for directory_name in directory_names:
            yield current / directory_name
        for file_name in sorted(name for name in file_names if name != ".git"):
            yield current / file_name


__all__ = ["LocalEnvironment"]
