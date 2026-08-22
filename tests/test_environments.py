from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from pydantic import ValidationError
from vedex.environments import (
    Environment,
    EnvironmentCancelledError,
    EnvironmentExportError,
    EnvironmentFileError,
    EnvironmentLimits,
    EnvironmentStateError,
    LocalEnvironment,
    WorkspacePathError,
    normalize_workspace_path,
)

from .conftest import run_async


class _CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


def _python_script_command(script_name: str) -> str:
    return f'"{sys.executable}" "{script_name}"'


def _run_git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_environment_contract_lifecycle_identity_and_tools(tmp_path: Path) -> None:
    environment = LocalEnvironment(tmp_path)
    second = LocalEnvironment(tmp_path)
    contract: Environment = environment

    assert isinstance(environment, Environment)
    assert environment.state == "created"
    assert environment.workspace == second.workspace
    assert environment.workspace.model_root == "."
    assert [tool.name for tool in environment.tools] == ["read", "write", "edit", "bash"]
    with pytest.raises(EnvironmentStateError, match="started"):
        run_async(contract.read_bytes("file.txt"))

    async def lifecycle() -> None:
        await environment.start()
        await environment.start()
        assert str(environment.state) == "started"
        await environment.stop()
        await environment.stop()
        assert str(environment.state) == "stopped"
        await environment.start()
        assert str(environment.state) == "started"
        await environment.stop()

    run_async(lifecycle())


def test_environment_start_rejects_missing_or_non_directory_workspace(tmp_path: Path) -> None:
    missing = LocalEnvironment(tmp_path / "missing")
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a directory", encoding="utf-8")
    file_environment = LocalEnvironment(file_path)

    with pytest.raises(EnvironmentStateError, match="does not exist"):
        run_async(missing.start())
    with pytest.raises(EnvironmentStateError, match="not a directory"):
        run_async(file_environment.start())


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("file.txt", "file.txt"),
        ("nested/./file.txt", "nested/file.txt"),
        ("nested/../file.txt", "file.txt"),
        (r"nested\file.txt", "nested/file.txt"),
        (".", "."),
    ],
)
def test_workspace_path_normalization(path: str, expected: str) -> None:
    assert normalize_workspace_path(path) == expected


@pytest.mark.parametrize(
    "path",
    ["", "../outside", "/absolute", r"C:\absolute", r"\\server\share", "a/../../b"],
)
def test_workspace_path_normalization_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(WorkspacePathError):
        normalize_workspace_path(path)


def test_local_environment_reads_and_atomically_writes_exact_bytes(tmp_path: Path) -> None:
    environment = LocalEnvironment(tmp_path)

    async def exercise() -> None:
        await environment.start()
        try:
            await environment.write_bytes("nested/data.bin", b"\xff\x00first")
            assert await environment.read_bytes("nested/data.bin") == b"\xff\x00first"
            await environment.write_bytes("nested/data.bin", b"replacement")
            assert await environment.read_bytes("nested/data.bin") == b"replacement"

            with pytest.raises(EnvironmentFileError) as missing:
                await environment.read_bytes("missing.txt")
            assert missing.value.kind == "not_found"

            (tmp_path / "directory").mkdir()
            with pytest.raises(EnvironmentFileError) as directory:
                await environment.read_bytes("directory")
            assert directory.value.kind == "is_directory"
            with pytest.raises(EnvironmentFileError) as write_directory:
                await environment.write_bytes("directory", b"no")
            assert write_directory.value.kind == "is_directory"
        finally:
            await environment.stop()

    run_async(exercise())
    assert not list((tmp_path / "nested").glob(".data.bin.*.tmp"))


def test_atomic_write_failure_preserves_original_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"original")
    environment = LocalEnvironment(tmp_path)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("vedex.environments.local.os.replace", fail_replace)

    async def exercise() -> None:
        await environment.start()
        try:
            with pytest.raises(EnvironmentFileError, match="replace failed"):
                await environment.write_bytes("file.txt", b"new")
        finally:
            await environment.stop()

    run_async(exercise())
    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob(".file.txt.*.tmp"))


def test_local_environment_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")

    environment = LocalEnvironment(workspace)

    async def exercise() -> None:
        await environment.start()
        try:
            with pytest.raises(WorkspacePathError, match="outside"):
                await environment.read_bytes("link.txt")
        finally:
            await environment.stop()

    run_async(exercise())


def test_local_commands_capture_output_status_timeout_and_cancellation(tmp_path: Path) -> None:
    (tmp_path / "success.py").write_text(
        "import os, sys\nprint(os.getcwd())\nprint('stderr', file=sys.stderr)\n",
        encoding="utf-8",
    )
    (tmp_path / "failure.py").write_text(
        "print('failed')\nraise SystemExit(7)\n",
        encoding="utf-8",
    )
    (tmp_path / "sleep.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    environment = LocalEnvironment(tmp_path)
    token = _CancellationToken()

    async def exercise() -> None:
        await environment.start()
        try:
            success = await environment.run_command(
                _python_script_command("success.py"),
                timeout_seconds=5,
            )
            failure = await environment.run_command(
                _python_script_command("failure.py"),
                timeout_seconds=5,
            )
            timed_out = await environment.run_command(
                _python_script_command("sleep.py"),
                timeout_seconds=0.1,
            )

            cancellation_task = asyncio.create_task(
                environment.run_command(
                    _python_script_command("sleep.py"),
                    timeout_seconds=5,
                    signal=token,
                )
            )
            await asyncio.sleep(0.1)
            token.cancel()
            cancelled = await cancellation_task

            assert success.exit_code == 0
            assert str(tmp_path).encode() in success.stdout
            assert b"stderr" in success.stdout
            assert failure.exit_code == 7
            assert b"failed" in failure.stdout
            assert timed_out.timed_out is True
            assert timed_out.cancelled is False
            assert cancelled.cancelled is True
            assert cancelled.timed_out is False
        finally:
            await environment.stop()

    run_async(exercise())


def test_stopping_environment_cancels_active_command(tmp_path: Path) -> None:
    (tmp_path / "sleep.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    environment = LocalEnvironment(tmp_path)

    async def exercise() -> None:
        await environment.start()
        command_task = asyncio.create_task(
            environment.run_command(
                _python_script_command("sleep.py"),
                timeout_seconds=5,
            )
        )
        await asyncio.sleep(0.1)
        await environment.stop()
        result = await command_task
        assert result.cancelled is True
        assert environment.state == "stopped"

    run_async(exercise())


def test_command_timeout_terminates_descendant_processes(tmp_path: Path) -> None:
    marker = tmp_path / "child-finished.txt"
    (tmp_path / "child.py").write_text(
        (
            "import pathlib, time\n"
            "time.sleep(1)\n"
            "pathlib.Path('child-finished.txt').write_text('alive')\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "parent.py").write_text(
        (
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, 'child.py'])\n"
            "time.sleep(30)\n"
        ),
        encoding="utf-8",
    )
    environment = LocalEnvironment(tmp_path)

    async def exercise() -> None:
        await environment.start()
        try:
            result = await environment.run_command(
                _python_script_command("parent.py"),
                timeout_seconds=0.2,
            )
            assert result.timed_out is True
            await asyncio.sleep(1.2)
            assert not marker.exists()
        finally:
            await environment.stop()

    run_async(exercise())


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_patch_collection_includes_tracked_and_untracked_changes(tmp_path: Path) -> None:
    (tmp_path / "tracked.txt").write_text("original\n", encoding="utf-8")
    _run_git(tmp_path, "init", "--quiet")
    _run_git(tmp_path, "add", "tracked.txt")
    _run_git(
        tmp_path,
        "-c",
        "user.name=Vedex Tests",
        "-c",
        "user.email=vedex@example.test",
        "commit",
        "--quiet",
        "-m",
        "initial",
    )
    (tmp_path / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("new file\n", encoding="utf-8")
    environment = LocalEnvironment(tmp_path)

    async def exercise() -> None:
        await environment.start()
        try:
            patch = await environment.collect_patch()
            metadata = await environment.get_metadata()
            assert patch.is_git_repository is True
            assert patch.changed is True
            assert patch.base_revision is not None
            assert len(patch.base_revision) == 40
            assert "tracked.txt" in patch.text
            assert "untracked.txt" in patch.text
            assert "+changed" in patch.text
            assert "+new file" in patch.text
            assert metadata.environment_type == "local"
            assert metadata.git_revision == patch.base_revision
            assert metadata.git_dirty is True
            assert metadata.details["max_command_seconds"] == 600.0
        finally:
            await environment.stop()

    run_async(exercise())


def test_patch_collection_reports_non_git_workspace(tmp_path: Path) -> None:
    environment = LocalEnvironment(tmp_path)

    async def exercise() -> None:
        await environment.start()
        try:
            patch = await environment.collect_patch()
            assert patch.is_git_repository is False
            assert patch.changed is False
            assert patch.text == ""
        finally:
            await environment.stop()

    run_async(exercise())


def test_workspace_export_is_deterministic_and_excludes_git_data(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    (workspace / "nested").mkdir(parents=True)
    (workspace / "empty").mkdir()
    (workspace / ".git").mkdir()
    (workspace / "nested" / "file.txt").write_text("contents", encoding="utf-8")
    (workspace / ".git" / "secret").write_text("omit", encoding="utf-8")
    environment = LocalEnvironment(workspace)
    first_path = output / "first.tar.gz"
    second_path = output / "second.tar.gz"

    async def exercise() -> None:
        await environment.start()
        try:
            first = await environment.export_workspace(first_path)
            second = await environment.export_workspace(second_path)
            assert first.file_count == 1
            assert first.sha256 == second.sha256
            assert first.size_bytes == second.size_bytes
            with pytest.raises(EnvironmentExportError, match="outside"):
                await environment.export_workspace(workspace / "inside.tar.gz")
        finally:
            await environment.stop()

    run_async(exercise())
    assert first_path.read_bytes() == second_path.read_bytes()
    with tarfile.open(first_path, "r:gz") as archive:
        names = archive.getnames()
        assert "nested/file.txt" in names
        assert "empty" in names
        assert not any(name == ".git" or name.startswith(".git/") for name in names)
        extracted = archive.extractfile("nested/file.txt")
        assert extracted is not None
        assert extracted.read() == b"contents"


def test_environment_cancellation_and_limit_contracts(tmp_path: Path) -> None:
    token = _CancellationToken()
    token.cancel()
    environment = LocalEnvironment(tmp_path)

    async def exercise() -> None:
        await environment.start()
        try:
            with pytest.raises(EnvironmentCancelledError):
                await environment.read_bytes("anything", signal=token)
            result = await environment.run_command(
                "ignored",
                timeout_seconds=1,
                signal=token,
            )
            assert result.cancelled is True
        finally:
            await environment.stop()

    run_async(exercise())
    with pytest.raises(ValidationError):
        EnvironmentLimits(max_command_seconds=0)
