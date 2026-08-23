from __future__ import annotations

import asyncio
import io
import json
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError
from vedex.environments import (
    DockerEnvironment,
    DockerEnvironmentConfig,
    DockerWorkspaceMode,
    Environment,
    EnvironmentCancelledError,
    EnvironmentExportError,
    EnvironmentFileError,
    EnvironmentLimits,
    EnvironmentStateError,
    LocalEnvironment,
    WorkspacePathError,
)
from vedex.environments.docker import DockerCLIResult
from vedex.headless import HeadlessExitCode, run_headless
from vedex.models import (
    FakeAdapter,
    ModelCompletedEvent,
    ModelSettings,
    ModelStartEvent,
)
from vedex.schema import AssistantMessage, ToolCall

from .conftest import run_async


@dataclass(frozen=True, slots=True)
class _DockerCall:
    arguments: tuple[str, ...]
    stdin: bytes | None
    timeout_seconds: float | None
    merge_stderr: bool


class _CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


class _FakeDockerCLI:
    def __init__(self, *, files: dict[str, bytes] | None = None) -> None:
        self.calls: list[_DockerCall] = []
        self.files = dict(files or {})
        self.directories: set[str] = set()
        self.symlinks: set[str] = set()
        self.container_name: str | None = None
        self.container_exists = False
        self.running = False
        self.fail_start = False
        self.fail_remove = False
        self.block_remove = False
        self.remove_started = asyncio.Event()
        self.remove_release = asyncio.Event()
        self.fail_image_inspection = False
        self.unsafe_archive = False
        self.user_result = DockerCLIResult(returncode=0, stdout=b"command output\n")
        self.block_user_command = False
        self.command_started = asyncio.Event()
        self.command_cancelled = False
        self.git_repository = True
        self.git_revision = "a" * 40
        self.tracked_diff = b"diff --git a/tracked.txt b/tracked.txt\n+changed\n"
        self.untracked_diffs: dict[str, bytes] = {
            "new.txt": b"diff --git a/new.txt b/new.txt\n+new\n"
        }

    async def run(
        self,
        arguments: Sequence[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: float | None = None,
        merge_stderr: bool = False,
    ) -> DockerCLIResult:
        args = tuple(arguments)
        self.calls.append(
            _DockerCall(
                arguments=args,
                stdin=stdin,
                timeout_seconds=timeout_seconds,
                merge_stderr=merge_stderr,
            )
        )

        if args[:2] == ("version", "--format"):
            return self._json_result(
                {"Client": {"Version": "28.4.0"}, "Server": {"Version": "28.3.3"}}
            )
        if args[:2] == ("container", "create"):
            self.container_name = args[args.index("--name") + 1]
            self.container_exists = True
            return DockerCLIResult(returncode=0, stdout=b"container-id\n")
        if args[:2] == ("container", "start"):
            if self.fail_start:
                return DockerCLIResult(returncode=1, stderr=b"start failed")
            self.running = True
            return DockerCLIResult(returncode=0, stdout=b"container-id\n")
        if args[:2] == ("container", "kill"):
            self.running = False
            return DockerCLIResult(returncode=0, stdout=b"container-id\n")
        if args[:3] == ("container", "rm", "--force"):
            if self.block_remove:
                self.remove_started.set()
                await self.remove_release.wait()
            if self.fail_remove:
                return DockerCLIResult(returncode=1, stderr=b"remove failed")
            self.running = False
            self.container_exists = False
            return DockerCLIResult(returncode=0, stdout=b"container-id\n")
        if args[:2] == ("container", "inspect"):
            return self._json_result({"Image": "sha256:image-id"})
        if args[:2] == ("image", "inspect"):
            if self.fail_image_inspection:
                return DockerCLIResult(returncode=1, stderr=b"inspect failed")
            return self._json_result(
                {
                    "Id": "sha256:image-id",
                    "Architecture": "amd64",
                    "Os": "linux",
                    "RepoDigests": ["example/image@sha256:resolved"],
                }
            )
        if args[:2] == ("container", "cp"):
            if args[-1] == "-":
                return DockerCLIResult(returncode=0, stdout=self._workspace_tar())
            self._copy_host_workspace(Path(args[2][:-2]))
            return DockerCLIResult(returncode=0)
        if args[:2] == ("container", "exec"):
            result = await self._run_exec(args, stdin)
            if merge_stderr and result.stderr:
                return DockerCLIResult(
                    returncode=result.returncode,
                    stdout=result.stdout + result.stderr,
                )
            return result
        raise AssertionError(f"Unexpected fake Docker command: {args}")

    async def _run_exec(
        self,
        arguments: tuple[str, ...],
        stdin: bytes | None,
    ) -> DockerCLIResult:
        if self.container_name is None:
            raise AssertionError("Container has not been created")
        command = arguments[arguments.index(self.container_name) + 1 :]
        if len(command) >= 4 and command[3] == "vedex-resolve-workdir":
            return DockerCLIResult(returncode=0, stdout=f"{command[-1]}\n".encode())
        if len(command) >= 6 and command[3] == "vedex-read":
            relative = self._relative_path(command[-1], command[-2])
            if relative in self.symlinks:
                return DockerCLIResult(returncode=48)
            if relative in self.directories or relative == ".":
                return DockerCLIResult(returncode=45)
            if relative not in self.files:
                return DockerCLIResult(returncode=44)
            return DockerCLIResult(returncode=0, stdout=self.files[relative])
        if len(command) >= 6 and command[3] == "vedex-write":
            relative = self._relative_path(command[-1], command[-2])
            if relative in self.symlinks:
                return DockerCLIResult(returncode=48)
            if relative in self.directories or relative == ".":
                return DockerCLIResult(returncode=45)
            self.files[relative] = stdin or b""
            return DockerCLIResult(returncode=0)
        if command[:1] == ("git",):
            return self._run_git(command)
        if command == ("uname", "-r"):
            return DockerCLIResult(returncode=0, stdout=b"6.8.0-test\n")
        if command in (("python3", "--version"), ("python", "--version")):
            return DockerCLIResult(returncode=0, stdout=b"Python 3.12.9\n")

        if self.block_user_command:
            self.command_started.set()
            try:
                await asyncio.Future[None]()
            except asyncio.CancelledError:
                self.command_cancelled = True
                raise
        return self.user_result

    def _run_git(self, command: tuple[str, ...]) -> DockerCLIResult:
        if command[1:] == ("rev-parse", "--is-inside-work-tree"):
            if self.git_repository:
                return DockerCLIResult(returncode=0, stdout=b"true\n")
            return DockerCLIResult(returncode=128, stderr=b"not a git repository")
        if command[1:] == ("rev-parse", "HEAD"):
            if not self.git_repository:
                return DockerCLIResult(returncode=128)
            return DockerCLIResult(returncode=0, stdout=f"{self.git_revision}\n".encode())
        if command[1:3] == ("status", "--porcelain=v1"):
            dirty = bool(self.tracked_diff or self.untracked_diffs)
            return DockerCLIResult(returncode=0, stdout=b" M tracked.txt\n" if dirty else b"")
        if command[1:3] == ("ls-files", "--others"):
            output = b"\0".join(path.encode() for path in self.untracked_diffs)
            return DockerCLIResult(returncode=0, stdout=output + (b"\0" if output else b""))
        if "--no-index" in command:
            path = command[-1]
            return DockerCLIResult(
                returncode=1,
                stdout=self.untracked_diffs.get(path, b""),
            )
        if command[1] == "diff":
            return DockerCLIResult(
                returncode=0 if not self.tracked_diff else 1,
                stdout=self.tracked_diff,
            )
        raise AssertionError(f"Unexpected fake git command: {command}")

    def _copy_host_workspace(self, source: Path) -> None:
        for path in source.rglob("*"):
            relative = path.relative_to(source).as_posix()
            if path.is_dir():
                self.directories.add(relative)
            elif path.is_file():
                self.files[relative] = path.read_bytes()

    def _workspace_tar(self) -> bytes:
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            root = tarfile.TarInfo(".")
            root.type = tarfile.DIRTYPE
            archive.addfile(root)
            directory_names = set(self.directories)
            for name in self.files:
                parent = Path(name).parent
                while parent != Path("."):
                    directory_names.add(parent.as_posix())
                    parent = parent.parent
            for name in sorted(directory_names):
                info = tarfile.TarInfo(name)
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            for name, data in sorted(self.files.items()):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 12345
                archive.addfile(info, io.BytesIO(data))
            git_data = b"internal"
            git_info = tarfile.TarInfo(".git/config")
            git_info.size = len(git_data)
            archive.addfile(git_info, io.BytesIO(git_data))
            if self.unsafe_archive:
                unsafe = tarfile.TarInfo("../escape.txt")
                unsafe.size = 1
                archive.addfile(unsafe, io.BytesIO(b"x"))
        return output.getvalue()

    @staticmethod
    def _relative_path(target: str, root: str) -> str:
        if target == root:
            return "."
        return target.removeprefix(f"{root.rstrip('/')}/")

    @staticmethod
    def _json_result(value: object) -> DockerCLIResult:
        return DockerCLIResult(returncode=0, stdout=json.dumps(value).encode())


def _calls_starting_with(
    cli: _FakeDockerCLI,
    prefix: tuple[str, ...],
) -> list[_DockerCall]:
    return [call for call in cli.calls if call.arguments[: len(prefix)] == prefix]


def test_docker_configuration_rejects_incoherent_or_unsafe_values(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="workspace is required"):
        DockerEnvironmentConfig(image="image", workspace_mode="copy")
    with pytest.raises(ValidationError, match="must be omitted"):
        DockerEnvironmentConfig(image="image", workspace=tmp_path)
    with pytest.raises(ValidationError, match="absolute container path"):
        DockerEnvironmentConfig(image="image", workdir="relative")
    with pytest.raises(ValidationError, match="absolute container path"):
        DockerEnvironmentConfig(image="image", workdir="/work/../escape")
    with pytest.raises(ValidationError, match="NAME=value"):
        DockerEnvironmentConfig(image="image", environment={"BAD=NAME": "value"})
    with pytest.raises(ValidationError):
        DockerEnvironmentConfig(image="image", cpus=0)


def test_docker_lifecycle_builds_isolated_container_and_records_metadata() -> None:
    first_cli = _FakeDockerCLI()
    second_cli = _FakeDockerCLI()
    config = DockerEnvironmentConfig(
        image="example/image:tag",
        environment={"TOKEN": "secret", "MODE": "test"},
        cpus=1.5,
        memory="2g",
    )
    first = DockerEnvironment(config, cli=first_cli)
    second = DockerEnvironment(config, cli=second_cli)
    contract: Environment = first

    assert isinstance(first, Environment)
    assert first.container_name != second.container_name
    assert first.state == "created"
    assert contract.workspace.model_root == "."

    async def exercise() -> None:
        await first.start()
        await first.start()
        metadata = await first.get_metadata()
        assert first.state == "started"
        assert metadata.environment_type == "docker"
        assert metadata.operating_system == "linux"
        assert metadata.operating_system_release == "6.8.0-test"
        assert metadata.architecture == "amd64"
        assert metadata.python_version == "Python 3.12.9"
        assert metadata.git_revision == "a" * 40
        assert metadata.git_dirty is True
        assert metadata.details["engine_version"] == "28.3.3"
        assert metadata.details["image_reference"] == "example/image:tag"
        assert metadata.details["resolved_image_reference"] == ("example/image@sha256:resolved")
        assert metadata.details["resolved_image_digest"] == "sha256:resolved"
        assert metadata.details["network"] == "none"
        assert metadata.details["resource_limits"] == {"cpus": 1.5, "memory": "2g"}
        assert metadata.details["environment_variable_names"] == ["MODE", "TOKEN"]
        assert metadata.details["lifecycle_timeouts_seconds"] == {
            "startup": 120.0,
            "operation": 30.0,
            "cleanup": 30.0,
        }
        assert "secret" not in metadata.model_dump_json()
        await first.stop()
        await first.stop()

    run_async(exercise())
    create_call = _calls_starting_with(first_cli, ("container", "create"))[0]
    assert create_call.arguments[create_call.arguments.index("--network") + 1] == "none"
    assert create_call.arguments[create_call.arguments.index("--cpus") + 1] == "1.5"
    assert create_call.arguments[create_call.arguments.index("--memory") + 1] == "2g"
    assert "TOKEN=secret" in create_call.arguments
    assert len(_calls_starting_with(first_cli, ("container", "create"))) == 1
    assert len(_calls_starting_with(first_cli, ("container", "rm", "--force"))) == 1
    assert str(first.state) == "stopped"


@pytest.mark.parametrize("workspace_mode", ["image", "copy", "mount"])
def test_docker_workspace_modes(
    workspace_mode: DockerWorkspaceMode,
    tmp_path: Path,
) -> None:
    source = tmp_path / workspace_mode
    source.mkdir()
    (source / "source.txt").write_text("copied", encoding="utf-8")
    cli = _FakeDockerCLI()
    config = DockerEnvironmentConfig(
        image="example/image:tag",
        workspace_mode=workspace_mode,
        workspace=source if workspace_mode != "image" else None,
    )
    environment = DockerEnvironment(config, cli=cli)

    async def exercise() -> None:
        async with environment:
            if workspace_mode == "copy":
                assert await environment.read_bytes("source.txt") == b"copied"

    run_async(exercise())
    create_call = _calls_starting_with(cli, ("container", "create"))[0]
    copy_calls = _calls_starting_with(cli, ("container", "cp"))
    if workspace_mode == "mount":
        mount = create_call.arguments[create_call.arguments.index("--mount") + 1]
        assert mount == f"type=bind,source={source.resolve()},target=/workspace"
        assert copy_calls == []
    elif workspace_mode == "copy":
        assert len(copy_calls) == 1
        assert "--mount" not in create_call.arguments
    else:
        assert copy_calls == []
        assert "--mount" not in create_call.arguments


def test_docker_copy_and_mount_require_an_existing_directory(tmp_path: Path) -> None:
    missing = DockerEnvironment(
        DockerEnvironmentConfig(
            image="image",
            workspace_mode="copy",
            workspace=tmp_path / "missing",
        ),
        cli=_FakeDockerCLI(),
    )
    file_path = tmp_path / "file.txt"
    file_path.write_text("file", encoding="utf-8")
    file_environment = DockerEnvironment(
        DockerEnvironmentConfig(
            image="image",
            workspace_mode="mount",
            workspace=file_path,
        ),
        cli=_FakeDockerCLI(),
    )

    with pytest.raises(EnvironmentStateError, match="does not exist"):
        run_async(missing.start())
    with pytest.raises(EnvironmentStateError, match="not a directory"):
        run_async(file_environment.start())


def test_docker_missing_engine_failure_is_explicit(tmp_path: Path) -> None:
    environment = DockerEnvironment(
        DockerEnvironmentConfig(
            image="image",
            engine=str(tmp_path / "missing-container-engine"),
        )
    )

    with pytest.raises(EnvironmentStateError, match="Could not start container engine"):
        run_async(environment.start())
    assert environment.state == "stopped"


def test_docker_byte_io_and_tools_share_the_environment_contract(tmp_path: Path) -> None:
    cli = _FakeDockerCLI(files={"source.txt": b"one\ntwo"})
    cli.directories.add("directory")
    cli.user_result = DockerCLIResult(
        returncode=7,
        stdout=b"stdout\n",
        stderr=b"stderr\n",
    )
    environment = DockerEnvironment(
        DockerEnvironmentConfig(image="image"),
        cli=cli,
    )

    async def exercise() -> None:
        await environment.start()
        try:
            await environment.write_bytes("nested/data.bin", b"\xff\x00data")
            assert await environment.read_bytes("nested/data.bin") == b"\xff\x00data"
            with pytest.raises(EnvironmentFileError) as missing:
                await environment.read_bytes("missing.txt")
            assert missing.value.kind == "not_found"
            with pytest.raises(EnvironmentFileError) as directory:
                await environment.write_bytes("directory", b"no")
            assert directory.value.kind == "is_directory"
            cli.symlinks.add("link.txt")
            with pytest.raises(WorkspacePathError, match="outside"):
                await environment.read_bytes("link.txt")
            with pytest.raises(WorkspacePathError):
                await environment.read_bytes("../outside")

            tools = {tool.name: tool for tool in environment.tools}
            write = await tools["write"].execute({"path": "new.txt", "content": "hello"})
            edit = await tools["edit"].execute(
                {
                    "path": "source.txt",
                    "edits": [{"oldText": "one", "newText": "ONE"}],
                }
            )
            read = await tools["read"].execute({"path": "source.txt", "limit": 1})
            bash = await tools["bash"].execute({"command": "test", "timeout": 2})
            assert write.ok is True
            assert edit.ok is True
            assert read.content.startswith("     1  ONE")
            assert bash.ok is False
            assert "stdout\nstderr" in bash.content
            assert cli.files["new.txt"] == b"hello"
            assert cli.files["source.txt"] == b"ONE\ntwo"
        finally:
            await environment.stop()

    run_async(exercise())
    local = LocalEnvironment(tmp_path)
    assert [tool.name for tool in environment.tools] == [tool.name for tool in local.tools]
    assert [tool.input_schema for tool in environment.tools] == [
        tool.input_schema for tool in local.tools
    ]


def test_docker_root_workdir_keeps_model_paths_relative(tmp_path: Path) -> None:
    cli = _FakeDockerCLI()
    environment = DockerEnvironment(
        DockerEnvironmentConfig(image="image", workdir="/"),
        cli=cli,
    )

    async def exercise() -> None:
        async with environment:
            await environment.write_bytes("nested/file.txt", b"contents")
            assert await environment.read_bytes("nested/file.txt") == b"contents"
            await environment.export_workspace(tmp_path / "workspace.tar.gz")

    run_async(exercise())
    copy_call = _calls_starting_with(cli, ("container", "cp"))[0]
    assert copy_call.arguments[2].endswith(":/.")


def test_docker_command_timeout_and_cancellation_restart_the_container() -> None:
    async def timeout_case() -> None:
        cli = _FakeDockerCLI()
        cli.block_user_command = True
        environment = DockerEnvironment(
            DockerEnvironmentConfig(image="image"),
            cli=cli,
        )
        await environment.start()
        try:
            result = await environment.run_command("sleep", timeout_seconds=0.01)
            assert result.timed_out is True
            assert result.cancelled is False
            assert cli.command_cancelled is True
            assert len(_calls_starting_with(cli, ("container", "kill"))) == 1
            assert len(_calls_starting_with(cli, ("container", "start"))) == 2
            assert environment.state == "started"
        finally:
            await environment.stop()

    async def cancellation_case() -> None:
        cli = _FakeDockerCLI()
        cli.block_user_command = True
        environment = DockerEnvironment(
            DockerEnvironmentConfig(image="image"),
            cli=cli,
        )
        token = _CancellationToken()
        await environment.start()
        try:
            task = asyncio.create_task(
                environment.run_command("sleep", timeout_seconds=5, signal=token)
            )
            await cli.command_started.wait()
            token.cancel()
            result = await task
            assert result.cancelled is True
            assert result.timed_out is False
            assert len(_calls_starting_with(cli, ("container", "kill"))) == 1
            assert len(_calls_starting_with(cli, ("container", "start"))) == 2
        finally:
            await environment.stop()

    run_async(timeout_case())
    run_async(cancellation_case())


def test_stopping_docker_environment_cancels_command_without_restarting() -> None:
    cli = _FakeDockerCLI()
    cli.block_user_command = True
    environment = DockerEnvironment(DockerEnvironmentConfig(image="image"), cli=cli)

    async def exercise() -> None:
        await environment.start()
        command_task = asyncio.create_task(environment.run_command("sleep", timeout_seconds=5))
        await cli.command_started.wait()
        await environment.stop()
        result = await command_task
        assert result.cancelled is True
        assert environment.state == "stopped"

    run_async(exercise())
    assert _calls_starting_with(cli, ("container", "kill")) == []
    assert len(_calls_starting_with(cli, ("container", "start"))) == 1
    assert len(_calls_starting_with(cli, ("container", "rm", "--force"))) == 1


def test_docker_cancellation_and_command_limit_contracts() -> None:
    cli = _FakeDockerCLI()
    environment = DockerEnvironment(
        DockerEnvironmentConfig(image="image"),
        limits=EnvironmentLimits(max_command_seconds=3),
        cli=cli,
    )
    token = _CancellationToken()
    token.cancel()

    async def exercise() -> None:
        await environment.start()
        try:
            with pytest.raises(EnvironmentCancelledError):
                await environment.read_bytes("file.txt", signal=token)
            cancelled = await environment.run_command(
                "ignored",
                timeout_seconds=1,
                signal=token,
            )
            assert cancelled.cancelled is True
            with pytest.raises(ValueError, match="must not exceed"):
                await environment.run_command("ignored", timeout_seconds=4)
        finally:
            await environment.stop()

    run_async(exercise())


def test_docker_patch_and_deterministic_workspace_export(tmp_path: Path) -> None:
    cli = _FakeDockerCLI(
        files={
            "tracked.txt": b"changed\n",
            "nested/file.txt": b"contents",
        }
    )
    cli.directories.add("empty")
    environment = DockerEnvironment(DockerEnvironmentConfig(image="image"), cli=cli)
    first_path = tmp_path / "first.tar.gz"
    second_path = tmp_path / "second.tar.gz"

    async def exercise() -> None:
        await environment.start()
        try:
            patch = await environment.collect_patch()
            assert patch.is_git_repository is True
            assert patch.changed is True
            assert patch.base_revision == "a" * 40
            assert "tracked.txt" in patch.text
            assert "new.txt" in patch.text
            first = await environment.export_workspace(first_path)
            second = await environment.export_workspace(second_path)
            assert first.file_count == 2
            assert first.sha256 == second.sha256
            assert first.size_bytes == second.size_bytes
        finally:
            await environment.stop()

    run_async(exercise())
    assert first_path.read_bytes() == second_path.read_bytes()
    with tarfile.open(first_path, "r:gz") as archive:
        names = archive.getnames()
        assert "tracked.txt" in names
        assert "nested/file.txt" in names
        assert "empty" in names
        assert not any(name == ".git" or name.startswith(".git/") for name in names)
        extracted = archive.extractfile("nested/file.txt")
        assert extracted is not None
        assert extracted.read() == b"contents"


def test_docker_patch_reports_non_git_workspace() -> None:
    cli = _FakeDockerCLI()
    cli.git_repository = False
    environment = DockerEnvironment(DockerEnvironmentConfig(image="image"), cli=cli)

    async def exercise() -> None:
        async with environment:
            patch = await environment.collect_patch()
            assert patch.is_git_repository is False
            assert patch.changed is False
            assert patch.text == ""

    run_async(exercise())


def test_docker_export_rejects_unsafe_archive_and_mounted_destination(
    tmp_path: Path,
) -> None:
    cli = _FakeDockerCLI(files={"file.txt": b"data"})
    cli.unsafe_archive = True
    image_environment = DockerEnvironment(
        DockerEnvironmentConfig(image="image"),
        cli=cli,
    )

    async def unsafe_export() -> None:
        async with image_environment:
            with pytest.raises(EnvironmentExportError, match="unsafe path"):
                await image_environment.export_workspace(tmp_path / "unsafe.tar.gz")

    run_async(unsafe_export())
    assert not (tmp_path / "unsafe.tar.gz").exists()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mounted = DockerEnvironment(
        DockerEnvironmentConfig(
            image="image",
            workspace_mode="mount",
            workspace=workspace,
        ),
        cli=_FakeDockerCLI(),
    )

    async def mounted_export() -> None:
        async with mounted:
            with pytest.raises(EnvironmentExportError, match="outside"):
                await mounted.export_workspace(workspace / "output.tar.gz")

    run_async(mounted_export())


def test_docker_cleans_up_after_context_and_startup_failures() -> None:
    context_cli = _FakeDockerCLI()
    context_environment = DockerEnvironment(
        DockerEnvironmentConfig(image="image"),
        cli=context_cli,
    )

    async def context_failure() -> None:
        with pytest.raises(RuntimeError, match="body failed"):
            async with context_environment:
                raise RuntimeError("body failed")

    run_async(context_failure())
    assert context_environment.state == "stopped"
    assert len(_calls_starting_with(context_cli, ("container", "rm", "--force"))) == 1

    startup_cli = _FakeDockerCLI()
    startup_cli.fail_image_inspection = True
    startup_environment = DockerEnvironment(
        DockerEnvironmentConfig(image="image"),
        cli=startup_cli,
    )
    with pytest.raises(EnvironmentStateError, match="image inspection"):
        run_async(startup_environment.start())
    assert startup_environment.state == "stopped"
    assert len(_calls_starting_with(startup_cli, ("container", "rm", "--force"))) == 1


def test_docker_cleanup_failure_is_explicit_and_state_is_stopped() -> None:
    cli = _FakeDockerCLI()
    environment = DockerEnvironment(DockerEnvironmentConfig(image="image"), cli=cli)

    async def exercise() -> None:
        await environment.start()
        cli.fail_remove = True
        with pytest.raises(EnvironmentStateError, match="remove failed"):
            await environment.stop()
        assert environment.state == "stopped"
        assert cli.container_exists is True
        cli.fail_remove = False
        await environment.stop()
        assert cli.container_exists is False

    run_async(exercise())


def test_docker_cleanup_finishes_when_stop_caller_is_cancelled() -> None:
    cli = _FakeDockerCLI()
    cli.block_remove = True
    environment = DockerEnvironment(DockerEnvironmentConfig(image="image"), cli=cli)

    async def exercise() -> None:
        await environment.start()
        stop_task = asyncio.create_task(environment.stop())
        await cli.remove_started.wait()
        stop_task.cancel()
        cli.remove_release.set()
        with pytest.raises(asyncio.CancelledError):
            await stop_task
        assert environment.state == "stopped"
        assert cli.container_exists is False

    run_async(exercise())


def test_docker_environment_can_be_used_across_sequential_event_loops() -> None:
    cli = _FakeDockerCLI(files={"file.txt": b"contents"})
    environment = DockerEnvironment(DockerEnvironmentConfig(image="image"), cli=cli)

    run_async(environment.start())
    assert run_async(environment.read_bytes("file.txt")) == b"contents"
    run_async(environment.stop())
    assert str(environment.state) == "stopped"


def test_headless_runtime_uses_docker_environment_end_to_end(tmp_path: Path) -> None:
    cli = _FakeDockerCLI()
    environment = DockerEnvironment(DockerEnvironmentConfig(image="image"), cli=cli)
    adapter = FakeAdapter(
        [
            [
                ModelStartEvent(),
                ModelCompletedEvent(
                    message=AssistantMessage(
                        tool_calls=[
                            ToolCall(
                                id="write-1",
                                name="write",
                                arguments={"path": "result.txt", "content": "created"},
                            )
                        ]
                    )
                ),
            ],
            [
                ModelStartEvent(),
                ModelCompletedEvent(message=AssistantMessage(content="Finished")),
            ],
        ]
    )
    stdout = StringIO()
    stderr = StringIO()

    result = run_async(
        run_headless(
            task="Create result.txt",
            adapter=adapter,
            environment=environment,
            settings=ModelSettings(model="fake"),
            workspace=tmp_path,
            stdout=stdout,
            stderr=stderr,
        )
    )

    assert result.status == "completed"
    assert result.exit_code == HeadlessExitCode.COMPLETED
    assert result.tool_calls == 1
    assert cli.files["result.txt"] == b"created"
    assert stdout.getvalue() == "Finished\n"
    assert stderr.getvalue() == ""
    assert environment.state == "stopped"
    assert len(_calls_starting_with(cli, ("container", "rm", "--force"))) == 1
