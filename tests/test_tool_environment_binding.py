from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from vedex.environments import (
    CommandResult,
    Environment,
    EnvironmentLimits,
    EnvironmentMetadata,
    EnvironmentState,
    WorkspaceExport,
    WorkspaceIdentity,
    WorkspacePatch,
    normalize_workspace_path,
)
from vedex.schema import AgentTool, CancellationToken
from vedex.tools import create_coding_tools

from .conftest import run_async


class _MemoryEnvironment:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {"source.txt": b"one\ntwo"}
        self.operations: list[tuple[str, str]] = []
        self._tools: tuple[AgentTool, ...] = ()

    @property
    def state(self) -> EnvironmentState:
        return "started"

    @property
    def workspace(self) -> WorkspaceIdentity:
        return WorkspaceIdentity(id="memory-workspace", name="memory")

    @property
    def limits(self) -> EnvironmentLimits:
        return EnvironmentLimits()

    @property
    def tools(self) -> Sequence[AgentTool]:
        if not self._tools:
            self._tools = tuple(create_coding_tools(environment=self))
        return self._tools

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def normalize_path(self, path: str) -> str:
        return normalize_workspace_path(path)

    async def read_bytes(
        self,
        path: str,
        *,
        signal: CancellationToken | None = None,
    ) -> bytes:
        del signal
        normalized = self.normalize_path(path)
        self.operations.append(("read", normalized))
        return self.files[normalized]

    async def write_bytes(
        self,
        path: str,
        data: bytes,
        *,
        signal: CancellationToken | None = None,
    ) -> None:
        del signal
        normalized = self.normalize_path(path)
        self.operations.append(("write", normalized))
        self.files[normalized] = data

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float,
        signal: CancellationToken | None = None,
    ) -> CommandResult:
        del timeout_seconds, signal
        self.operations.append(("command", command))
        return CommandResult(command=command, stdout=b"memory output", exit_code=0)

    async def collect_patch(
        self,
        *,
        signal: CancellationToken | None = None,
    ) -> WorkspacePatch:
        del signal
        return WorkspacePatch(
            text="",
            is_git_repository=False,
            changed=False,
        )

    async def export_workspace(
        self,
        destination: Path,
        *,
        signal: CancellationToken | None = None,
    ) -> WorkspaceExport:
        del signal
        return WorkspaceExport(
            path=str(destination),
            file_count=len(self.files),
            size_bytes=0,
            sha256="0" * 64,
        )

    async def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            environment_type="memory",
            workspace=self.workspace,
            operating_system="test",
            operating_system_release="test",
            architecture="test",
            python_version="test",
            shell="test",
        )


def test_all_coding_tools_use_only_the_environment_contract() -> None:
    memory = _MemoryEnvironment()
    environment: Environment = memory
    tools = {tool.name: tool for tool in create_coding_tools(environment=environment)}

    read_result = run_async(tools["read"].execute({"path": "nested/../source.txt", "limit": 1}))
    write_result = run_async(tools["write"].execute({"path": "new.txt", "content": "héllo"}))
    edit_result = run_async(
        tools["edit"].execute(
            {
                "path": "source.txt",
                "edits": [{"oldText": "one", "newText": "ONE"}],
            }
        )
    )
    bash_result = run_async(tools["bash"].execute({"command": "test command"}))

    assert read_result.content.startswith("     1  one")
    assert write_result.data == {"path": "new.txt", "characters": 5, "bytes": 6}
    assert memory.files["new.txt"] == "héllo".encode()
    assert edit_result.content == "Edited source.txt: 1 replacement(s)."
    assert memory.files["source.txt"] == b"ONE\ntwo"
    assert bash_result.content == "memory output"
    assert memory.operations == [
        ("read", "source.txt"),
        ("write", "new.txt"),
        ("read", "source.txt"),
        ("write", "source.txt"),
        ("command", "test command"),
    ]
