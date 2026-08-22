from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import vedex.cli as cli
from typer.testing import CliRunner
from vedex.environments import LocalEnvironment
from vedex.models import (
    FakeAdapter,
    ModelCancelledEvent,
    ModelCompletedEvent,
    ModelEvent,
    ModelRequest,
    ModelSettings,
    ModelStartEvent,
    ModelTextDeltaEvent,
)
from vedex.resources import ResourcePaths
from vedex.runtime import AppRuntime
from vedex.schema import AssistantMessage, CancellationToken, UserMessage
from vedex.workspace import ReloadCategorySummary, ReloadSummary

from .conftest import run_async


def _completed_stream(content: str) -> list[ModelEvent]:
    return [
        ModelStartEvent(),
        ModelTextDeltaEvent(delta=content),
        ModelCompletedEvent(message=AssistantMessage(content=content)),
    ]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class _BlockingThenCompletedAdapter:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.requests: list[ModelRequest] = []
        self._call_count = 0

    def stream(
        self,
        request: ModelRequest,
        *,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ModelEvent]:
        self.requests.append(request.model_copy(deep=True))
        self._call_count += 1
        call_number = self._call_count

        async def events() -> AsyncIterator[ModelEvent]:
            yield ModelStartEvent()
            if call_number == 1:
                self.started.set()
                while signal is None or not signal.is_cancelled():
                    await asyncio.sleep(0)
                yield ModelCancelledEvent()
                return
            yield ModelTextDeltaEvent(delta="recovered")
            yield ModelCompletedEvent(message=AssistantMessage(content="recovered"))

        return events()


def test_public_cli_requires_startup_adapter_and_model(tmp_path: Path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(cli.app, ["--help"])
    run_result = runner.invoke(
        cli.app,
        [
            "--adapter",
            "vedex.models:FakeAdapter",
            "--model",
            "fake",
            "--cwd",
            str(tmp_path),
        ],
        input="/exit\n",
    )

    assert help_result.exit_code == 0
    assert "--adapter" in help_result.output
    assert "--model" in help_result.output
    assert "--session" not in help_result.output
    assert run_result.exit_code == 0


def test_adapter_loader_validates_factory_contract() -> None:
    adapter = cli.load_adapter("vedex.models:FakeAdapter")

    assert isinstance(adapter, FakeAdapter)
    with pytest.raises(ValueError, match="module:factory"):
        cli.load_adapter("invalid")
    with pytest.raises(ValueError, match="not callable"):
        cli.load_adapter("vedex.models:__all__")


def test_repl_uses_shared_runtime_resources_and_ephemeral_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    resources = tmp_path / "resources"
    project.mkdir()
    _write(project / "pyproject.toml", "[project]\nname='fixture'\n")
    instructions = project / "AGENTS.md"
    _write(instructions, "Original project instruction")
    _write(
        resources / "skills" / "review" / "SKILL.md",
        "---\ndescription: Review code\n---\nFull review instructions",
    )
    _write(
        resources / "prompts" / "fix.md",
        "---\ndescription: Fix a target\n---\nFix {{arguments}}",
    )
    adapter = FakeAdapter([_completed_stream(f"answer {number}") for number in range(1, 6)])
    cleared: list[bool] = []
    commands = iter(
        [
            "/help",
            "/skills",
            "/prompts",
            "/context",
            "/skill:review inspect parser",
            "/fix parser",
            "!echo shell-output",
            "/model other",
            "/reload",
            "after reload",
            "/reset",
            "after reset",
            "/clear",
            "/exit",
        ]
    )

    def read_input(_prompt: str = "") -> str:
        command = next(commands)
        if command == "/reload":
            instructions.write_text("Reloaded project instruction", encoding="utf-8")
        return command

    monkeypatch.setattr("builtins.input", read_input)
    monkeypatch.setattr(cli, "_clear_screen", lambda: cleared.append(True))

    run_async(
        cli.run_repl(
            adapter=adapter,
            settings=ModelSettings(model="fake-model"),
            workspace_path=project,
            initial_prompt="first task",
            resource_paths=ResourcePaths(root=resources),
        )
    )

    captured = capsys.readouterr()
    assert "Available skills:" in captured.out
    assert "/skill:review — Review code" in captured.out
    assert "Available prompt templates:" in captured.out
    assert "Messages in memory:" in captured.out
    assert "shell-output" in captured.out
    assert "system prompt updated" in captured.out
    assert "Conversation memory reset." in captured.out
    assert "Unknown command: /model" in captured.err
    assert cleared == [True]
    assert len(adapter.requests) == 5
    skill_message = adapter.requests[1].messages[-1]
    assert isinstance(skill_message, UserMessage)
    assert "Full review instructions" in skill_message.content
    assert adapter.requests[2].messages[-1] == UserMessage(content="Fix parser")
    assert adapter.requests[3].system != adapter.requests[0].system
    assert "Reloaded project instruction" in adapter.requests[3].system
    assert len(adapter.requests[3].messages) > 1
    assert adapter.requests[4].messages == [UserMessage(content="after reset")]
    assert not (resources / "sessions").exists()
    assert list(project.rglob("*.jsonl")) == []


def test_ctrl_c_cancels_active_turn_and_next_prompt_still_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter = _BlockingThenCompletedAdapter()
    runtime = AppRuntime(
        adapter=adapter,
        environment=LocalEnvironment(tmp_path),
        settings=ModelSettings(model="fake"),
        workspace_path=tmp_path,
        resource_paths=ResourcePaths(root=tmp_path / "resources"),
    )

    async def exercise() -> None:
        async with runtime:
            active_turn = asyncio.create_task(cli._run_agent_turn(runtime, "cancel me"))
            await asyncio.wait_for(adapter.started.wait(), timeout=1)
            active_turn.cancel()
            await active_turn

            assert runtime.agent.is_running is False
            cancelled_result = runtime.agent.last_result
            assert cancelled_result is not None
            assert cancelled_result.status == "cancelled"

            await cli._run_agent_turn(runtime, "next prompt")
            completed_result = runtime.agent.last_result
            assert completed_result is not None
            assert completed_result.status == "completed"

    run_async(exercise())

    captured = capsys.readouterr()
    assert "Cancelled." in captured.err
    assert "recovered" in captured.out
    assert len(adapter.requests) == 2


def test_direct_shell_uses_runtime_tool_without_adding_model_history(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = AppRuntime(
        adapter=FakeAdapter(),
        environment=LocalEnvironment(tmp_path),
        settings=ModelSettings(model="fake"),
        workspace_path=tmp_path,
        resource_paths=ResourcePaths(root=tmp_path / "resources"),
    )

    async def exercise() -> None:
        async with runtime:
            await cli._run_terminal_command(runtime, "echo direct-output")
            assert runtime.agent.messages == ()

    run_async(exercise())

    assert "direct-output" in capsys.readouterr().out


def test_command_helpers_are_stable() -> None:
    summary = ReloadSummary(
        skills=ReloadCategorySummary(before=1, after=2, changed=True),
        prompt_templates=ReloadCategorySummary(before=1, after=1, changed=False),
        context_files=ReloadCategorySummary(before=0, after=1, changed=True),
        system_prompt_rebuilt=True,
    )

    assert cli._split_command("/skill:review request") == ("/skill:review", "request")
    assert "system prompt updated" in cli._format_reload_summary(summary)
    with pytest.raises(ValueError, match="does not accept"):
        cli._require_no_argument("/reset", "unexpected")
