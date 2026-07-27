from __future__ import annotations

import builtins
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import vedex.cli as cli
from typer.testing import CliRunner
from vedex.core import OllamaModelInfo
from vedex.resources import PromptTemplate, ResourceError, ResourcePaths, Skill, VedexPaths
from vedex.schema import AgentTool, AgentToolResult, UserMessage
from vedex.session import Session, SessionStore
from vedex.workspace import ReloadCategorySummary, ReloadSummary, Workspace

from .conftest import make_tool, native_ollama_client, native_tags_response, run_async


def _patch_session_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> VedexPaths:
    paths = VedexPaths(home=tmp_path / ".vedex")
    monkeypatch.setattr(cli, "VedexPaths", lambda: paths)
    return paths


def test_public_cli_help_and_model_lookup() -> None:
    result = CliRunner().invoke(cli.app, ["--help"])
    models = [
        OllamaModelInfo(name="one:latest", context_length=4096, supports_tools=True),
        OllamaModelInfo(name="two:latest", context_length=None, supports_tools=False),
    ]

    assert result.exit_code == 0
    assert "--session" in result.output
    assert cli._find_model(models, "one").name == "one:latest"  # type: ignore[union-attr]
    assert cli._find_model(models, "two:latest").name == "two:latest"  # type: ignore[union-attr]
    assert cli._find_model(models, "missing") is None


def test_new_resolve_list_and_preview_session_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _patch_session_home(monkeypatch, tmp_path)
    paths.sessions_dir.mkdir(parents=True)
    (paths.sessions_dir / "aaaaaa.jsonl").write_text(
        '{"role":"user","content":"first user prompt"}\n', encoding="utf-8"
    )
    (paths.sessions_dir / "bbbbbb.jsonl").write_text("not json\n", encoding="utf-8")
    (paths.sessions_dir / "ignored-name.jsonl").write_text("", encoding="utf-8")

    choices = cli._list_session_choices()
    assert {choice.identifier for choice in choices} == {"aaaaaa", "bbbbbb"}
    assert (
        next(choice.preview for choice in choices if choice.identifier == "aaaaaa")
        == "first user prompt"
    )
    assert (
        next(choice.preview for choice in choices if choice.identifier == "bbbbbb")
        == "(invalid session file)"
    )
    assert cli._resolve_session_store("aaaaaa").path.name == "aaaaaa.jsonl"
    direct_path = tmp_path / "direct.jsonl"
    direct_path.write_text("", encoding="utf-8")
    assert cli._resolve_session_store(str(direct_path)).path == direct_path.resolve()
    with pytest.raises(RuntimeError, match="Unknown session ID"):
        cli._resolve_session_store("cccccc")

    (paths.sessions_dir / "000000.jsonl").write_text("", encoding="utf-8")
    generated = iter(["000000", "dddddd"])
    monkeypatch.setattr("vedex.cli.secrets.token_hex", lambda _size: next(generated))
    store = cli._new_session_store()
    assert store.path.name == "dddddd.jsonl"
    assert store.path.read_text(encoding="utf-8") == ""


def test_model_selection_and_unavailable_ollama_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = [OllamaModelInfo(name="local:latest", context_length=2048, supports_tools=True)]

    async def list_models() -> list[OllamaModelInfo]:
        return models

    monkeypatch.setattr(cli, "list_model_info", list_models)
    assert run_async(cli._select_model("local")).context_length == 2048  # type: ignore[union-attr]

    inputs = iter(["bad", "1"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))
    assert run_async(cli._select_model("")) == models[0]

    async def unavailable() -> list[OllamaModelInfo]:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(cli, "list_model_info", unavailable)
    with pytest.raises(RuntimeError, match="Ollama is unavailable"):
        run_async(cli._available_models())


def test_skill_prompt_and_context_selection_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill = Skill(name="review", path=tmp_path / "review.md", content="body", description="Review")
    template = PromptTemplate(name="fix", path=tmp_path / "fix.md", content="body")

    assert cli._select_skill((skill,), "REVIEW") is skill
    assert cli._select_prompt_template((template,), "fix") is template
    with pytest.raises(ResourceError):
        cli._select_skill((skill,), "missing")

    inputs: Iterator[str] = iter(["1", "fix"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))
    assert cli._select_skill((skill,), "") is skill
    assert cli._select_prompt_template((template,), "") is template
    cli._show_skill(skill)
    cli._show_prompt_template(template)
    assert "/skill:review" in capsys.readouterr().out


def test_terminal_modes_persist_only_single_bang_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = native_ollama_client(lambda _request: native_tags_response())
    session = Session(
        cwd=tmp_path,
        model="local:latest",
        system_prompt="system",
        tools=[],
        client=client,
        store=SessionStore(tmp_path / "session.jsonl"),
        context_window_tokens=4096,
    )
    bash = make_tool(
        name="bash",
        result=AgentToolResult(tool_call_id="", name="bash", ok=True, content="command output"),
    )
    try:
        run_async(cli._run_terminal_command(command="echo one", tools=[bash], session=session))
        run_async(cli._run_terminal_command(command="echo two", tools=[bash], session=None))
    finally:
        run_async(session.close())

    assert session.messages == [
        UserMessage(content="Terminal command:\n$ echo one\n\nOutput:\ncommand output")
    ]
    output = capsys.readouterr().out
    assert "[added to context]" in output
    assert "[terminal only]" in output


def test_command_formatters_and_read_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = ReloadSummary(
        skills=ReloadCategorySummary(before=1, after=2, changed=True),
        prompt_templates=ReloadCategorySummary(before=1, after=1, changed=False),
        context_files=ReloadCategorySummary(before=0, after=1, changed=True),
        system_prompt_rebuilt=True,
    )
    assert cli._split_command("/model local") == ("/model", "local")
    assert "system prompt updated" in cli._format_reload_summary(summary)

    monkeypatch.setattr(builtins, "input", lambda _prompt="": " choice ")
    assert cli._read_choice("pick") == "choice"
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "")
    assert cli._read_choice("pick") is None


def test_repl_routes_commands_to_workspace_session_and_session_picker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _patch_session_home(monkeypatch, tmp_path)
    resource_root = tmp_path / "resources"
    project = tmp_path / "project"
    (resource_root / "skills").mkdir(parents=True)
    (resource_root / "prompts").mkdir(parents=True)
    (resource_root / "AGENTS.md").write_text("global context", encoding="utf-8")
    (resource_root / "skills" / "review.md").write_text("# Review skill", encoding="utf-8")
    (resource_root / "prompts" / "fix.md").write_text("Fix {{arguments}}", encoding="utf-8")

    info = OllamaModelInfo(name="local:latest", context_length=2048, supports_tools=True)
    selected = OllamaModelInfo(name="other:latest", context_length=8192, supports_tools=True)
    seen_prompts: list[str] = []
    cleared: list[bool] = []

    async def resolve_initial(_requested: str | None) -> OllamaModelInfo:
        return info

    async def select_model(_argument: str) -> OllamaModelInfo | None:
        return selected

    async def run_agent_turn(_session: Session, prompt: str) -> None:
        seen_prompts.append(prompt)

    def workspace_factory(*, cwd: Path, tools: list[AgentTool]) -> Workspace:
        return Workspace(
            cwd=cwd,
            tools=tools,
            resource_paths=ResourcePaths(root=resource_root),
        )

    monkeypatch.setattr(cli, "_resolve_initial_model", resolve_initial)
    monkeypatch.setattr(cli, "_select_model", select_model)
    monkeypatch.setattr(cli, "_run_agent_turn", run_agent_turn)
    monkeypatch.setattr(
        cli,
        "create_coding_tools",
        lambda *, cwd: [make_tool(name="read"), make_tool(name="bash")],
    )
    monkeypatch.setattr(cli, "Workspace", workspace_factory)
    monkeypatch.setattr(cli, "_clear_screen", lambda: cleared.append(True))
    inputs = iter(
        [
            "/help",
            "/clear",
            "/model other",
            "/skills review",
            "/prompts fix",
            "/skill:review inspect",
            "/fix parser",
            "/context",
            "/reload",
            "/session",
            "/new",
            "/resume",
            "1",
            "/exit",
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    run_async(
        cli._run_repl(
            requested_model="local:latest",
            cwd=project,
            session_ref=None,
            initial_prompt=None,
        )
    )

    output = capsys.readouterr().out
    assert cleared == [True]
    assert "Current model: other:latest" in output
    assert "Skill: review" in output
    assert "Prompt template: fix" in output
    assert "Active project context:" in output
    assert "Session:" in output
    assert "New session:" in output
    assert "Resumed session:" in output
    assert len(list(paths.sessions_dir.glob("*.jsonl"))) == 2
    assert any("Review skill" in prompt and prompt.endswith("inspect") for prompt in seen_prompts)
    assert "Fix parser" in seen_prompts


@pytest.mark.parametrize("command", ["/quit", "/exit"])
def test_repl_exit_commands_stop_cleanly(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_home(monkeypatch, tmp_path)
    info = OllamaModelInfo(name="local:latest", context_length=2048, supports_tools=True)

    async def resolve_initial(_requested: str | None) -> OllamaModelInfo:
        return info

    monkeypatch.setattr(cli, "_resolve_initial_model", resolve_initial)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": command)

    run_async(
        cli._run_repl(
            requested_model="local:latest",
            cwd=tmp_path,
            session_ref=None,
            initial_prompt=None,
        )
    )
