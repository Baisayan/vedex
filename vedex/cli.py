from __future__ import annotations

import asyncio
import contextlib
import os
import re
import secrets
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer

from .core import OLLAMA_HOST, OllamaClient, OllamaModelInfo, list_model_info
from .environments import LocalEnvironment
from .rendering import CommandLineRenderer
from .resources import PromptTemplate, ResourceError, Skill, VedexPaths
from .schema import AgentMessage, AgentTool, UserMessage
from .session import Session, SessionError, SessionStore
from .tools import create_coding_tools
from .workspace import ReloadSummary, Workspace

_SESSION_ID_RE = re.compile(r"[0-9a-f]{6}")
_SESSION_PREVIEW_LENGTH = 72


def _is_utf8_encoding(encoding: str | None) -> bool:
    if encoding is None:
        return False
    return encoding.lower().replace("-", "").replace("_", "") == "utf8"


def _force_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if _is_utf8_encoding(getattr(stream, "encoding", None)):
            continue
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


_force_utf8_streams()

app = typer.Typer(
    name="vedex",
    help="Vedex coding-agent harness.",
    add_completion=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


@dataclass(frozen=True, slots=True)
class _SessionChoice:
    path: Path
    modified_at: datetime
    preview: str

    @property
    def identifier(self) -> str:
        return self.path.stem


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt_args: Annotated[
        list[str] | None,
        typer.Argument(help="Initial prompt to run in interactive mode."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Locally available Ollama model."),
    ] = None,
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory for built-in coding tools."),
    ] = None,
    session: Annotated[
        str | None,
        typer.Option("--session", help="Six-character session ID or existing JSONL file."),
    ] = None,
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    initial_prompt = " ".join(prompt_args) if prompt_args else None
    resolved_cwd = (cwd or Path.cwd()).expanduser().resolve()
    try:
        asyncio.run(
            _run_repl(
                requested_model=model,
                cwd=resolved_cwd,
                session_ref=session,
                initial_prompt=initial_prompt,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


async def _run_repl(
    *,
    requested_model: str | None,
    cwd: Path,
    session_ref: str | None,
    initial_prompt: str | None,
) -> None:
    model_info = await _resolve_initial_model(requested_model)
    environment = LocalEnvironment(cwd)
    await environment.start()
    try:
        tools = create_coding_tools(environment=environment)
        workspace = Workspace(
            cwd=cwd,
            tools=tools,
            model_cwd=environment.workspace.model_root,
        )
        store = _resolve_session_store(session_ref) if session_ref else _new_session_store()
        session = _create_session(
            cwd=cwd,
            model_info=model_info,
            workspace=workspace,
            tools=tools,
            store=store,
        )
    except BaseException:
        await environment.stop()
        raise

    try:
        if initial_prompt is not None:
            await _run_agent_turn(session, initial_prompt)

        while True:
            try:
                raw = input("> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                typer.echo()
                break

            text = raw.strip()
            if not text:
                continue

            try:
                if text.startswith("!!"):
                    await _run_terminal_command(
                        command=text[2:].strip(),
                        tools=tools,
                        session=None,
                    )
                    continue
                if text.startswith("!"):
                    await _run_terminal_command(
                        command=text[1:].strip(),
                        tools=tools,
                        session=session,
                    )
                    continue

                if not text.startswith("/"):
                    await _run_agent_turn(session, text)
                    continue

                command, argument = _split_command(text)
                if command in {"/exit", "/quit"}:
                    break
                if command == "/help":
                    typer.echo(_HELP_TEXT)
                    continue
                if command == "/clear":
                    _clear_screen()
                    continue
                if command == "/model":
                    selected_model = await _select_model(argument)
                    if selected_model is not None:
                        model_info = selected_model
                        session.set_model(selected_model.name, selected_model.context_length)
                        typer.echo(f"Current model: {selected_model.name}")
                    continue
                if command == "/skills":
                    skill = _select_skill(workspace.skills, argument)
                    if skill is not None:
                        _show_skill(skill)
                    continue
                if command == "/prompts":
                    template = _select_prompt_template(workspace.prompt_templates, argument)
                    if template is not None:
                        _show_prompt_template(template)
                    continue
                if command == "/context":
                    _show_context(workspace)
                    continue
                if command == "/reload":
                    summary = workspace.reload()
                    if summary.system_prompt_rebuilt:
                        session.set_system_prompt(workspace.system_prompt)
                    typer.echo(_format_reload_summary(summary))
                    continue
                if command == "/session":
                    typer.echo(_format_session_status(session))
                    continue
                if command == "/new":
                    next_session = _create_session(
                        cwd=cwd,
                        model_info=model_info,
                        workspace=workspace,
                        tools=tools,
                        store=_new_session_store(),
                    )
                    await session.close()
                    session = next_session
                    typer.echo(f"New session: {session.store.path.stem}")
                    continue
                if command == "/resume":
                    next_store = _select_session_store(argument)
                    if next_store is None:
                        continue
                    next_session = _create_session(
                        cwd=cwd,
                        model_info=model_info,
                        workspace=workspace,
                        tools=tools,
                        store=next_store,
                    )
                    await session.close()
                    session = next_session
                    typer.echo(f"Resumed session: {session.store.path.stem}")
                    continue

                expanded_prompt = workspace.expand_prompt_text(text)
                if expanded_prompt != text:
                    await _run_agent_turn(session, expanded_prompt)
                    continue

                typer.echo(f"Error: Unknown command: {command}", err=True)
            except (OSError, ResourceError, SessionError, ValueError) as exc:
                typer.echo(f"Error: {exc}", err=True)
    finally:
        try:
            await session.close()
        finally:
            await environment.stop()


async def _resolve_initial_model(requested_model: str | None) -> OllamaModelInfo:
    models = await _available_models()
    if requested_model is not None:
        selected_model = _find_model(models, requested_model)
        if selected_model is None:
            raise RuntimeError(f"Ollama model is not available locally: {requested_model}")
        return selected_model

    selected_model = _pick_model(models)
    if selected_model is None:
        raise RuntimeError("No Ollama model selected")
    return selected_model


async def _select_model(argument: str) -> OllamaModelInfo | None:
    models = await _available_models()
    if argument:
        selected_model = _find_model(models, argument)
        if selected_model is None:
            raise RuntimeError(f"Ollama model is not available locally: {argument}")
        return selected_model
    return _pick_model(models)


async def _available_models() -> list[OllamaModelInfo]:
    try:
        models = await list_model_info()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Ollama is unavailable at {OLLAMA_HOST}") from exc
    if not models:
        raise RuntimeError("No local Ollama models are available")
    return models


def _find_model(models: list[OllamaModelInfo], requested: str) -> OllamaModelInfo | None:
    name = requested.strip()
    exact_name = name if ":" in name else f"{name}:latest"
    for model in models:
        if model.name == name or model.name == exact_name:
            return model
    return None


def _pick_model(models: list[OllamaModelInfo]) -> OllamaModelInfo | None:
    typer.echo("Available local models:")
    for index, model in enumerate(models, start=1):
        context = f", context {model.context_length}" if model.context_length else ""
        typer.echo(f"  {index}. {model.name}{context}")

    while True:
        choice = _read_choice("Select model (number or name, empty to cancel): ")
        if choice is None:
            return None
        if choice.isdecimal():
            index = int(choice)
            if 1 <= index <= len(models):
                return models[index - 1]
        selected_model = _find_model(models, choice)
        if selected_model is not None:
            return selected_model
        typer.echo(f"Invalid model: {choice}")


def _create_session(
    *,
    cwd: Path,
    model_info: OllamaModelInfo,
    workspace: Workspace,
    tools: list[AgentTool],
    store: SessionStore,
) -> Session:
    return Session(
        cwd=cwd,
        model=model_info.name,
        system_prompt=workspace.system_prompt,
        tools=tools,
        client=OllamaClient(),
        store=store,
        context_window_tokens=model_info.context_length,
    )


def _resolve_session_store(session_ref: str) -> SessionStore:
    if _SESSION_ID_RE.fullmatch(session_ref):
        path = VedexPaths().sessions_dir / f"{session_ref}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Unknown session ID: {session_ref}")
        return SessionStore(path)

    path = Path(session_ref).expanduser()
    if not path.is_file():
        raise RuntimeError(f"Unknown session file: {session_ref}")
    return SessionStore(path.resolve())


def _new_session_store() -> SessionStore:
    sessions_dir = VedexPaths().sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    while True:
        path = sessions_dir / f"{secrets.token_hex(3)}.jsonl"
        if not path.exists():
            store = SessionStore(path)
            store.rewrite([])
            return store


def _select_session_store(argument: str) -> SessionStore | None:
    if argument:
        return _resolve_session_store(argument)

    choices = _list_session_choices()
    if not choices:
        typer.echo("No saved sessions.")
        return None

    typer.echo("Saved sessions:")
    for index, choice in enumerate(choices, start=1):
        timestamp = choice.modified_at.strftime("%Y-%m-%d %H:%M")
        typer.echo(f"  {index}. {choice.identifier}  {timestamp}  {choice.preview}")

    while True:
        selected = _read_choice("Resume session (number or ID, empty to cancel): ")
        if selected is None:
            return None
        if selected.isdecimal():
            index = int(selected)
            if 1 <= index <= len(choices):
                return SessionStore(choices[index - 1].path)
        for choice in choices:
            if choice.identifier == selected:
                return SessionStore(choice.path)
        typer.echo(f"Invalid session: {selected}")


def _list_session_choices() -> list[_SessionChoice]:
    sessions_dir = VedexPaths().sessions_dir
    if not sessions_dir.is_dir():
        return []

    choices: list[_SessionChoice] = []
    for path in sessions_dir.glob("*.jsonl"):
        if not _SESSION_ID_RE.fullmatch(path.stem):
            continue
        try:
            messages = SessionStore(path).load()
            preview = _first_user_preview(messages)
        except (OSError, SessionError):
            preview = "(invalid session file)"
        choices.append(
            _SessionChoice(
                path=path,
                modified_at=datetime.fromtimestamp(path.stat().st_mtime),
                preview=preview,
            )
        )
    return sorted(choices, key=lambda choice: choice.modified_at, reverse=True)


def _first_user_preview(messages: Sequence[AgentMessage]) -> str:
    for message in messages:
        if isinstance(message, UserMessage):
            preview = " ".join(message.content.split())
            if len(preview) > _SESSION_PREVIEW_LENGTH:
                return f"{preview[: _SESSION_PREVIEW_LENGTH - 1].rstrip()}…"
            return preview or "(empty user message)"
    return "(no user message)"


async def _run_terminal_command(
    *,
    command: str,
    tools: list[AgentTool],
    session: Session | None,
) -> None:
    if not command:
        raise ValueError("Shell command cannot be empty")

    bash_tool = next((tool for tool in tools if tool.name == "bash"), None)
    if bash_tool is None:
        raise RuntimeError("The bash tool is unavailable")
    result = await bash_tool.execute({"command": command})
    context_status = "added to context" if session is not None else "terminal only"
    typer.echo(f"$ {command}\n[{context_status}]\n{result.content}")
    if session is not None:
        session.add_user_message(f"Terminal command:\n$ {command}\n\nOutput:\n{result.content}")


async def _run_agent_turn(session: Session, prompt: str) -> None:
    renderer = CommandLineRenderer()
    try:
        async for event in session.prompt(prompt):
            renderer.render(event)
    except (KeyboardInterrupt, asyncio.CancelledError):
        typer.echo("\nCancelled.")
    renderer.finish()


def _split_command(text: str) -> tuple[str, str]:
    command, separator, argument = text.partition(" ")
    return command.lower(), argument.strip() if separator else ""


def _select_skill(skills: tuple[Skill, ...], argument: str) -> Skill | None:
    if argument:
        skill = _find_skill(skills, argument)
        if skill is None:
            raise ResourceError(f"Unknown skill: {argument}")
        return skill
    if not skills:
        typer.echo("No skills available.")
        return None

    typer.echo("Available skills:")
    for index, skill in enumerate(skills, start=1):
        description = f" — {skill.description}" if skill.description else ""
        typer.echo(f"  {index}. {skill.name}{description}")
    while True:
        selected = _read_choice("Select skill (number or name, empty to cancel): ")
        if selected is None:
            return None
        if selected.isdecimal():
            index = int(selected)
            if 1 <= index <= len(skills):
                return skills[index - 1]
        skill = _find_skill(skills, selected)
        if skill is not None:
            return skill
        typer.echo(f"Invalid skill: {selected}")


def _select_prompt_template(
    templates: tuple[PromptTemplate, ...],
    argument: str,
) -> PromptTemplate | None:
    if argument:
        template = _find_prompt_template(templates, argument)
        if template is None:
            raise ResourceError(f"Unknown prompt template: {argument}")
        return template
    if not templates:
        typer.echo("No prompt templates available.")
        return None

    typer.echo("Available prompt templates:")
    for index, template in enumerate(templates, start=1):
        description = f" — {template.description}" if template.description else ""
        typer.echo(f"  {index}. {template.name}{description}")
    while True:
        selected = _read_choice("Select prompt template (number or name, empty to cancel): ")
        if selected is None:
            return None
        if selected.isdecimal():
            index = int(selected)
            if 1 <= index <= len(templates):
                return templates[index - 1]
        template = _find_prompt_template(templates, selected)
        if template is not None:
            return template
        typer.echo(f"Invalid prompt template: {selected}")


def _find_skill(skills: tuple[Skill, ...], name: str) -> Skill | None:
    normalized_name = name.strip().lower()
    return next((skill for skill in skills if skill.name.lower() == normalized_name), None)


def _find_prompt_template(
    templates: tuple[PromptTemplate, ...],
    name: str,
) -> PromptTemplate | None:
    normalized_name = name.strip().removeprefix("/").lower()
    return next(
        (template for template in templates if template.name.lower() == normalized_name), None
    )


def _show_skill(skill: Skill) -> None:
    typer.echo(f"Skill: {skill.name}\nPath: {skill.path}")
    if skill.description:
        typer.echo(f"Description: {skill.description}")
    typer.echo(f"Use: /skill:{skill.name} [request]")


def _show_prompt_template(template: PromptTemplate) -> None:
    typer.echo(f"Prompt template: {template.name}\nPath: {template.path}")
    if template.description:
        typer.echo(f"Description: {template.description}")
    typer.echo(f"Use: /{template.name} [arguments]")


def _show_context(workspace: Workspace) -> None:
    if not workspace.context_files:
        typer.echo("No project context files are active.")
        return
    typer.echo("Active project context:")
    for context_file in workspace.context_files:
        typer.echo(f"  - {context_file.path}")


def _format_reload_summary(summary: ReloadSummary) -> str:
    def format_category(name: str, before: int, after: int, changed: bool) -> str:
        state = "changed" if changed else "unchanged"
        return f"{name} {before}→{after} ({state})"

    parts = [
        format_category(
            "skills", summary.skills.before, summary.skills.after, summary.skills.changed
        ),
        format_category(
            "prompts",
            summary.prompt_templates.before,
            summary.prompt_templates.after,
            summary.prompt_templates.changed,
        ),
        format_category(
            "context",
            summary.context_files.before,
            summary.context_files.after,
            summary.context_files.changed,
        ),
    ]
    prompt_state = "updated" if summary.system_prompt_rebuilt else "unchanged"
    return f"Reloaded: {', '.join(parts)}; system prompt {prompt_state}."


def _format_session_status(session: Session) -> str:
    context_window = (
        str(session.context_window_tokens)
        if session.context_window_tokens is not None
        else "unknown"
    )
    context_limit = (
        str(session.context_token_limit) if session.context_token_limit is not None else "unknown"
    )
    return "\n".join(
        (
            f"Session: {session.store.path.stem}",
            f"Model: {session.model}",
            f"Working directory: {session.cwd}",
            f"Messages: {len(session.messages)}",
            f"Estimated context tokens: {session.context_token_estimate}",
            f"Context window: {context_window} (history limit: {context_limit})",
            f"Loaded tools: {len(session.tools)}",
        )
    )


def _read_choice(prompt: str) -> str | None:
    try:
        choice = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        typer.echo()
        return None
    return choice or None


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


_HELP_TEXT = """Commands:
  /help                     Show this help.
  /clear                    Clear the terminal.
  /exit, /quit              Exit Vedex.
  /model [name]             Pick or change a local Ollama model.
  /skills [name]            Pick or inspect a loaded skill.
  /skill:<name> [request]   Send a skill-guided request to the agent.
  /prompts [name]           Pick or inspect a prompt template.
  /<prompt> [arguments]     Render a loaded prompt template and send it.
  /context                  List project context injected into the prompt.
  /reload                   Reload skills, prompts, and project context.
  /session                  Show current runtime/session details.
  /new                      Create and switch to a new session.
  /resume [id]              Pick or resume a six-character session ID.
  !command                  Run bash and add its output to the conversation.
  !!command                 Run bash for terminal output only."""
