from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import sys
from pathlib import Path
from typing import Annotated

import typer

from .coding_session import (
    CodingSession,
    CodingSessionConfig,
    TerminalCommandResult,
    parse_terminal_command,
)
from .core import OLLAMA_HOST, list_model_info
from .rendering import CommandLineRenderer
from .resources import VedexPaths
from .session import SessionStore
from .tools import create_coding_tools
from .workspace import Workspace


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


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt_args: Annotated[
        list[str] | None,
        typer.Argument(help="Initial prompt to run in interactive mode."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name to request from Ollama."),
    ] = None,
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory for built-in coding tools."),
    ] = None,
    session: Annotated[
        str | None,
        typer.Option("--session", help="Session ID or path to JSONL session file."),
    ] = None,
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    initial_prompt = " ".join(prompt_args) if prompt_args else None
    resolved_model = model or "llama3.2"
    resolved_cwd = (cwd or Path.cwd()).expanduser().resolve()

    try:
        asyncio.run(
            _run_repl(
                initial_model=resolved_model,
                cwd=resolved_cwd,
                session_ref=session,
                initial_prompt=initial_prompt,
            )
        )
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


async def _run_repl(
    initial_model: str,
    cwd: Path,
    session_ref: str | None,
    initial_prompt: str | None,
) -> None:
    config = _build_session_config(
        initial_model=initial_model,
        cwd=cwd,
        session_ref=session_ref,
    )
    session = await CodingSession.load(config)

    try:
        if initial_prompt is not None:
            renderer = CommandLineRenderer()
            try:
                async for event in session.prompt(initial_prompt):
                    renderer.render(event)
            except (KeyboardInterrupt, asyncio.CancelledError):
                typer.echo("\nCancelled.")
            renderer.finish()

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

            terminal = parse_terminal_command(text)
            if terminal is not None:
                result = await session.run_terminal_command(
                    terminal.command,
                    add_to_context=terminal.add_to_context,
                )
                typer.echo(_format_terminal_command_result(result))
                continue

            command_result = await session.handle_command(text)
            if command_result.handled:
                if command_result.exit_requested:
                    break
                if command_result.clear_requested:
                    _clear_screen()
                    continue
                if command_result.model_picker_requested:
                    model_name = await _pick_model()
                    if model_name is not None:
                        session.set_model(model_name)
                        typer.echo(f"Current model set to: {model_name}")
                    continue
                if command_result.message:
                    typer.echo(command_result.message)
                continue

            renderer = CommandLineRenderer()
            try:
                async for event in session.prompt(text):
                    renderer.render(event)
            except (KeyboardInterrupt, asyncio.CancelledError):
                typer.echo("\nCancelled.")
            renderer.finish()
    finally:
        await session.aclose()


def _build_session_config(
    initial_model: str,
    cwd: Path,
    session_ref: str | None,
) -> CodingSessionConfig:
    tools = create_coding_tools(cwd=cwd)
    workspace = Workspace(cwd=cwd, tools=tools)
    if session_ref:
        candidate_path = Path(session_ref).expanduser()
        if len(session_ref) == 6 and all(char in "0123456789abcdef" for char in session_ref):
            candidate_path = VedexPaths().sessions_dir / f"{session_ref}.jsonl"
        if candidate_path.exists():
            return CodingSessionConfig(
                ollama_host=OLLAMA_HOST,
                model=initial_model,
                cwd=cwd,
                storage=SessionStore(candidate_path),
                tools=tools,
                workspace=workspace,
            )

        raise RuntimeError(f"Unknown session file: {session_ref}")

    return CodingSessionConfig(
        ollama_host=OLLAMA_HOST,
        model=initial_model,
        cwd=cwd,
        storage=_new_session_store(),
        tools=tools,
        workspace=workspace,
    )


def _new_session_store() -> SessionStore:
    sessions_dir = VedexPaths().sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    while True:
        path = sessions_dir / f"{secrets.token_hex(3)}.jsonl"
        if not path.exists():
            return SessionStore(path)


async def _pick_model() -> str | None:
    try:
        models = await list_model_info()
    except Exception as exc:
        typer.echo(f"Could not connect to Ollama: {exc}", err=True)
        return None

    if not models:
        typer.echo("No models available.", err=True)
        return None

    typer.echo("Available models:")
    for i, m in enumerate(models, 1):
        typer.echo(f"  {i}. {m.name}")

    while True:
        try:
            choice = input("Select model (number or name, empty to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not choice:
            return None

        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                return models[idx - 1].name
        except ValueError:
            pass

        for m in models:
            if m.name == choice:
                return m.name

        typer.echo(f"Invalid choice: {choice}")


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _format_terminal_command_result(result: TerminalCommandResult) -> str:
    context_status = "added to context" if result.added_to_context else "not added to context"
    return f"$ {result.command}\n[{context_status}]\n{result.output}"
