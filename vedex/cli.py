from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from core import OLLAMA_HOST, list_model_info
from rendering import CommandLineRenderer
from coding_session import (
    CodingSession,
    CodingSessionConfig,
    TerminalCommandResult,
    jsonl_session_storage,
    parse_terminal_command,
)
from session_manager import SessionManager


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
        raise typer.Exit(1)


async def _run_repl(
    initial_model: str,
    cwd: Path,
    session_ref: str | None,
    initial_prompt: str | None,
) -> None:
    config, manager = _build_session_config(
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

            result = await session.handle_command(text)
            if result.handled:
                if result.exit_requested:
                    break
                if result.clear_requested:
                    _clear_screen()
                    continue
                if result.new_session_requested:
                    try:
                        msg = await session.new_session()
                    except ValueError as exc:
                        typer.echo(str(exc), err=True)
                        continue
                    typer.echo(msg)
                    continue
                if result.resume_session_id is not None:
                    try:
                        msg = await session.resume(result.resume_session_id)
                    except ValueError as exc:
                        typer.echo(str(exc), err=True)
                        continue
                    typer.echo(msg)
                    continue
                if result.resume_picker_requested and manager is not None:
                    sid = _pick_session(manager, cwd)
                    if sid is not None:
                        try:
                            msg = await session.resume(sid)
                        except ValueError as exc:
                            typer.echo(str(exc), err=True)
                            continue
                        typer.echo(msg)
                    continue
                if result.model_picker_requested:
                    model_name = await _pick_model()
                    if model_name is not None:
                        session.set_model(model_name)
                        typer.echo(f"Current model set to: {model_name}")
                    continue
                if result.message:
                    typer.echo(result.message)
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
) -> tuple[CodingSessionConfig, SessionManager | None]:
    manager = SessionManager()

    if session_ref:
        existing = manager.get_session(session_ref)
        if existing is not None:
            return CodingSessionConfig(
                ollama_host=OLLAMA_HOST,
                model=existing.model,
                cwd=existing.cwd,
                storage=jsonl_session_storage(existing.path),
                session_id=existing.id,
                session_manager=manager,
            ), manager

        candidate_path = Path(session_ref).expanduser()
        if candidate_path.exists():
            return CodingSessionConfig(
                ollama_host=OLLAMA_HOST,
                model=initial_model,
                cwd=cwd,
                storage=jsonl_session_storage(candidate_path),
            ), None

        raise RuntimeError(f"Unknown session or file: {session_ref}")

    record = manager.get_or_create_default_session(cwd=cwd, model=initial_model)
    return CodingSessionConfig(
        ollama_host=OLLAMA_HOST,
        model=record.model,
        cwd=record.cwd,
        storage=jsonl_session_storage(record.path),
        session_id=record.id,
        session_manager=manager,
    ), manager


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


def _pick_session(manager: SessionManager, cwd: Path) -> str | None:
    records = manager.list_sessions(cwd)
    if not records:
        typer.echo("No sessions found for this directory.")
        return None

    typer.echo("Sessions:")
    for i, record in enumerate(records, 1):
        title = record.title or "Untitled"
        typer.echo(f"  {i}. {record.id}  {title}  {record.model}")

    while True:
        try:
            choice = input("Select session (number or id, empty to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not choice:
            return None

        try:
            idx = int(choice)
            if 1 <= idx <= len(records):
                return records[idx - 1].id
        except ValueError:
            pass

        for r in records:
            if r.id == choice:
                return choice

        typer.echo(f"Invalid choice: {choice}")


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _format_terminal_command_result(result: TerminalCommandResult) -> str:
    context_status = "added to context" if result.added_to_context else "not added to context"
    return f"$ {result.command}\n[{context_status}]\n{result.output}"
