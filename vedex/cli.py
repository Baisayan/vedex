from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, cast

import typer

from .agent import AgentLimits
from .environments import Environment, LocalEnvironment
from .models import ModelAdapter, ModelSettings
from .rendering import CommandLineRenderer
from .resources import PromptTemplate, ResourceError, ResourcePaths, Skill
from .runtime import AppRuntime
from .schema import AgentTool
from .workspace import ReloadSummary

_CANCELLATION_GRACE_SECONDS = 2.0


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
    help="Vedex provider-neutral terminal coding agent.",
    add_completion=False,
)


@app.command()
def main(
    adapter: Annotated[
        str,
        typer.Option(
            "--adapter",
            help="Import path to a zero-argument ModelAdapter factory (module:attribute).",
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model identifier passed to the adapter."),
    ],
    prompt_args: Annotated[
        list[str] | None,
        typer.Argument(help="Optional initial prompt, followed by interactive mode."),
    ] = None,
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Workspace used by coding tools."),
    ] = None,
) -> None:
    workspace_path = (cwd or Path.cwd()).expanduser().resolve(strict=False)
    initial_prompt = " ".join(prompt_args) if prompt_args else None
    try:
        model_adapter = load_adapter(adapter)
        asyncio.run(
            run_repl(
                adapter=model_adapter,
                settings=ModelSettings(model=model),
                workspace_path=workspace_path,
                initial_prompt=initial_prompt,
            )
        )
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.", err=True)
        raise typer.Exit(130) from None
    except (ImportError, OSError, ResourceError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


def load_adapter(specification: str) -> ModelAdapter:
    """Load an adapter factory without coupling the REPL to a model provider."""
    module_name, separator, attribute_name = specification.strip().partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Adapter must use the form 'module:factory'")

    module = importlib.import_module(module_name)
    factory_value = getattr(module, attribute_name, None)
    if not callable(factory_value):
        raise ValueError(f"Adapter factory is not callable: {specification}")
    factory = cast(Callable[[], object], factory_value)
    try:
        candidate = factory()
    except Exception as exc:
        raise RuntimeError(f"Adapter factory failed: {specification}: {exc}") from exc
    if not callable(getattr(candidate, "stream", None)):
        raise ValueError(f"Adapter factory did not return a ModelAdapter: {specification}")
    return cast(ModelAdapter, candidate)


async def run_repl(
    *,
    adapter: ModelAdapter,
    settings: ModelSettings,
    workspace_path: Path,
    initial_prompt: str | None = None,
    environment: Environment | None = None,
    limits: AgentLimits | None = None,
    resource_paths: ResourcePaths | None = None,
) -> None:
    """Run one ephemeral interactive process around the shared AppRuntime."""
    active_environment = environment or LocalEnvironment(workspace_path)
    runtime = AppRuntime(
        adapter=adapter,
        environment=active_environment,
        settings=settings,
        workspace_path=workspace_path,
        limits=limits,
        resource_paths=resource_paths,
    )

    async with runtime:
        if initial_prompt is not None:
            await _run_agent_turn(runtime, initial_prompt)

        while True:
            try:
                raw = input("> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                typer.echo("\nUse /exit to quit.", err=True)
                continue

            text = raw.strip()
            if not text:
                continue

            try:
                if text.startswith("!"):
                    await _run_terminal_command(runtime, text[1:].strip())
                    continue

                if not text.startswith("/"):
                    await _run_agent_turn(runtime, text)
                    continue

                command, argument = _split_command(text)
                if command == "/exit":
                    _require_no_argument(command, argument)
                    break
                if command == "/help":
                    _require_no_argument(command, argument)
                    typer.echo(_HELP_TEXT)
                    continue
                if command == "/clear":
                    _require_no_argument(command, argument)
                    _clear_screen()
                    continue
                if command == "/skills":
                    _require_no_argument(command, argument)
                    _show_skills(runtime.workspace.skills)
                    continue
                if command == "/prompts":
                    _require_no_argument(command, argument)
                    _show_prompt_templates(runtime.workspace.prompt_templates)
                    continue
                if command == "/context":
                    _require_no_argument(command, argument)
                    _show_context(runtime)
                    continue
                if command == "/reload":
                    _require_no_argument(command, argument)
                    typer.echo(_format_reload_summary(runtime.reload_resources()))
                    continue
                if command == "/reset":
                    _require_no_argument(command, argument)
                    runtime.reset()
                    typer.echo("Conversation memory reset.")
                    continue
                if command.startswith("/skill:"):
                    await _run_agent_turn(runtime, text)
                    continue
                if runtime.workspace.expand_prompt_template_command(text) is not None:
                    await _run_agent_turn(runtime, text)
                    continue

                typer.echo(f"Error: Unknown command: {command}", err=True)
            except (OSError, ResourceError, RuntimeError, ValueError) as exc:
                typer.echo(f"Error: {exc}", err=True)


async def _run_agent_turn(runtime: AppRuntime, prompt: str) -> None:
    renderer = CommandLineRenderer()

    async def consume_events() -> None:
        async for event in runtime.prompt(prompt):
            renderer.render(event)

    turn_task = asyncio.create_task(consume_events())
    try:
        await asyncio.shield(turn_task)
    except asyncio.CancelledError:
        runtime.cancel()
        _uncancel_current_task()
        try:
            await asyncio.wait_for(
                asyncio.shield(turn_task),
                timeout=_CANCELLATION_GRACE_SECONDS,
            )
        except (TimeoutError, asyncio.CancelledError):
            turn_task.cancel()
            await asyncio.gather(turn_task, return_exceptions=True)
        typer.echo("Cancelled.", err=True)
    finally:
        renderer.finish()


async def _run_terminal_command(runtime: AppRuntime, command: str) -> None:
    if not command:
        raise ValueError("Shell command cannot be empty")

    bash_tool = _find_tool(runtime.agent.tools, "bash")
    if bash_tool is None:
        raise RuntimeError("The bash tool is unavailable")
    try:
        result = await bash_tool.execute({"command": command})
    except asyncio.CancelledError:
        _uncancel_current_task()
        typer.echo("Command cancelled.", err=True)
        return

    status = "ok" if result.ok else "failed"
    typer.echo(f"$ {command}\n[{status}; terminal only]\n{result.content}")


def _uncancel_current_task() -> None:
    task = asyncio.current_task()
    if task is not None:
        task.uncancel()


def _find_tool(tools: tuple[AgentTool, ...], name: str) -> AgentTool | None:
    return next((tool for tool in tools if tool.name == name), None)


def _split_command(text: str) -> tuple[str, str]:
    command, separator, argument = text.partition(" ")
    return command.lower(), argument.strip() if separator else ""


def _require_no_argument(command: str, argument: str) -> None:
    if argument:
        raise ValueError(f"{command} does not accept arguments")


def _show_skills(skills: tuple[Skill, ...]) -> None:
    if not skills:
        typer.echo("No skills available.")
        return
    typer.echo("Available skills:")
    for skill in skills:
        description = f" — {skill.description}" if skill.description else ""
        typer.echo(f"  /skill:{skill.name}{description}")


def _show_prompt_templates(templates: tuple[PromptTemplate, ...]) -> None:
    if not templates:
        typer.echo("No prompt templates available.")
        return
    typer.echo("Available prompt templates:")
    for template in templates:
        description = f" — {template.description}" if template.description else ""
        typer.echo(f"  /{template.name} [arguments]{description}")


def _show_context(runtime: AppRuntime) -> None:
    typer.echo(f"Model: {runtime.settings.model}")
    typer.echo(f"Workspace: {runtime.workspace_path}")
    typer.echo(f"Messages in memory: {len(runtime.agent.messages)}")
    typer.echo(f"System prompt: {len(runtime.system_prompt)} characters")
    if not runtime.workspace.context_files:
        typer.echo("Project instructions: none")
        return
    typer.echo("Project instructions:")
    for context_file in runtime.workspace.context_files:
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


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


_HELP_TEXT = """Commands:
  /help                     Show this help.
  /skills                   List loaded skills.
  /skill:<name> [request]   Run a request with the skill's full instructions.
  /prompts                  List loaded prompt templates.
  /<prompt> [arguments]     Expand and run a loaded prompt template.
  /context                  Inspect in-memory and project context.
  /reload                   Reload resources without changing message history.
  /reset                    Clear in-memory conversation history.
  /clear                    Clear the terminal.
  /exit                     Exit Vedex.
  !command                  Run a foreground shell command outside model history."""


__all__ = ["app", "load_adapter", "main", "run_repl"]
