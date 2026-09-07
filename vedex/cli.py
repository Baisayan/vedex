"""Command-line entry points for Vedex."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer

from .agent import AgentConfig, DefaultAgent
from .environments import DockerEnvironment, LocalEnvironment
from .models import LiteLLMModel

app = typer.Typer(
    name="vedex",
    help="A small terminal coding agent.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="Task for the coding agent.")],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", envvar="VEDEX_MODEL", help="LiteLLM model name."),
    ] = None,
    environment: Annotated[
        str,
        typer.Option("--environment", "-e", help="Execution environment: local or docker."),
    ] = "local",
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Local workspace directory."),
    ] = None,
    image: Annotated[
        str | None,
        typer.Option("--image", help="Docker image when using the Docker environment."),
    ] = None,
    container_cwd: Annotated[
        str,
        typer.Option("--container-cwd", help="Working directory inside the container."),
    ] = "/",
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Command timeout in seconds."),
    ] = 30.0,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Trajectory output path."),
    ] = None,
) -> None:
    """Run one task without interactive prompting."""
    try:
        asyncio.run(
            _run_task(
                task=task,
                model_name=_require_model(model),
                environment_name=environment,
                cwd=cwd,
                image=image,
                container_cwd=container_cwd,
                timeout=timeout,
                output=output,
            )
        )
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def repl(
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", envvar="VEDEX_MODEL", help="LiteLLM model name."),
    ] = None,
    environment: Annotated[
        str,
        typer.Option("--environment", "-e", help="Execution environment: local or docker."),
    ] = "local",
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Local workspace directory."),
    ] = None,
    image: Annotated[
        str | None,
        typer.Option("--image", help="Docker image when using the Docker environment."),
    ] = None,
    container_cwd: Annotated[
        str,
        typer.Option("--container-cwd", help="Working directory inside the container."),
    ] = "/",
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Command timeout in seconds."),
    ] = 30.0,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Trajectory output path."),
    ] = None,
) -> None:
    """Run repeated tasks in a simple terminal prompt."""
    try:
        asyncio.run(
            _run_repl(
                model_name=_require_model(model),
                environment_name=environment,
                cwd=cwd,
                image=image,
                container_cwd=container_cwd,
                timeout=timeout,
                output=output,
            )
        )
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


async def _run_task(
    *,
    task: str,
    model_name: str,
    environment_name: str,
    cwd: Path | None,
    image: str | None,
    container_cwd: str,
    timeout: float,
    output: Path | None,
) -> dict[str, Any]:
    environment = _build_environment(
        environment_name,
        cwd=cwd,
        image=image,
        container_cwd=container_cwd,
        timeout=timeout,
    )
    try:
        trajectory = await _run_agent(
            model=LiteLLMModel(model_name),
            environment=environment,
            task=task,
            output=output,
        )
        _print_result(trajectory, output)
        return trajectory
    finally:
        _cleanup(environment)


async def _run_repl(
    *,
    model_name: str,
    environment_name: str,
    cwd: Path | None,
    image: str | None,
    container_cwd: str,
    timeout: float,
    output: Path | None,
) -> None:
    environment = _build_environment(
        environment_name,
        cwd=cwd,
        image=image,
        container_cwd=container_cwd,
        timeout=timeout,
    )
    model = LiteLLMModel(model_name)
    turn = 0
    try:
        while True:
            try:
                task = input("vedex> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                typer.echo()
                continue

            if not task:
                continue
            if task in {"/exit", "/quit", "exit", "quit"}:
                break
            if task == "/help":
                typer.echo("Enter a task, or use /exit to quit.")
                continue

            turn += 1
            try:
                trajectory = await _run_agent(
                    model=model,
                    environment=environment,
                    task=task,
                    output=_repl_output(output, turn),
                )
                _print_result(trajectory, _repl_output(output, turn))
            except (OSError, RuntimeError, ValueError) as exc:
                typer.echo(f"Error: {exc}", err=True)
    finally:
        _cleanup(environment)


async def _run_agent(
    *,
    model: LiteLLMModel,
    environment: Any,
    task: str,
    output: Path | None,
) -> dict[str, Any]:
    agent = DefaultAgent(
        model,
        environment,
        config=AgentConfig(output_path=output),
    )
    return await agent.run(task)


def _build_environment(
    name: str,
    *,
    cwd: Path | None,
    image: str | None,
    container_cwd: str,
    timeout: float,
) -> LocalEnvironment | DockerEnvironment:
    if timeout <= 0:
        raise ValueError("--timeout must be greater than zero")

    normalized = name.casefold()
    if normalized == "local":
        workspace = (cwd or Path.cwd()).expanduser().resolve(strict=False)
        return LocalEnvironment(cwd=str(workspace), timeout=timeout)
    if normalized == "docker":
        if not image:
            raise ValueError("--image is required with --environment docker")
        return DockerEnvironment(image=image, cwd=container_cwd, timeout=timeout)
    raise ValueError(f"Unknown environment {name!r}; use 'local' or 'docker'")


def _require_model(model: str | None) -> str:
    if model and model.strip():
        return model.strip()
    raise ValueError("--model or VEDEX_MODEL is required")


def _cleanup(environment: Any) -> None:
    cleanup = getattr(environment, "cleanup", None)
    if callable(cleanup):
        cleanup()


def _repl_output(output: Path | None, turn: int) -> Path | None:
    if output is None or turn == 1:
        return output
    suffix = output.suffix or ".json"
    return output.with_name(f"{output.stem}-{turn}{suffix}")


def _print_result(trajectory: dict[str, Any], output: Path | None) -> None:
    _print_trace(trajectory.get("messages", []))
    info = trajectory.get("info", {})
    status = info.get("exit_status", "unknown")
    submission = info.get("submission", "")
    typer.echo(f"\n[{status}]")
    if submission:
        typer.echo(str(submission))
    if output is not None:
        typer.echo(f"Trajectory: {output}")


def _print_trace(messages: Any) -> None:
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            actions = message.get("extra", {}).get("actions", [])
            for action in actions if isinstance(actions, list) else []:
                if isinstance(action, dict) and action.get("command"):
                    typer.echo(f"$ {action['command']}")
        if message.get("role") == "tool":
            raw_output = message.get("extra", {}).get("raw_output", "")
            if raw_output:
                typer.echo(str(raw_output).rstrip())


__all__ = ["app", "repl", "run"]
