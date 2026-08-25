from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from contextlib import aclosing
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from time import monotonic
from typing import Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field

from .agent import AgentLimits, AgentRunResult
from .artifacts import (
    RunArtifactConfig,
    RunArtifactRecorder,
    RunStatus,
    RunTiming,
)
from .environments import Environment, WorkspaceExport, WorkspacePatch
from .models import ModelAdapter, ModelSettings, Usage
from .resources import ProjectContextFile, ResourcePaths
from .runtime import AppRuntime
from .schema import (
    AgentEvent,
    AgentMessage,
    AssistantMessage,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
)

type HeadlessOutputMode = Literal["plain", "jsonl"]


class HeadlessExitCode(IntEnum):
    COMPLETED = 0
    MODEL_FAILURE = 10
    MALFORMED_MODEL_STREAM = 11
    TURN_LIMIT = 20
    TOOL_CALL_LIMIT = 21
    CONTEXT_LIMIT = 22
    FATAL_ENVIRONMENT_FAILURE = 30
    RUNTIME_FAILURE = 40
    CLEANUP_FAILURE = 41
    ARTIFACT_FAILURE = 42
    TIMED_OUT = 124
    CANCELLED = 130


_EXIT_CODES: dict[RunStatus, HeadlessExitCode] = {
    "completed": HeadlessExitCode.COMPLETED,
    "model_failure": HeadlessExitCode.MODEL_FAILURE,
    "cancelled": HeadlessExitCode.CANCELLED,
    "timed_out": HeadlessExitCode.TIMED_OUT,
    "turn_limit": HeadlessExitCode.TURN_LIMIT,
    "tool_call_limit": HeadlessExitCode.TOOL_CALL_LIMIT,
    "context_limit": HeadlessExitCode.CONTEXT_LIMIT,
    "malformed_model_stream": HeadlessExitCode.MALFORMED_MODEL_STREAM,
    "fatal_environment_failure": HeadlessExitCode.FATAL_ENVIRONMENT_FAILURE,
    "runtime_failure": HeadlessExitCode.RUNTIME_FAILURE,
    "cleanup_failure": HeadlessExitCode.CLEANUP_FAILURE,
    "artifact_failure": HeadlessExitCode.ARTIFACT_FAILURE,
}


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: RunStatus
    exit_code: int = Field(ge=0, le=255)
    timing: RunTiming
    usage: Usage = Field(default_factory=Usage)
    turns: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    error: str | None = None
    messages: list[AgentMessage] = Field(default_factory=list)
    artifact_path: str | None = None


class HeadlessEventWriter:
    """Render normalized events without mixing diagnostics into machine output."""

    def __init__(
        self,
        *,
        mode: HeadlessOutputMode,
        stdout: TextIO,
        stderr: TextIO,
    ) -> None:
        self._mode = mode
        self._stdout = stdout
        self._stderr = stderr
        self._assistant_open = False
        self._assistant_wrote_delta = False
        self._last_plain_character = ""

    def write_event(self, event: AgentEvent) -> None:
        if self._mode == "jsonl":
            self._stdout.write(f"{event.model_dump_json()}\n")
            self._stdout.flush()
        else:
            self._write_plain_event(event)

        if isinstance(event, ErrorEvent):
            self.write_diagnostic(event.message)

    def write_diagnostic(self, message: str) -> None:
        self._stderr.write(f"vedex: {message}\n")
        self._stderr.flush()

    def finish(self) -> None:
        if self._mode == "plain" and self._assistant_open:
            self._finish_assistant_message()

    def _write_plain_event(self, event: AgentEvent) -> None:
        if isinstance(event, MessageStartEvent) and event.message_role == "assistant":
            self._assistant_open = True
            self._assistant_wrote_delta = False
            self._last_plain_character = ""
            return

        if isinstance(event, MessageDeltaEvent):
            self._assistant_open = True
            self._assistant_wrote_delta = self._assistant_wrote_delta or bool(event.delta)
            self._write_plain(event.delta)
            return

        if isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage):
            if not self._assistant_wrote_delta and event.message.content:
                self._write_plain(event.message.content)
            self._finish_assistant_message()

    def _write_plain(self, value: str) -> None:
        if not value:
            return
        self._stdout.write(value)
        self._stdout.flush()
        self._last_plain_character = value[-1]

    def _finish_assistant_message(self) -> None:
        if self._last_plain_character and self._last_plain_character != "\n":
            self._stdout.write("\n")
            self._stdout.flush()
        self._assistant_open = False
        self._assistant_wrote_delta = False
        self._last_plain_character = ""


async def run_headless(
    *,
    task: str,
    adapter: ModelAdapter,
    environment: Environment,
    settings: ModelSettings,
    workspace: Path,
    limits: AgentLimits | None = None,
    resource_paths: ResourcePaths | None = None,
    custom_system_prompt: str | None = None,
    append_system_prompt: str | None = None,
    context_files: Sequence[ProjectContextFile] = (),
    output_mode: HeadlessOutputMode = "plain",
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    artifact: RunArtifactConfig | None = None,
    benchmark_instance_id: str | None = None,
) -> RunResult:
    """Run one ephemeral task through AppRuntime and return a process-ready result."""

    resolved_limits = limits or AgentLimits()
    output = HeadlessEventWriter(
        mode=output_mode,
        stdout=stdout if stdout is not None else sys.stdout,
        stderr=stderr if stderr is not None else sys.stderr,
    )
    runtime = AppRuntime(
        adapter=adapter,
        environment=environment,
        settings=settings,
        workspace_path=workspace,
        limits=resolved_limits,
        resource_paths=resource_paths,
        custom_system_prompt=custom_system_prompt,
        append_system_prompt=append_system_prompt,
        context_files=context_files,
    )
    recorder = (
        RunArtifactRecorder(
            task=task,
            benchmark_instance_id=benchmark_instance_id,
            adapter=adapter,
            settings=settings,
            agent_limits=resolved_limits,
            environment_limits=environment.limits,
        )
        if artifact is not None
        else None
    )

    started_at = datetime.now(UTC)
    started_monotonic = monotonic()
    status: RunStatus = "runtime_failure"
    error: str | None = None
    usage = Usage()
    turns = 0
    tool_calls = 0
    messages: list[AgentMessage] = []
    patch: WorkspacePatch | None = None
    workspace_export: WorkspaceExport | None = None
    artifact_errors: list[str] = []
    artifact_path: Path | None = None

    interrupted: BaseException | None = None
    try:
        try:
            await runtime.start()
            if recorder is not None:
                try:
                    recorder.capture_environment(await environment.get_metadata())
                except Exception as exc:
                    detail = _error_detail(exc)
                    artifact_errors.append(f"Could not capture environment metadata: {detail}")
                try:
                    recorder.capture_resources(runtime.workspace)
                except Exception as exc:
                    detail = _error_detail(exc)
                    artifact_errors.append(f"Could not capture resource metadata: {detail}")

            async with aclosing(runtime.prompt(task)) as event_stream:
                async for event in event_stream:
                    if recorder is not None:
                        recorder.record(event)
                    output.write_event(event)

            agent_result = runtime.agent.last_result
            if agent_result is None:
                raise RuntimeError("Agent stream ended without a terminal result")
            status, error, usage, turns, tool_calls = _from_agent_result(agent_result)
            messages = [message.model_copy(deep=True) for message in runtime.agent.messages]
        except asyncio.CancelledError:
            runtime.cancel()
            status = "cancelled"
            error = "Headless run cancelled"
            output.write_diagnostic(error)
            if runtime.state == "started":
                messages = [message.model_copy(deep=True) for message in runtime.agent.messages]
                usage = runtime.agent.usage
        except KeyboardInterrupt:
            runtime.cancel()
            status = "cancelled"
            error = "Headless run interrupted"
            output.write_diagnostic(error)
            if runtime.state == "started":
                messages = [message.model_copy(deep=True) for message in runtime.agent.messages]
                usage = runtime.agent.usage
        except Exception as exc:
            status = "runtime_failure"
            error = _error_detail(exc)
            output.write_diagnostic(error)
            if runtime.state == "started":
                messages = [message.model_copy(deep=True) for message in runtime.agent.messages]
                usage = runtime.agent.usage

        if recorder is not None and runtime.state == "started":
            try:
                if artifact is None:
                    raise AssertionError("Artifact configuration unexpectedly missing")
                if artifact.submission == "patch":
                    patch = await environment.collect_patch()
                else:
                    workspace_export = await environment.export_workspace(
                        artifact.resolved_export_path()
                    )
            except Exception as exc:
                detail = f"Could not collect run submission: {_error_detail(exc)}"
                artifact_errors.append(detail)
    except BaseException as exc:
        interrupted = exc
        runtime.cancel()
    finally:
        try:
            await runtime.close()
        except BaseException as close_error:
            if interrupted is not None:
                raise BaseExceptionGroup(
                    "Headless run and environment cleanup both failed",
                    [interrupted, close_error],
                ) from None
            if not isinstance(close_error, Exception):
                raise
            detail = f"Environment cleanup failed: {_error_detail(close_error)}"
            status = "cleanup_failure"
            error = _combine_errors(error, detail)
            output.write_diagnostic(detail)

    if interrupted is not None:
        raise interrupted

    if artifact_errors:
        if status != "cleanup_failure":
            status = "artifact_failure"
        for detail in artifact_errors:
            error = _combine_errors(error, detail)
            output.write_diagnostic(detail)

    output.finish()
    finished_at = datetime.now(UTC)
    timing = RunTiming(
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=monotonic() - started_monotonic,
    )

    if recorder is not None:
        try:
            if artifact is None:
                raise AssertionError("Artifact configuration unexpectedly missing")
            run_artifact = recorder.finalize(
                timing=timing,
                messages=messages,
                usage=usage,
                turns=turns,
                tool_calls=tool_calls,
                status=status,
                error=error,
                patch=patch,
                workspace_export=workspace_export,
            )
            artifact_path = await recorder.write(artifact.path, run_artifact)
        except Exception as exc:
            detail = f"Could not write run artifact: {_error_detail(exc)}"
            status = "artifact_failure"
            error = _combine_errors(error, detail)
            output.write_diagnostic(detail)
            artifact_path = None

    return RunResult(
        status=status,
        exit_code=int(exit_code_for_status(status)),
        timing=timing,
        usage=usage,
        turns=turns,
        tool_calls=tool_calls,
        error=error,
        messages=messages,
        artifact_path=str(artifact_path) if artifact_path is not None else None,
    )


def exit_code_for_status(status: RunStatus) -> HeadlessExitCode:
    return _EXIT_CODES[status]


def _from_agent_result(result: AgentRunResult) -> tuple[RunStatus, str | None, Usage, int, int]:
    return (
        result.status,
        result.message,
        result.usage.model_copy(deep=True),
        result.turns,
        result.tool_calls,
    )


def _error_detail(error: BaseException) -> str:
    return str(error) or type(error).__name__


def _combine_errors(current: str | None, additional: str) -> str:
    return f"{current}; {additional}" if current else additional


__all__ = [
    "HeadlessEventWriter",
    "HeadlessExitCode",
    "HeadlessOutputMode",
    "RunResult",
    "exit_code_for_status",
    "run_headless",
]
