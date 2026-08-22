from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from io import StringIO
from pathlib import Path

import pytest
from vedex.artifacts import RunStatus
from vedex.environments import LocalEnvironment
from vedex.headless import HeadlessExitCode, RunResult, exit_code_for_status, run_headless
from vedex.models import (
    FakeAdapter,
    ModelCompletedEvent,
    ModelEvent,
    ModelFailureEvent,
    ModelRequest,
    ModelSettings,
    ModelStartEvent,
    ModelTextDeltaEvent,
    Usage,
)
from vedex.schema import AssistantMessage, CancellationToken

from .conftest import run_async


class _CleanupFailureEnvironment(LocalEnvironment):
    async def stop(self) -> None:
        await super().stop()
        raise RuntimeError("cleanup exploded")


class _BlockingAdapter:
    def stream(
        self,
        request: ModelRequest,
        *,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ModelEvent]:
        del request

        async def events() -> AsyncIterator[ModelEvent]:
            yield ModelStartEvent()
            while signal is None or not signal.is_cancelled():
                await asyncio.sleep(1)

        return events()


def _text_stream(*deltas: str, content: str | None = None) -> list[ModelEvent]:
    completed_content = content if content is not None else "".join(deltas)
    return [
        ModelStartEvent(),
        *(ModelTextDeltaEvent(delta=delta) for delta in deltas),
        ModelCompletedEvent(
            message=AssistantMessage(content=completed_content),
            usage=Usage(input_tokens=7, output_tokens=3, cached_tokens=2),
        ),
    ]


def test_headless_plain_mode_streams_only_assistant_text(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()
    adapter = FakeAdapter([_text_stream("Hello", " world")])
    environment = LocalEnvironment(tmp_path)

    result = run_async(
        run_headless(
            task="Say hello",
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
    assert result.error is None
    assert result.usage == Usage(input_tokens=7, output_tokens=3, cached_tokens=2)
    assert result.turns == 1
    assert result.tool_calls == 0
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert stdout.getvalue() == "Hello world\n"
    assert stderr.getvalue() == ""
    assert environment.state == "stopped"
    assert adapter.requests[0].messages[0].content == "Say hello"
    assert [tool.name for tool in adapter.requests[0].tools] == ["read", "write", "edit", "bash"]


def test_headless_plain_mode_falls_back_to_completed_message_without_deltas(
    tmp_path: Path,
) -> None:
    stdout = StringIO()
    result = run_async(
        run_headless(
            task="task",
            adapter=FakeAdapter([_text_stream(content="complete only")]),
            environment=LocalEnvironment(tmp_path),
            settings=ModelSettings(model="fake"),
            workspace=tmp_path,
            stdout=stdout,
            stderr=StringIO(),
        )
    )

    assert result.status == "completed"
    assert stdout.getvalue() == "complete only\n"


def test_headless_jsonl_keeps_events_on_stdout_and_diagnostics_on_stderr(
    tmp_path: Path,
) -> None:
    stdout = StringIO()
    stderr = StringIO()
    result = run_async(
        run_headless(
            task="fail",
            adapter=FakeAdapter(
                [[ModelFailureEvent(kind="unavailable", message="model unavailable")]]
            ),
            environment=LocalEnvironment(tmp_path),
            settings=ModelSettings(model="fake"),
            workspace=tmp_path,
            output_mode="jsonl",
            stdout=stdout,
            stderr=stderr,
        )
    )

    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert result.status == "model_failure"
    assert result.exit_code == HeadlessExitCode.MODEL_FAILURE
    assert records[0] == {"type": "agent_start"}
    assert records[-1]["type"] == "agent_end"
    assert records[-1]["status"] == "model_failure"
    assert any(record["type"] == "error" for record in records)
    assert all("vedex:" not in line for line in stdout.getvalue().splitlines())
    assert stderr.getvalue() == "vedex: model unavailable\n"


def test_headless_returns_runtime_failure_when_environment_cannot_start(tmp_path: Path) -> None:
    missing_workspace = tmp_path / "missing"
    environment = LocalEnvironment(missing_workspace)
    stdout = StringIO()
    stderr = StringIO()

    result = run_async(
        run_headless(
            task="task",
            adapter=FakeAdapter(),
            environment=environment,
            settings=ModelSettings(model="fake"),
            workspace=missing_workspace,
            output_mode="jsonl",
            stdout=stdout,
            stderr=stderr,
        )
    )

    assert result.status == "runtime_failure"
    assert result.exit_code == HeadlessExitCode.RUNTIME_FAILURE
    assert result.messages == []
    assert stdout.getvalue() == ""
    assert "Workspace does not exist" in stderr.getvalue()
    assert environment.state == "stopped"


def test_headless_cleanup_failure_cannot_report_success(tmp_path: Path) -> None:
    stderr = StringIO()
    result = run_async(
        run_headless(
            task="task",
            adapter=FakeAdapter([_text_stream("done")]),
            environment=_CleanupFailureEnvironment(tmp_path),
            settings=ModelSettings(model="fake"),
            workspace=tmp_path,
            stdout=StringIO(),
            stderr=stderr,
        )
    )

    assert result.status == "cleanup_failure"
    assert result.exit_code == HeadlessExitCode.CLEANUP_FAILURE
    assert result.error == "Environment cleanup failed: cleanup exploded"
    assert stderr.getvalue() == "vedex: Environment cleanup failed: cleanup exploded\n"


def test_headless_task_cancellation_returns_stable_result_and_closes_environment(
    tmp_path: Path,
) -> None:
    environment = LocalEnvironment(tmp_path)
    stderr = StringIO()

    async def exercise() -> RunResult:
        task = asyncio.create_task(
            run_headless(
                task="wait",
                adapter=_BlockingAdapter(),
                environment=environment,
                settings=ModelSettings(model="fake"),
                workspace=tmp_path,
                output_mode="jsonl",
                stdout=StringIO(),
                stderr=stderr,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        return await task

    result = run_async(exercise())

    assert result.status == "cancelled"
    assert result.exit_code == HeadlessExitCode.CANCELLED
    assert result.error == "Headless run cancelled"
    assert stderr.getvalue() == "vedex: Headless run cancelled\n"
    assert environment.state == "stopped"


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [
        ("completed", HeadlessExitCode.COMPLETED),
        ("model_failure", HeadlessExitCode.MODEL_FAILURE),
        ("malformed_model_stream", HeadlessExitCode.MALFORMED_MODEL_STREAM),
        ("turn_limit", HeadlessExitCode.TURN_LIMIT),
        ("tool_call_limit", HeadlessExitCode.TOOL_CALL_LIMIT),
        ("context_limit", HeadlessExitCode.CONTEXT_LIMIT),
        ("fatal_environment_failure", HeadlessExitCode.FATAL_ENVIRONMENT_FAILURE),
        ("runtime_failure", HeadlessExitCode.RUNTIME_FAILURE),
        ("cleanup_failure", HeadlessExitCode.CLEANUP_FAILURE),
        ("artifact_failure", HeadlessExitCode.ARTIFACT_FAILURE),
        ("timed_out", HeadlessExitCode.TIMED_OUT),
        ("cancelled", HeadlessExitCode.CANCELLED),
    ],
)
def test_headless_exit_codes_are_explicit_and_stable(
    status: RunStatus,
    exit_code: HeadlessExitCode,
) -> None:
    assert exit_code_for_status(status) == exit_code
