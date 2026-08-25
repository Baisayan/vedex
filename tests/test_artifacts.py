from __future__ import annotations

import shutil
import subprocess
import tarfile
from io import StringIO
from pathlib import Path
from typing import override

import pytest
from vedex.artifacts import RUN_ARTIFACT_SCHEMA_VERSION, RunArtifact, RunArtifactConfig
from vedex.environments import EnvironmentMetadata, LocalEnvironment
from vedex.headless import HeadlessExitCode, run_headless
from vedex.models import (
    FakeAdapter,
    ModelCompletedEvent,
    ModelEvent,
    ModelFailureEvent,
    ModelSettings,
    ModelStartEvent,
    ModelTextDeltaEvent,
    Usage,
)
from vedex.resources import ResourcePaths
from vedex.schema import AssistantMessage, ToolCall

from .conftest import run_async


class _MetadataFailureEnvironment(LocalEnvironment):
    @override
    async def get_metadata(self) -> EnvironmentMetadata:
        raise RuntimeError("metadata unavailable")


def _run_git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _final_stream(content: str = "finished", *, usage: Usage | None = None) -> list[ModelEvent]:
    return [
        ModelStartEvent(),
        ModelTextDeltaEvent(delta=content),
        ModelCompletedEvent(
            message=AssistantMessage(content=content),
            usage=usage or Usage(input_tokens=3, output_tokens=2),
        ),
    ]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_run_artifact_records_reproducible_patch_trajectory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    global_root = tmp_path / "global"
    artifact_path = tmp_path / "artifacts" / "instance.run.json"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (project / "AGENTS.md").write_text("Follow project rules.", encoding="utf-8")
    (project / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    (global_root / "skills" / "review").mkdir(parents=True)
    (global_root / "prompts").mkdir()
    (global_root / "skills" / "review" / "SKILL.md").write_text(
        "---\ndescription: Review carefully\n---\nReview instructions.",
        encoding="utf-8",
    )
    (global_root / "prompts" / "check.md").write_text(
        "Check {{ arguments }}",
        encoding="utf-8",
    )
    _run_git(project, "init", "--quiet")
    _run_git(project, "add", ".")
    _run_git(
        project,
        "-c",
        "user.name=Vedex Tests",
        "-c",
        "user.email=vedex@example.test",
        "commit",
        "--quiet",
        "-m",
        "baseline",
    )

    tool_stream: list[ModelEvent] = [
        ModelStartEvent(),
        ModelCompletedEvent(
            message=AssistantMessage(
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write",
                        arguments={"path": "solution.txt", "content": "solved\n"},
                    )
                ]
            ),
            usage=Usage(input_tokens=5, output_tokens=1, thinking_tokens=2),
        ),
    ]
    adapter = FakeAdapter(
        [
            tool_stream,
            _final_stream(usage=Usage(input_tokens=7, output_tokens=3, cached_tokens=4)),
        ]
    )
    stdout = StringIO()
    stderr = StringIO()

    result = run_async(
        run_headless(
            task="Create the solution",
            benchmark_instance_id="owner__repo__1",
            adapter=adapter,
            environment=LocalEnvironment(project),
            settings=ModelSettings(model="fake-model", options={"temperature": 0}),
            workspace=project,
            resource_paths=ResourcePaths(root=global_root),
            output_mode="jsonl",
            stdout=stdout,
            stderr=stderr,
            artifact=RunArtifactConfig(path=artifact_path),
        )
    )

    artifact = RunArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    assert result.status == "completed"
    assert result.exit_code == HeadlessExitCode.COMPLETED
    assert result.artifact_path == str(artifact_path.resolve())
    assert stderr.getvalue() == ""
    assert artifact.schema_version == RUN_ARTIFACT_SCHEMA_VERSION
    assert artifact.task == "Create the solution"
    assert artifact.benchmark_instance_id == "owner__repo__1"
    assert artifact.adapter.adapter_type == "vedex.models.fake.FakeAdapter"
    assert artifact.adapter.settings == ModelSettings(
        model="fake-model", options={"temperature": 0}
    )
    assert artifact.environment is not None
    assert artifact.environment.environment_type == "local"
    assert artifact.environment.git_revision is not None
    assert artifact.limits.agent.context_reserve_tokens == 0
    assert artifact.limits.environment.max_command_seconds == 600
    assert artifact.timing.finished_at >= artifact.timing.started_at
    assert artifact.timing.duration_seconds >= 0
    assert artifact.resources is not None
    assert len(artifact.resources.aggregate_sha256) == 64
    assert len(artifact.resources.system_prompt_sha256) == 64
    assert {entry.kind for entry in artifact.resources.entries} == {
        "project_instruction",
        "skill",
        "prompt_template",
    }
    assert [message.role for message in artifact.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert artifact.events[0].type == "agent_start"
    assert artifact.events[-1].type == "agent_end"
    assert artifact.usage == Usage(
        input_tokens=12,
        output_tokens=4,
        cached_tokens=4,
        thinking_tokens=2,
    )
    assert artifact.turns == 2
    assert artifact.tool_calls == 1
    assert artifact.status == "completed"
    assert artifact.error is None
    assert artifact.patch is not None
    assert artifact.patch.changed is True
    assert "solution.txt" in artifact.patch.text
    assert "+solved" in artifact.patch.text
    assert artifact.workspace_export is None
    assert (project / "solution.txt").read_text(encoding="utf-8") == "solved\n"


def test_run_artifact_can_export_workspace_submission(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "answer.txt").write_text("answer", encoding="utf-8")
    artifact_path = tmp_path / "outputs" / "export.run.json"

    result = run_async(
        run_headless(
            task="Inspect",
            adapter=FakeAdapter([_final_stream()]),
            environment=LocalEnvironment(project),
            settings=ModelSettings(model="fake"),
            workspace=project,
            stdout=StringIO(),
            stderr=StringIO(),
            artifact=RunArtifactConfig(path=artifact_path, submission="export"),
        )
    )

    artifact = RunArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    expected_export = artifact_path.with_name("export.run.submission.tar.gz").resolve()
    assert result.status == "completed"
    assert artifact.patch is None
    assert artifact.workspace_export is not None
    assert artifact.workspace_export.path == str(expected_export)
    assert expected_export.is_file()
    with tarfile.open(expected_export, "r:gz") as archive:
        assert "answer.txt" in archive.getnames()


def test_terminal_failure_still_writes_inspectable_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "failed.run.json"
    stderr = StringIO()
    result = run_async(
        run_headless(
            task="Fail predictably",
            adapter=FakeAdapter([[ModelFailureEvent(message="expected failure")]]),
            environment=LocalEnvironment(tmp_path),
            settings=ModelSettings(model="fake"),
            workspace=tmp_path,
            output_mode="jsonl",
            stdout=StringIO(),
            stderr=stderr,
            artifact=RunArtifactConfig(path=artifact_path),
        )
    )

    artifact = RunArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    assert result.status == "model_failure"
    assert artifact.status == "model_failure"
    assert artifact.error == "expected failure"
    assert artifact.messages[0].role == "user"
    assert artifact.events[-1].type == "agent_end"
    assert artifact.patch is not None
    assert stderr.getvalue() == "vedex: expected failure\n"


def test_metadata_failure_is_recorded_without_losing_the_trajectory(tmp_path: Path) -> None:
    artifact_path = tmp_path / "metadata-failed.run.json"
    result = run_async(
        run_headless(
            task="Complete despite metadata failure",
            adapter=FakeAdapter([_final_stream()]),
            environment=_MetadataFailureEnvironment(tmp_path),
            settings=ModelSettings(model="fake"),
            workspace=tmp_path,
            stdout=StringIO(),
            stderr=StringIO(),
            artifact=RunArtifactConfig(path=artifact_path),
        )
    )

    artifact = RunArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    assert result.status == "artifact_failure"
    assert result.exit_code == HeadlessExitCode.ARTIFACT_FAILURE
    assert artifact.status == "artifact_failure"
    assert artifact.environment is None
    assert artifact.resources is not None
    assert artifact.messages[-1].role == "assistant"
    assert artifact.events[-1].type == "agent_end"
    assert artifact.error is not None
    assert "metadata unavailable" in artifact.error


def test_atomic_artifact_write_failure_preserves_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "existing.run.json"
    artifact_path.write_text("old artifact", encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("vedex.artifacts.os.replace", fail_replace)
    stderr = StringIO()
    result = run_async(
        run_headless(
            task="task",
            adapter=FakeAdapter([_final_stream()]),
            environment=LocalEnvironment(tmp_path),
            settings=ModelSettings(model="fake"),
            workspace=tmp_path,
            stdout=StringIO(),
            stderr=stderr,
            artifact=RunArtifactConfig(path=artifact_path),
        )
    )

    assert result.status == "artifact_failure"
    assert result.artifact_path is None
    assert artifact_path.read_text(encoding="utf-8") == "old artifact"
    assert not list(tmp_path.glob(".existing.run.json.*.tmp"))
    assert "Could not write run artifact: replace failed" in stderr.getvalue()


def test_artifact_config_rejects_export_path_for_patch_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="submission='export'"):
        RunArtifactConfig(
            path=tmp_path / "run.json",
            submission="patch",
            export_path=tmp_path / "submission.tar.gz",
        )
