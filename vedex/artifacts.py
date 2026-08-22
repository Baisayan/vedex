from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .agent import AgentLimits
from .environments import (
    EnvironmentLimits,
    EnvironmentMetadata,
    WorkspaceExport,
    WorkspacePatch,
)
from .models import ModelAdapter, ModelSettings, Usage
from .schema import AgentEvent, AgentMessage, AgentStatus
from .workspace import Workspace

RUN_ARTIFACT_SCHEMA_VERSION: Literal["vedex.run-artifact.v1"] = "vedex.run-artifact.v1"

type RuntimeFailureStatus = Literal["runtime_failure", "cleanup_failure", "artifact_failure"]
type RunStatus = AgentStatus | RuntimeFailureStatus
type SubmissionMode = Literal["patch", "export"]
type ResourceKind = Literal["project_instruction", "skill", "prompt_template"]


class _ArtifactContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AgentLimitsSnapshot(_ArtifactContract):
    max_turns: int | None = None
    max_tool_calls: int | None = None
    time_limit_seconds: float | None = None
    max_context_tokens: int | None = None
    context_reserve_tokens: int = 0

    @classmethod
    def from_limits(cls, limits: AgentLimits) -> AgentLimitsSnapshot:
        return cls(
            max_turns=limits.max_turns,
            max_tool_calls=limits.max_tool_calls,
            time_limit_seconds=limits.time_limit_seconds,
            max_context_tokens=limits.max_context_tokens,
            context_reserve_tokens=limits.context_reserve_tokens,
        )


class RunLimitsSnapshot(_ArtifactContract):
    agent: AgentLimitsSnapshot
    environment: EnvironmentLimits


class AdapterSnapshot(_ArtifactContract):
    adapter_type: str = Field(min_length=1)
    settings: ModelSettings


class ResourceHash(_ArtifactContract):
    kind: ResourceKind
    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResourceManifest(_ArtifactContract):
    aggregate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: list[ResourceHash] = Field(default_factory=list)


class RunTiming(_ArtifactContract):
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)


class RunArtifact(_ArtifactContract):
    schema_version: Literal["vedex.run-artifact.v1"] = RUN_ARTIFACT_SCHEMA_VERSION
    task: str
    benchmark_instance_id: str | None = None
    adapter: AdapterSnapshot
    environment: EnvironmentMetadata | None = None
    limits: RunLimitsSnapshot
    timing: RunTiming
    resources: ResourceManifest | None = None
    messages: list[AgentMessage] = Field(default_factory=list)
    events: list[AgentEvent] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    turns: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    status: RunStatus
    error: str | None = None
    patch: WorkspacePatch | None = None
    workspace_export: WorkspaceExport | None = None


@dataclass(frozen=True, slots=True)
class RunArtifactConfig:
    path: Path
    submission: SubmissionMode = "patch"
    export_path: Path | None = None

    def __post_init__(self) -> None:
        if self.submission not in ("patch", "export"):
            raise ValueError(f"Unsupported artifact submission mode: {self.submission}")
        if self.submission == "patch" and self.export_path is not None:
            raise ValueError("export_path requires submission='export'")

    def resolved_export_path(self) -> Path:
        if self.submission != "export":
            raise ValueError("An export path is only available for export submissions")
        if self.export_path is not None:
            return self.export_path.expanduser().resolve(strict=False)
        artifact_path = self.path.expanduser().resolve(strict=False)
        return artifact_path.with_name(f"{artifact_path.stem}.submission.tar.gz")


class RunArtifactRecorder:
    """Collect normalized run state and serialize it independently of the Agent."""

    def __init__(
        self,
        *,
        task: str,
        adapter: ModelAdapter,
        settings: ModelSettings,
        agent_limits: AgentLimits,
        environment_limits: EnvironmentLimits,
        benchmark_instance_id: str | None = None,
    ) -> None:
        self._task = task
        self._benchmark_instance_id = benchmark_instance_id
        adapter_class = type(adapter)
        self._adapter = AdapterSnapshot(
            adapter_type=f"{adapter_class.__module__}.{adapter_class.__qualname__}",
            settings=settings.model_copy(deep=True),
        )
        self._limits = RunLimitsSnapshot(
            agent=AgentLimitsSnapshot.from_limits(agent_limits),
            environment=environment_limits.model_copy(deep=True),
        )
        self._environment: EnvironmentMetadata | None = None
        self._resources: ResourceManifest | None = None
        self._events: list[AgentEvent] = []

    @property
    def events(self) -> tuple[AgentEvent, ...]:
        return tuple(event.model_copy(deep=True) for event in self._events)

    def record(self, event: AgentEvent) -> None:
        self._events.append(event.model_copy(deep=True))

    def capture_environment(self, metadata: EnvironmentMetadata) -> None:
        self._environment = metadata.model_copy(deep=True)

    def capture_resources(self, workspace: Workspace) -> None:
        self._resources = build_resource_manifest(workspace)

    def finalize(
        self,
        *,
        timing: RunTiming,
        messages: Sequence[AgentMessage],
        usage: Usage,
        turns: int,
        tool_calls: int,
        status: RunStatus,
        error: str | None,
        patch: WorkspacePatch | None = None,
        workspace_export: WorkspaceExport | None = None,
    ) -> RunArtifact:
        if patch is not None and workspace_export is not None:
            raise ValueError("A run artifact cannot contain both patch and export submissions")
        return RunArtifact(
            task=self._task,
            benchmark_instance_id=self._benchmark_instance_id,
            adapter=self._adapter,
            environment=self._environment,
            limits=self._limits,
            timing=timing,
            resources=self._resources,
            messages=[message.model_copy(deep=True) for message in messages],
            events=[event.model_copy(deep=True) for event in self._events],
            usage=usage.model_copy(deep=True),
            turns=turns,
            tool_calls=tool_calls,
            status=status,
            error=error,
            patch=patch,
            workspace_export=workspace_export,
        )

    async def write(self, path: Path, artifact: RunArtifact) -> Path:
        resolved = path.expanduser().resolve(strict=False)
        serialized = f"{artifact.model_dump_json(indent=2)}\n"
        await asyncio.to_thread(_write_text_atomic, resolved, serialized)
        return resolved


def build_resource_manifest(workspace: Workspace) -> ResourceManifest:
    entries: list[ResourceHash] = []
    for context_file in workspace.context_files:
        entries.append(
            _resource_hash(
                kind="project_instruction",
                name=Path(context_file.path).name or context_file.path,
                source=context_file.path,
                payload={"content": context_file.content},
            )
        )
    for skill in workspace.skills:
        entries.append(
            _resource_hash(
                kind="skill",
                name=skill.name,
                source=str(skill.path),
                payload={"content": skill.content, "description": skill.description},
            )
        )
    for template in workspace.prompt_templates:
        entries.append(
            _resource_hash(
                kind="prompt_template",
                name=template.name,
                source=str(template.path),
                payload={"content": template.content, "description": template.description},
            )
        )

    entries.sort(key=lambda entry: (entry.kind, entry.name.casefold(), entry.source))
    aggregate_payload = [entry.model_dump(mode="json") for entry in entries]
    aggregate = _sha256_text(_canonical_json(aggregate_payload))
    return ResourceManifest(
        aggregate_sha256=aggregate,
        system_prompt_sha256=_sha256_text(workspace.system_prompt),
        entries=entries,
    )


def _resource_hash(
    *,
    kind: ResourceKind,
    name: str,
    source: str,
    payload: dict[str, str | None],
) -> ResourceHash:
    normalized_source = source.replace("\\", "/")
    digest_payload = {"name": name, "source": normalized_source, **payload}
    return ResourceHash(
        kind=kind,
        name=name,
        source=normalized_source,
        sha256=_sha256_text(_canonical_json(digest_payload)),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


__all__ = [
    "RUN_ARTIFACT_SCHEMA_VERSION",
    "AdapterSnapshot",
    "AgentLimitsSnapshot",
    "ResourceHash",
    "ResourceManifest",
    "RunArtifact",
    "RunArtifactConfig",
    "RunArtifactRecorder",
    "RunLimitsSnapshot",
    "RunStatus",
    "RunTiming",
    "SubmissionMode",
    "build_resource_manifest",
]
