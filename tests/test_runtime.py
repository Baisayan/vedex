from __future__ import annotations

from pathlib import Path

import pytest
import vedex.runtime as runtime_module
from vedex.agent import AgentLimits
from vedex.environments import LocalEnvironment
from vedex.models import (
    FakeAdapter,
    ModelCompletedEvent,
    ModelEvent,
    ModelSettings,
    ModelStartEvent,
    ModelTextDeltaEvent,
    Usage,
)
from vedex.resources import ResourceError, ResourcePaths
from vedex.runtime import AppRuntime
from vedex.schema import AgentEvent, AssistantMessage, UserMessage
from vedex.workspace import Workspace

from .conftest import run_async


def _completed_stream(content: str = "done") -> list[ModelEvent]:
    return [
        ModelStartEvent(),
        ModelTextDeltaEvent(delta=content),
        ModelCompletedEvent(
            message=AssistantMessage(content=content),
            usage=Usage(input_tokens=4, output_tokens=2),
        ),
    ]


def test_app_runtime_composes_resources_tools_agent_and_prompt_execution(tmp_path: Path) -> None:
    project = tmp_path / "project"
    global_root = tmp_path / "global"
    (project / ".vedex" / "prompts").mkdir(parents=True)
    global_root.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (project / "AGENTS.md").write_text("Project instruction", encoding="utf-8")
    (project / ".vedex" / "prompts" / "greet.md").write_text(
        "Hello {{ arguments }}",
        encoding="utf-8",
    )

    adapter = FakeAdapter([_completed_stream()])
    environment = LocalEnvironment(project)
    limits = AgentLimits(max_turns=3, max_tool_calls=4)
    runtime = AppRuntime(
        adapter=adapter,
        environment=environment,
        settings=ModelSettings(model="fake-model", options={"temperature": 0}),
        workspace_path=project,
        limits=limits,
        resource_paths=ResourcePaths(root=global_root),
    )

    async def exercise() -> list[AgentEvent]:
        assert runtime.state == "created"
        async with runtime:
            assert str(runtime.state) == "started"
            assert environment.state == "started"
            assert runtime.limits is limits
            assert runtime.settings.model == "fake-model"
            assert [tool.name for tool in runtime.agent.tools] == ["read", "write", "edit", "bash"]
            assert "Project instruction" in runtime.system_prompt
            return [event async for event in runtime.prompt("/greet Ada")]

    events = run_async(exercise())

    assert events[-1].type == "agent_end"
    assert runtime.state == "closed"
    assert environment.state == "stopped"
    assert adapter.requests[0].system == runtime.system_prompt
    assert [tool.name for tool in adapter.requests[0].tools] == ["read", "write", "edit", "bash"]
    assert adapter.requests[0].messages == [UserMessage(content="Hello Ada")]
    assert runtime.agent.last_result is not None
    assert runtime.agent.last_result.status == "completed"
    assert runtime.agent.usage == Usage(input_tokens=4, output_tokens=2)


def test_app_runtime_rejects_prompt_before_start_and_is_one_shot(tmp_path: Path) -> None:
    runtime = AppRuntime(
        adapter=FakeAdapter([_completed_stream()]),
        environment=LocalEnvironment(tmp_path),
        settings=ModelSettings(model="fake"),
        workspace_path=tmp_path,
    )

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="not started"):
            await anext(runtime.prompt("task"))
        await runtime.start()
        await runtime.close()
        await runtime.close()
        with pytest.raises(RuntimeError, match="Cannot start"):
            await runtime.start()

    run_async(exercise())


def test_reload_updates_prompt_without_history_and_reset_only_clears_memory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resources"
    skill_path = root / "skills" / "review" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("---\ndescription: First\n---\nsecret body", encoding="utf-8")
    adapter = FakeAdapter([_completed_stream("first"), _completed_stream("second")])
    runtime = AppRuntime(
        adapter=adapter,
        environment=LocalEnvironment(tmp_path),
        settings=ModelSettings(model="fake"),
        workspace_path=tmp_path,
        resource_paths=ResourcePaths(root=root),
    )

    async def exercise() -> None:
        async with runtime:
            _ = [event async for event in runtime.prompt("one")]
            history = runtime.agent.messages
            original_prompt = runtime.agent.system_prompt
            skill_path.write_text(
                "---\ndescription: Changed\n---\nchanged secret body",
                encoding="utf-8",
            )

            summary = runtime.reload_resources()

            assert summary.system_prompt_rebuilt is True
            assert runtime.agent.messages == history
            assert runtime.agent.system_prompt == runtime.system_prompt
            assert runtime.agent.system_prompt != original_prompt
            _ = [event async for event in runtime.prompt("two")]
            runtime.reset()
            assert runtime.agent.messages == ()
            assert runtime.agent.system_prompt == runtime.system_prompt

    run_async(exercise())

    assert adapter.requests[1].system == runtime.system_prompt
    assert skill_path.is_file()


def test_app_runtime_closes_environment_when_resource_composition_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = LocalEnvironment(tmp_path)
    runtime = AppRuntime(
        adapter=FakeAdapter(),
        environment=environment,
        settings=ModelSettings(model="fake"),
        workspace_path=tmp_path,
    )

    def fail_workspace(**_arguments: object) -> Workspace:
        raise ResourceError("required resource failed")

    monkeypatch.setattr(runtime_module, "Workspace", fail_workspace)

    with pytest.raises(ResourceError, match="required resource failed"):
        run_async(runtime.start())

    assert runtime.state == "closed"
    assert environment.state == "stopped"
