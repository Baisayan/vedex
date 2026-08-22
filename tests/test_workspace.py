from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from vedex.resources import ProjectContextFile, PromptTemplate, ResourceError, ResourcePaths, Skill
from vedex.workspace import (
    Workspace,
    build_system_prompt,
    format_project_context,
    format_skill_invocation,
    render_prompt_template,
)

from .conftest import make_tool


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_default_and_custom_system_prompts_include_expected_context(tmp_path: Path) -> None:
    read_tool = make_tool(name="read")
    skill = Skill(
        name="review", path=tmp_path / "review.md", content="# Review", description="Review code"
    )
    context = ProjectContextFile(path="C:/project/<AGENTS>.md", content="follow <rules>")

    prompt = build_system_prompt(
        cwd=tmp_path,
        tools=[read_tool],
        skills=[skill],
        custom_prompt=None,
        append_system_prompt="extra",
        context_files=[context],
        current_date=date(2026, 7, 28),
    )
    custom = build_system_prompt(
        cwd=tmp_path,
        tools=[],
        skills=[],
        custom_prompt="custom",
        append_system_prompt="append",
        context_files=[context],
        current_date=date(2026, 7, 28),
    )

    assert "Available tools:" in prompt
    assert "<available_skills>" in prompt
    assert "C:/project/&lt;AGENTS&gt;.md" in prompt
    assert "Current date: 2026-07-28" in prompt
    assert "Current working directory: ." in prompt
    assert str(tmp_path).replace("\\", "/") not in prompt
    assert custom.startswith("custom\n\nappend")
    assert "<available_skills>" not in custom


def test_template_rendering_and_skill_expansion_behave_as_commands(tmp_path: Path) -> None:
    root = tmp_path / "global"
    project = tmp_path / "project"
    _write(
        root / "skills" / "review" / "SKILL.md",
        "---\ndescription: Review code\n---\n# Review instructions\nSecret workflow",
    )
    _write(root / "prompts" / "fix.md", "Fix: {{arguments}}")
    _write(root / "prompts" / "plain.md", "Do the work")
    workspace = Workspace(
        cwd=project,
        tools=[make_tool(name="read")],
        resource_paths=ResourcePaths(root=root),
    )

    assert workspace.expand_prompt_template_command("/fix parser") == "Fix: parser"
    assert workspace.expand_prompt_template_command("/plain parser") == "Do the work\n\nparser"
    assert workspace.expand_prompt_template_command("//fix parser") is None
    expanded_skill = workspace.expand_skill_command("/skill:review inspect parser")
    assert expanded_skill is not None
    assert '<skill name="review"' in expanded_skill
    assert "Secret workflow" not in workspace.system_prompt
    assert str(root / "skills" / "review" / "SKILL.md") not in workspace.system_prompt
    assert "Review instructions" in expanded_skill
    assert "Secret workflow" in expanded_skill
    assert expanded_skill.endswith("inspect parser")
    assert workspace.expand_prompt_text("ordinary text") == "ordinary text"

    with pytest.raises(ResourceError, match="must include"):
        workspace.expand_skill_command("/skill:")
    with pytest.raises(ResourceError, match="Unknown skill"):
        workspace.expand_skill_command("/skill:missing")
    with pytest.raises(ResourceError, match="Missing prompt"):
        render_prompt_template(
            PromptTemplate(name="strict", path=tmp_path / "strict.md", content="{{required}}"),
            {},
        )


def test_workspace_reload_reports_changes_and_rebuilds_only_system_prompt_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "global"
    project = tmp_path / "project"
    skill_path = root / "skills" / "review" / "SKILL.md"
    prompt_path = root / "prompts" / "review.md"
    _write(skill_path, "# Review")
    _write(prompt_path, "review {{arguments}}")
    workspace = Workspace(
        cwd=project,
        tools=[make_tool(name="read")],
        resource_paths=ResourcePaths(root=root),
        context_files=[ProjectContextFile(path="explicit", content="explicit context")],
    )
    initial_prompt = workspace.system_prompt

    _write(prompt_path, "changed {{arguments}}")
    template_only = workspace.reload()
    assert template_only.prompt_templates.changed is True
    assert template_only.system_prompt_rebuilt is False
    assert workspace.system_prompt == initial_prompt

    _write(skill_path, "# Changed review")
    skill_changed = workspace.reload()
    assert skill_changed.skills.changed is True
    assert skill_changed.system_prompt_rebuilt is True
    assert workspace.system_prompt != initial_prompt


def test_context_and_skill_formatting_are_deduplicated_and_escaped(tmp_path: Path) -> None:
    context = ProjectContextFile(path="same", content="one")
    workspace = Workspace(
        cwd=tmp_path,
        tools=[],
        resource_paths=ResourcePaths(root=tmp_path / "global"),
        context_files=[context, context],
    )
    skill = Skill(name="x", path=tmp_path / "skill.md", content=" body ")

    assert workspace.context_files == (context,)
    assert '<project_instructions path="same">' in format_project_context([context])
    assert format_skill_invocation(skill) == ('<skill name="x">\nbody\n</skill>')
