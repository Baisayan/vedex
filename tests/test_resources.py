from __future__ import annotations

from pathlib import Path

import pytest
from vedex.resources import (
    ResourcePaths,
    VedexPaths,
    derive_description,
    discover_project_context,
    load_prompt_templates,
    load_skills,
    parse_markdown_resource,
    resource_paths_with_cwd,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_vedex_paths_and_resource_paths_only_use_global_and_project_vedex(tmp_path: Path) -> None:
    global_root = tmp_path / "global" / ".vedex"
    project = tmp_path / "project"
    legacy_global = tmp_path / "global" / ".agents"
    legacy_project = project / ".agents"
    _write(global_root / "skills" / "global.md", "# Global")
    _write(project / ".vedex" / "skills" / "project.md", "# Project")
    _write(legacy_global / "skills" / "legacy.md", "# Legacy")
    _write(legacy_project / "skills" / "legacy.md", "# Legacy")

    paths = ResourcePaths(root=global_root, cwd=project)

    assert [skill.name for skill in load_skills(paths)] == ["global", "project"]
    assert VedexPaths(home=global_root).sessions_dir == global_root / "sessions"
    assert paths.skills_dirs == (global_root / "skills", project / ".vedex" / "skills")
    assert paths.prompts_dirs == (global_root / "prompts", project / ".vedex" / "prompts")


def test_markdown_metadata_and_description_derivation() -> None:
    metadata, body = parse_markdown_resource(
        "---\r\ndescription: 'Useful task'\r\nignored\r\n---\r\n# Heading\r\nBody"
    )

    assert metadata == {"description": "Useful task"}
    assert body == "# Heading\nBody"
    assert derive_description("\n# Heading\nBody") == "Heading"
    assert derive_description("\nplain description\nBody") == "plain description"
    assert parse_markdown_resource("---\nmissing end")[0] == {}


def test_skill_and_prompt_loading_handles_directories_duplicates_and_warnings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    global_root = tmp_path / "global"
    project = tmp_path / "project"
    paths = ResourcePaths(root=global_root, cwd=project)
    _write(global_root / "skills" / "single.md", "---\ndescription: One\n---\nbody")
    _write(global_root / "skills" / "nested" / "SKILL.md", "# Nested skill")
    _write(global_root / "skills" / "AGENTS.md", "must not become a skill")
    _write(project / ".vedex" / "skills" / "single.md", "# Duplicate")
    _write(global_root / "prompts" / "review.md", "# Review")
    _write(project / ".vedex" / "prompts" / "review.md", "# Duplicate")

    skills = load_skills(paths)
    prompts = load_prompt_templates(paths)

    assert [(skill.name, skill.description) for skill in skills] == [
        ("SKILL", "Nested skill"),
        ("single", "One"),
    ]
    assert [template.name for template in prompts] == ["review"]
    warning = capsys.readouterr().err
    assert "duplicate skill 'single'" in warning
    assert "duplicate prompt 'review'" in warning


def test_unreadable_optional_resource_warns_and_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "global"
    unreadable = root / "skills" / "bad.md"
    _write(unreadable, "# bad")
    original_read_text = Path.read_text

    def read_text(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if path == unreadable:
            raise OSError("denied")
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text)

    assert load_skills(ResourcePaths(root=root)) == []
    assert "could not read skill" in capsys.readouterr().err


def test_project_context_discovery_uses_global_repo_and_project_vedex_only(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    project = tmp_path / "project"
    nested = project / "src" / "feature"
    (project / ".git").mkdir(parents=True)
    _write(global_root / "AGENTS.md", "global")
    _write(project / "AGENTS.md", "project")
    _write(project / "src" / "AGENTS.md", "nested")
    _write(nested / ".vedex" / "AGENTS.md", "project vedex")
    _write(tmp_path / ".agents" / "AGENTS.md", "legacy global")
    _write(project / ".agents" / "AGENTS.md", "legacy project")

    context = discover_project_context(ResourcePaths(root=global_root, cwd=nested))

    assert [item.content for item in context] == ["global", "project", "nested", "project vedex"]
    assert [item.path for item in context] == list(dict.fromkeys(item.path for item in context))


def test_resource_paths_with_cwd_fills_missing_cwd_and_preserves_explicit_cwd(
    tmp_path: Path,
) -> None:
    root = tmp_path / "global"
    inferred = resource_paths_with_cwd(ResourcePaths(root=root), tmp_path / "project")
    explicit = ResourcePaths(root=root, cwd=tmp_path / "explicit")

    assert inferred.cwd == tmp_path / "project"
    assert resource_paths_with_cwd(explicit, tmp_path / "ignored") is explicit
