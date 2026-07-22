from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from coding.resources import (
    ResourceDiagnostic,
    ResourceError,
    ResourcePaths,
    derive_description,
    parse_markdown_resource,
)


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    path: Path
    content: str
    description: str | None = None


def load_skills_with_diagnostics(
    paths: ResourcePaths | None = None,
) -> tuple[list[Skill], list[ResourceDiagnostic]]:
    resource_paths = paths or ResourcePaths()
    skills_by_name: dict[str, Skill] = {}
    diagnostics: list[ResourceDiagnostic] = []

    for skills_dir in resource_paths.skills_dirs:
        skills, directory_diagnostics = _load_skills_from_dir_with_diagnostics(skills_dir)
        diagnostics.extend(directory_diagnostics)
        for skill in skills:
            previous = skills_by_name.get(skill.name)
            if previous is not None:
                diagnostics.append(
                    ResourceDiagnostic(
                        kind="skill",
                        name=skill.name,
                        path=skill.path,
                        message=f"overrides lower-precedence resource at {previous.path}",
                    )
                )
            skills_by_name[skill.name] = skill

    return sorted(skills_by_name.values(), key=lambda skill: skill.name), diagnostics


def expand_skill_command(text: str, skills: Sequence[Skill]) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("/skill:"):
        return None

    command, separator, request = stripped.partition(" ")
    name = command.removeprefix("/skill:").strip()
    if not name:
        raise ResourceError("Skill command must include a skill name")

    skill_by_name = {skill.name: skill for skill in skills}
    skill = skill_by_name.get(name)
    if skill is None:
        raise ResourceError(f"Unknown skill: {name}")

    additional_instructions = request.strip() if separator else None
    return format_skill_invocation(skill, additional_instructions)


def format_skill_invocation(
    skill: Skill,
    additional_instructions: str | None = None,
) -> str:
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.path}">\n'
        f"References are relative to {skill.path.parent}.\n\n"
        f"{skill.content.strip()}\n"
        "</skill>"
    )
    if additional_instructions and additional_instructions.strip():
        return f"{skill_block}\n\n{additional_instructions.strip()}"
    return skill_block


def _load_skills_from_dir_with_diagnostics(
    skills_dir: Path,
) -> tuple[list[Skill], list[ResourceDiagnostic]]:
    if not skills_dir.exists() or not skills_dir.is_dir():
        return [], []

    skills: list[Skill] = []
    diagnostics: list[ResourceDiagnostic] = []
    seen: set[str] = set()
    for path in sorted(skills_dir.iterdir(), key=lambda item: item.name):
        skill_path: Path | None = None
        name = path.stem
        if path.is_dir():
            skill_path = path / "SKILL.md"
            name = path.name
            if not skill_path.exists():
                continue
        elif path.is_file() and path.suffix.lower() == ".md":
            if path.name.upper() == "AGENTS.MD":
                continue
            skill_path = path
        else:
            continue

        if name in seen:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="skill",
                    name=name,
                    path=skill_path,
                    message=f"Duplicate skill name ignored in {skills_dir}",
                )
            )
            continue
        seen.add(name)
        try:
            skills.append(_load_skill(name, skill_path))
        except (OSError, UnicodeDecodeError) as exc:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="skill",
                    name=name,
                    path=skill_path,
                    message=f"could not read skill: {exc}",
                    severity="error",
                )
            )
    return skills, diagnostics


def _load_skill(name: str, path: Path) -> Skill:
    raw = path.read_text(encoding="utf-8")
    metadata, content = parse_markdown_resource(raw)
    description = metadata.get("description") or derive_description(content)
    return Skill(name=name, path=path, content=content, description=description)