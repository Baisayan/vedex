from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from .resources import (
    ProjectContextFile,
    PromptTemplate,
    ResourceError,
    ResourcePaths,
    Skill,
    discover_project_context,
    load_prompt_templates,
    load_skills,
    resource_paths_with_cwd,
)
from .schema import AgentTool

_TEMPLATE_VARIABLE_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")
_ARGUMENT_TEMPLATE_VARIABLES = {"arguments", "args"}


class Workspace:
    """Synchronous local resources and the system prompt built from them."""

    def __init__(
        self,
        *,
        cwd: Path,
        tools: Sequence[AgentTool],
        resource_paths: ResourcePaths | None = None,
        custom_system_prompt: str | None = None,
        append_system_prompt: str | None = None,
        context_files: Sequence[ProjectContextFile] = (),
    ) -> None:
        self.cwd = cwd
        self._tools = tuple(tools)
        self._resource_paths = resource_paths_with_cwd(resource_paths, cwd)
        self._custom_system_prompt = custom_system_prompt
        self._append_system_prompt = append_system_prompt
        self._explicit_context_files = tuple(context_files)
        self._skills: tuple[Skill, ...] = ()
        self._prompt_templates: tuple[PromptTemplate, ...] = ()
        self._context_files: tuple[ProjectContextFile, ...] = ()
        self._system_prompt = ""
        self._load_resources()
        self._system_prompt = self._build_system_prompt()

    @property
    def skills(self) -> tuple[Skill, ...]:
        return self._skills

    @property
    def prompt_templates(self) -> tuple[PromptTemplate, ...]:
        return self._prompt_templates

    @property
    def context_files(self) -> tuple[ProjectContextFile, ...]:
        return self._context_files

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def reload(self) -> ReloadSummary:
        before_skills = _skill_signatures(self._skills)
        before_prompt_templates = _prompt_template_signatures(self._prompt_templates)
        before_context_files = _context_file_signatures(self._context_files)
        before_system_prompt_inputs = _system_prompt_resource_signatures(
            skills=self._skills,
            context_files=self._context_files,
        )

        self._load_resources()

        after_skills = _skill_signatures(self._skills)
        after_prompt_templates = _prompt_template_signatures(self._prompt_templates)
        after_context_files = _context_file_signatures(self._context_files)
        after_system_prompt_inputs = _system_prompt_resource_signatures(
            skills=self._skills,
            context_files=self._context_files,
        )
        system_prompt_rebuilt = before_system_prompt_inputs != after_system_prompt_inputs
        if system_prompt_rebuilt:
            self._system_prompt = self._build_system_prompt()

        return ReloadSummary(
            skills=_category_summary(before_skills, after_skills),
            prompt_templates=_category_summary(before_prompt_templates, after_prompt_templates),
            context_files=_category_summary(before_context_files, after_context_files),
            system_prompt_rebuilt=system_prompt_rebuilt,
        )

    def expand_prompt_text(self, text: str) -> str:
        expanded_template = self.expand_prompt_template_command(text)
        if expanded_template is not None:
            return expanded_template
        expanded_skill = self.expand_skill_command(text)
        return expanded_skill if expanded_skill is not None else text

    def expand_prompt_template_command(self, text: str) -> str | None:
        stripped = text.strip()
        if (
            not stripped.startswith("/")
            or stripped.startswith("//")
            or stripped.startswith("/skill:")
        ):
            return None

        name, args = _parse_prompt_template_command(stripped)
        if not name:
            return None
        template = _find_prompt_template(name, self._prompt_templates)
        if template is None:
            return None

        rendered = render_prompt_template(template, {"arguments": args, "args": args}, missing="")
        if args and not _template_references_arguments(template.content):
            return f"{rendered.rstrip()}\n\n{args}"
        return rendered

    def expand_skill_command(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped.startswith("/skill:"):
            return None

        command, separator, request = stripped.partition(" ")
        name = command.removeprefix("/skill:").strip()
        if not name:
            raise ResourceError("Skill command must include a skill name")

        skill_by_name = {skill.name: skill for skill in self._skills}
        skill = skill_by_name.get(name)
        if skill is None:
            raise ResourceError(f"Unknown skill: {name}")

        additional_instructions = request.strip() if separator else None
        return format_skill_invocation(skill, additional_instructions)

    def _load_resources(self) -> None:
        self._skills = tuple(load_skills(self._resource_paths))
        self._prompt_templates = tuple(load_prompt_templates(self._resource_paths))
        discovered_context = discover_project_context(self._resource_paths)
        self._context_files = _merge_context_files(
            self._explicit_context_files,
            discovered_context,
        )

    def _build_system_prompt(self) -> str:
        return build_system_prompt(
            cwd=self.cwd,
            tools=self._tools,
            skills=self._skills,
            custom_prompt=self._custom_system_prompt,
            append_system_prompt=self._append_system_prompt,
            context_files=self._context_files,
        )


@dataclass(frozen=True, slots=True)
class ReloadCategorySummary:
    before: int
    after: int
    changed: bool

    @property
    def delta(self) -> int:
        return self.after - self.before


@dataclass(frozen=True, slots=True)
class ReloadSummary:
    skills: ReloadCategorySummary
    prompt_templates: ReloadCategorySummary
    context_files: ReloadCategorySummary
    system_prompt_rebuilt: bool


def build_system_prompt(
    *,
    cwd: Path,
    tools: Sequence[AgentTool],
    skills: Sequence[Skill],
    custom_prompt: str | None,
    append_system_prompt: str | None,
    context_files: Sequence[ProjectContextFile],
    current_date: date | None = None,
    extra_guidelines: Sequence[str] = (),
) -> str:
    current_date = current_date or date.today()
    formatted_cwd = str(cwd).replace("\\", "/")
    append_section = f"\n\n{append_system_prompt}" if append_system_prompt else ""

    if custom_prompt is not None:
        prompt = custom_prompt + append_section
        prompt += format_project_context(context_files)
        if _has_tool(tools, "read"):
            prompt += format_skills_for_prompt(skills)
        return _append_runtime_context(prompt, current_date, formatted_cwd)

    prompt = (
        "You are an expert coding assistant operating inside Vedex, a coding agent harness. "
        "You help users by reading files, executing commands, editing code, and writing new files."
        f"\n\nAvailable tools:\n{format_available_tools(tools)}"
        "\n\nIn addition to the tools above, you may have access to other custom tools "
        "depending on the project."
        f"\n\nGuidelines:\n{format_guidelines(tools, extra_guidelines)}"
    )
    prompt += append_section
    prompt += format_project_context(context_files)
    if _has_tool(tools, "read"):
        prompt += format_skills_for_prompt(skills)
    return _append_runtime_context(prompt, current_date, formatted_cwd)


def render_prompt_template(
    template: PromptTemplate,
    variables: Mapping[str, str],
    *,
    missing: str | None = None,
) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = variables.get(name)
        if value is None:
            if missing is None:
                raise ResourceError(f"Missing prompt template variable: {name}")
            return missing
        return value

    return _TEMPLATE_VARIABLE_RE.sub(replace, template.content)


def format_skill_invocation(skill: Skill, additional_instructions: str | None = None) -> str:
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.path}">\n'
        f"References are relative to {skill.path.parent}.\n\n"
        f"{skill.content.strip()}\n"
        "</skill>"
    )
    if additional_instructions and additional_instructions.strip():
        return f"{skill_block}\n\n{additional_instructions.strip()}"
    return skill_block


def format_available_tools(tools: Sequence[AgentTool]) -> str:
    lines = [f"- {tool.name}: {tool.prompt_snippet}" for tool in tools if tool.prompt_snippet]
    return "\n".join(lines) if lines else "(none)"


def format_guidelines(
    tools: Sequence[AgentTool],
    extra_guidelines: Sequence[str] = (),
) -> str:
    guidelines: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            guidelines.append(normalized)

    add("Use bash for file operations like ls, rg, find")
    for tool in tools:
        for guideline in tool.prompt_guidelines:
            add(guideline)
    for guideline in extra_guidelines:
        add(guideline)
    add("Be concise in your responses")
    add("Show file paths clearly when working with files")
    return "\n".join(f"- {guideline}" for guideline in guidelines)


def format_project_context(context_files: Sequence[ProjectContextFile]) -> str:
    if not context_files:
        return ""

    lines = [
        "\n\n<project_context>",
        "",
        "Project-specific instructions and guidelines:",
        "",
    ]
    for context_file in context_files:
        lines.append(f'<project_instructions path="{escape(context_file.path)}">')
        lines.append(context_file.content)
        lines.append("</project_instructions>")
        lines.append("")
    lines.append("</project_context>")
    return "\n".join(lines)


def format_skills_for_prompt(skills: Sequence[Skill]) -> str:
    if not skills:
        return ""

    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Read the full skill file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory "
        "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in sorted(skills, key=lambda item: item.name):
        description = skill.description or "No description"
        lines.extend(
            [
                "  <skill>",
                f"    <name>{escape(skill.name)}</name>",
                f"    <description>{escape(description)}</description>",
                f"    <location>{escape(str(skill.path))}</location>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def _merge_context_files(
    explicit: Sequence[ProjectContextFile],
    discovered: Sequence[ProjectContextFile],
) -> tuple[ProjectContextFile, ...]:
    merged: list[ProjectContextFile] = []
    seen: set[str] = set()
    for context_file in (*explicit, *discovered):
        if context_file.path not in seen:
            seen.add(context_file.path)
            merged.append(context_file)
    return tuple(merged)


def _category_summary(
    before: tuple[tuple[object, ...], ...],
    after: tuple[tuple[object, ...], ...],
) -> ReloadCategorySummary:
    return ReloadCategorySummary(before=len(before), after=len(after), changed=before != after)


def _skill_signatures(skills: Sequence[Skill]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (skill.name, str(skill.path), skill.description, skill.content) for skill in skills
    )


def _prompt_template_signatures(
    templates: Sequence[PromptTemplate],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (template.name, str(template.path), template.description, template.content)
        for template in templates
    )


def _context_file_signatures(
    context_files: Sequence[ProjectContextFile],
) -> tuple[tuple[object, ...], ...]:
    return tuple((context_file.path, context_file.content) for context_file in context_files)


def _system_prompt_resource_signatures(
    *,
    skills: Sequence[Skill],
    context_files: Sequence[ProjectContextFile],
) -> tuple[tuple[object, ...], tuple[tuple[object, ...], ...]]:
    prompt_skills = tuple(
        (skill.name, str(skill.path), skill.description)
        for skill in sorted(skills, key=lambda item: item.name)
    )
    return prompt_skills, _context_file_signatures(context_files)


def _parse_prompt_template_command(text: str) -> tuple[str, str]:
    command, separator, args = text[1:].partition(" ")
    return command.strip().lower(), args.strip() if separator else ""


def _find_prompt_template(
    name: str,
    templates: Sequence[PromptTemplate],
) -> PromptTemplate | None:
    normalized_name = name.strip().removeprefix("/").lower()
    for template in templates:
        if template.name.lower() == normalized_name:
            return template
    return None


def _template_references_arguments(content: str) -> bool:
    return any(
        match.group(1) in _ARGUMENT_TEMPLATE_VARIABLES
        for match in _TEMPLATE_VARIABLE_RE.finditer(content)
    )


def _has_tool(tools: Sequence[AgentTool], name: str) -> bool:
    return any(tool.name == name for tool in tools)


def _append_runtime_context(prompt: str, current_date: date, cwd: str) -> str:
    return f"{prompt}\nCurrent date: {current_date.isoformat()}\nCurrent working directory: {cwd}"
