from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from hashlib import sha256
from pathlib import Path
from xml.sax.saxutils import escape

from .schema import AgentTool

# Paths


@dataclass(frozen=True, slots=True)
class VedexPaths:
    home: Path = field(default_factory=lambda: Path.home() / ".vedex")
    agents_home: Path = field(default_factory=lambda: Path.home() / ".agents")

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def user_skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def user_prompts_dir(self) -> Path:
        return self.home / "prompts"

    @property
    def user_agents_skills_dir(self) -> Path:
        return self.agents_home / "skills"

    @property
    def user_agents_prompts_dir(self) -> Path:
        return self.agents_home / "prompts"

    def project_vedex_dir(self, cwd: Path) -> Path:
        return cwd / ".vedex"

    def project_agents_dir(self, cwd: Path) -> Path:
        return cwd / ".agents"

    def project_skills_dir(self, cwd: Path) -> Path:
        return self.project_vedex_dir(cwd) / "skills"

    def project_prompts_dir(self, cwd: Path) -> Path:
        return self.project_vedex_dir(cwd) / "prompts"

    def project_agents_skills_dir(self, cwd: Path) -> Path:
        return self.project_agents_dir(cwd) / "skills"

    def project_agents_prompts_dir(self, cwd: Path) -> Path:
        return self.project_agents_dir(cwd) / "prompts"

    def project_session_dir(self, cwd: Path) -> Path:
        resolved = cwd.resolve()
        digest = sha256(str(resolved).encode("utf-8")).hexdigest()[:6]
        slug = _slugify_path(resolved)
        return self.sessions_dir / f"{slug or 'project'}-{digest}"

    def default_session_path(self, cwd: Path) -> Path:
        path = self.project_session_dir(cwd) / "default.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _slugify_path(path: Path, *, max_length: int = 72) -> str:
    parts = [part for part in path.parts if part not in (path.anchor, "")]
    try:
        relative_to_home = path.relative_to(Path.home())
    except ValueError:
        pass
    else:
        parts = ["home", *relative_to_home.parts]

    slug_parts = [
        normalized
        for part in parts
        if (normalized := re.sub(r"[^a-zA-Z0-9._-]+", "-", part).strip(".-_").lower())
    ]
    slug = "-".join(slug_parts)
    if len(slug) <= max_length:
        return slug

    suffix_parts: list[str] = []
    suffix_length = 0
    for part in reversed(slug_parts):
        next_length = suffix_length + len(part) + (1 if suffix_parts else 0)
        if next_length > max_length:
            break
        suffix_parts.append(part)
        suffix_length = next_length
    return "-".join(reversed(suffix_parts)) or slug[-max_length:].strip("-")


# Resources


class ResourceError(ValueError):
    """Raised when resources are invalid."""


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


@dataclass(frozen=True, slots=True)
class ResourcePaths:
    root: Path = field(default_factory=lambda: Path.home() / ".vedex")
    cwd: Path | None = None
    agents_root: Path | None = field(default_factory=lambda: Path.home() / ".agents")
    paths: VedexPaths | None = None

    @property
    def skills_dir(self) -> Path:
        return self.root / "skills"

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    @property
    def skills_dirs(self) -> tuple[Path, ...]:
        paths = self._paths()
        dirs = [self.skills_dir]
        if self.agents_root is not None:
            dirs.append(self.agents_root / "skills")
        if self.cwd is not None:
            dirs.extend(
                [
                    paths.project_skills_dir(self.cwd),
                    paths.project_agents_skills_dir(self.cwd),
                ]
            )
        return tuple(_dedupe_paths(dirs))

    @property
    def prompts_dirs(self) -> tuple[Path, ...]:
        paths = self._paths()
        dirs = [self.prompts_dir]
        if self.agents_root is not None:
            dirs.append(self.agents_root / "prompts")
        if self.cwd is not None:
            dirs.extend(
                [
                    paths.project_prompts_dir(self.cwd),
                    paths.project_agents_prompts_dir(self.cwd),
                ]
            )
        return tuple(_dedupe_paths(dirs))

    def _paths(self) -> VedexPaths:
        agents_home = self.agents_root or Path.home() / ".agents"
        return self.paths or VedexPaths(home=self.root, agents_home=agents_home)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def resource_paths_with_cwd(
    paths: ResourcePaths | None,
    cwd: Path,
) -> ResourcePaths:
    if paths is None:
        return ResourcePaths(cwd=cwd)
    if paths.cwd is not None:
        return paths
    return ResourcePaths(
        root=paths.root,
        cwd=cwd,
        agents_root=paths.agents_root,
        paths=paths.paths,
    )


def parse_markdown_resource(text: str) -> tuple[dict[str, str], str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized

    end = normalized.find("\n---", 4)
    if end == -1:
        return {}, normalized

    raw_frontmatter = normalized[4:end]
    body = normalized[end + len("\n---") :]
    if body.startswith("\n"):
        body = body[1:]

    metadata: dict[str, str] = {}
    for line in raw_frontmatter.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, body


def derive_description(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
        return stripped
    return None


@dataclass(frozen=True, slots=True)
class _MarkdownResource:
    name: str
    path: Path
    content: str
    description: str | None = None


def _load_markdown_resources(
    dir_paths: Sequence[Path],
    resource_kind: str,
    *,
    skip_filename: str | None = None,
    include_subdirs: bool = False,
) -> list[_MarkdownResource]:
    all_resources: list[_MarkdownResource] = []
    seen_names: set[str] = set()

    for dir_path in dir_paths:
        if not dir_path.exists() or not dir_path.is_dir():
            continue

        files: list[Path] = []
        if include_subdirs:
            for item in sorted(dir_path.iterdir(), key=lambda item: item.name):
                if item.is_dir():
                    sub_path = item / "SKILL.md"
                    if sub_path.exists():
                        files.append(sub_path)
                elif item.is_file() and item.suffix.lower() == ".md":
                    if skip_filename and item.name.upper() == skip_filename.upper():
                        continue
                    files.append(item)
        else:
            for item in sorted(dir_path.glob("*.md"), key=lambda item: item.name):
                if item.is_file():
                    files.append(item)

        for path in files:
            name = path.stem
            if name in seen_names:
                _warn_optional_resource(f"duplicate {resource_kind} '{name}' ignored: {path}")
                continue
            seen_names.add(name)
            try:
                raw = path.read_text(encoding="utf-8")
                metadata, content = parse_markdown_resource(raw)
                description = metadata.get("description") or derive_description(content)
                all_resources.append(
                    _MarkdownResource(
                        name=name, path=path, content=content, description=description
                    )
                )
            except (OSError, UnicodeDecodeError) as exc:
                _warn_optional_resource(f"could not read {resource_kind} {path}: {exc}")

    return all_resources


def _warn_optional_resource(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


# Skills


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    path: Path
    content: str
    description: str | None = None


def load_skills(paths: ResourcePaths | None = None) -> list[Skill]:
    resource_paths = paths or ResourcePaths()
    raw_resources = _load_markdown_resources(
        resource_paths.skills_dirs,
        "skill",
        skip_filename="AGENTS.md",
        include_subdirs=True,
    )
    skills = [
        Skill(name=r.name, path=r.path, content=r.content, description=r.description)
        for r in raw_resources
    ]
    return skills


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


# Prompt templates

_TEMPLATE_VARIABLE_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")
_ARGUMENT_TEMPLATE_VARIABLES = {"arguments", "args"}


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    path: Path
    content: str
    description: str | None = None


def load_prompt_templates(paths: ResourcePaths | None = None) -> list[PromptTemplate]:
    resource_paths = paths or ResourcePaths()
    raw_resources = _load_markdown_resources(
        resource_paths.prompts_dirs,
        "prompt",
    )
    templates = [
        PromptTemplate(name=r.name, path=r.path, content=r.content, description=r.description)
        for r in raw_resources
    ]
    return templates


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


def expand_prompt_template_command(
    text: str,
    templates: Sequence[PromptTemplate],
) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("/") or stripped.startswith("//") or stripped.startswith("/skill:"):
        return None

    name, args = _parse_prompt_template_command(stripped)
    if not name:
        return None

    template = _find_prompt_template(name, templates)
    if template is None:
        return None

    rendered = render_prompt_template(template, {"arguments": args, "args": args}, missing="")
    if args and not _template_references_arguments(template.content):
        return f"{rendered.rstrip()}\n\n{args}"
    return rendered


def _template_references_arguments(content: str) -> bool:
    return any(
        match.group(1) in _ARGUMENT_TEMPLATE_VARIABLES
        for match in _TEMPLATE_VARIABLE_RE.finditer(content)
    )


def _find_prompt_template(
    name: str,
    templates: Sequence[PromptTemplate],
) -> PromptTemplate | None:
    normalized_name = name.strip().removeprefix("/").lower()
    for template in templates:
        if template.name.lower() == normalized_name:
            return template
    return None


def _parse_prompt_template_command(text: str) -> tuple[str, str]:
    command, separator, args = text[1:].partition(" ")
    return command.strip().lower(), args.strip() if separator else ""


# Project context

PROJECT_MARKERS = (".git", "pyproject.toml", "uv.lock", "setup.py", "package.json")


@dataclass(frozen=True, slots=True)
class ProjectContextFile:
    path: str
    content: str


def discover_project_context(
    paths: ResourcePaths | None = None,
) -> tuple[ProjectContextFile, ...]:
    resource_paths = paths or ResourcePaths()
    context_files: list[ProjectContextFile] = []
    for path in _context_file_candidates(resource_paths):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            _warn_optional_resource(f"could not read project context {path}: {exc}")
            continue
        context_files.append(ProjectContextFile(path=str(path), content=content))
    return tuple(context_files)


def _context_file_candidates(paths: ResourcePaths) -> tuple[Path, ...]:
    candidates: list[Path] = [paths.root / "AGENTS.md"]
    if paths.agents_root is not None:
        candidates.append(paths.agents_root / "AGENTS.md")

    if paths.cwd is not None:
        cwd = paths.cwd.expanduser().resolve()
        project_root = _find_project_root(cwd)
        candidates.extend(_ancestor_agents_files(project_root, cwd))
        vedex_paths = paths._paths()
        candidates.extend(
            [
                vedex_paths.project_vedex_dir(cwd) / "AGENTS.md",
                vedex_paths.project_agents_dir(cwd) / "AGENTS.md",
            ]
        )

    existing = [path for path in candidates if path.is_file()]
    return tuple(_dedupe_resolved_paths(existing))


def _find_project_root(cwd: Path) -> Path:
    for path in (cwd, *cwd.parents):
        if any((path / marker).exists() for marker in PROJECT_MARKERS):
            return path
    return cwd


def _ancestor_agents_files(project_root: Path, cwd: Path) -> list[Path]:
    try:
        relative = cwd.relative_to(project_root)
    except ValueError:
        return [cwd / "AGENTS.md"]

    paths = [project_root / "AGENTS.md"]
    current = project_root
    for part in relative.parts:
        current = current / part
        paths.append(current / "AGENTS.md")
    return paths


def _dedupe_resolved_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


# System prompt


@dataclass(frozen=True, slots=True)
class BuildSystemPromptOptions:
    cwd: Path
    tools: Sequence[AgentTool] = ()
    skills: Sequence[Skill] = ()
    custom_prompt: str | None = None
    append_system_prompt: str | None = None
    context_files: Sequence[ProjectContextFile] = ()
    current_date: date | None = None
    extra_guidelines: Sequence[str] = field(default_factory=tuple)


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    current_date = options.current_date or date.today()
    cwd = _format_path(options.cwd)
    append_section = f"\n\n{options.append_system_prompt}" if options.append_system_prompt else ""

    if options.custom_prompt is not None:
        prompt = options.custom_prompt
        prompt += append_section
        prompt += format_project_context(options.context_files)
        if _has_tool(options.tools, "read"):
            prompt += format_skills_for_prompt(options.skills)
        prompt += f"\nCurrent date: {current_date.isoformat()}"
        prompt += f"\nCurrent working directory: {cwd}"
        return prompt

    prompt = (
        "You are an expert coding assistant operating inside Vedex, a coding agent harness. "
        "You help users by reading files, executing commands, editing code, and writing new files."
        f"\n\nAvailable tools:\n{format_available_tools(options.tools)}"
        "\n\nIn addition to the tools above, you may have access to other custom tools "
        "depending on the project."
        f"\n\nGuidelines:\n{format_guidelines(options.tools, options.extra_guidelines)}"
    )

    prompt += append_section
    prompt += format_project_context(options.context_files)
    if _has_tool(options.tools, "read"):
        prompt += format_skills_for_prompt(options.skills)
    prompt += f"\nCurrent date: {current_date.isoformat()}"
    prompt += f"\nCurrent working directory: {cwd}"
    return prompt


def format_available_tools(tools: Sequence[AgentTool]) -> str:
    lines = [f"- {tool.name}: {tool.prompt_snippet}" for tool in tools if tool.prompt_snippet]
    return "\n".join(lines) if lines else "(none)"


def collect_prompt_guidelines(
    tools: Sequence[AgentTool],
    extra_guidelines: Sequence[str] = (),
) -> list[str]:
    guidelines: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
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
    return guidelines


def format_guidelines(
    tools: Sequence[AgentTool],
    extra_guidelines: Sequence[str] = (),
) -> str:
    return "\n".join(
        f"- {guideline}" for guideline in collect_prompt_guidelines(tools, extra_guidelines)
    )


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


def _has_tool(tools: Sequence[AgentTool], name: str) -> bool:
    return any(tool.name == name for tool in tools)


def _format_path(path: Path) -> str:
    return str(path).replace("\\", "/")
