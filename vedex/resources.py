from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


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


class ResourceError(ValueError):
    """Raised when a selected resource is invalid."""


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
        directories = [self.skills_dir]
        if self.agents_root is not None:
            directories.append(self.agents_root / "skills")
        if self.cwd is not None:
            directories.extend(
                [
                    paths.project_skills_dir(self.cwd),
                    paths.project_agents_skills_dir(self.cwd),
                ]
            )
        return tuple(_dedupe_paths(directories))

    @property
    def prompts_dirs(self) -> tuple[Path, ...]:
        paths = self._paths()
        directories = [self.prompts_dir]
        if self.agents_root is not None:
            directories.append(self.agents_root / "prompts")
        if self.cwd is not None:
            directories.extend(
                [
                    paths.project_prompts_dir(self.cwd),
                    paths.project_agents_prompts_dir(self.cwd),
                ]
            )
        return tuple(_dedupe_paths(directories))

    def _paths(self) -> VedexPaths:
        agents_home = self.agents_root or Path.home() / ".agents"
        return self.paths or VedexPaths(home=self.root, agents_home=agents_home)


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
        if separator:
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
class Skill:
    name: str
    path: Path
    content: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    path: Path
    content: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectContextFile:
    path: str
    content: str


@dataclass(frozen=True, slots=True)
class _MarkdownResource:
    name: str
    path: Path
    content: str
    description: str | None = None


def load_skills(paths: ResourcePaths | None = None) -> list[Skill]:
    resources = _load_markdown_resources(
        (paths or ResourcePaths()).skills_dirs,
        "skill",
        skip_filename="AGENTS.md",
        include_subdirs=True,
    )
    return [Skill(r.name, r.path, r.content, r.description) for r in resources]


def load_prompt_templates(paths: ResourcePaths | None = None) -> list[PromptTemplate]:
    resources = _load_markdown_resources((paths or ResourcePaths()).prompts_dirs, "prompt")
    return [PromptTemplate(r.name, r.path, r.content, r.description) for r in resources]


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


def _load_markdown_resources(
    directories: Sequence[Path],
    resource_kind: str,
    *,
    skip_filename: str | None = None,
    include_subdirs: bool = False,
) -> list[_MarkdownResource]:
    resources: list[_MarkdownResource] = []
    seen_names: set[str] = set()
    for directory in directories:
        if not directory.is_dir():
            continue

        files: list[Path] = []
        if include_subdirs:
            for item in sorted(directory.iterdir(), key=lambda item: item.name):
                if item.is_dir():
                    skill_path = item / "SKILL.md"
                    if skill_path.exists():
                        files.append(skill_path)
                elif item.is_file() and item.suffix.lower() == ".md":
                    if skip_filename and item.name.upper() == skip_filename.upper():
                        continue
                    files.append(item)
        else:
            files = [item for item in sorted(directory.glob("*.md")) if item.is_file()]

        for path in files:
            name = path.stem
            if name in seen_names:
                _warn_optional_resource(f"duplicate {resource_kind} '{name}' ignored: {path}")
                continue
            seen_names.add(name)
            try:
                metadata, content = parse_markdown_resource(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError) as exc:
                _warn_optional_resource(f"could not read {resource_kind} {path}: {exc}")
                continue
            resources.append(
                _MarkdownResource(
                    name=name,
                    path=path,
                    content=content,
                    description=metadata.get("description") or derive_description(content),
                )
            )
    return resources


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
    return tuple(_dedupe_resolved_paths([path for path in candidates if path.is_file()]))


def _find_project_root(cwd: Path) -> Path:
    markers = (".git", "pyproject.toml", "uv.lock", "setup.py", "package.json")
    for path in (cwd, *cwd.parents):
        if any((path / marker).exists() for marker in markers):
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


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def _dedupe_resolved_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def _warn_optional_resource(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)
