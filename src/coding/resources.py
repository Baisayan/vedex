from dataclasses import dataclass, field
from pathlib import Path

from coding.paths import VedexPaths


class ResourceError(ValueError):
    """Raised when resources are invalid."""


@dataclass(frozen=True, slots=True)
class ResourceDiagnostic:
    kind: str
    message: str
    path: Path | None = None
    name: str | None = None
    severity: str = "warning"

    def format(self) -> str:
        parts = [self.severity, self.kind]
        if self.name is not None:
            parts.append(self.name)
        label = " ".join(parts)
        if self.path is None:
            return f"{label}: {self.message}"
        return f"{label}: {self.message} ({self.path})"


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
