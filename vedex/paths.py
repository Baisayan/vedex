"""All paths for VedeX user and projects."""

import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VedexPaths:
    home: Path = field(default_factory=lambda: Path.home() / ".vedex")
    agents_home: Path = field(default_factory=lambda: Path.home() / ".agents")

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def agent_calls_log_path(self) -> Path:
        return self.logs_dir / "agent-calls.jsonl"

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