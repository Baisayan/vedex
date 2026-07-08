from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReloadCategorySummary:
    before: int
    after: int
    changed: bool

    @property
    def delta(self) -> int:
        return self.after - self.before


@dataclass(frozen=True, slots=True)
class CodingReloadSummary:
    skills: ReloadCategorySummary
    prompt_templates: ReloadCategorySummary
    context_files: ReloadCategorySummary
    diagnostics: ReloadCategorySummary
    system_prompt_rebuilt: bool