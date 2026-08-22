# Vedex

Vedex is a small, provider-neutral terminal coding-agent harness. Its current
runtime combines a typed model adapter, an in-memory Agent, environment-bound
coding tools, project instructions and skills, deterministic headless output,
and versioned run artifacts.

The implementation is being migrated in deliberate milestones. The new
headless path is the canonical path for future benchmark runners. The existing
interactive REPL is still on the legacy runtime and will be moved onto the same
`AppRuntime` in a later milestone.

## Implemented runtime

- `ModelAdapter` and normalized model streams, with `FakeAdapter` for offline
  deterministic tests.
- An ephemeral Agent with explicit limits, cancellation, usage, context policy,
  sequential tool calls, and terminal statuses.
- `Environment` and `LocalEnvironment` contracts for byte-based file access,
  foreground commands, patch collection, workspace export, and reproducibility
  metadata.
- Environment-bound `read`, `write`, `edit`, and `bash` tools.
- `AppRuntime` for environment lifecycle, project resources, system-prompt
  construction, and Agent composition.
- `run_headless()` with plain-text or JSONL event streaming, typed `RunResult`
  values, stable exit codes, and stderr-only diagnostics.
- Versioned JSON run artifacts containing configuration, hashes, messages,
  normalized events, usage, timing, status, and a patch or workspace export.

Docker execution, production model adapters, benchmark runners, and migration
of the interactive REPL are not implemented yet. No benchmark score is claimed.

## Headless API

Benchmarks and other automation call `vedex.headless.run_headless`. It accepts
the task, adapter, environment, model settings, Agent limits, and host project
path. `output_mode="plain"` writes assistant text to stdout;
`output_mode="jsonl"` writes one normalized `AgentEvent` per line. Diagnostics
always go to stderr.

Passing a `RunArtifactConfig` records a patch by default. Set
`submission="export"` for a deterministic `.tar.gz` workspace submission.
The returned `RunResult.exit_code` is suitable for a process wrapper.

The runtime and recorder are also usable independently:

- `vedex.runtime.AppRuntime`
- `vedex.artifacts.RunArtifactRecorder`
- `vedex.artifacts.RunArtifact`

Run artifacts are explicit outputs only. They are never loaded as conversation
memory and are not durable sessions.

## Development

Vedex requires Python 3.12+ and uses `uv` for local development.

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy vedex tests
```

Normal tests are offline and use `FakeAdapter`; they do not require a provider
account or paid API usage.

## References

Vedex takes architectural inspiration from the
[Tau repository](https://github.com/huggingface/tau) and benchmark discipline
from the
[mini-SWE-agent repository](https://github.com/SWE-agent/mini-swe-agent).
Planned benchmark integrations target the official
[SWE-bench repository](https://github.com/SWE-bench/SWE-bench) first and the
[ProgramBench repository](https://github.com/facebookresearch/programbench)
later.
