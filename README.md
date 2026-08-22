# Vedex

Vedex is a small, provider-neutral terminal coding-agent harness. Its current
runtime combines a typed model adapter, an in-memory Agent, environment-bound
coding tools, project instructions and skills, deterministic headless output,
and versioned run artifacts.

Interactive and headless execution now share the same ephemeral `AppRuntime`,
Agent, resources, tools, and environment lifecycle. Conversation history lives
only in memory and disappears when the process exits.

## Implemented runtime

- `ModelAdapter` and normalized model streams, with `FakeAdapter` for offline
  deterministic tests.
- An ephemeral Agent with explicit limits, cancellation, usage, context policy,
  sequential tool calls, and terminal statuses.
- `Environment` and `LocalEnvironment` contracts for byte-based file access,
  foreground commands, patch collection, workspace export, and reproducibility
  metadata.
- Environment-bound `read`, `write`, `edit`, and `bash` tools whose published
  schemas are their only accepted input contracts.
- `AppRuntime` for environment lifecycle, project resources, system-prompt
  construction, and the package's only Agent composition point.
- A Rich interactive REPL with streamed Agent events, cancellable turns,
  project resources, and focused context/reload/reset commands.
- `run_headless()` with plain-text or JSONL event streaming, typed `RunResult`
  values, stable exit codes, and stderr-only diagnostics.
- Versioned JSON run artifacts containing configuration, hashes, messages,
  normalized events, usage, timing, status, and a patch or workspace export.

The base package has no provider client or model-discovery path and no durable
conversation store. Adapters normalize provider events and failures before the
Agent sees them; filesystem and process mechanics remain inside environments.

Docker execution, built-in production model adapters, and benchmark runners are
not implemented yet. No benchmark score is claimed.

## Interactive REPL

The CLI chooses an adapter and model once at startup. An adapter entry point is
a zero-argument factory using the `module:attribute` form; it returns an object
implementing `ModelAdapter`. This keeps provider packages and credentials out of
the REPL itself.

```bash
vedex --adapter your_package.adapters:create_adapter --model model-id --cwd .
```

Vedex currently ships only `FakeAdapter` for deterministic offline tests. A
production adapter can be supplied externally through the same contract until
built-in provider support is added.

The interactive command set is intentionally small:

- `/help`, `/skills`, `/prompts`, and `/context` inspect runtime state.
- `/skill:<name> [request]` injects that skill's complete instructions.
- `/<prompt> [arguments]` expands a loaded prompt template before the user
  message is appended.
- `/reload` reloads resources without changing history; `/reset` clears only
  in-memory messages.
- `/clear`, `/exit`, and `!command` provide terminal controls. Direct shell
  output is not added to model history.

Skills use `skills/<name>/SKILL.md` under `~/.vedex` or the project's `.vedex`
directory. Only their names and descriptions appear in the base system prompt.
Project `AGENTS.md` instructions are loaded deterministically from repository
root toward the active directory.

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
