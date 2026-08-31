# Vedex

Vedex is a small, provider-neutral terminal coding agent. It combines a typed
model adapter boundary, an in-memory Agent, environment-bound coding tools,
project instructions, skills, an interactive REPL, and headless execution.

The runtime is deliberately ephemeral: conversation history exists only while
the process is running. Docker and local environments share the same tool and
Agent contracts. Run artifacts are optional per-run records for debugging,
reproducibility, and future benchmark submissions; they are not sessions.

## Current status

The core runtime is implemented and covered by the automated suite:

- `ModelAdapter` and deterministic `FakeAdapter` model streams.
- Optional SDK-backed OpenAI and Gemini adapters.
- Agent limits, cancellation, context trimming, sequential tools, and terminal statuses.
- Local and Docker environments with read, write, edit, bash, patch collection,
  workspace export, and metadata.
- Shared `AppRuntime`, Rich REPL, headless output, and versioned run artifacts.

The SWE-bench runner is the next benchmark milestone. No benchmark score is
claimed until the official evaluator has completed.

## Install

Vedex requires Python 3.12 or newer and uses `uv` for project management.

```powershell
uv sync
```

Provider SDKs are optional. Install only the adapter you intend to use:

```powershell
uv sync --extra gemini
# or
uv sync --extra openai
```

The base package and `FakeAdapter` do not require network access or provider
credentials.

## Gemini setup

Create a Gemini API key in Google AI Studio, then add a Windows user environment
variable named `GEMINI_API_KEY`. Open a new terminal after saving the variable.
The official `google-genai` SDK and Vedex's Gemini adapter read this variable
automatically.

Run the interactive agent with the adapter factory:

```powershell
uv run vedex `
  --adapter vedex.models.gemini:create_adapter `
  --model your-gemini-model `
  --cwd .
```

Do not put the key directly in source code or command history. To check that
the variable exists without printing its value:

```powershell
if ([string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY)) {
  Write-Error "GEMINI_API_KEY is not set"
} else {
  Write-Output "GEMINI_API_KEY is set"
}
```

## Interactive use

The CLI selects one adapter and model at startup:

```text
vedex --adapter module:factory --model model-id --cwd .
```

The REPL supports `/help`, `/skills`, `/prompts`, `/skill:<name>`, `/context`,
`/reload`, `/reset`, `/clear`, `/exit`, and direct `!command` execution.

## Docker

Docker Desktop must be running with Linux containers enabled. Verify the daemon
before using a Docker environment:

```powershell
docker info
```

Benchmark environments use image-native workspaces, workspace-relative model
paths, disabled networking by default, explicit lifecycle cleanup, and recorded
image/runtime metadata.

## Development checks

Run the complete local quality gate:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
```

Normal tests do not require a live provider, paid usage, or a running Docker
daemon. Docker-specific real smoke checks and benchmark runs require Docker
Desktop and a callable model adapter.

## References

- [SWE-bench repository](https://github.com/SWE-bench/SWE-bench)
- [SWE-bench website](https://www.swebench.com/)
- [Tau repository](https://github.com/huggingface/tau)
- [mini-SWE-agent repository](https://github.com/SWE-agent/mini-swe-agent)
