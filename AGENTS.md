# Vedex Agent Instructions

This is the target specification for Vedex. It defines the features, architecture, boundaries, quality requirements, and benchmark goals.

## Goal

Vedex is a terminal coding agent. It should combine:

- A small, understandable agent core.
- Practical coding tools and project awareness.
- An adapter-based model layer that can grow to multiple providers.
- Local and isolated execution environments.
- Interactive streaming and reliable headless execution.
- Strong automated tests and reproducible benchmark integrations.

## Architecture

The runtime is composed of these independent responsibilities:

- **User experience:** interactive CLI, headless runner, event rendering, skills,
  project instructions, and resource reload.
- **Agent:** a provider/environment-neutral coordinator with a narrow
  responsibility boundary. Keep it cohesive, readable, and easy to test; use
  design clarity rather than an arbitrary size target.
- **Model layer:** a `ModelAdapter` protocol producing normalized `ModelEvent`
  values. `FakeAdapter` provides deterministic tests, while optional real
  adapters keep their SDKs and configuration behind the same protocol.
- **Environment layer:** a common environment contract with base, local, and
  Docker implementations.
- **Tools:** focused read, write, edit, and bash capabilities exposed through
  the active environment.
- **Rendering:** consumes normalized `AgentEvent` values and knows nothing about
  provider response formats.
- **Benchmarks:** thin runners that invoke the same headless Agent path and produce official submission artifacts.

Vedex remains one distributable Python package. Components have clear ownership; the Agent must not absorb provider, CLI, environment, or benchmark logic.

## Features

### User experience

- Read, write, edit, and bash tools.
- Assistant/tool event streaming.
- Simple blocking terminal interaction with Rich rendering.
- Headless execution for scripts, CI, and benchmark runners.
- Project instruction discovery.
- Skills and prompt templates.
- Context inspection, resource reload, and in-memory reset interactions.
- Stable structured output and exit statuses in headless mode.
- No TUI, GUI, web dashboard, plugin marketplace, or approval system.

### Agent behavior

- The loop requests a model response, forwards deltas, accepts a completed assistant message, executes tool calls, appends tool results, and continues.
- Tool calls execute sequentially by default for predictable file/shell changes.
- Completion, model failure, tool failure, cancellation, timeout, context, and
  step/turn limits have explicit terminal results.
- Context trimming preserves coherent user/tool turns.
- The Agent emits normalized events and never parses provider-specific streams.

### Model extensibility

- `ModelAdapter` is the only model boundary.
- `FakeAdapter` is the deterministic fake model used by offline tests.
- The adapter contract supports streaming text, optional thinking deltas, tool calls, usage, completion, cancellation, and normalized failure.
- Provider-specific conversion and SDK details stay inside adapters.
- The base package and FakeAdapter work without network access or a provider account.
- FakeAdapter and every supported real adapter pass the shared adapter contract.
- Native function/tool calling is the initial action interface; free-form text action parsing is not part of the target.

### Environments

- A base environment contract owns lifecycle, workspace, tools, limits, metadata, and patch/submission collection.
- Local execution uses the current filesystem and subprocesses.
- Docker execution exposes the same tool names and schemas in an isolated workspace.
- Model-visible paths are workspace-relative.
- Startup, timeout, cancellation, failure, and cleanup behavior is explicit.
- Docker benchmark runs default to disabled networking and record image, architecture, and resource details.
- Local and Docker implementations pass equivalent tool-behavior and cleanup checks.

### Workspace and resources

- Global/project skills and prompt templates remain first-class features.
- Project instructions are discovered from the repository and supported parent paths.
- Resource loading and system-prompt construction remain separate from Agent execution and message memory.
- Required resources fail clearly; optional unreadable resources produce a concise warning.

## Core contracts

### ModelAdapter

The adapter accepts normalized system/messages/tools/settings and returns an asynchronous normalized `ModelEvent` stream. Provider-specific objects never escape this boundary.

Normalized model events cover:

- Start.
- Text delta.
- Optional thinking delta.
- Completed assistant message with assembled tool calls and usage.
- Normalized failure or cancellation.

Messages may carry opaque JSON-serializable provider metadata when an adapter needs to preserve provider-specific continuation data. The Agent never interprets this metadata.

### AgentEvent

The Agent exposes normalized events for rendering and headless recording:

- Run/turn start and completion.
- Assistant text/thinking deltas.
- Tool start and completion.
- Tool failures, limits, cancellation, and terminal errors.

Renderers and recorders consume AgentEvent values only.

### Environment

The environment supplies the active tools, owns execution location/lifecycle,
and can collect a patch or benchmark submission. The Agent receives tools, not
subprocess objects, Docker clients, or filesystem implementation details.

### Tools

Each tool has a stable name, JSON-compatible schema, concise model description,
bounded output, and deterministic success/failure tests. Ordinary tool errors
become unsuccessful tool results that remain available to the model as context.

## Quality requirements

After every feature or meaningful refactor, complete all of the following:

- Run focused Pytest tests while developing.
- Run the full Pytest suite.
- Run Ruff lint and formatting verification.
- Run BasedPyright for the package and tests using the checked-in project configuration.
- Run the relevant mocked or real integration check for CLI, adapter, environment, or benchmark changes.
- Commit each completed end-to-end feature or meaningful milestone after its tests and quality checks pass; keep commits focused and coherent. Don't commit work in progress or incomplete features or small diffs.

The test suite must cover:

- Message, tool, event, usage, and serialization contracts.
- FakeAdapter streaming, tool calls, malformed responses, failures, limits, and
  cancellation.
- Agent completion, sequential tools, context trimming, and terminal statuses.
- Tool bounds, filesystem behavior, exact edits, subprocess timeout, and errors.
- Local environment behavior and Docker parity/cleanup.
- Interactive resource behavior and ephemeral state.
- Headless output and stable result/exit behavior.
- Benchmark serialization and runners using tiny fake fixtures without live API calls.

Live provider tests are optional. Normal tests must not require network access,
paid usage, or a particular provider.

Boundary failures must be explicit: malformed model data, tool errors, provider
errors, stream interruption, cancellation, timeout, context overflow, invalid
configuration, and environment cleanup failures must produce a clear result or error. Do not silently continue after a terminal failure or report success after one.

Keep strict typing at all public boundaries. Narrow untyped JSON/SDK/subprocess data immediately and use typed protocols/models for messages, tools, events, adapters, and environments.

## Reference projects

Keep only these external references in project documentation:

- [SWE-bench repository](https://github.com/SWE-bench/SWE-bench)
- [SWE-bench website](https://www.swebench.com/)
- [Tau repository](https://github.com/huggingface/tau)
- [mini-SWE-agent repository](https://github.com/SWE-agent/mini-swe-agent)
