# Vedex Agent Instructions

Vedex is a local-first Ollama only CLI coding agent inspired by Pi's minimalist agent harness architecture.

The goal is to build a fast, understandable, and maintainable coding agent that runs entirely on the local machine using Ollama.

# Architecture

Preserve separation of concerns.

```
ollama     native Ollama client and streaming
agent      reusable agent runtime, loop, events, tools, sessions
coding     CLI application, resources, prompts, skills, commands
```

The core agent package must remain independent of CLI concerns, prompt loading, session locations, and application-specific resources.

---

# Ollama Integration

Vedex communicates exclusively with the native Ollama API.

Assume Ollama is already running at:

```
http://localhost:11434
```

Do not implement installation, startup, or lifecycle management for Ollama.

Use native endpoints whenever possible.

---

# Model Management

Models are discovered directly from Ollama.

Users may switch models through CLI commands.

The model list should always come from the local Ollama instance.

Model metadata such as context window size, thinking capability, and other runtime information should come from the native Ollama API rather than hardcoded values.

---

# CLI

Vedex currently targets a print-mode CLI.

Do not introduce:

- Textual TUI
- GUI frameworks

The CLI consumes agent events and renders them to stdout.

---

# Python Guidelines

- Target the Python version declared in `pyproject.toml`.

- Prefer typed dataclasses or schema models.

- Keep async boundaries explicit.

- Use fake provider and fake tools for deterministic agent-loop tests.

---
