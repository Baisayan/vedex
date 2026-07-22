# Vedex Agent Instructions

Vedex is a local-first Ollama-native CLI coding agent. The goal is to build a lightning fast, readable, and maintainable assistant that runs entirely on the local machine.

# Architecture



---

# Ollama Integration

- Vedex communicates exclusively with the native Ollama API at `http://localhost:11434`.
- **No Lifecycle Management:** Assume Ollama is already running. Do not implement installation or startup scripts.
- **Native Models:** Model discovery, context limits, and capabilities must be queried directly from Ollama's `/api/tags` endpoint.

---

# Model Management

Models are discovered directly from Ollama. Ollama lists all the available models on a device. Users may switch models through CLI commands. The model list should always come from the local Ollama instance.

Model metadata such as context window size, thinking capability, and other runtime information should come from the native Ollama API rather than hardcoded values.

---

# CLI

- Vedex targets a premium, print-mode terminal experience.
- Do not introduce Textual, TUI frameworks, or any GUI frameworks. 
- The CLI operates as a simple, blocking Read-Eval-Print Loop (REPL) that consumes the synchronous stream of events from the engine.

---

# Python Guidelines

- Target the Python version declared in `pyproject.toml`.
- Prefer strict typed `dataclasses` or Pydantic models.
- Keep async boundaries isolated strictly to the HTTP streaming layer.
- For deterministic testing, mock the `OllamaClient` HTTP responses rather than building fake provider abstractions.

---
