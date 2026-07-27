# Vedex Agent Instructions

Vedex is a local-first, Ollama-native CLI coding agent. The goal is a fast,
minimal, readable, maintainable assistant that runs entirely on the local
machine.

## File structure

Keep the runtime small and give each file one clear responsibility:

```text
vedex/
  cli.py         # REPL, terminal UX, slash-command parsing and pickers.
  session.py     # Message-only JSONL storage, conversation history, Ollama run loop,
                 # persistence, and strict context truncation.
  workspace.py   # Skills, prompt templates, project context, system-prompt building, reload, and prompt expansion.
  core.py        # Native Ollama HTTP client, model discovery, streaming, and tool loop.
  tools/
    base.py       # Shared tool types, validation, and common helpers.
    read.py       # file-reading tool.
    write.py      # file-writing tool.
    edit.py       # precise-edit tool.
    bash.py       # shell-command tool.
  resources.py   # Filesystem paths plus skill, template, and project-context loaders.
  schema.py      # Pydantic message, event, tool-call, and tool-result contracts.
  rendering.py   # Print-mode terminal rendering only.
```

Do not introduce provider abstractions, database layers, event-sourcing
frameworks, plugin registries, TUI frameworks, or GUI frameworks.

## Core architecture

- `Session` is the one owner of conversation history and agent execution.
- `Session` receives concrete dependencies directly: working directory, model,
  system prompt, tools, Ollama client, message store, and the model context
  length.
- `SessionStore` is synchronous and stores only validated `AgentMessage` JSONL
  rows. Pydantic validation remains mandatory for message and tool-call safety.
- Persist each accepted user, completed assistant, and tool-result message at
  the moment it enters history. Do not persist partial assistant output unless
  the engine has explicitly added it to history.
- Context compaction means strict truncation only. Do not add LLM summaries,
  deterministic summaries, compaction events, replacement IDs, or replay
  logic.
- `Workspace` owns resource loading and prompt construction only. It must not
  own the Ollama execution loop, message storage, session identifiers, or
  session switching.
- `cli.py` assembles a `Workspace` and a `Session`, runs the REPL, renders
  events, and owns interactive terminal pickers.

## Sessions

- Sessions are message-only JSONL files in `~/.vedex/sessions/`.
- Each newly created session uses a six-character lowercase hexadecimal
  identifier as its filename: `<6-char-id>.jsonl`. Do not use long UUIDs or
  decorative names.
- The filesystem is the session database. Discover sessions by scanning that
  directory.
- `/resume` without an argument lists available session files in a numbered
  terminal picker. `/resume <6-char-id>` resolves that identifier directly.
- `/new` creates and switches to a new UUID JSONL file.
- `--session <6-char-id>` resolves the same global file. Supporting a direct JSONL
  path is acceptable when useful for local development.
- Because session files contain only messages, resuming uses the current CLI
  model and working directory. Do not add hidden model, title, timestamp, or
  working-directory records to session JSONL.
- Invalid JSONL is an error. Do not support or migrate the old event-ledger
  format.

## Ollama integration and models

- Vedex communicates exclusively with the native Ollama API at
  `http://localhost:11434`.
- Assume Ollama is already running. Do not add installation, startup, or
  lifecycle-management code.
- Discover models, tool capability, and context limits directly from Ollama's
  `/api/tags` endpoint. Do not hardcode model metadata.
- `/model` with no argument opens a numbered picker of locally available
  models. `/model <name>` selects and validates that local Ollama model.
- Changing models refreshes the active context-window metadata.

## CLI and user experience

- Use a simple blocking, print-mode REPL. Do not introduce Textual or another
  TUI framework.
- Keep command parsing direct and readable in `cli.py`; do not retain a generic
  command registry merely for abstraction.
- Keep these user-facing commands:
  - `/help`
  - `/clear`
  - `/exit` and `/quit`
  - `/model [name]`
  - `/skills [name]` and `/skill:<name> [request]`
  - `/prompts [name]`
  - `/context`
  - `/reload`
  - `/session` for current-session details
  - `/new`
- `/skills` without an argument lists available skills in a numbered picker;
  `/skills <name>` selects the named skill directly. `/skill:<name> [request]`
  remains the prompt-expansion form for sending a skill to the agent.
- `/prompts` without an argument lists available prompt templates in a numbered
  picker; `/prompts <name>` selects the named template directly.
- `/resume` lists session files in a numbered, readable form. A useful preview
  may be derived from the file name, modification time, and first user message;
  do not persist separate session metadata for it.
- `/session` shows current conversation/runtime status: session file ID, model,
  working directory, message count, estimated token usage, context-window
  usage, and loaded tool count. It does not list project instruction files.
- `/context` lists the active project context files and their paths—the
  instructions/resources injected into the system prompt. It does not show
  conversation history or session runtime statistics.
- Keep `!command` for a shell command whose output is added to the conversation
  and `!!command` for terminal-only output.
- Print startup and runtime failures once, concisely, to stderr or through the
  normal event renderer. Do not collect diagnostic logs for later display.

## Resources and tools

- Keep skills, prompt templates, and project context working after every
  cleanup. They are user features, not optional diagnostics.
- Required resources raise concise errors. Optional unreadable resources may
  emit one short terminal warning at load time, then continue.
- `/reload` reloads skills, templates, and project context, then rebuilds the
  system prompt when its inputs changed.
- Tool failures remain normal agent context via `AgentToolResult(ok=False, ...)`.
- Keep Pydantic for tool and message contracts.

## Python guidelines

- Target the Python version declared in `pyproject.toml`.
- Prefer strict typed dataclasses and Pydantic models where they protect an
  external or model-produced contract.
- Keep asynchronous work at the Ollama HTTP streaming boundary. Filesystem
  session storage and resource discovery stay synchronous.
- For deterministic tests, mock native `OllamaClient` HTTP responses rather
  than building fake provider abstractions.
- Before considering code complete, run Ruff checks and formatting checks, then
  run strict Mypy checks. No Ruff or Mypy errors should remain.
