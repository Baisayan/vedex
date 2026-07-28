# Vedex

**A small local coding agent built for Ollama and the terminal.**

Vedex is an Ollama-native terminal coding agent. Point it at a repository, pick
a model already on your machine, and let it read files, make exact edits, write
new ones, and run commands. It requires no cloud account, provider setup, or
approval workflow.

It is intentionally small: native Ollama HTTP, a blocking terminal REPL, four
tools, Markdown resources, and JSONL sessions on your own filesystem.

## Install and run

You need Python 3.12+, [Ollama](https://ollama.com/) running at
`http://localhost:11434`, and at least one local model:

```bash
ollama list
```

Install Vedex directly from GitHub with pip:

```bash
python -m pip install git+https://github.com/Baisayan/vedex.git
```

Or install it as a `uv` tool:

```bash
uv tool install git+https://github.com/Baisayan/vedex.git
```

For a pinned release, add a tag to the URL:

```bash
python -m pip install git+https://github.com/Baisayan/vedex.git@v0.1.0
```

Then enter the repository you want to work in and start a session:

```bash
cd path/to/project
vedex --model <your-local-model-name> --cwd .
```

Leave out `--model` to get a numbered picker of models discovered from your
local Ollama installation. Vedex does not hardcode a default model.

## Using Vedex

Vedex is a simple REPL. These options get you in the right place:

| Command | What it does |
| --- | --- |
| `vedex` | Start Vedex and pick a local model. |
| `vedex --model <name>` | Start with a specific locally installed Ollama model. |
| `vedex --cwd <path>` | Give tools a different working directory. |
| `vedex --session <id-or-path>` | Resume a six-character session ID or a JSONL file. |

Once inside, these are the useful slash commands:

| Command | What it does |
| --- | --- |
| `/help` | Show the command cheat sheet. |
| `/clear` | Clear the terminal. |
| `/exit`, `/quit` | Leave gracefully. |
| `/model [name]` | Pick a local model or select one by name. |
| `/skills [name]` | List, pick, or inspect a skill. |
| `/skill:<name> [request]` | Send a skill-guided request to the agent. |
| `/prompts [name]` | List, pick, or inspect a prompt template. |
| `/<prompt> [arguments]` | Render a loaded prompt template and send it. |
| `/context` | Show instruction files injected into the system prompt. |
| `/reload` | Reload skills, templates, and context from disk. |
| `/session` | Show session ID, model, cwd, token estimate, and tool count. |
| `/new` | Create and switch to a fresh session. |
| `/resume [id]` | Pick a saved session or resume one directly. |

`!command` adds shell output to conversation history; `!!command` prints shell
output only and leaves history unchanged.

## Tools

Vedex gives the model four focused tools:

| Tool | Job |
| --- | --- |
| `read` | Read bounded ranges from UTF-8 text files, with line numbers and offsets. |
| `write` | Create or replace a complete text file, including parent directories. |
| `edit` | Apply one or more exact, unique, non-overlapping text replacements. |
| `bash` | Run a foreground shell command with a timeout and bounded combined output. |

The most reliable prompting pattern is to read before editing, make the
smallest change, then run the relevant command.

Want `grep`, `ls`, or another tool later? Add one focused module under
`vedex/tools/` and register it in `create_coding_tools()`. No plugin system is
required.

## Resources and instructions

Vedex loads Markdown resources from global `~/.vedex/` and the current
project's `.vedex/` directory. Skills live in `skills/`; prompt templates live
in `prompts/`. Global resources load first, and later duplicates are ignored
with one concise warning. Resource names match case-insensitively.

`AGENTS.md` files are always-present instructions. Vedex loads the global file,
the project-root file, any `AGENTS.md` files between that root and the working
directory, plus `.vedex/AGENTS.md`. `/context` reports the files injected into
the system prompt.

Skills are task-specific instructions selected through `/skills` or
`/skill:<name>`. Prompt templates are reusable slash commands that receive
their trailing arguments. Use `/reload` after changing any resource.

## Sessions and context

Every session is a plain message-only JSONL file at
`~/.vedex/sessions/<six-character-id>.jsonl`.

The six-character lowercase hexadecimal filename is the session ID. The file
contains user messages, completed assistant messages, and tool results—enough
context to resume the conversation, without a database, index, title, or hidden
metadata.

When the model context window gets crowded, Vedex strictly drops the oldest
valid history that no longer fits. It does not ask another model for a summary,
invent compaction events, or keep a replay ledger. The newest coherent turns
remain in history.

## Troubleshooting

| Symptom | Usually means | Try this |
| --- | --- | --- |
| `Ollama is unavailable` | Ollama is not reachable on its default local port. | Start Ollama, then try again. |
| No models appear | Ollama has no local models to offer. | Check `ollama list`, then pull a model. |
| Model is unavailable | The selected name is not installed locally. | Use `/model` to pick from discovered models. |
| A skill or prompt is missing | It was added after startup or has a duplicate name. | Run `/reload`, then inspect `/skills` or `/prompts`. |
| A tool returns an error | The model made a bad request or the filesystem/shell rejected it. | Read the returned error and retry with a narrower request. |
| A session will not resume | Its JSONL contains an invalid row. | Vedex reports the exact file and line; repair or discard that test session. |

## Architecture, for curious contributors

Vedex keeps its runtime responsibilities separate. Each file has one job:

```text
vedex/
  cli.py         # REPL, slash commands, terminal pickers
  session.py     # JSONL messages, history, persistence, execution, truncation
  workspace.py   # skills, prompts, AGENTS.md context, system-prompt reloads
  core.py        # native Ollama HTTP, streaming, model discovery, tool loop
  tools/
    base.py      # shared tool types and helpers
    read.py      # read text files
    write.py     # write complete files
    edit.py      # exact text edits
    bash.py      # foreground shell commands
  resources.py   # resource paths and Markdown loaders
  schema.py      # Pydantic message, event, and tool contracts
  rendering.py   # terminal event rendering
```

`Session` owns one conversation and its message-only store. `Workspace` owns
prompt resources and knows nothing about Ollama or sessions. `core.py` speaks
native Ollama; tools remain small and explicit. These boundaries keep the
runtime readable and easy to extend.
