from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .core import list_model_info
from .resources import (
    ProjectContextFile,
    PromptTemplate,
    Skill,
)
from .schema import AgentTool
from .workspace import ReloadCategorySummary, ReloadSummary


class CommandSession(Protocol):
    @property
    def cwd(self) -> Path: ...

    @property
    def model(self) -> str: ...

    @property
    def tools(self) -> Sequence[AgentTool]: ...

    @property
    def skills(self) -> Sequence[Skill]: ...

    @property
    def prompt_templates(self) -> Sequence[PromptTemplate]: ...

    @property
    def context_files(self) -> Sequence[ProjectContextFile]: ...

    @property
    def context_token_estimate(self) -> int: ...

    @property
    def auto_compact_token_threshold(self) -> int | None: ...

    @property
    def context_window_tokens(self) -> int: ...

    @property
    def system_prompt(self) -> str: ...

    def set_model(self, model: str) -> None: ...

    def reload(self) -> ReloadSummary: ...


@dataclass(frozen=True, slots=True)
class CommandResult:
    handled: bool
    exit_requested: bool = False
    clear_requested: bool = False
    model_picker_requested: bool = False
    message: str | None = None


@dataclass(frozen=True, slots=True)
class CommandContext:
    session: CommandSession
    registry: "CommandRegistry"
    text: str
    name: str
    args: str


CommandHandler = Callable[[CommandContext], Awaitable[CommandResult]]


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    description: str
    usage: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()
    search_terms: tuple[str, ...] = ()


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._aliases: dict[str, str] = {}

    def register(self, command: SlashCommand) -> None:
        """Register a slash command and its aliases."""
        name = _normalize_name(command.name)
        if name in self._commands:
            raise ValueError(f"Duplicate slash command: /{name}")
        self._commands[name] = command
        for alias in command.aliases:
            normalized_alias = _normalize_name(alias)
            if normalized_alias in self._commands or normalized_alias in self._aliases:
                raise ValueError(f"Duplicate slash command alias: /{normalized_alias}")
            self._aliases[normalized_alias] = name

    def get(self, name: str) -> SlashCommand | None:
        """Return a command by name or alias."""
        normalized = _normalize_name(name)
        command_name = self._aliases.get(normalized, normalized)
        return self._commands.get(command_name)

    def list_commands(self) -> tuple[SlashCommand, ...]:
        """Return registered commands sorted by name."""
        return tuple(self._commands[name] for name in sorted(self._commands))

    async def execute(self, session: CommandSession, text: str) -> CommandResult:
        """Execute a slash command, or return unhandled for ordinary prompts."""
        stripped = text.strip()
        if not stripped.startswith("/"):
            return CommandResult(handled=False)

        if stripped.startswith("/skill:"):
            return CommandResult(handled=False)

        name, args = _parse_command(stripped)
        if not name:
            return CommandResult(handled=False)

        command = self.get(name)
        if command is None:
            return CommandResult(handled=True, message=f"Unknown command: /{name}")

        return await command.handler(
            CommandContext(session=session, registry=self, text=stripped, name=name, args=args)
        )


def create_default_command_registry() -> CommandRegistry:
    registry = CommandRegistry()
    registry.register(
        SlashCommand(
            name="help",
            usage="/help",
            description="Show available commands.",
            handler=_help_command,
            aliases=("?",),
        )
    )
    registry.register(
        SlashCommand(
            name="quit",
            usage="/quit",
            description="Exit the current session.",
            handler=_exit_command,
            aliases=("exit", "bye"),
        )
    )
    registry.register(
        SlashCommand(
            name="session",
            usage="/session",
            description="Show session info and stats.",
            handler=_status_command,
            search_terms=("info", "status"),
        )
    )
    registry.register(
        SlashCommand(
            name="system",
            usage="/system",
            description="Show the active system prompt without saving it.",
            handler=_system_command,
            search_terms=("prompt", "instructions"),
        )
    )
    registry.register(
        SlashCommand(
            name="skills",
            usage="/skills",
            description="List available skills.",
            handler=_skills_command,
        )
    )
    registry.register(
        SlashCommand(
            name="skill",
            usage="/skill:<name> [request]",
            description="Expand a loaded skill into your prompt.",
            handler=_skill_command,
            search_terms=("skills",),
        )
    )
    registry.register(
        SlashCommand(
            name="context",
            usage="/context",
            description="List active project context files.",
            handler=_context_command,
        )
    )
    registry.register(
        SlashCommand(
            name="reload",
            usage="/reload",
            description="Reload local resources and project context.",
            handler=_reload_command,
        )
    )
    registry.register(
        SlashCommand(
            name="model",
            usage="/model",
            description="Choose the active model.",
            handler=_model_command,
        )
    )
    return registry


async def _help_command(context: CommandContext) -> CommandResult:
    lines = ["Available commands:"]
    for command in context.registry.list_commands():
        lines.append(f"{command.usage}\t{command.description}")
    return CommandResult(handled=True, message="\n".join(lines))


async def _exit_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, exit_requested=True, message="Exiting session.")


async def _status_command(context: CommandContext) -> CommandResult:
    session = context.session
    context_usage = getattr(session, "context_usage", None)
    lines = [
        f"Model: {session.model}",
        f"CWD: {session.cwd}",
        f"Tools: {len(session.tools)}",
        f"Skills: {len(session.skills)}",
        f"Prompt templates: {len(session.prompt_templates)}",
        f"Context files: {len(session.context_files)}",
        f"Estimated context tokens: {session.context_token_estimate}",
        f"Context window: {session.context_window_tokens}",
    ]
    if context_usage is not None:
        lines.append(
            "Context token breakdown: "
            f"system={context_usage.system_tokens}, "
            f"messages={context_usage.message_tokens}, "
            f"tools={context_usage.tool_tokens}",
        )
    return CommandResult(handled=True, message="\n".join(lines))


async def _system_command(context: CommandContext) -> CommandResult:
    if context.args:
        return CommandResult(handled=True, message="Usage: /system")
    return CommandResult(handled=True, message=context.session.system_prompt)


async def _skills_command(context: CommandContext) -> CommandResult:
    if not context.session.skills:
        return CommandResult(handled=True, message="No skills loaded.")

    lines = ["Available skills:"]
    for skill in sorted(context.session.skills, key=lambda item: item.name):
        description = skill.description or "No description"
        lines.append(f"- {skill.name}: {description}")
    lines.append("Use a skill with /skill:<name> [request].")
    return CommandResult(handled=True, message="\n".join(lines))


async def _reload_command(context: CommandContext) -> CommandResult:
    try:
        summary = context.session.reload()
    except ValueError as exc:
        return CommandResult(handled=True, message=f"Could not reload: {exc}")

    return CommandResult(
        handled=True,
        message=_format_reload_summary(summary),
    )


async def _context_command(context: CommandContext) -> CommandResult:
    session = context.session
    if not session.context_files:
        return CommandResult(handled=True, message="No project context files loaded.")

    lines = ["Active project context files:"]
    lines.extend(f"- {context_file.path}" for context_file in session.context_files)
    return CommandResult(handled=True, message="\n".join(lines))


async def _skill_command(context: CommandContext) -> CommandResult:
    return CommandResult(
        handled=True,
        message="Use /skill:<name> [request] to expand a loaded skill into your prompt.",
    )


async def _model_command(context: CommandContext) -> CommandResult:
    if context.args:
        model = context.args.strip()
        try:
            models_info = await list_model_info()
            available_models = {info.name for info in models_info}
            if available_models and model not in available_models:
                models_str = ", ".join(sorted(available_models))
                return CommandResult(
                    handled=True,
                    message=f"Unknown local model: {model}\nAvailable Ollama models: {models_str}",
                )
        except Exception as exc:
            return CommandResult(handled=True, message=f"Could not connect to Ollama: {exc}")

        context.session.set_model(model)
        return CommandResult(handled=True, message=f"Current model set to: {model}")

    return CommandResult(handled=True, model_picker_requested=True)


def _format_reload_summary(summary: ReloadSummary) -> str:
    lines = [
        "Reloaded local coding resources and project context.",
        "Resources:",
        f"- Skills: {_format_reload_category(summary.skills)}",
        f"- Prompt templates: {_format_reload_category(summary.prompt_templates)}",
        "Context:",
        f"- Project context files: {_format_reload_category(summary.context_files)}",
        "- Next-turn system prompt: "
        + ("rebuilt" if summary.system_prompt_rebuilt else "unchanged"),
    ]
    return "\n".join(lines)


def _format_reload_category(summary: ReloadCategorySummary) -> str:
    status = "changed" if summary.changed else "unchanged"
    delta = _format_count_delta(summary.delta)
    suffix = f", {delta}" if delta is not None else ""
    return f"{summary.after} total ({status}{suffix})"


def _format_count_delta(delta: int) -> str | None:
    if delta == 0:
        return None
    return f"{delta:+d}"


def _parse_command(text: str) -> tuple[str, str]:
    command, separator, args = text[1:].partition(" ")
    return _normalize_name(command), args.strip() if separator else ""


def _normalize_name(name: str) -> str:
    return name.strip().removeprefix("/").lower()
