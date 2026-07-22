from collections.abc import Callable, Sequence, Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent import AgentTool, list_model_info
from coding.prompt_templates import PromptTemplate
from coding.reload import CodingReloadSummary, ReloadCategorySummary
from coding.resources import ResourceDiagnostic
from coding.session_manager import SessionManager
from coding.skills import Skill
from coding.system_prompt import ProjectContextFile


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
    def resource_diagnostics(self) -> Sequence[ResourceDiagnostic]: ...

    @property
    def system_prompt(self) -> str: ...

    @property
    def session_id(self) -> str | None: ...

    @property
    def session_title(self) -> str | None: ...

    @property
    def session_manager(self) -> SessionManager | None: ...

    def ensure_session_indexed(self) -> None: ...

    def set_model(self, model: str) -> None: ...

    def reload(self) -> CodingReloadSummary: ...


@dataclass(frozen=True, slots=True)
class CommandResult:

    handled: bool
    exit_requested: bool = False
    clear_requested: bool = False
    new_session_requested: bool = False
    resume_session_id: str | None = None
    resume_picker_requested: bool = False
    model_picker_requested: bool = False
    message: str | None = None


@dataclass(frozen=True, slots=True)
class CommandContext:

    session: CommandSession
    registry: 'CommandRegistry'
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
            name="new",
            usage="/new",
            description="Start a new session.",
            handler=_new_command,
            search_terms=("clear", "reset"),
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
            name="resume",
            usage="/resume [session-id]",
            description="Resume a previous session.",
            handler=_resume_command,
            search_terms=("history", "previous"),
        )
    )
    registry.register(
        SlashCommand(
            name="name",
            usage="/name <new name>",
            description="Rename the current session.",
            handler=_name_command,
            search_terms=("rename", "title"),
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


async def _new_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, new_session_requested=True)


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
    lines.append(f"Resource diagnostics: {len(session.resource_diagnostics)}")
    if session.session_id is not None:
        lines.append(f"Session: {session.session_id}")
    if session.session_title:
        lines.append(f"Session name: {session.session_title}")
    return CommandResult(handled=True, message="\n".join(lines))


async def _system_command(context: CommandContext) -> CommandResult:
    if context.args:
        return CommandResult(handled=True, message="Usage: /system")
    return CommandResult(handled=True, message=context.session.system_prompt)


async def _skills_command(context: CommandContext) -> CommandResult:
    if not context.session.skills:
        lines = ["No skills loaded."]
        if context.session.resource_diagnostics:
            lines.append("")
            lines.extend(_format_diagnostics(context.session.resource_diagnostics, kind="skill"))
        return CommandResult(handled=True, message="\n".join(lines))

    lines = ["Available skills:"]
    for skill in sorted(context.session.skills, key=lambda item: item.name):
        description = skill.description or "No description"
        lines.append(f"- {skill.name}: {description}")
    lines.append("Use a skill with /skill:<name> [request].")
    if context.session.resource_diagnostics:
        lines.append("")
        lines.extend(_format_diagnostics(context.session.resource_diagnostics, kind="skill"))
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
        lines = ["No project context files loaded."]
        if session.resource_diagnostics:
            lines.append("")
            lines.extend(_format_diagnostics(session.resource_diagnostics, kind="context"))
        return CommandResult(handled=True, message="\n".join(lines))

    lines = ["Active project context files:"]
    lines.extend(f"- {context_file.path}" for context_file in session.context_files)
    if session.resource_diagnostics:
        lines.append("")
        lines.extend(_format_diagnostics(session.resource_diagnostics, kind="context"))
    return CommandResult(handled=True, message="\n".join(lines))


async def _skill_command(context: CommandContext) -> CommandResult:
    return CommandResult(
        handled=True,
        message="Use /skill:<name> [request] to expand a loaded skill into your prompt.",
    )


async def _resume_command(context: CommandContext) -> CommandResult:
    if not context.args:
        return CommandResult(handled=True, resume_picker_requested=True)
    manager = context.session.session_manager
    if manager is None:
        return CommandResult(handled=True, message="Session manager is not available.")
    session_id = context.args.strip()
    if manager.get_session(session_id) is None:
        return CommandResult(handled=True, message=f"Unknown session: {session_id}")
    return CommandResult(
        handled=True,
        resume_session_id=session_id,
    )


async def _name_command(context: CommandContext) -> CommandResult:
    manager = context.session.session_manager
    session_id = context.session.session_id
    if manager is None or session_id is None:
        return CommandResult(handled=True, message="Session manager is not available.")

    if not context.args:
        record = manager.get_session(session_id)
        title = (
            record.title if record is not None else context.session.session_title
        ) or "Untitled session"
        return CommandResult(
            handled=True,
            message=f"Current session name: {title}\nUsage: /name <new name>",
        )

    try:
        name = _validated_session_name(context.args)
    except ValueError as exc:
        return CommandResult(handled=True, message=str(exc))

    if manager.get_session(session_id) is None:
        context.session.ensure_session_indexed()

    updated = manager.touch_session(
        session_id,
        model=context.session.model,
        title=name,
    )
    if updated is None:
        return CommandResult(handled=True, message=f"Unknown current session: {session_id}")
    return CommandResult(handled=True, message=f"Session renamed: {updated.title}")


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


def _format_diagnostics(
    diagnostics: Sequence[ResourceDiagnostic], *, kind: str | None = None
) -> list[str]:
    filtered = [diagnostic for diagnostic in diagnostics if kind is None or diagnostic.kind == kind]
    if not filtered:
        return ["Resource diagnostics: none"]
    lines = ["Resource diagnostics:"]
    lines.extend(f"- {diagnostic.format()}" for diagnostic in filtered)
    return lines


def _format_reload_summary(summary: CodingReloadSummary) -> str:
    lines = [
        "Reloaded local coding resources and project context.",
        "Resources:",
        f"- Skills: {_format_reload_category(summary.skills)}",
        f"- Prompt templates: {_format_reload_category(summary.prompt_templates)}",
        "Context:",
        f"- Project context files: {_format_reload_category(summary.context_files)}",
        "- Next-turn system prompt: "
        + ("rebuilt" if summary.system_prompt_rebuilt else "unchanged"),
        "Diagnostics:",
        f"- Resource diagnostics: {_format_reload_category(summary.diagnostics)}"
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


def _validated_session_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise ValueError("Usage: /name <new name>")
    if any(char in name for char in "\r\n\t"):
        raise ValueError("Session name must be a single line.")
    return name


def _normalize_name(name: str) -> str:
    return name.strip().removeprefix("/").lower()