from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path

from .commands import CommandRegistry, CommandResult, create_default_command_registry
from .context_window import (
    DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    ContextUsageEstimate,
    auto_compaction_threshold_for_context_window,
    estimate_context_usage,
    estimate_message_tokens,
)
from .core import OllamaClient, get_model_info, run_agent_loop
from .resources import ProjectContextFile, PromptTemplate, ResourceError, ResourcePaths, Skill
from .schema import (
    AgentEvent,
    AgentMessage,
    AgentTool,
    ErrorEvent,
    MessageEndEvent,
    ToolExecutionEndEvent,
    UserMessage,
)
from .session import SessionStore
from .tools import create_bash_tool, create_coding_tools
from .workspace import ReloadSummary, Workspace


@dataclass(frozen=True, slots=True)
class TerminalCommandResult:
    command: str
    output: str
    exit_code: int | None
    ok: bool
    added_to_context: bool


@dataclass(frozen=True, slots=True)
class TerminalCommandRequest:
    command: str
    add_to_context: bool


@dataclass(frozen=True, slots=True)
class CodingSessionConfig:
    ollama_host: str
    model: str
    storage: SessionStore
    cwd: Path
    system: str | None = None
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None
    context_files: tuple[ProjectContextFile, ...] = ()
    tools: list[AgentTool] | None = None
    workspace: Workspace | None = None
    resource_paths: ResourcePaths | None = None
    command_registry: CommandRegistry | None = None
    auto_compact_token_threshold: int | None = None
    auto_compact_enabled: bool = True


class CodingSession:
    """Temporary runner that uses the message-only store before Session replaces it."""

    def __init__(
        self,
        config: CodingSessionConfig,
        *,
        messages: list[AgentMessage],
        system_prompt: str,
        tools: list[AgentTool],
        client: OllamaClient,
        context_window_tokens: int,
        workspace: Workspace,
        command_registry: CommandRegistry | None = None,
    ) -> None:
        self._config = config
        self._messages = messages
        self._system_prompt = system_prompt
        self._tools = tools
        self._client = client
        self._context_window_tokens = context_window_tokens
        self._workspace = workspace
        self._command_registry = command_registry or create_default_command_registry()
        self._auto_compact_token_threshold = config.auto_compact_token_threshold
        self._auto_compact_enabled = config.auto_compact_enabled
        self._context_usage_cache: ContextUsageEstimate | None = None

    @classmethod
    async def load(cls, config: CodingSessionConfig) -> CodingSession:
        tools = config.tools if config.tools is not None else create_coding_tools(cwd=config.cwd)
        workspace = config.workspace or Workspace(
            cwd=config.cwd,
            tools=tools,
            resource_paths=config.resource_paths,
            custom_system_prompt=config.custom_system_prompt,
            append_system_prompt=config.append_system_prompt,
            context_files=config.context_files,
        )
        client = OllamaClient(config.ollama_host)
        context_window_tokens = DEFAULT_CONTEXT_WINDOW_TOKENS
        try:
            model_info = await get_model_info(config.model, client=client)
            context_window_tokens = model_info.context_length or DEFAULT_CONTEXT_WINDOW_TOKENS
        except LookupError:
            pass

        return cls(
            config,
            messages=config.storage.load(),
            system_prompt=config.system or workspace.system_prompt,
            tools=tools,
            client=client,
            context_window_tokens=context_window_tokens,
            workspace=workspace,
            command_registry=config.command_registry,
        )

    @property
    def cwd(self) -> Path:
        return self._config.cwd

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        return tuple(self._tools)

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return tuple(self._messages)

    @property
    def storage(self) -> SessionStore:
        return self._config.storage

    @property
    def skills(self) -> tuple[Skill, ...]:
        return self._workspace.skills

    @property
    def prompt_templates(self) -> tuple[PromptTemplate, ...]:
        return self._workspace.prompt_templates

    @property
    def context_files(self) -> tuple[ProjectContextFile, ...]:
        return self._workspace.context_files

    @property
    def context_token_estimate(self) -> int:
        return self.context_usage.total_tokens

    @property
    def context_usage(self) -> ContextUsageEstimate:
        if self._context_usage_cache is None:
            self._context_usage_cache = estimate_context_usage(
                system=self._system_prompt,
                messages=tuple(self._messages),
                tools=self.tools,
            )
        return self._context_usage_cache

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def auto_compact_token_threshold(self) -> int | None:
        if not self._auto_compact_enabled:
            return None
        if self._auto_compact_token_threshold is not None:
            return self._auto_compact_token_threshold
        return auto_compaction_threshold_for_context_window(self.context_window_tokens)

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens

    @property
    def command_registry(self) -> CommandRegistry:
        return self._command_registry

    def set_model(self, model: str) -> None:
        self._config = replace(self._config, model=model)

    def reload(self) -> ReloadSummary:
        summary = self._workspace.reload()
        if self._config.system is None and summary.system_prompt_rebuilt:
            self._system_prompt = self._workspace.system_prompt
            self._invalidate_context_usage_cache()
        return summary

    async def aclose(self) -> None:
        await self._client.aclose()

    async def handle_command(self, text: str) -> CommandResult:
        if self._workspace.expand_prompt_template_command(text) is not None:
            return CommandResult(handled=False)
        return await self._command_registry.execute(self, text)

    def expand_prompt_text(self, text: str) -> str:
        return self._workspace.expand_prompt_text(text)

    async def run_terminal_command(
        self,
        command: str,
        *,
        add_to_context: bool,
    ) -> TerminalCommandResult:
        normalized_command = command.strip()
        if not normalized_command:
            raise ValueError("Terminal command cannot be empty")

        result = await create_bash_tool(cwd=self.cwd).execute({"command": normalized_command})
        exit_code = result.data.get("exit_code") if result.data is not None else None
        if not isinstance(exit_code, int):
            exit_code = None

        if add_to_context:
            self._messages.append(
                UserMessage(
                    content=(
                        "Terminal command executed by the user.\n\n"
                        f"Command:\n```bash\n{normalized_command}\n```\n\n"
                        f"Output:\n```text\n{result.content}\n```"
                    )
                )
            )
            self._invalidate_context_usage_cache()
            self._persist_messages_since(len(self._messages) - 1)

        return TerminalCommandResult(
            command=normalized_command,
            output=result.content,
            exit_code=exit_code,
            ok=result.ok,
            added_to_context=add_to_context,
        )

    async def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        try:
            expanded_content = self.expand_prompt_text(content)
        except ResourceError as exc:
            yield ErrorEvent(message=str(exc), recoverable=True)
            return

        self._messages.append(UserMessage(content=expanded_content))
        self._invalidate_context_usage_cache()
        self._try_auto_truncate()
        persisted_count = len(self._messages) - 1
        overflow_event: ErrorEvent | None = None
        try:
            async for event in run_agent_loop(
                client=self._client,
                model=self.model,
                system=self._system_prompt,
                messages=self._messages,
                tools=self._tools,
            ):
                if isinstance(event, MessageEndEvent):
                    persisted_count = self._persist_messages_since(persisted_count)
                if isinstance(event, ToolExecutionEndEvent):
                    self._invalidate_context_usage_cache()
                if (
                    isinstance(event, ErrorEvent)
                    and not event.recoverable
                    and _is_context_overflow(event)
                ):
                    overflow_event = event
                yield event

            persisted_count = self._persist_messages_since(persisted_count)
            if overflow_event is not None and self._truncate_history():
                retry_persisted_count = len(self._messages)
                async for event in run_agent_loop(
                    client=self._client,
                    model=self.model,
                    system=self._system_prompt,
                    messages=self._messages,
                    tools=self._tools,
                ):
                    if isinstance(event, MessageEndEvent):
                        retry_persisted_count = self._persist_messages_since(retry_persisted_count)
                    if isinstance(event, ToolExecutionEndEvent):
                        self._invalidate_context_usage_cache()
                    yield event
                self._persist_messages_since(retry_persisted_count)
                return
            self._try_auto_truncate()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            yield ErrorEvent(message=f"Agent run failed: {exc}")

    async def continue_(self) -> AsyncIterator[AgentEvent]:
        persisted_count = len(self._messages)
        try:
            async for event in run_agent_loop(
                client=self._client,
                model=self.model,
                system=self._system_prompt,
                messages=self._messages,
                tools=self._tools,
            ):
                if isinstance(event, MessageEndEvent):
                    persisted_count = self._persist_messages_since(persisted_count)
                if isinstance(event, ToolExecutionEndEvent):
                    self._invalidate_context_usage_cache()
                yield event
            self._persist_messages_since(persisted_count)
            self._try_auto_truncate()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            yield ErrorEvent(message=f"Agent run failed: {exc}")

    def _persist_messages_since(self, persisted_count: int) -> int:
        for message in self._messages[persisted_count:]:
            self.storage.append(message)
        self._invalidate_context_usage_cache()
        return len(self._messages)

    def _try_auto_truncate(self) -> bool:
        threshold = self.auto_compact_token_threshold
        if threshold is None or self.context_token_estimate <= threshold:
            return False
        return self._truncate_history()

    def _truncate_history(self) -> bool:
        first_kept_index = _first_recent_message_index(
            self._messages,
            keep_recent_tokens=DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
        )
        if first_kept_index <= 0:
            return False
        kept_messages = self._messages[first_kept_index:]
        self.storage.rewrite(kept_messages)
        self._messages = kept_messages
        self._invalidate_context_usage_cache()
        return True

    def _invalidate_context_usage_cache(self) -> None:
        self._context_usage_cache = None


def _first_recent_message_index(
    messages: list[AgentMessage],
    *,
    keep_recent_tokens: int,
) -> int:
    if len(messages) < 2:
        return 0
    if keep_recent_tokens <= 0:
        return _next_user_message_index(messages, start=1) or len(messages)

    accumulated_tokens = 0
    candidate_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        accumulated_tokens += estimate_message_tokens(messages[index])
        if accumulated_tokens >= keep_recent_tokens:
            candidate_index = index
            break
    if candidate_index is None:
        return 0
    if messages[candidate_index].role == "user":
        return (
            candidate_index
            if candidate_index > 0
            else _next_user_message_index(messages, start=1) or 0
        )
    return _next_user_message_index(messages, start=candidate_index + 1) or len(messages)


def _next_user_message_index(messages: list[AgentMessage], *, start: int) -> int | None:
    for index in range(start, len(messages)):
        if messages[index].role == "user":
            return index
    return None


def _is_context_overflow(event: ErrorEvent) -> bool:
    text = f"{event.message} {event.data or ''}".lower()
    markers = (
        "context length",
        "context window",
        "context limit",
        "maximum context",
        "max context",
        "input is too long",
        "input length",
        "prompt is too long",
        "too many tokens",
        "token limit",
        "exceeds the limit",
        "exceeded the limit",
    )
    return any(marker in text for marker in markers)


def parse_terminal_command(text: str) -> TerminalCommandRequest | None:
    stripped = text.strip()
    if stripped.startswith("!!"):
        command = stripped[2:].strip()
        return TerminalCommandRequest(command, add_to_context=False) if command else None
    if stripped.startswith("!"):
        command = stripped[1:].strip()
        return TerminalCommandRequest(command, add_to_context=True) if command else None
    return None
