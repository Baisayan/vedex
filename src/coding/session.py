from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path

from agent import (
    AgentEvent,
    AgentMessage,
    AgentTool,
    ErrorEvent,
    MessageEndEvent,
    OllamaClient,
    ToolExecutionEndEvent,
    UserMessage,
    get_model_info,
    run_agent_loop,
)
from agent.session import (
    CompactionEntry,
    JsonlSessionStorage,
    MessageEntry,
    ModelChangeEntry,
    SessionEntry,
    SessionInfoEntry,
    SessionState,
    SessionStorage,
    SessionJsonlError,
    entry_from_json_line,
)
from coding.commands import CommandRegistry, CommandResult, create_default_command_registry
from coding.context import discover_project_context_with_diagnostics
from coding.context_window import (
    DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    ContextUsageEstimate,
    auto_compaction_threshold_for_context_window,
    build_truncation_summary,
    estimate_context_usage,
    estimate_message_tokens,
)
from coding.diagnostics import (
    AgentCallDiagnosticContext,
    AgentCallDiagnosticLogger,
    new_agent_call_run_id,
)
from coding.prompt_templates import (
    PromptTemplate,
    expand_prompt_template_command,
    load_prompt_templates_with_diagnostics,
)
from coding.reload import CodingReloadSummary, ReloadCategorySummary
from coding.resources import (
    ResourceDiagnostic,
    ResourceError,
    ResourcePaths,
    resource_paths_with_cwd,
)
from coding.session_manager import SessionManager
from coding.skills import Skill, expand_skill_command, load_skills_with_diagnostics
from coding.system_prompt import (
    BuildSystemPromptOptions,
    ProjectContextFile,
    build_system_prompt,
)
from coding.tools import create_bash_tool, create_coding_tools


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
class SessionResources:
    skills: tuple[Skill, ...]
    prompt_templates: tuple[PromptTemplate, ...]
    context_files: tuple[ProjectContextFile, ...]
    diagnostics: tuple[ResourceDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    replace_entry_ids: tuple[str, ...]
    messages_to_summarize: tuple[AgentMessage, ...]


@dataclass(frozen=True, slots=True)
class CodingSessionConfig:
    ollama_host: str
    model: str
    storage: SessionStorage
    cwd: Path
    system: str | None = None
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None
    context_files: tuple[ProjectContextFile, ...] = ()
    tools: list[AgentTool] | None = None
    resource_paths: ResourcePaths | None = None
    session_id: str | None = None
    session_manager: SessionManager | None = None
    command_registry: CommandRegistry | None = None
    auto_compact_token_threshold: int | None = None
    auto_compact_enabled: bool = True


class CodingSession:

    def __init__(
        self,
        config: CodingSessionConfig,
        *,
        state: SessionState,
        model: str,
        system_prompt: str,
        tools: list[AgentTool],
        client: OllamaClient,
        ollama_context_length: int,
        skills: tuple[Skill, ...] = (),
        prompt_templates: tuple[PromptTemplate, ...] = (),
        context_files: tuple[ProjectContextFile, ...] = (),
        resource_diagnostics: tuple[ResourceDiagnostic, ...] = (),
        command_registry: CommandRegistry | None = None,
    ) -> None:
        self._config = config
        self._state = state
        self._model = model
        self._system_prompt = system_prompt
        self._messages: list[AgentMessage] = list(state.messages)
        self._tools = tools
        self._client = client
        self._ollama_context_length = ollama_context_length
        self._skills = skills
        self._prompt_templates = prompt_templates
        self._context_files = context_files
        self._resource_diagnostics = resource_diagnostics
        self._command_registry = command_registry or create_default_command_registry()
        self._resource_paths = resource_paths_with_cwd(config.resource_paths, config.cwd)
        self._auto_compact_token_threshold = config.auto_compact_token_threshold
        self._auto_compact_enabled = config.auto_compact_enabled
        self._context_usage_cache: ContextUsageEstimate | None = None
        self._diagnostic_logger = AgentCallDiagnosticLogger.from_paths(self._resource_paths.paths)
        self._last_diagnostic_log_path: Path | None = None

    @classmethod
    async def load(cls, config: CodingSessionConfig) -> CodingSession:
        entries = await _read_compatible_entries(config.storage)
        if not entries:
            info = SessionInfoEntry(cwd=str(config.cwd))
            model_entry = ModelChangeEntry(model=config.model)
            entries = [info, model_entry]
            await config.storage.append(info)
            await config.storage.append(model_entry)

        state = SessionState.from_entries(entries)
        tools = (
            config.tools
            if config.tools is not None
            else create_coding_tools(cwd=config.cwd)
        )
        resource_paths = resource_paths_with_cwd(config.resource_paths, config.cwd)
        resources = _load_session_resources(resource_paths, config.context_files)
        system = (
            config.system
            if config.system is not None
            else build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=config.cwd,
                    tools=tools,
                    skills=resources.skills,
                    custom_prompt=config.custom_system_prompt,
                    append_system_prompt=config.append_system_prompt,
                    context_files=resources.context_files,
                )
            )
        )
        client = OllamaClient(config.ollama_host)
        resolved_model = state.model or config.model
        ollama_context_length = DEFAULT_CONTEXT_WINDOW_TOKENS
        try:
            model_info = await get_model_info(resolved_model, client=client)
            ollama_context_length = model_info.context_length or DEFAULT_CONTEXT_WINDOW_TOKENS
        except LookupError:
            pass

        return cls(
            config,
            state=state,
            model=resolved_model,
            system_prompt=system,
            tools=tools,
            client=client,
            ollama_context_length=ollama_context_length,
            skills=resources.skills,
            prompt_templates=resources.prompt_templates,
            context_files=resources.context_files,
            resource_diagnostics=resources.diagnostics,
            command_registry=config.command_registry,
        )

    @property
    def cwd(self) -> Path:
        return self._config.cwd

    @property
    def model(self) -> str:
        return self._model

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        return tuple(self._tools)

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return tuple(self._messages)

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def storage(self) -> SessionStorage:
        return self._config.storage

    @property
    def skills(self) -> tuple[Skill, ...]:
        return self._skills

    @property
    def prompt_templates(self) -> tuple[PromptTemplate, ...]:
        return self._prompt_templates

    @property
    def context_files(self) -> tuple[ProjectContextFile, ...]:
        return self._context_files

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
        return self._ollama_context_length

    @property
    def command_registry(self) -> CommandRegistry:
        return self._command_registry

    @property
    def resource_diagnostics(self) -> tuple[ResourceDiagnostic, ...]:
        return self._resource_diagnostics

    @property
    def session_id(self) -> str | None:
        return self._config.session_id

    @property
    def session_title(self) -> str | None:
        if self._config.session_id is None or self._config.session_manager is None:
            return None
        record = self._config.session_manager.get_session(self._config.session_id)
        if record is None:
            return None
        return record.title

    @property
    def session_manager(self) -> SessionManager | None:
        return self._config.session_manager

    @property
    def last_diagnostic_log_path(self) -> Path | None:
        return self._last_diagnostic_log_path

    def set_model(self, model: str) -> None:
        self._model = model
        if self._config.session_id is not None and self._config.session_manager is not None:
            self._config.session_manager.touch_session(
                self._config.session_id,
                model=model,
            )

    def reload(self) -> CodingReloadSummary:
        before_skills = _skill_signatures(self._skills)
        before_prompt_templates = _prompt_template_signatures(self._prompt_templates)
        before_context_files = _context_file_signatures(self._context_files)
        before_diagnostics = _diagnostic_signatures(self._resource_diagnostics)
        before_system_prompt_inputs = _system_prompt_resource_signatures(
            skills=self._skills,
            context_files=self._context_files,
        )

        resources = _load_session_resources(self._resource_paths, self._config.context_files)

        after_skills = _skill_signatures(resources.skills)
        after_prompt_templates = _prompt_template_signatures(resources.prompt_templates)
        after_context_files = _context_file_signatures(resources.context_files)
        after_diagnostics = _diagnostic_signatures(resources.diagnostics)
        after_system_prompt_inputs = _system_prompt_resource_signatures(
            skills=resources.skills,
            context_files=resources.context_files,
        )

        rebuilt_system_prompt: str | None = None
        system_prompt_rebuilt = False
        if (
            self._config.system is None
            and before_system_prompt_inputs != after_system_prompt_inputs
        ):
            rebuilt_system_prompt = build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=self._config.cwd,
                    tools=self._tools,
                    skills=resources.skills,
                    custom_prompt=self._config.custom_system_prompt,
                    append_system_prompt=self._config.append_system_prompt,
                    context_files=resources.context_files,
                )
            )
            system_prompt_rebuilt = True

        self._skills = resources.skills
        self._prompt_templates = resources.prompt_templates
        self._context_files = resources.context_files
        self._resource_diagnostics = resources.diagnostics
        if rebuilt_system_prompt is not None:
            self._system_prompt = rebuilt_system_prompt
            self._invalidate_context_usage_cache()

        return CodingReloadSummary(
            skills=_category_summary(before_skills, after_skills),
            prompt_templates=_category_summary(
                before_prompt_templates,
                after_prompt_templates,
            ),
            context_files=_category_summary(before_context_files, after_context_files),
            diagnostics=_category_summary(before_diagnostics, after_diagnostics),
            system_prompt_rebuilt=system_prompt_rebuilt,
        )

    async def resume(self, session_id: str) -> str:
        manager = self._config.session_manager
        if manager is None:
            raise ValueError("Session manager is not available")
        record = manager.get_session(session_id)
        if record is None:
            raise ValueError(f"Unknown session: {session_id}")

        replacement = await type(self).load(
            CodingSessionConfig(
                ollama_host=self._config.ollama_host,
                model=record.model,
                cwd=record.cwd,
                storage=jsonl_session_storage(record.path),
                system=self._config.system,
                custom_system_prompt=self._config.custom_system_prompt,
                append_system_prompt=self._config.append_system_prompt,
                context_files=self._config.context_files,
                resource_paths=self._config.resource_paths,
                session_id=record.id,
                session_manager=manager,
                command_registry=self._command_registry,
                auto_compact_token_threshold=self._auto_compact_token_threshold,
                auto_compact_enabled=self._auto_compact_enabled,
            )
        )
        self._adopt_replacement(replacement)
        return f"Resumed session: {record.id}"

    async def new_session(self) -> str:
        manager = self._config.session_manager
        if manager is None:
            raise ValueError("Session manager is not available")

        model = self._model

        record = manager.prepare_session(
            cwd=self.cwd,
            model=model,
        )
        replacement = await type(self).load(
            replace(
                self._config,
                model=record.model or model,
                cwd=record.cwd,
                storage=jsonl_session_storage(record.path),
                session_id=record.id,
            )
        )
        self._adopt_replacement(replacement)
        return f"Started new session: {record.id}"

    def _adopt_replacement(self, replacement: CodingSession) -> None:
        self._config = replacement._config
        self._state = replacement._state
        self._model = replacement._model
        self._system_prompt = replacement._system_prompt
        self._messages = replacement._messages
        self._tools = replacement._tools
        self._client = replacement._client
        self._ollama_context_length = replacement._ollama_context_length
        self._invalidate_context_usage_cache()
        self._skills = replacement._skills
        self._prompt_templates = replacement._prompt_templates
        self._context_files = replacement._context_files
        self._resource_diagnostics = replacement._resource_diagnostics
        self._command_registry = replacement._command_registry
        self._resource_paths = replacement._resource_paths
        self._auto_compact_token_threshold = replacement._auto_compact_token_threshold
        self._auto_compact_enabled = replacement._auto_compact_enabled

    async def compact(self) -> str:
        plan = self._manual_compaction_plan()
        summary = build_truncation_summary(plan.messages_to_summarize)
        compaction = await self._append_compaction(
            summary,
            replace_entry_ids=plan.replace_entry_ids,
        )
        return f"Compacted {len(compaction.replaces_entry_ids)} context entries."

    async def aclose(self) -> None:
        await self._client.aclose()

    async def handle_command(self, text: str) -> CommandResult:
        if expand_prompt_template_command(text, self._prompt_templates) is not None:
            return CommandResult(handled=False)
        return await self._command_registry.execute(self, text)

    def ensure_session_indexed(self) -> None:
        if self._config.session_id is None or self._config.session_manager is None:
            return
        if self._config.session_manager.get_session(self._config.session_id) is None:
            self._config.session_manager.create_session(
                cwd=self.cwd,
                model=self._model,
                session_id=self._config.session_id,
            )

    def expand_prompt_text(self, text: str) -> str:
        expanded_prompt = expand_prompt_template_command(text, self._prompt_templates)
        if expanded_prompt is not None:
            return expanded_prompt
        expanded_skill = expand_skill_command(text, self._skills)
        return expanded_skill if expanded_skill is not None else text

    async def run_terminal_command(
        self,
        command: str,
        *,
        add_to_context: bool,
    ) -> TerminalCommandResult:
        normalized_command = command.strip()
        if not normalized_command:
            raise ValueError("Terminal command cannot be empty")

        bash_tool = create_bash_tool(cwd=self.cwd)
        result = await bash_tool.execute({"command": normalized_command})
        exit_code = None
        if result.data is not None:
            raw_exit_code = result.data.get("exit_code")
            exit_code = raw_exit_code if isinstance(raw_exit_code, int) else None

        if add_to_context:
            before_count = len(self._messages)
            self._messages.append(
                UserMessage(
                    content=_terminal_command_context_message(
                        normalized_command,
                        result.content,
                    )
                )
            )
            self._invalidate_context_usage_cache()
            await self._persist_messages_since(before_count)

        return TerminalCommandResult(
            command=normalized_command,
            output=result.content,
            exit_code=exit_code,
            ok=result.ok,
            added_to_context=add_to_context,
        )

    async def prompt(
        self,
        content: str,
    ) -> AsyncIterator[AgentEvent]:
        context = self._diagnostic_context()
        try:
            expanded_content = self.expand_prompt_text(content)
        except ResourceError:
            raise
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="expand_prompt",
                exc=exc,
            )
            raise

        self._messages.append(UserMessage(content=expanded_content))

        await self._try_auto_compact(context=context, phase="auto_compact_before_prompt")
        persisted_count = len(self._messages) - 1
        overflow_event: ErrorEvent | None = None
        try:
            events = run_agent_loop(
                client=self._client,
                model=self._model,
                system=self._system_prompt,
                messages=self._messages,
                tools=self._tools,
            )
            self._invalidate_context_usage_cache()
            async for event in events:
                if isinstance(event, MessageEndEvent):
                    persisted_count = await self._persist_messages_since(persisted_count)
                if isinstance(event, ToolExecutionEndEvent):
                    self._invalidate_context_usage_cache()
                if isinstance(event, ErrorEvent) and not event.recoverable:
                    self._last_diagnostic_log_path = self._diagnostic_logger.log_error_event(
                        context=context,
                        phase="agent_loop",
                        event=event,
                    )
                    if _is_context_overflow_error(event):
                        overflow_event = event
                yield event
            persisted_count = await self._persist_messages_since(persisted_count)
            if overflow_event is not None:
                compacted = await self._try_overflow_compact(context=context)
                if compacted:
                    retry_persisted_count = len(self._messages)
                    retry_events = run_agent_loop(
                        client=self._client,
                        model=self._model,
                        system=self._system_prompt,
                        messages=self._messages,
                        tools=self._tools,
                    )
                    self._invalidate_context_usage_cache()
                    async for retry_event in retry_events:
                        if isinstance(retry_event, MessageEndEvent):
                            retry_persisted_count = await self._persist_messages_since(
                                retry_persisted_count
                            )
                        if isinstance(retry_event, ToolExecutionEndEvent):
                            self._invalidate_context_usage_cache()
                        if isinstance(retry_event, ErrorEvent) and not retry_event.recoverable:
                            self._last_diagnostic_log_path = (
                                self._diagnostic_logger.log_error_event(
                                    context=context,
                                    phase="agent_loop_retry",
                                    event=retry_event,
                                )
                            )
                        yield retry_event
                    await self._persist_messages_since(retry_persisted_count)
                return
            await self._try_auto_compact(context=context, phase="auto_compact_after_prompt")
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="agent_loop",
                exc=exc,
            )
            raise

    async def continue_(self) -> AsyncIterator[AgentEvent]:
        context = self._diagnostic_context()
        persisted_count = len(self._messages)
        try:
            events = run_agent_loop(
                client=self._client,
                model=self._model,
                system=self._system_prompt,
                messages=self._messages,
                tools=self._tools,
            )
            self._invalidate_context_usage_cache()
            async for event in events:
                if isinstance(event, MessageEndEvent):
                    persisted_count = await self._persist_messages_since(persisted_count)
                if isinstance(event, ToolExecutionEndEvent):
                    self._invalidate_context_usage_cache()
                if isinstance(event, ErrorEvent) and not event.recoverable:
                    self._last_diagnostic_log_path = self._diagnostic_logger.log_error_event(
                        context=context,
                        phase="agent_loop",
                        event=event,
                    )
                yield event
            await self._persist_messages_since(persisted_count)
            await self._try_auto_compact(context=context, phase="auto_compact_after_continue")
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="agent_loop",
                exc=exc,
            )
            raise

    def _diagnostic_context(self) -> AgentCallDiagnosticContext:
        return AgentCallDiagnosticContext(
            model=self._model,
            cwd=self.cwd,
            session_id=self.session_id,
            run_id=new_agent_call_run_id(),
        )

    async def _persist_messages_since(self, persisted_count: int) -> int:
        new_messages = self._messages[persisted_count:]
        if not new_messages:
            return persisted_count

        for message in new_messages:
            entry = MessageEntry(message=message)
            await self._append_session_entry(entry)

        await self._refresh_persisted_state()
        self._invalidate_context_usage_cache()
        return persisted_count + len(new_messages)

    def _invalidate_context_usage_cache(self) -> None:
        self._context_usage_cache = None

    async def _refresh_persisted_state(self) -> None:
        entries = await self._read_session_entries()
        self._state = SessionState.from_entries(entries)
        if self._config.session_id is not None and self._config.session_manager is not None:
            self._config.session_manager.touch_session(
                self._config.session_id,
                model=self._model,
            )

    async def _read_session_entries(self) -> list[SessionEntry]:
        return await _read_compatible_entries(self._config.storage)

    async def _append_session_entry(self, entry: SessionEntry) -> None:
        await self._config.storage.append(entry)

    async def _try_auto_compact(
        self,
        *,
        context: AgentCallDiagnosticContext,
        phase: str,
    ) -> bool:
        try:
            return await self._maybe_auto_compact()
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase=phase,
                exc=exc,
            )
            return False

    async def _try_overflow_compact(
        self,
        *,
        context: AgentCallDiagnosticContext,
    ) -> bool:
        try:
            plan = self._recent_preserving_compaction_plan()
            if plan is None:
                return False
            summary = build_truncation_summary(plan.messages_to_summarize)
            await self._append_compaction(summary, replace_entry_ids=plan.replace_entry_ids)
            return True
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="overflow_compact",
                exc=exc,
            )
            return False

    async def _maybe_auto_compact(self) -> bool:
        threshold = self.auto_compact_token_threshold
        if threshold is None or threshold <= 0:
            return False
        if len(self._state.context_entry_ids) < 2:
            return False
        if self.context_token_estimate <= threshold:
            return False
        plan = self._recent_preserving_compaction_plan()
        if plan is None:
            return False
        summary = build_truncation_summary(plan.messages_to_summarize)
        await self._append_compaction(summary, replace_entry_ids=plan.replace_entry_ids)
        return True

    def _manual_compaction_plan(self) -> CompactionPlan:
        rows = self._active_context_rows()
        if not rows:
            raise ValueError("No active context messages to compact")
        return CompactionPlan(
            replace_entry_ids=tuple(entry_id for entry_id, _message in rows),
            messages_to_summarize=tuple(message for _entry_id, message in rows),
        )

    def _recent_preserving_compaction_plan(self) -> CompactionPlan | None:
        rows = self._active_context_rows()
        if len(rows) < 2:
            return None

        first_kept_index = _first_recent_context_index(
            rows,
            keep_recent_tokens=DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
        )
        if first_kept_index <= 0:
            return None

        replaced = rows[:first_kept_index]
        if not replaced:
            return None
        return CompactionPlan(
            replace_entry_ids=tuple(entry_id for entry_id, _message in replaced),
            messages_to_summarize=tuple(message for _entry_id, message in replaced),
        )

    def _active_context_rows(self) -> tuple[tuple[str, AgentMessage], ...]:
        return tuple(zip(self._state.context_entry_ids, self._state.messages, strict=True))

    async def _append_compaction(
        self,
        summary: str,
        *,
        replace_entry_ids: tuple[str, ...],
    ) -> CompactionEntry:
        if not replace_entry_ids:
            raise ValueError("No active context messages to compact")

        compaction = CompactionEntry(
            summary=summary,
            replaces_entry_ids=list(replace_entry_ids),
        )
        await self._append_session_entry(compaction)

        await self._refresh_persisted_state()
        self._messages = list(self._state.messages)
        self._invalidate_context_usage_cache()
        return compaction


async def _read_compatible_entries(storage: SessionStorage) -> list[SessionEntry]:
    if not isinstance(storage, JsonlSessionStorage):
        return await storage.read_all()
    try:
        return await storage.read_all()
    except SessionJsonlError:
        if not storage.path.exists():
            return []
        known_types = {"message", "model_change", "session_info", "compaction"}
        entries: list[SessionEntry] = []
        for line in storage.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            entry_type = raw.get("type")
            if entry_type not in known_types:
                continue
            try:
                entry = entry_from_json_line(stripped)
                entries.append(entry)
            except SessionJsonlError:
                continue
        return entries


def _first_recent_context_index(
    rows: tuple[tuple[str, AgentMessage], ...],
    *,
    keep_recent_tokens: int,
) -> int:
    if keep_recent_tokens <= 0:
        return len(rows)

    accumulated_tokens = 0
    candidate_index: int | None = None
    for index in range(len(rows) - 1, -1, -1):
        _entry_id, message = rows[index]
        accumulated_tokens += estimate_message_tokens(message)
        if accumulated_tokens >= keep_recent_tokens:
            candidate_index = index
            break

    if candidate_index is None:
        return 0

    candidate_message = rows[candidate_index][1]
    if candidate_message.role == "user":
        if candidate_index > 0:
            return candidate_index
        next_user_index = _next_user_message_index(rows, start=1)
        return next_user_index if next_user_index is not None else 0

    next_user_index = _next_user_message_index(rows, start=candidate_index + 1)
    if next_user_index is not None:
        return next_user_index

    for index in range(candidate_index, len(rows)):
        if rows[index][1].role != "tool":
            return index
    return len(rows)


def _next_user_message_index(
    rows: tuple[tuple[str, AgentMessage], ...],
    *,
    start: int,
) -> int | None:
    for index in range(start, len(rows)):
        if rows[index][1].role == "user":
            return index
    return None


def _is_context_overflow_error(event: ErrorEvent) -> bool:
    text = event.message
    if event.data is not None:
        text = f"{text} {event.data}"
    normalized = text.lower()
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
    return any(marker in normalized for marker in markers)


def _terminal_command_context_message(command: str, output: str) -> str:
    return (
        "Terminal command executed by the user.\n\n"
        f"Command:\n```bash\n{command}\n```\n\n"
        f"Output:\n```text\n{output}\n```"
    )


def parse_terminal_command(text: str) -> TerminalCommandRequest | None:
    stripped = text.strip()
    if stripped.startswith("!!"):
        command = stripped[2:].strip()
        if not command:
            return None
        return TerminalCommandRequest(command=command, add_to_context=False)
    if stripped.startswith("!"):
        command = stripped[1:].strip()
        if not command:
            return None
        return TerminalCommandRequest(command=command, add_to_context=True)
    return None


def _category_summary(
    before: tuple[tuple[object, ...], ...],
    after: tuple[tuple[object, ...], ...],
) -> ReloadCategorySummary:
    return ReloadCategorySummary(
        before=len(before),
        after=len(after),
        changed=before != after,
    )


def _skill_signatures(skills: tuple[Skill, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (skill.name, str(skill.path), skill.description, skill.content) for skill in skills
    )


def _prompt_template_signatures(
    prompt_templates: tuple[PromptTemplate, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (template.name, str(template.path), template.description, template.content)
        for template in prompt_templates
    )


def _context_file_signatures(
    context_files: tuple[ProjectContextFile, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple((context_file.path, context_file.content) for context_file in context_files)


def _diagnostic_signatures(
    diagnostics: tuple[ResourceDiagnostic, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            diagnostic.kind,
            diagnostic.message,
            str(diagnostic.path) if diagnostic.path is not None else None,
            diagnostic.name,
            diagnostic.severity,
        )
        for diagnostic in diagnostics
    )


def _system_prompt_resource_signatures(
    *,
    skills: tuple[Skill, ...],
    context_files: tuple[ProjectContextFile, ...],
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    prompt_skills = tuple(
        (skill.name, str(skill.path), skill.description)
        for skill in sorted(skills, key=lambda item: item.name)
    )
    return (prompt_skills, _context_file_signatures(context_files))


def _load_session_resources(
    resource_paths: ResourcePaths,
    explicit_context_files: tuple[ProjectContextFile, ...],
) -> SessionResources:
    loaded_skills, skill_diagnostics = load_skills_with_diagnostics(resource_paths)
    loaded_prompt_templates, prompt_diagnostics = load_prompt_templates_with_diagnostics(
        resource_paths
    )
    discovered_context, context_diagnostics = discover_project_context_with_diagnostics(
        resource_paths
    )
    return SessionResources(
        skills=tuple(loaded_skills),
        prompt_templates=tuple(loaded_prompt_templates),
        context_files=_merge_context_files(explicit_context_files, discovered_context),
        diagnostics=tuple([*skill_diagnostics, *prompt_diagnostics, *context_diagnostics]),
    )


def _merge_context_files(
    explicit: tuple[ProjectContextFile, ...],
    discovered: tuple[ProjectContextFile, ...],
) -> tuple[ProjectContextFile, ...]:
    merged: list[ProjectContextFile] = []
    seen: set[str] = set()
    for context_file in (*explicit, *discovered):
        if context_file.path in seen:
            continue
        seen.add(context_file.path)
        merged.append(context_file)
    return tuple(merged)


def jsonl_session_storage(path: str | Path) -> JsonlSessionStorage:
    return JsonlSessionStorage(path)
