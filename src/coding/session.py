from __future__ import annotations

import string
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from agent import (
    AgentEvent,
    AgentHarness,
    AgentHarnessConfig,
    ErrorEvent,
    MessageEndEvent,
    QueuedMessages,
    QueueUpdateEvent,
    ToolExecutionEndEvent,
)
from agent.messages import AgentMessage, AssistantMessage, ToolResultMessage, UserMessage
from agent.session import (
    CompactionEntry,
    JsonlSessionStorage,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
    SessionState,
    SessionStorage
)
from agent.session import entry_to_json_line, SessionEntry
from agent.tools import AgentTool
from agent.events import ProviderErrorEvent, ProviderResponseEndEvent, ProviderTextDeltaEvent
from agent.loop import Provider
from agent.models import get_model_info
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
from coding.paths import VedexPaths
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

StreamingBehavior = Literal["steer", "follow_up"]
SESSION_NAME_SYSTEM_PROMPT = (
    "You write concise coding-agent session names. Reply with only a short title, "
    "maximum four words, no quotes, no punctuation-only output."
)
TREE_RUNNING_MESSAGE = "Vedex is still working. Press Escape to interrupt before using /tree."


@dataclass(frozen=True, slots=True)
class TerminalCommandResult:
    command: str
    output: str
    exit_code: int | None
    ok: bool
    added_to_context: bool


@dataclass(frozen=True, slots=True)
class SessionTreeChoice:
    entry_id: str
    label: str
    active: bool = False
    is_tool_call: bool = False


@dataclass(frozen=True, slots=True)
class SessionTreeBranchResult:
    message: str
    input_prefill: str | None = None


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
    provider: Provider
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
    index_on_first_persist: bool = False
    shell_command_prefix: str | None = None


class CodingSession:
    
    def __init__(
        self,
        config: CodingSessionConfig,
        *,
        state: SessionState,
        harness: AgentHarness,
        last_parent_id: str | None,
        skills: tuple[Skill, ...] = (),
        prompt_templates: tuple[PromptTemplate, ...] = (),
        context_files: tuple[ProjectContextFile, ...] = (),
        resource_diagnostics: tuple[ResourceDiagnostic, ...] = (),
        command_registry: CommandRegistry | None = None,
        pending_initial_entries: tuple[SessionEntry, ...] = (),
    ) -> None:
        self._config = config
        self._state = state
        self._harness = harness
        self._last_parent_id = last_parent_id
        self._pending_initial_entries = pending_initial_entries
        self._skills = skills
        self._prompt_templates = prompt_templates
        self._context_files = context_files
        self._resource_diagnostics = resource_diagnostics
        self._command_registry = command_registry or create_default_command_registry()
        self._resource_paths = resource_paths_with_cwd(config.resource_paths, config.cwd)
        self._auto_compact_token_threshold = config.auto_compact_token_threshold
        self._auto_compact_enabled = config.auto_compact_enabled
        self._context_usage_cache: ContextUsageEstimate | None = None
        self._ollama_context_length: int = DEFAULT_CONTEXT_WINDOW_TOKENS
        self._diagnostic_logger = AgentCallDiagnosticLogger.from_paths(self._resource_paths.paths)
        self._last_diagnostic_log_path: Path | None = None

    @classmethod
    async def load(cls, config: CodingSessionConfig) -> CodingSession:
        entries = await config.storage.read_all()
        pending_initial_entries: tuple[SessionEntry, ...] = ()
        if not entries:
            info = SessionInfoEntry(cwd=str(config.cwd))
            model = ModelChangeEntry(
                parent_id=info.id,
                model=config.model,
            )
            entries = [info, model]
            pending_initial_entries = (info, model)
        else:
            entries = _detach_missing_parents(entries)

        linear_state = SessionState.from_entries(entries)
        latest_leaf = _latest_leaf_entry(entries)
        state = (
            SessionState.from_entries(entries, leaf_id=latest_leaf.entry_id)
            if latest_leaf is not None
            else linear_state
        )
        tools = (
            config.tools
            if config.tools is not None
            else create_coding_tools(
                cwd=config.cwd,
                shell_command_prefix=config.shell_command_prefix,
            )
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
        harness = AgentHarness(
            AgentHarnessConfig(
                provider=config.provider,
                model=state.model or config.model,
                system=system,
                tools=tools,
            ),
            messages=state.messages,
        )
        session = cls(
            config,
            state=state,
            harness=harness,
            last_parent_id=_last_parent_id_from_state(state),
            skills=resources.skills,
            prompt_templates=resources.prompt_templates,
            context_files=resources.context_files,
            resource_diagnostics=resources.diagnostics,
            command_registry=config.command_registry,
            pending_initial_entries=pending_initial_entries,
        )
        await session._persist_loaded_interrupted_tool_repairs()
        try:
            model_info = await get_model_info(state.model or config.model)
            session._ollama_context_length = model_info.context_length or DEFAULT_CONTEXT_WINDOW_TOKENS
        except LookupError:
            pass
        return session

    @property
    def cwd(self) -> Path:
        return self._config.cwd

    @property
    def model(self) -> str:
        return self._harness.config.model

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        return tuple(self._harness.config.tools)

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return self._harness.messages

    @property
    def state(self) -> SessionState:
        return self._state

    async def tree_choices(self) -> tuple[SessionTreeChoice, ...]:
        entries = await self._read_session_entries()
        branch_indents = _tree_branch_indents(entries)
        return tuple(
            SessionTreeChoice(
                entry_id=entry.id,
                label=_tree_choice_label(entry, branch_indent=branch_indents.get(entry.id, 0)),
                active=entry.id == self._state.active_leaf_id,
                is_tool_call=_is_tool_call_tree_entry(entry),
            )
            for entry in _ordered_tree_entries(entries)
            if _is_branchable_tree_entry(entry)
        )

    async def branch_to_entry(
        self,
        entry_id: str,
        *,
        summarize: bool = False,
    ) -> SessionTreeBranchResult:
        if self._harness.is_running:
            raise RuntimeError(TREE_RUNNING_MESSAGE)
        entries = await self._read_session_entries()
        by_id = {entry.id: entry for entry in entries}
        if entry_id not in by_id:
            raise ValueError(f"Unknown session entry: {entry_id}")
        selected_entry = by_id[entry_id]
        if not _is_branchable_tree_entry(selected_entry):
            raise ValueError(f"Session entry cannot be branched from: {entry_id}")

        target_id: str | None = entry_id
        input_prefill: str | None = None
        summary_entry: BranchSummaryEntry | None = None
        if summarize:
            abandoned_messages = _messages_after_entry_on_active_path(
                entries,
                entry_id,
                self._last_parent_id,
            )
            if abandoned_messages:
                summary = build_truncation_summary(abandoned_messages)
                summary_entry = BranchSummaryEntry(
                    parent_id=entry_id,
                    branch_root_id=entry_id,
                    summary=summary,
                )
                await self._append_session_entry(summary_entry)
                target_id = summary_entry.id
        elif selected_entry.type == "message" and isinstance(selected_entry.message, UserMessage):
            target_id = selected_entry.parent_id
            input_prefill = selected_entry.message.content

        leaf = LeafEntry(parent_id=target_id, entry_id=target_id)
        await self._append_session_entry(leaf)
        self._last_parent_id = target_id

        await self._refresh_persisted_state(leaf_id=target_id)
        self._harness.replace_messages(self._state.messages)
        self._invalidate_context_usage_cache()
        suffix = " with branch summary" if summary_entry is not None else ""
        if input_prefill is not None:
            return SessionTreeBranchResult(
                message=f"Branched session before {entry_id}.",
                input_prefill=input_prefill,
            )
        return SessionTreeBranchResult(message=f"Branched session at {target_id}{suffix}.")

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
                system=self._harness.config.system,
                messages=self._harness.messages,
                tools=tuple(self._harness.config.tools),
            )
        return self._context_usage_cache

    @property
    def system_prompt(self) -> str:
        return self._harness.config.system

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
    def is_running(self) -> bool:
        return self._harness.is_running

    @property
    def queued_messages(self) -> QueuedMessages:
        return self._harness.queued_messages

    @property
    def queued_steering_messages(self) -> tuple[str, ...]:
        return tuple(message.content for message in self._harness.queued_messages.steering)

    @property
    def queued_follow_up_messages(self) -> tuple[str, ...]:
        return tuple(message.content for message in self._harness.queued_messages.follow_up)

    @property
    def last_diagnostic_log_path(self) -> Path | None:
        return self._last_diagnostic_log_path

    def cancel(self) -> None:
        self._harness.cancel()

    def queue_update_event(self) -> QueueUpdateEvent:
        return self._harness.queue_update_event()

    def clear_queued_messages(self) -> QueuedMessages:
        return self._harness.clear_queues()

    def pop_latest_follow_up_message(self) -> str | None:
        message = self._harness.pop_latest_follow_up()
        return None if message is None else message.content

    def pop_latest_steering_message(self) -> str | None:
        message = self._harness.pop_latest_steering()
        return None if message is None else message.content

    def set_model(self, model: str) -> None:
        self._harness.config.model = model
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
                    tools=self._harness.config.tools,
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
            self._harness.config.system = rebuilt_system_prompt
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
                provider=self._harness.config.provider,
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
                shell_command_prefix=self._config.shell_command_prefix,
            )
        )
        self._adopt_replacement(replacement)
        return f"Resumed session: {record.id}"

    async def new_session(self) -> str:
        manager = self._config.session_manager
        if manager is None:
            raise ValueError("Session manager is not available")

        model = self.model

        record = manager.prepare_session(
            cwd=self.cwd,
            model=model,
        )
        replacement = await type(self).load(
            replace(
                self._config,
                provider=self._harness.config.provider,
                model=record.model or model,
                cwd=record.cwd,
                storage=jsonl_session_storage(record.path),
                session_id=record.id,
                index_on_first_persist=True,
            )
        )
        self._adopt_replacement(replacement)
        return f"Started new session: {record.id}"

    def _adopt_replacement(self, replacement: CodingSession) -> None:
        self._config = replacement._config
        self._state = replacement._state
        self._harness = replacement._harness
        self._invalidate_context_usage_cache()
        self._last_parent_id = replacement._last_parent_id
        self._skills = replacement._skills
        self._prompt_templates = replacement._prompt_templates
        self._context_files = replacement._context_files
        self._resource_diagnostics = replacement._resource_diagnostics
        self._command_registry = replacement._command_registry
        self._resource_paths = replacement._resource_paths
        self._auto_compact_token_threshold = replacement._auto_compact_token_threshold
        self._auto_compact_enabled = replacement._auto_compact_enabled
        self._ollama_context_length = replacement._ollama_context_length

    async def compact(self) -> str:
        plan = self._manual_compaction_plan()
        summary = build_truncation_summary(plan.messages_to_summarize)
        compaction = await self._append_compaction(
            summary,
            replace_entry_ids=plan.replace_entry_ids,
        )
        return f"Compacted {len(compaction.replaces_entry_ids)} context entries."

    async def aclose(self) -> None:
        provider: Any = self._harness.config.provider
        if hasattr(provider, 'aclose'):
            await provider.aclose()

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
                model=self.model,
                session_id=self._config.session_id,
            )
        self._config = replace(self._config, index_on_first_persist=False)
        self._ensure_session_file_initialized()

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

        bash_tool = create_bash_tool(
            cwd=self.cwd,
            shell_command_prefix=self._config.shell_command_prefix,
        )
        result = await bash_tool.execute({"command": normalized_command})
        exit_code = None
        if result.data is not None:
            raw_exit_code = result.data.get("exit_code")
            exit_code = raw_exit_code if isinstance(raw_exit_code, int) else None

        if add_to_context:
            before_count = len(self._harness.messages)
            self._harness.append_message(
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
        *,
        streaming_behavior: StreamingBehavior | None = None,
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

        if self._harness.is_running:
            if streaming_behavior == "steer":
                yield self._harness.steer(expanded_content)
                return
            if streaming_behavior == "follow_up":
                yield self._harness.follow_up(expanded_content)
                return
            raise RuntimeError(
                "CodingSession is already running; pass streaming_behavior to queue a message."
            )

        await self._try_auto_compact(context=context, phase="auto_compact_before_prompt")
        persisted_count = len(self._harness.messages)
        auto_name_attempted = False
        overflow_event: ErrorEvent | None = None
        try:
            events = self._harness.prompt(expanded_content)
            self._invalidate_context_usage_cache()
            async for event in events:
                if isinstance(event, MessageEndEvent):
                    persisted_count = await self._persist_messages_since(persisted_count)
                    if not auto_name_attempted and isinstance(event.message, UserMessage):
                        auto_name_attempted = True
                        await self._try_auto_name_session(event.message.content, context=context)
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
                    retry_persisted_count = len(self._harness.messages)
                    retry_events = self._harness.continue_()
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
        persisted_count = len(self._harness.messages)
        try:
            events = self._harness.continue_()
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
            model=self.model,
            cwd=self.cwd,
            session_id=self.session_id,
            run_id=new_agent_call_run_id(),
        )

    async def _persist_loaded_interrupted_tool_repairs(self) -> None:
        repair = _interrupted_tool_repair_plan(
            self._state.messages,
            context_entry_ids=self._state.context_entry_ids,
        )
        if repair is None:
            return

        parent_id, suffix = repair
        for message in suffix:
            entry = MessageEntry(parent_id=parent_id, message=message)
            await self._append_session_entry(entry)
            parent_id = entry.id
        leaf = LeafEntry(parent_id=parent_id, entry_id=parent_id)
        await self._append_session_entry(leaf)
        self._last_parent_id = parent_id
        await self._refresh_persisted_state(leaf_id=parent_id)
        self._harness = AgentHarness(
            AgentHarnessConfig(
                provider=self._harness.config.provider,
                model=self._harness.config.model,
                system=self._harness.config.system,
                tools=self._harness.config.tools,
                max_turns=self._harness.config.max_turns,
                queue_mode=self._harness.config.queue_mode,
            ),
            messages=self._state.messages,
        )

    async def _persist_messages_since(self, persisted_count: int) -> int:
        new_messages = self._harness.messages[persisted_count:]
        if not new_messages:
            return persisted_count

        for message in new_messages:
            entry = MessageEntry(parent_id=self._last_parent_id, message=message)
            await self._append_session_entry(entry)
            self._last_parent_id = entry.id
            leaf = LeafEntry(parent_id=entry.id, entry_id=entry.id)
            await self._append_session_entry(leaf)

        await self._refresh_persisted_state(leaf_id=self._last_parent_id)
        self._invalidate_context_usage_cache()
        return persisted_count + len(new_messages)

    def _invalidate_context_usage_cache(self) -> None:
        self._context_usage_cache = None

    async def _refresh_persisted_state(self, *, leaf_id: str | None) -> None:
        entries = await self._read_session_entries()
        self._state = SessionState.from_entries(entries, leaf_id=leaf_id)
        if self._config.session_id is not None and self._config.session_manager is not None:
            self._config.session_manager.touch_session(
                self._config.session_id,
                model=self.model,
            )

    async def _read_session_entries(self) -> list[SessionEntry]:
        return _detach_missing_parents(await self._config.storage.read_all())

    async def _append_session_entry(self, entry: SessionEntry) -> None:
        await self._ensure_session_initialized()
        await self._config.storage.append(entry)

    async def _ensure_session_initialized(self) -> None:
        if not self._pending_initial_entries:
            return
        await self._write_pending_initial_entries()
        if self._config.index_on_first_persist:
            self._index_current_session()

    async def _write_pending_initial_entries(self) -> None:
        for entry in self._pending_initial_entries:
            await self._config.storage.append(entry)
        self._pending_initial_entries = ()

    def _ensure_session_file_initialized(self) -> None:
        if not self._pending_initial_entries:
            return
        for entry in self._pending_initial_entries:
            _append_session_entry_sync(self._config.storage, entry)
        self._pending_initial_entries = ()

    def _index_current_session(self) -> None:
        if self._config.session_id is None or self._config.session_manager is None:
            return
        existing = self._config.session_manager.get_session(self._config.session_id)
        if existing is not None:
            return
        self._config.session_manager.create_session(
            cwd=self.cwd,
            model=self.model,
            session_id=self._config.session_id,
        )

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

    async def _try_auto_name_session(
        self,
        first_message: str,
        *,
        context: AgentCallDiagnosticContext,
    ) -> None:
        if not self._should_auto_name_session():
            return
        try:
            title = await self._generate_session_name(first_message)
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="auto_name_session",
                exc=exc,
            )
            title = _sanitize_session_name(first_message)
        if title is None:
            title = _sanitize_session_name(first_message)
        if title is None:
            return
        self._set_auto_session_title(title)

    def _should_auto_name_session(self) -> bool:
        if self._config.session_id is None or self._config.session_manager is None:
            return False
        record = self._config.session_manager.get_session(self._config.session_id)
        if record is not None and record.title:
            return False
        return sum(isinstance(message, UserMessage) for message in self._harness.messages) == 1

    async def _generate_session_name(self, first_message: str) -> str | None:
        prompt = (
            "Create a concise session name for this first user message. "
            "Use at most four words.\n\n"
            f"User message:\n{first_message}"
        )
        text_parts: list[str] = []
        final_text: str | None = None
        async for event in self._harness.config.provider.stream_response(
            model=self.model,
            system=SESSION_NAME_SYSTEM_PROMPT,
            messages=[UserMessage(content=prompt)],
            tools=[],
        ):
            if isinstance(event, ProviderTextDeltaEvent):
                text_parts.append(event.delta)
            elif isinstance(event, ProviderResponseEndEvent):
                final_text = event.message.content
            elif isinstance(event, ProviderErrorEvent):
                details = f": {event.data}" if event.data is not None else ""
                raise RuntimeError(f"Session naming failed: {event.message}{details}")
        return _sanitize_session_name(final_text if final_text is not None else "".join(text_parts))

    def _set_auto_session_title(self, title: str) -> None:
        if self._config.session_id is None or self._config.session_manager is None:
            return
        existing = self._config.session_manager.get_session(self._config.session_id)
        if existing is not None and existing.title:
            return
        self._config.session_manager.touch_session(
            self._config.session_id,
            model=self.model,
            title=title,
        )

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
            parent_id=self._last_parent_id,
            summary=summary,
            replaces_entry_ids=list(replace_entry_ids),
        )
        await self._append_session_entry(compaction)
        leaf = LeafEntry(parent_id=compaction.id, entry_id=compaction.id)
        await self._append_session_entry(leaf)
        self._last_parent_id = compaction.id

        await self._refresh_persisted_state(leaf_id=compaction.id)
        self._harness.replace_messages(self._state.messages)
        self._invalidate_context_usage_cache()
        return compaction


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


def _detach_missing_parents(entries: list[SessionEntry]) -> list[SessionEntry]:
    entry_ids = {entry.id for entry in entries}
    return [
        entry.model_copy(update={"parent_id": None})
        if entry.parent_id is not None and entry.parent_id not in entry_ids
        else entry
        for entry in entries
    ]


def _last_parent_id_from_state(state: SessionState) -> str | None:
    if state.active_leaf_id is not None:
        return state.active_leaf_id
    if state.entries:
        return state.entries[-1].id
    return None


def _latest_leaf_entry(entries: list[SessionEntry]) -> LeafEntry | None:
    for entry in reversed(entries):
        if isinstance(entry, LeafEntry):
            return entry
    return None


def _is_branchable_tree_entry(entry: SessionEntry) -> bool:
    if entry.type in {"compaction", "branch_summary"}:
        return True
    if entry.type != "message":
        return False
    return isinstance(entry.message, UserMessage | AssistantMessage)


def _tree_choice_label(entry: SessionEntry, *, branch_indent: int = 0) -> str:
    prefix = "  " * branch_indent
    return f"{prefix}{_tree_entry_title(entry)}"


def _tree_branch_indents(entries: list[SessionEntry]) -> dict[str, int]:
    children_by_parent: dict[str | None, list[str]] = {}
    for entry in entries:
        if entry.type != "leaf":
            children_by_parent.setdefault(entry.parent_id, []).append(entry.id)

    sibling_indexes = {
        child_id: index
        for children in children_by_parent.values()
        for index, child_id in enumerate(children)
    }
    indents: dict[str, int] = {}
    for entry in entries:
        if entry.type == "leaf":
            continue
        parent_indent = indents.get(entry.parent_id, 0) if entry.parent_id is not None else 0
        sibling_index = sibling_indexes.get(entry.id, 0)
        indents[entry.id] = parent_indent + (1 if sibling_index > 0 else 0)
    return indents


def _ordered_tree_entries(entries: list[SessionEntry]) -> tuple[SessionEntry, ...]:
    children_by_parent: dict[str | None, list[SessionEntry]] = {}
    for entry in entries:
        if entry.type != "leaf":
            children_by_parent.setdefault(entry.parent_id, []).append(entry)

    ordered: list[SessionEntry] = []
    seen: set[str] = set()
    expanded: set[str | None] = set()

    def append_descendants(root_parent_id: str | None) -> None:
        stack: list[str | None] = [root_parent_id]
        while stack:
            parent_id = stack.pop()
            if parent_id in expanded:
                continue
            expanded.add(parent_id)
            children = children_by_parent.get(parent_id, [])
            for child in children:
                if child.id not in seen:
                    ordered.append(child)
                    seen.add(child.id)
            for child in reversed(children):
                stack.append(child.id)

    append_descendants(None)
    for entry in entries:
        if entry.type != "leaf" and entry.id not in seen:
            ordered.append(entry)
            seen.add(entry.id)
            append_descendants(entry.id)
    return tuple(ordered)


def _is_tool_call_tree_entry(entry: SessionEntry) -> bool:
    return (
        entry.type == "message"
        and isinstance(entry.message, AssistantMessage)
        and bool(entry.message.tool_calls)
    )


def _tree_entry_title(entry: SessionEntry) -> str:
    match entry.type:
        case "message":
            message = entry.message
            if isinstance(message, AssistantMessage) and message.tool_calls and not message.content:
                tool_names = ", ".join(call.name for call in message.tool_calls)
                return f"tool call: {tool_names}"
            return f"{message.role}: {_message_text_preview(message)}"
        case "compaction":
            return f"compaction summary: {_short_preview(entry.summary)}"
        case "branch_summary":
            return f"branch summary: {_short_preview(entry.summary)}"
        case _:
            return entry.type


def _message_text_preview(message: AgentMessage) -> str:
    return _short_preview(str(message.content))


def _short_preview(text: str, *, limit: int = 72) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized or "(empty)"
    return f"{normalized[: limit - 1]}..."


def _messages_after_entry_on_active_path(
    entries: list[SessionEntry],
    entry_id: str,
    active_leaf_id: str | None,
) -> tuple[AgentMessage, ...]:
    if active_leaf_id is None:
        return ()
    try:
        active_path = path_to_entry(entries, active_leaf_id)
    except SessionTreeError:
        return ()
    try:
        target_index = next(
            index for index, entry in enumerate(active_path) if entry.id == entry_id
        )
    except StopIteration:
        return ()
    return tuple(
        entry.message for entry in active_path[target_index + 1 :] if entry.type == "message"
    )


def _sanitize_session_name(text: str) -> str | None:
    cleaned = " ".join(text.split()).strip()
    cleaned = cleaned.strip("\"'`“”‘’")
    cleaned = cleaned.strip(string.punctuation + " ")
    words = [word.strip(string.punctuation + "\"'`“”‘’") for word in cleaned.split()]
    words = [word for word in words if word]
    if not words:
        return None
    return " ".join(words[:4])


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


def _interrupted_tool_repair_plan(
    messages: tuple[AgentMessage, ...],
    *,
    context_entry_ids: tuple[str, ...],
) -> tuple[str, tuple[AgentMessage, ...]] | None:
    repaired: list[AgentMessage] = []
    returned_ids = {
        message.tool_call_id for message in messages if isinstance(message, ToolResultMessage)
    }
    for message in messages:
        repaired.append(message)
        if not isinstance(message, AssistantMessage):
            continue
        for tool_call in message.tool_calls:
            if tool_call.id in returned_ids:
                continue
            returned_ids.add(tool_call.id)
            content = "Tool call interrupted by user"
            repaired.append(
                ToolResultMessage(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=content,
                    ok=False,
                    error=content,
                )
            )

    if tuple(repaired) == messages:
        return None

    common_prefix_length = 0
    for old_message, repaired_message in zip(messages, repaired, strict=False):
        if old_message != repaired_message:
            break
        common_prefix_length += 1
    if common_prefix_length == 0:
        return None
    return context_entry_ids[common_prefix_length - 1], tuple(repaired[common_prefix_length:])


def default_session_path(cwd: Path) -> Path:
    return VedexPaths().default_session_path(cwd)


def jsonl_session_storage(path: str | Path) -> JsonlSessionStorage:
    return JsonlSessionStorage(path)


def _append_session_entry_sync(storage: SessionStorage, entry: SessionEntry) -> None:
    if isinstance(storage, JsonlSessionStorage):
        storage.path.parent.mkdir(parents=True, exist_ok=True)
        with storage.path.open("a", encoding="utf-8") as file:
            file.write(entry_to_json_line(entry))
        return
    raise RuntimeError("Session storage does not support synchronous initialization")