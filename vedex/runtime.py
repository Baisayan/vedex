from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Literal

from .agent import Agent, AgentLimits
from .environments import Environment
from .models import ModelAdapter, ModelSettings
from .resources import ProjectContextFile, ResourcePaths
from .schema import AgentEvent
from .workspace import Workspace

type AppRuntimeState = Literal["created", "starting", "started", "closing", "closed"]


class AppRuntime:
    """Ephemeral composition root shared by interactive and headless frontends."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        environment: Environment,
        settings: ModelSettings,
        workspace_path: Path,
        limits: AgentLimits | None = None,
        resource_paths: ResourcePaths | None = None,
        custom_system_prompt: str | None = None,
        append_system_prompt: str | None = None,
        context_files: Sequence[ProjectContextFile] = (),
    ) -> None:
        self._adapter = adapter
        self._environment = environment
        self._settings = settings.model_copy(deep=True)
        self._workspace_path = workspace_path.expanduser().resolve(strict=False)
        self._limits = limits or AgentLimits()
        self._resource_paths = resource_paths
        self._custom_system_prompt = custom_system_prompt
        self._append_system_prompt = append_system_prompt
        self._context_files = tuple(context_files)
        self._state: AppRuntimeState = "created"
        self._workspace: Workspace | None = None
        self._agent: Agent | None = None

    @property
    def state(self) -> AppRuntimeState:
        return self._state

    @property
    def adapter(self) -> ModelAdapter:
        return self._adapter

    @property
    def environment(self) -> Environment:
        return self._environment

    @property
    def settings(self) -> ModelSettings:
        return self._settings.model_copy(deep=True)

    @property
    def limits(self) -> AgentLimits:
        return self._limits

    @property
    def workspace_path(self) -> Path:
        return self._workspace_path

    @property
    def workspace(self) -> Workspace:
        if self._workspace is None:
            raise RuntimeError("AppRuntime has not started")
        return self._workspace

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            raise RuntimeError("AppRuntime has not started")
        return self._agent

    @property
    def system_prompt(self) -> str:
        return self.workspace.system_prompt

    async def __aenter__(self) -> AppRuntime:
        await self.start()
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        await self.close()

    async def start(self) -> None:
        if self._state == "started":
            return
        if self._state != "created":
            raise RuntimeError(f"Cannot start AppRuntime from state {self._state!r}")

        self._state = "starting"
        try:
            await self._environment.start()
            tools = tuple(self._environment.tools)
            workspace = Workspace(
                cwd=self._workspace_path,
                tools=tools,
                resource_paths=self._resource_paths,
                custom_system_prompt=self._custom_system_prompt,
                append_system_prompt=self._append_system_prompt,
                context_files=self._context_files,
                model_cwd=self._environment.workspace.model_root,
            )
            agent = Agent(
                adapter=self._adapter,
                settings=self._settings,
                system_prompt=workspace.system_prompt,
                tools=tools,
                limits=self._limits,
            )
        except BaseException as start_error:
            self._state = "closing"
            try:
                await self._environment.stop()
            except BaseException as close_error:
                self._state = "closed"
                raise BaseExceptionGroup(
                    "AppRuntime startup and environment cleanup both failed",
                    [start_error, close_error],
                ) from None
            self._state = "closed"
            raise

        self._workspace = workspace
        self._agent = agent
        self._state = "started"

    async def close(self) -> None:
        if self._state == "closed":
            return
        if self._state == "closing":
            return

        self._state = "closing"
        if self._agent is not None:
            self._agent.cancel()
        try:
            await self._environment.stop()
        finally:
            self._state = "closed"

    def cancel(self) -> None:
        if self._agent is not None:
            self._agent.cancel()

    def reset(self) -> None:
        self.agent.reset()

    async def prompt(self, task: str) -> AsyncGenerator[AgentEvent, None]:
        if self._state != "started":
            raise RuntimeError(f"AppRuntime is not started; current state is {self._state!r}")
        expanded_task = self.workspace.expand_prompt_text(task)
        async for event in self.agent.run(expanded_task):
            yield event


__all__ = ["AppRuntime", "AppRuntimeState"]
