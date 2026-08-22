from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterator, Mapping
from pathlib import Path

import pytest
from vedex.environments import LocalEnvironment
from vedex.resources import ResourcePaths
from vedex.schema import AgentTool, AgentToolResult, CancellationToken, JSONValue


def run_async[Result](awaitable: Awaitable[Result]) -> Result:
    """Run one coroutine without adding a pytest async-plugin dependency."""

    async def await_result() -> Result:
        return await awaitable

    return asyncio.run(await_result())


def make_tool(
    *,
    name: str = "test_tool",
    result: AgentToolResult | None = None,
    raises: Exception | None = None,
) -> AgentTool:
    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        if raises is not None:
            raise raises
        return result or AgentToolResult(
            tool_call_id="",
            name=name,
            ok=True,
            content="tool output",
        )

    return AgentTool(
        name=name,
        description=f"{name} description",
        input_schema={"type": "object"},
        executor=execute,
        prompt_snippet=f"Run {name}",
    )


@pytest.fixture
def local_environment(tmp_path: Path) -> Iterator[LocalEnvironment]:
    environment = LocalEnvironment(tmp_path)
    run_async(environment.start())
    try:
        yield environment
    finally:
        run_async(environment.stop())


@pytest.fixture
def resource_paths(tmp_path: Path) -> ResourcePaths:
    return ResourcePaths(root=tmp_path / "global-vedex", cwd=tmp_path / "project")
