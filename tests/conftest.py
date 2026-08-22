from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from vedex.core import OllamaClient
from vedex.environments import LocalEnvironment
from vedex.resources import ResourcePaths
from vedex.schema import AgentTool, AgentToolResult, CancellationToken, JSONValue


def run_async[Result](awaitable: Awaitable[Result]) -> Result:
    """Run one coroutine without adding a pytest async-plugin dependency."""

    async def await_result() -> Result:
        return await awaitable

    return asyncio.run(await_result())


def ndjson_response(*items: dict[str, Any], status_code: int = 200) -> httpx.Response:
    body = b"".join(json.dumps(item).encode("utf-8") + b"\n" for item in items)
    return httpx.Response(status_code, content=body)


def native_ollama_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    max_retries: int = 0,
) -> OllamaClient:
    """Build the real client against an in-memory native HTTP transport."""

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaClient(
        "http://ollama.test",
        http_client=http_client,
        max_retries=max_retries,
    )


def native_tags_response(
    *,
    name: str = "local:latest",
    context_length: int | None = 4096,
    supports_tools: bool = True,
) -> httpx.Response:
    details: dict[str, Any] = {}
    if context_length is not None:
        details["context_length"] = context_length
    model: dict[str, Any] = {"name": name, "details": details}
    if supports_tools:
        model["capabilities"] = ["tools"]
    return httpx.Response(200, json={"models": [model]})


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
