from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import vedex.core as core
from vedex.core import get_model_info, list_model_info

from .conftest import native_ollama_client, native_tags_response, ndjson_response, run_async


async def _collect(stream: AsyncIterator[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item async for item in stream]


def test_client_get_and_stream_use_native_paths_and_skip_blank_ndjson() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/tags":
            return native_tags_response()
        return httpx.Response(200, content=b'\n{"message": "one"}\n\n')

    client = native_ollama_client(handler)
    try:
        assert run_async(client.get("/api/tags"))["models"]
        assert run_async(_collect(client.stream("/api/chat", body={}))) == [{"message": "one"}]
    finally:
        run_async(client.aclose())

    assert [request.url.path for request in requests] == ["/api/tags", "/api/chat"]


@pytest.mark.parametrize(
    "content",
    [b"not json\n", b"[1, 2, 3]\n"],
)
def test_client_stream_rejects_malformed_or_non_object_ndjson(content: bytes) -> None:
    client = native_ollama_client(lambda _request: httpx.Response(200, content=content))
    try:
        with pytest.raises(ValueError):
            run_async(_collect(client.stream("/api/chat", body={})))
    finally:
        run_async(client.aclose())


def test_client_retries_connection_failure_before_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def no_sleep(_delay: float) -> None:
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("offline", request=request)
        return ndjson_response({"done": True})

    monkeypatch.setattr(core, "sleep", no_sleep)
    client = native_ollama_client(handler, max_retries=1)
    try:
        assert run_async(_collect(client.stream("/api/chat", body={}))) == [{"done": True}]
    finally:
        run_async(client.aclose())
    assert calls == 2


def test_client_does_not_retry_a_failure_after_a_chunk() -> None:
    calls = 0

    class FailingStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'{"message": "first"}\n'
            raise httpx.ReadError("connection dropped")

        async def aclose(self) -> None:
            return None

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=FailingStream())

    client = native_ollama_client(handler, max_retries=2)
    try:
        with pytest.raises(httpx.ReadError):
            run_async(_collect(client.stream("/api/chat", body={})))
    finally:
        run_async(client.aclose())
    assert calls == 1


def test_client_http_errors_and_pre_cancelled_stream_do_not_make_a_request() -> None:
    calls = 0

    class Cancelled:
        def is_cancelled(self) -> bool:
            return True

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    client = native_ollama_client(handler)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            run_async(client.get("/api/tags"))
        assert run_async(_collect(client.stream("/api/chat", body={}, signal=Cancelled()))) == []
    finally:
        run_async(client.aclose())
    assert calls == 1


def test_model_discovery_parses_context_tools_and_ignores_invalid_entries() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "tool:latest",
                        "details": {"context_length": 8192},
                        "capabilities": ["tools"],
                    },
                    {"model": "plain:latest", "details": {"context_length": 0}},
                    {"name": ""},
                    "not an object",
                ]
            },
        )

    client = native_ollama_client(handler)
    try:
        models = run_async(list_model_info(client=client))
        assert [(model.name, model.context_length, model.supports_tools) for model in models] == [
            ("tool:latest", 8192, True),
            ("plain:latest", None, False),
        ]
        assert run_async(get_model_info("tool", client=client)).name == "tool:latest"
    finally:
        run_async(client.aclose())


def test_model_discovery_rejects_unavailable_model() -> None:
    client = native_ollama_client(lambda _request: native_tags_response(name="other:latest"))
    try:
        with pytest.raises(LookupError, match="not available locally"):
            run_async(get_model_info("missing", client=client))
    finally:
        run_async(client.aclose())
