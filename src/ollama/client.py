from __future__ import annotations

from asyncio import sleep
from collections.abc import AsyncIterator
from json import JSONDecodeError, loads
from typing import Any, Protocol

import httpx

OLLAMA_HOST = "http://localhost:11434"

_RETRY_BASE_DELAY = 0.5
_RETRY_MAX_DELAY = 8.0
_RETRY_POLL = 0.05
_TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}


class CancellationToken(Protocol):
    def is_cancelled(self) -> bool:
        ...


class OllamaClient:
    def __init__(
        self,
        host: str = OLLAMA_HOST,
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._http_client = http_client
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._http_client is not None and self._owns_client:
            await self._http_client.aclose()
            self._http_client = None

    async def get(self, path: str) -> dict[str, Any]:
        client = self._client()
        response = await client.get(f"{self._host}{path}")
        response.raise_for_status()
        return response.json()

    def stream(
        self,
        path: str,
        *,
        body: dict[str, Any],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        async def _iter() -> AsyncIterator[dict[str, Any]]:
            client = self._client()
            url = f"{self._host}{path}"

            emitted_chunk = False
            for attempt in range(self._max_retries + 1):
                if attempt > 0 and not await _backoff(attempt - 1, signal):
                    return

                if signal is not None and signal.is_cancelled():
                    return

                try:
                    async with client.stream("POST", url, json=body) as response:
                        if response.status_code >= 400:
                            await response.aread()
                            if attempt < self._max_retries and response.status_code in _TRANSIENT_STATUSES:
                                continue
                            response.raise_for_status()
                            return

                        async for line in response.aiter_lines():
                            if signal is not None and signal.is_cancelled():
                                return
                            chunk = _parse_object(line)
                            if chunk is not None:
                                emitted_chunk = True
                                yield chunk
                        return

                except httpx.HTTPStatusError:
                    raise
                except httpx.HTTPError:
                    if not emitted_chunk and attempt < self._max_retries:
                        continue
                    raise

        return _iter()

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self._timeout)
        return self._http_client


async def _backoff(attempt: int, signal: CancellationToken | None) -> bool:
    delay = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
    remaining = delay
    while remaining > 0:
        if signal is not None and signal.is_cancelled():
            return False
        interval = min(_RETRY_POLL, remaining)
        await sleep(interval)
        remaining -= interval
    return signal is None or not signal.is_cancelled()


def _parse_object(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        value = loads(line)
    except JSONDecodeError as exc:
        raise ValueError("Ollama returned malformed NDJSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Ollama returned a non-object NDJSON value")
    return value
