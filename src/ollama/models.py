from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ollama.client import OLLAMA_HOST, OllamaClient


@dataclass(frozen=True, slots=True)
class OllamaModelInfo:
    name: str
    context_length: int | None
    supports_tools: bool


async def list_model_info(
    host: str = OLLAMA_HOST,
    *,
    client: OllamaClient | None = None,
) -> list[OllamaModelInfo]:
    
    owns_client = client is None
    client = client or OllamaClient(host)
    try:
        data = await client.get("/api/tags")
    finally:
        if owns_client:
            await client.aclose()

    models = data.get("models")
    if not isinstance(models, list):
        return []

    result: list[OllamaModelInfo] = []
    for model in models:
        info = _parse_model_info(model)
        if info is not None:
            result.append(info)
    return result


async def get_model_info(
    model: str,
    host: str = OLLAMA_HOST,
    *,
    client: OllamaClient | None = None,
) -> OllamaModelInfo:
    """Return metadata for a locally available model from ``GET /api/tags``."""
    for info in await list_model_info(host, client=client):
        if info.name == model:
            return info
    raise LookupError(f"Ollama model is not available locally: {model}")


def _parse_model_info(value: Any) -> OllamaModelInfo | None:
    if not isinstance(value, dict):
        return None

    name = value.get("name") or value.get("model")
    if not isinstance(name, str) or not name:
        return None

    context_length: int | None = None
    details = value.get("details")
    if isinstance(details, dict):
        raw = details.get("context_length")
        if isinstance(raw, int) and raw > 0:
            context_length = raw

    raw_capabilities = value.get("capabilities")
    capabilities = (
        {item for item in raw_capabilities if isinstance(item, str)}
        if isinstance(raw_capabilities, list)
        else set()
    )

    return OllamaModelInfo(
        name=name,
        context_length=context_length,
        supports_tools="tools" in capabilities
    )
