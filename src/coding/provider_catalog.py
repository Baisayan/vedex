from dataclasses import dataclass
from typing import Literal

from coding.thinking import ThinkingLevel, ThinkingParameter

@dataclass(frozen=True, slots=True)
class ProviderCatalogEntry:
    name: str
    display_name: str
    base_url: str
    api_key_env: str
    credential_name: str
    models: tuple[str, ...]
    default_model: str
    docs_url: str
    kind: Literal["openai-compatible"] = "openai-compatible"
    context_windows: dict[str, int] | None = None
    thinking_levels: tuple[ThinkingLevel, ...] | None = None
    thinking_models: tuple[str, ...] = ()
    thinking_default: ThinkingLevel | None = None
    thinking_parameter: ThinkingParameter | None = None


BUILTIN_PROVIDER_CATALOG: tuple[ProviderCatalogEntry, ...] = (
    ProviderCatalogEntry(
        name="openai",
        display_name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        credential_name="openai",
        models=(
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.3-codex",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
        ),
        default_model="gpt-5.5",
        docs_url="https://platform.openai.com/docs",
        context_windows={
            "gpt-5.5": 272_000,
            "gpt-5.5-pro": 1_050_000,
            "gpt-5.4": 272_000,
            "gpt-5.4-mini": 400_000,
            "gpt-5.3-codex": 400_000,
            "gpt-5.2": 400_000,
            "gpt-5.1": 400_000,
            "gpt-5": 400_000,
            "gpt-5-mini": 400_000,
            "gpt-4.1": 1_047_576,
            "gpt-4.1-mini": 1_047_576,
        },
        thinking_levels=("off", "low", "medium", "high", "xhigh"),
        thinking_models=(
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.3-codex",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5",
            "gpt-5-mini",
        ),
        thinking_default="medium",
        thinking_parameter="reasoning_effort",
    ),
    ProviderCatalogEntry(
        name="openrouter",
        display_name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        credential_name="openrouter",
        models=(
            "openai/gpt-5.5",
            "openai/gpt-5.4",
            "openai/gpt-5.3-codex",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-opus-4.8",
            "google/gemini-3.5-pro",
            "moonshotai/kimi-k2.7-code",
            "moonshotai/kimi-k2-instruct",
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-flash",
            "z-ai/glm-5.2",
            "z-ai/glm-4.5",
            "minimax/minimax-m3",
            "qwen/qwen3-coder-plus",
            "qwen/qwen3-coder",
            "qwen/qwen3-235b-a22b-thinking-2507",
            "mistralai/codestral-2508",
            "meta-llama/llama-4-maverick",
        ),
        default_model="openai/gpt-5.5",
        docs_url="https://openrouter.ai/docs",
        context_windows={
            "openai/gpt-5.5": 1_050_000,
            "openai/gpt-5.4": 1_050_000,
            "openai/gpt-5.3-codex": 400_000,
            "anthropic/claude-sonnet-4.6": 1_000_000,
            "qwen/qwen3-coder-plus": 1_000_000,
            "mistralai/codestral-2508": 256_000,
        },
        thinking_levels=("off", "low", "medium", "high", "xhigh"),
        thinking_models=(
            "openai/gpt-5.5",
            "openai/gpt-5.4",
            "openai/gpt-5.3-codex",
            "qwen/qwen3-235b-a22b-thinking-2507",
        ),
        thinking_default="medium",
        thinking_parameter="reasoning_effort",
    ),
    ProviderCatalogEntry(
        name="huggingface",
        display_name="Hugging Face Inference Providers",
        base_url="https://router.huggingface.co/v1",
        api_key_env="HF_TOKEN",
        credential_name="huggingface",
        models=(
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "Qwen/Qwen3-Coder-Next",
            "Qwen/Qwen3-235B-A22B-Thinking-2507",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "moonshotai/Kimi-K2.7-Code",
            "deepseek-ai/DeepSeek-V4-Pro",
            "deepseek-ai/DeepSeek-V4-Flash",
            "deepseek-ai/DeepSeek-R1",
            "moonshotai/Kimi-K2-Instruct",
            "zai-org/GLM-5.2",
            "zai-org/GLM-4.5",
            "MiniMaxAI/MiniMax-M3",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            "mistralai/Codestral-22B-v0.1",
            "bigcode/starcoder2-15b",
        ),
        default_model="openai/gpt-oss-120b",
        docs_url="https://huggingface.co/inference/get-started",
        context_windows={
            "openai/gpt-oss-120b": 131_072,
            "openai/gpt-oss-20b": 131_072,
            "Qwen/Qwen3-Coder-480B-A35B-Instruct": 262_144,
        },
        thinking_levels=("low", "medium", "high"),
        thinking_models=(
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "Qwen/Qwen3-235B-A22B-Thinking-2507",
            "deepseek-ai/DeepSeek-R1",
        ),
        thinking_default="medium",
        thinking_parameter="reasoning_effort",
    ),
)


def builtin_provider_entry(name: str) -> ProviderCatalogEntry | None:
    for entry in BUILTIN_PROVIDER_CATALOG:
        if entry.name == name:
            return entry
    return None