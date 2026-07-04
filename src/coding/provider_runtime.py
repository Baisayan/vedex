from typing import Protocol

from ai import ModelProvider, OpenAICompatibleProvider
from coding.credentials import FileCredentialStore
from coding.provider_config import ProviderConfig, openai_compatible_config_from_provider
from coding.thinking import ThinkingLevel


class ClosableModelProvider(ModelProvider, Protocol):
    async def aclose(self) -> None:
        ...


def create_model_provider(
    provider: ProviderConfig,
    *,
    credential_store: FileCredentialStore | None = None,
    model: str | None = None,
    thinking_level: ThinkingLevel | None = None,
) -> ClosableModelProvider:
    credentials = credential_store or FileCredentialStore()
    
    return OpenAICompatibleProvider(
        openai_compatible_config_from_provider(
            provider,
            credential_reader=credentials,
            model=model,
            thinking_level=thinking_level,
        )
    )