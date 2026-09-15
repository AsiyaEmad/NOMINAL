from backend.app.core.config import ProviderKind, Settings
from backend.app.providers.base import ModelProvider
from backend.app.providers.errors import ProviderUnavailableError
from backend.app.providers.ollama import OllamaProvider
from backend.app.providers.openai_compatible import OpenAICompatibleProvider


class ProviderRegistry:
    """Provider lookup and construction isolated from HTTP routes."""

    def __init__(self, providers: dict[str, ModelProvider]) -> None:
        self._providers = providers

    @classmethod
    def from_settings(cls, settings: Settings) -> "ProviderRegistry":
        providers: dict[str, ModelProvider] = {}
        for name, config in settings.provider_configs.items():
            if config.kind is ProviderKind.OLLAMA:
                providers[name] = OllamaProvider(name, config)
            else:
                providers[name] = OpenAICompatibleProvider(name, config)
        return cls(providers)

    def get(self, provider_name: str) -> ModelProvider:
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ProviderUnavailableError(
                f"Provider '{provider_name}' is not configured"
            )
        return provider
