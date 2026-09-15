from abc import ABC, abstractmethod
from backend.app.models import ChatCompletionRequest, InternalChatCompletion, ModelDefinition


class ModelProvider(ABC):
    """Adapter boundary for an OpenAI-compatible model provider."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def chat_completion(
        self, model: ModelDefinition, request: ChatCompletionRequest
    ) -> InternalChatCompletion: ...

    @abstractmethod
    async def health_check(self) -> bool: ...
