from abc import ABC, abstractmethod
from typing import Sequence

from backend.app.models import ChatCompletionRequest, ModelDefinition, RequestProfile, RoutingDecision


class RequestProfiler(ABC):
    """Converts an OpenAI-compatible request into routing constraints."""

    @abstractmethod
    async def profile(
        self, request: ChatCompletionRequest, request_id: str | None = None
    ) -> RequestProfile: ...


class RoutingEngine(ABC):
    """Selects a model without coupling to a particular policy or provider."""

    @abstractmethod
    async def route(
        self, profile: RequestProfile, candidates: Sequence[ModelDefinition]
    ) -> RoutingDecision: ...
