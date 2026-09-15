from abc import ABC, abstractmethod
from backend.app.models import InternalChatCompletion, QualityResult, RequestProfile, RoutingDecision


class QualityEvaluator(ABC):
    """Evaluates a provider response independently of routing policy."""

    @abstractmethod
    async def evaluate(
        self,
        profile: RequestProfile,
        decision: RoutingDecision,
        response: InternalChatCompletion,
    ) -> QualityResult: ...
