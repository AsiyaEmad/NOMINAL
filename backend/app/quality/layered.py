import json
import re
import time
from collections.abc import Awaitable, Callable

from backend.app.models import InternalChatCompletion, QualityResult, RequestProfile, RoutingDecision
from backend.app.quality.base import QualityEvaluator

JudgeCallback = Callable[[RequestProfile, InternalChatCompletion], Awaitable[tuple[float, list[str]]]]


class LayeredQualityEvaluator(QualityEvaluator):
    """Cheap deterministic checks, with an injectable judge reserved for demo mode."""

    def __init__(self, *, judge_enabled: bool = False, judge: JudgeCallback | None = None) -> None:
        self._judge_enabled = judge_enabled
        self._judge = judge

    async def evaluate(
        self,
        profile: RequestProfile,
        decision: RoutingDecision,
        response: InternalChatCompletion,
    ) -> QualityResult:
        started = time.perf_counter()
        reasons: list[str] = []
        content = self._content(response)
        score = 0.95
        evaluator_type = "deterministic"
        if not content:
            score = 0.0
            reasons.append("response is empty")
        else:
            lowered = content.lower()
            if re.match(r"^(error|provider error|internal server error)\b", lowered):
                score = min(score, 0.10)
                reasons.append("response appears to contain a provider error")
            if len(content) < 8:
                score = min(score, 0.30)
                reasons.append("response is suspiciously short")
            if profile.intent.value == "coding" and not self._has_meaningful_code(content):
                score = min(score, 0.25)
                reasons.append("requested code but response contains no meaningful code")
            if self._expects_json(profile) and not self._is_json(content):
                score = min(score, 0.20)
                reasons.append("requested JSON but response is not valid JSON")
        if self._judge_enabled and self._judge is not None:
            judge_score, judge_reasons = await self._judge(profile, response)
            score = min(score, max(0.0, min(1.0, judge_score)))
            reasons.extend(judge_reasons)
            evaluator_type = "deterministic+judge"
        return QualityResult(
            request_id=profile.request_id,
            model_id=decision.selected_model.id,
            score=round(score, 3),
            passed=score >= profile.quality_target,
            reasons=reasons or ["deterministic checks passed"],
            evaluator_type=evaluator_type,
            evaluation_latency_ms=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _content(response: InternalChatCompletion) -> str:
        return "\n".join(
            choice.message.content.strip()
            for choice in response.choices
            if isinstance(choice.message.content, str) and choice.message.content.strip()
        )

    @staticmethod
    def _has_meaningful_code(content: str) -> bool:
        return "```" in content or bool(re.search(r"\b(def|class|function|return|import|const|SELECT)\b", content))

    @staticmethod
    def _expects_json(profile: RequestProfile) -> bool:
        return profile.expects_json

    @staticmethod
    def _is_json(content: str) -> bool:
        stripped = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.IGNORECASE).strip()
        try:
            json.loads(stripped)
            return True
        except json.JSONDecodeError:
            return False
