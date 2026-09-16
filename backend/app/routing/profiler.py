import math
import re
from collections.abc import Iterable

from backend.app.models import ChatCompletionRequest, ChatMessage, RequestIntent, RequestProfile, TaskType
from backend.app.routing.base import RequestProfiler


class HeuristicRequestProfiler(RequestProfiler):
    """Fast, deterministic routing features; it deliberately makes no LLM calls."""

    _KEYWORDS: dict[RequestIntent, tuple[str, ...]] = {
        RequestIntent.TRANSLATION: ("translate", "translation", "traduc", "Ã¼bersetz"),
        RequestIntent.SUMMARIZATION: ("summarize", "summary", "tl;dr", "condense"),
        RequestIntent.EXTRACTION: ("extract", "pull out", "find all", "entities", "fields"),
        RequestIntent.CLASSIFICATION: ("classify", "categorize", "label these", "sentiment"),
        RequestIntent.CODING: ("python", "javascript", "typescript", "function", "code", "bug", "api", "sql"),
        RequestIntent.REASONING: ("prove", "solve", "algorithm", "math", "calculate", "logic", "dynamic programming"),
        RequestIntent.ANALYSIS: ("analyze", "analysis", "tradeoff", "compare", "strategy", "root cause", "evaluate"),
        RequestIntent.CREATIVE: ("write a story", "poem", "creative", "brainstorm", "fiction"),
    }

    _BASELINES: dict[RequestIntent, dict[str, float]] = {
        RequestIntent.SIMPLE_QA: {"reasoning": 0.12, "coding": 0.0, "general": 0.42, "quality": 0.52, "latency": 0.85},
        RequestIntent.TRANSLATION: {"reasoning": 0.06, "coding": 0.0, "general": 0.55, "quality": 0.64, "latency": 0.75},
        RequestIntent.SUMMARIZATION: {"reasoning": 0.25, "coding": 0.0, "general": 0.70, "quality": 0.73, "latency": 0.60},
        RequestIntent.EXTRACTION: {"reasoning": 0.18, "coding": 0.0, "general": 0.60, "quality": 0.68, "latency": 0.72},
        RequestIntent.CLASSIFICATION: {"reasoning": 0.22, "coding": 0.0, "general": 0.52, "quality": 0.66, "latency": 0.70},
        RequestIntent.CODING: {"reasoning": 0.55, "coding": 0.85, "general": 0.45, "quality": 0.80, "latency": 0.42},
        RequestIntent.REASONING: {"reasoning": 0.90, "coding": 0.10, "general": 0.45, "quality": 0.86, "latency": 0.30},
        RequestIntent.ANALYSIS: {"reasoning": 0.78, "coding": 0.0, "general": 0.78, "quality": 0.90, "latency": 0.25},
        RequestIntent.CREATIVE: {"reasoning": 0.22, "coding": 0.0, "general": 0.80, "quality": 0.72, "latency": 0.50},
        RequestIntent.UNKNOWN: {"reasoning": 0.35, "coding": 0.0, "general": 0.50, "quality": 0.62, "latency": 0.55},
    }

    async def profile(
        self, request: ChatCompletionRequest, request_id: str | None = None
    ) -> RequestProfile:
        text = self._conversation_text(request.messages)
        normalized = text.lower()
        intent = self._detect_intent(normalized)
        baseline = self._BASELINES[intent].copy()
        estimated_input_tokens = max(1, math.ceil(len(text) / 4))
        complexity = min(1.0, 0.10 + estimated_input_tokens / 8_000)

        if re.search(r"\b(step.by.step|multi.step|edge cases?|constraints?|rigorous)\b", normalized):
            complexity = min(1.0, complexity + 0.18)
            baseline["reasoning"] = min(1.0, baseline["reasoning"] + 0.10)
        if "```" in text or re.search(r"\b(implement|debug|refactor)\b", normalized):
            baseline["coding"] = max(baseline["coding"], 0.78)
            complexity = min(1.0, complexity + 0.12)
        if re.search(r"\b(accurate|precise|comprehensive|detailed|thorough)\b", normalized):
            baseline["quality"] = min(1.0, baseline["quality"] + 0.03)
        if re.search(r"\b(quick|quickly|fast|brief)\b", normalized):
            baseline["latency"] = min(1.0, baseline["latency"] + 0.12)
        if re.search(r"\b(thorough|deep|detailed)\b", normalized):
            baseline["latency"] = max(0.0, baseline["latency"] - 0.10)

        context_requirement = min(1.0, estimated_input_tokens / 100_000)
        if re.search(r"\b(document|transcript|following text|attached)\b", normalized):
            context_requirement = max(context_requirement, 0.35)

        return RequestProfile(
            request_id=request_id or RequestProfile().request_id,
            intent=intent,
            task_type=self._task_type(intent),
            reasoning_requirement=baseline["reasoning"],
            coding_requirement=baseline["coding"],
            general_requirement=baseline["general"],
            context_requirement=context_requirement,
            estimated_complexity=complexity,
            quality_target=baseline["quality"],
            latency_priority=baseline["latency"],
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=request.max_tokens or 256,
            required_context_tokens=estimated_input_tokens + (request.max_tokens or 0),
            expects_json="json" in normalized,
        )

    def _detect_intent(self, text: str) -> RequestIntent:
        for intent in (
            RequestIntent.TRANSLATION,
            RequestIntent.SUMMARIZATION,
            RequestIntent.EXTRACTION,
            RequestIntent.CLASSIFICATION,
            RequestIntent.CODING,
            RequestIntent.REASONING,
            RequestIntent.ANALYSIS,
            RequestIntent.CREATIVE,
        ):
            if any(self._matches_keyword(text, keyword) for keyword in self._KEYWORDS[intent]):
                return intent
        return RequestIntent.SIMPLE_QA if "?" in text or text.strip() else RequestIntent.UNKNOWN

    @staticmethod
    def _matches_keyword(text: str, keyword: str) -> bool:
        """Match complete words or phrases, never an arbitrary character substring."""
        phrase = re.escape(keyword).replace(r"\ ", r"\s+")
        return bool(re.search(rf"(?<!\w){phrase}(?!\w)", text))

    @staticmethod
    def _conversation_text(messages: Iterable[ChatMessage]) -> str:
        parts: list[str] = []
        for message in messages:
            if isinstance(message.content, str):
                parts.append(message.content)
            elif message.content:
                parts.extend(str(part.get("text", part)) for part in message.content)
        return "\n".join(parts)

    @staticmethod
    def _task_type(intent: RequestIntent) -> TaskType:
        if intent is RequestIntent.CODING:
            return TaskType.CODING
        if intent in (RequestIntent.REASONING, RequestIntent.ANALYSIS):
            return TaskType.REASONING
        return TaskType.GENERAL
