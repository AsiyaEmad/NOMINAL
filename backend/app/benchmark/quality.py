"""Strategy-blind, offline quality evaluation for benchmark responses."""

import json
import time
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from backend.app.models import ChatCompletionRequest, ChatMessage, InternalChatCompletion, ModelDefinition
from backend.app.providers.errors import ProviderError
from backend.app.providers.registry import ProviderRegistry
from backend.app.telemetry.costs import CostCalculator
from backend.app.benchmark.retry import (
    BenchmarkOperationContext,
    BenchmarkOperationRetry,
    BenchmarkOperationTelemetry,
)


class CriterionResult(BaseModel):
    criterion: str
    passed: bool
    score: float = Field(ge=0, le=1)
    reason: str


class JudgePayload(BaseModel):
    score: float = Field(ge=0, le=1)
    passed: bool
    reasons: list[str] = Field(min_length=1)
    criteria_results: list[CriterionResult] = Field(min_length=1)


class BenchmarkEvaluation(BaseModel):
    score: float | None = Field(default=None, ge=0, le=1)
    passed: bool | None = None
    reasons: list[str] = Field(default_factory=list)
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    evaluator_type: str
    judge_model: str | None = None
    evaluation_cost: Decimal = Decimal("0")
    evaluation_latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failed: bool = False
    failure_reason: str | None = None
    operation: BenchmarkOperationTelemetry | None = None


class BenchmarkQualityEvaluator(ABC):
    """Offline evaluator; never participates in routing or production escalation."""

    @abstractmethod
    async def evaluate(
        self,
        item: dict[str, Any],
        response: InternalChatCompletion,
        operation_context: BenchmarkOperationContext | None = None,
    ) -> BenchmarkEvaluation: ...


class UnavailableBenchmarkQualityEvaluator(BenchmarkQualityEvaluator):
    """Makes missing benchmark judge configuration visible in persisted results."""

    async def evaluate(
        self,
        item: dict[str, Any],
        response: InternalChatCompletion,
        operation_context: BenchmarkOperationContext | None = None,
    ) -> BenchmarkEvaluation:
        return BenchmarkEvaluation(
            evaluator_type="benchmark_judge_unavailable",
            failed=True,
            failure_reason="benchmark_judge_not_configured",
            reasons=["Benchmark judge is not configured."],
        )


class JudgeModelBenchmarkQualityEvaluator(BenchmarkQualityEvaluator):
    """Uses one configured model with an identity-blind rubric for every strategy."""

    PASSING_SCORE = 0.75

    def __init__(
        self,
        *,
        providers: ProviderRegistry,
        judge_model: ModelDefinition,
        max_tokens: int,
        operation_retry: BenchmarkOperationRetry | None = None,
    ) -> None:
        self._providers = providers
        self._judge_model = judge_model
        self._max_tokens = max_tokens
        self._operation_retry = operation_retry

    async def evaluate(
        self,
        item: dict[str, Any],
        response: InternalChatCompletion,
        operation_context: BenchmarkOperationContext | None = None,
    ) -> BenchmarkEvaluation:
        started = time.perf_counter()
        request = self._request_for(item, self._response_text(response))
        completion: InternalChatCompletion | None = None
        try:
            provider = self._providers.get(self._judge_model.provider)
            completion = (
                await self._operation_retry.execute_provider_chat(
                    provider,
                    self._judge_model,
                    request,
                    operation_context,
                    operation_type="benchmark_judge",
                )
                if self._operation_retry is not None and operation_context is not None
                else await provider.chat_completion(self._judge_model, request)
            )
            payload = JudgePayload.model_validate_json(self._response_text(completion))
        except (ProviderError, ValidationError, ValueError, json.JSONDecodeError) as exc:
            # A syntactically invalid judge verdict can still be billable if
            # the provider supplied normalized usage. Keep it separate from
            # inference spend and never manufacture usage for transport errors.
            cost = (
                CostCalculator.calculate_cost(
                    self._judge_model, completion.usage.input_tokens, completion.usage.output_tokens
                )
                if completion is not None
                else Decimal("0")
            )
            return BenchmarkEvaluation(
                evaluator_type="benchmark_judge",
                judge_model=self._judge_model.id,
                evaluation_latency_ms=int((time.perf_counter() - started) * 1000),
                failed=True,
                failure_reason=self._failure_reason(exc),
                reasons=["Benchmark judge evaluation failed."],
                evaluation_cost=cost,
                input_tokens=completion.usage.input_tokens if completion is not None else 0,
                output_tokens=completion.usage.output_tokens if completion is not None else 0,
                operation=(
                    operation_context.operation("benchmark_judge") if operation_context is not None else None
                ),
            )

        cost = CostCalculator.calculate_cost(
            self._judge_model, completion.usage.input_tokens, completion.usage.output_tokens
        )
        return BenchmarkEvaluation(
            score=payload.score,
            passed=payload.passed and payload.score >= self.PASSING_SCORE,
            reasons=payload.reasons,
            criteria_results=payload.criteria_results,
            evaluator_type="benchmark_judge",
            judge_model=self._judge_model.id,
            evaluation_cost=cost,
            evaluation_latency_ms=(
                operation_context.operation("benchmark_judge").elapsed_ms
                if operation_context is not None and operation_context.operation("benchmark_judge") is not None
                else completion.provider_latency_ms
            ),
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            operation=(
                operation_context.operation("benchmark_judge") if operation_context is not None else None
            ),
        )

    def _request_for(self, item: dict[str, Any], candidate_response: str) -> ChatCompletionRequest:
        criterion = item.get("evaluation_criteria") or "No additional criterion supplied; evaluate against the prompt and category only."
        prompt = (
            "Evaluate one candidate answer against a benchmark task. You must be impartial. "
            "Do not infer or discuss the source of the answer.\n\n"
            f"Category: {item['category']}\n"
            f"Benchmark prompt: {item['prompt']}\n"
            f"Evaluation criterion: {criterion}\n"
            f"Candidate response: {candidate_response}\n\n"
            "Use this rubric: 0.90-1.00 fully satisfies; 0.75-0.89 substantially satisfies; "
            "0.50-0.74 partially satisfies; 0.25-0.49 has major deficiencies; "
            "0.00-0.24 is incorrect or non-responsive. Set passed true only for score >= 0.75."
        )
        return ChatCompletionRequest.model_validate({
            "model": self._judge_model.id,
            "messages": [
                {"role": "system", "content": "Return only the requested structured evaluation."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "benchmark_evaluation", "strict": True, "schema": self._schema()},
            },
        })

    @staticmethod
    def _response_text(completion: InternalChatCompletion) -> str:
        return "\n".join(
            choice.message.content.strip()
            for choice in completion.choices
            if isinstance(choice.message.content, str) and choice.message.content.strip()
        )

    @staticmethod
    def _failure_reason(error: Exception) -> str:
        if isinstance(error, ProviderError):
            return error.code
        if isinstance(error, ValidationError):
            return "malformed_judge_response"
        return "malformed_judge_response"

    @staticmethod
    def _schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["score", "passed", "reasons", "criteria_results"],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "passed": {"type": "boolean"},
                "reasons": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "criteria_results": {
                    "type": "array", "minItems": 1,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["criterion", "passed", "score", "reason"],
                        "properties": {
                            "criterion": {"type": "string"},
                            "passed": {"type": "boolean"},
                            "score": {"type": "number", "minimum": 0, "maximum": 1},
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
        }
