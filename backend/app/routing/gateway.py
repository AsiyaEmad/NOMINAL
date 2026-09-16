from datetime import datetime, timezone
from collections.abc import Awaitable, Callable
import time
from typing import Any
from uuid import uuid4

from backend.app.core.config import Settings
from backend.app.models import (
    ChatCompletionRequest,
    InternalChatCompletion,
    ModelDefinition,
    ProviderAttempt,
    QualityResult,
    RequestTrace,
    RoutingDecision,
)
from backend.app.providers.errors import (
    ModelNotAvailableError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from backend.app.providers.registry import ProviderRegistry
from backend.app.quality.base import QualityEvaluator
from backend.app.routing.base import RequestProfiler, RoutingEngine
from backend.app.routing.catalog import ModelCatalog
from backend.app.telemetry.base import TelemetryRepository
from backend.app.telemetry.costs import CostCalculator

_RETRYABLE_PROVIDER_ERRORS = (ProviderTimeoutError, ProviderRateLimitError, ProviderUnavailableError)
ProviderOperationExecutor = Callable[
    [Any, ModelDefinition, ChatCompletionRequest, Any], Awaitable[InternalChatCompletion]
]


class ChatCompletionGateway:
    """Executes routing, quality escalation, and infrastructure failover as distinct stages."""

    def __init__(
        self,
        catalog: ModelCatalog,
        providers: ProviderRegistry,
        telemetry: TelemetryRepository,
        settings: Settings,
        profiler: RequestProfiler,
        routing_engine: RoutingEngine,
        quality_evaluator: QualityEvaluator,
        cost_calculator: CostCalculator | None = None,
        provider_operation_executor: ProviderOperationExecutor | None = None,
    ) -> None:
        self._catalog = catalog
        self._providers = providers
        self._telemetry = telemetry
        self._settings = settings
        self._profiler = profiler
        self._routing_engine = routing_engine
        self._quality_evaluator = quality_evaluator
        self._cost_calculator = cost_calculator
        # This hook is only supplied by BenchmarkRunner's dedicated gateway.
        # The production gateway continues to call providers directly.
        self._provider_operation_executor = provider_operation_executor

    async def complete(
        self, request: ChatCompletionRequest, *, operation_context: Any | None = None
    ) -> tuple[str, InternalChatCompletion, RoutingDecision]:
        request_id = str(uuid4())
        started = time.perf_counter()
        profile = await self._profiler.profile(request, request_id)
        decision = await self._routing_engine.route(profile, self._candidate_models(request.model))
        initial_model = decision.selected_model
        current_model = initial_model
        failed_infrastructure_models: set[str] = set()
        fallback_reasons: list[str] = []
        fallback_count = 0
        escalation_count = 0
        escalation_reason: str | None = None
        quality_result: QualityResult | None = None
        executions: list[tuple[ModelDefinition, InternalChatCompletion]] = []

        try:
            while True:
                current_model, completion, attempt_fallbacks, attempt_reasons = (
                    await self._complete_with_failover(
                        request,
                        current_model,
                        decision,
                        failed_infrastructure_models,
                        operation_context,
                    )
                )
                fallback_count += attempt_fallbacks
                fallback_reasons.extend(attempt_reasons)
                executions.append((current_model, completion))

                evaluation_decision = decision.model_copy(update={"selected_model": current_model})
                quality_result = await self._quality_evaluator.evaluate(
                    profile, evaluation_decision, completion
                )
                if quality_result.passed:
                    break
                if escalation_count >= self._settings.max_quality_escalations:
                    escalation_reason = "; ".join(quality_result.reasons)
                    break
                next_model = self._next_higher_capability_model(current_model, decision)
                if next_model is None:
                    escalation_reason = "; ".join(quality_result.reasons)
                    break
                escalation_count += 1
                escalation_reason = "; ".join(quality_result.reasons)
                current_model = next_model

        except _RETRYABLE_PROVIDER_ERRORS as exc:
            exc.request_id = request_id
            fallback_count += getattr(exc, "fallback_count", 0)
            fallback_reasons.extend(getattr(exc, "fallback_reasons", []))
            await self._record_failure(
                request_id=request_id,
                profile=profile,
                decision=decision,
                initial_model=initial_model,
                final_model=current_model,
                fallback_count=fallback_count,
                fallback_reasons=fallback_reasons,
                escalation_count=escalation_count,
                escalation_reason=escalation_reason,
                executions=executions,
                total_latency_ms=int((time.perf_counter() - started) * 1000),
            )
            raise
        except ProviderError as exc:
            exc.request_id = request_id
            await self._record_failure(
                request_id=request_id,
                profile=profile,
                decision=decision,
                initial_model=initial_model,
                final_model=current_model,
                fallback_count=fallback_count,
                fallback_reasons=fallback_reasons,
                escalation_count=escalation_count,
                escalation_reason=escalation_reason,
                executions=executions,
                total_latency_ms=int((time.perf_counter() - started) * 1000),
            )
            raise

        accounting = self._cost_calculator.account_for(executions) if self._cost_calculator else None
        provider_latency_ms = sum(item.provider_latency_ms for _, item in executions)
        provider_attempts = self._provider_attempts(executions)
        trace = RequestTrace(
            request_id=request_id,
            profile=profile,
            routing_decision=decision,
            quality_result=quality_result,
            completed_at=datetime.now(timezone.utc),
            actual_latency_ms=completion.provider_latency_ms,
            provider_latency_ms=provider_latency_ms,
            total_latency_ms=int((time.perf_counter() - started) * 1000),
            provider=current_model.provider,
            model_used=completion.model,
            input_tokens=sum(attempt.input_tokens for attempt in provider_attempts),
            output_tokens=sum(attempt.output_tokens for attempt in provider_attempts),
            provider_attempts=provider_attempts,
            initial_model=initial_model.id,
            final_model=current_model.id,
            escalated=escalation_count > 0,
            escalation_reason=escalation_reason,
            quality_escalation_count=escalation_count,
            infrastructure_fallback_count=fallback_count,
            infrastructure_fallback_reasons=fallback_reasons,
            fallback_used=fallback_count > 0,
            fallback_reason="; ".join(fallback_reasons) or None,
            actual_cost=accounting.actual_cost if accounting else None,
            estimated_baseline_cost=accounting.estimated_baseline_cost if accounting else None,
            estimated_cost_saved=accounting.estimated_cost_saved if accounting else None,
            savings_percentage=accounting.savings_percentage if accounting else None,
            status="success" if quality_result and quality_result.passed else "quality_failed",
        )
        await self._telemetry.record(trace)
        return request_id, completion, decision

    async def _complete_with_failover(
        self,
        request: ChatCompletionRequest,
        selected_model: ModelDefinition,
        decision: RoutingDecision,
        failed_models: set[str],
        operation_context: Any | None = None,
    ) -> tuple[ModelDefinition, InternalChatCompletion, int, list[str]]:
        ordered_models = [selected_model] + [
            self._catalog_model(candidate.model_id)
            for candidate in decision.candidate_models
            if candidate.model_id != selected_model.id and candidate.model_id not in failed_models
        ]
        fallback_count = 0
        reasons: list[str] = []
        last_error: ProviderTimeoutError | ProviderRateLimitError | ProviderUnavailableError | None = None
        for index, model in enumerate(ordered_models):
            if model.id in failed_models:
                continue
            try:
                provider = self._providers.get(model.provider)
                completion = (
                    await self._provider_operation_executor(provider, model, request, operation_context)
                    if self._provider_operation_executor is not None and operation_context is not None
                    else await provider.chat_completion(model, request)
                )
                return model, completion, fallback_count, reasons
            except _RETRYABLE_PROVIDER_ERRORS as exc:
                failed_models.add(model.id)
                last_error = exc
                reasons.append(f"{model.id}: {exc.code}")
                if index >= self._settings.max_provider_failovers:
                    break
                fallback_count += 1

        if last_error is not None:
            last_error.fallback_count = fallback_count
            last_error.fallback_reasons = reasons
            raise last_error
        raise ProviderUnavailableError("No valid provider candidates are available")

    def _next_higher_capability_model(
        self, current_model: ModelDefinition, decision: RoutingDecision
    ) -> ModelDefinition | None:
        current_candidate = next(
            (candidate for candidate in decision.candidate_models if candidate.model_id == current_model.id),
            None,
        )
        current_quality = current_candidate.predicted_quality if current_candidate else 0.0
        higher = [
            candidate
            for candidate in decision.candidate_models
            if candidate.model_id != current_model.id and candidate.predicted_quality > current_quality
        ]
        if not higher:
            return None
        higher.sort(key=lambda candidate: (candidate.predicted_quality, candidate.estimated_cost))
        return self._catalog_model(higher[0].model_id)

    def _catalog_model(self, model_id: str) -> ModelDefinition:
        return next(model for model in self._catalog.enabled_models if model.id == model_id)

    @staticmethod
    def _provider_attempts(
        executions: list[tuple[ModelDefinition, InternalChatCompletion]],
    ) -> list[ProviderAttempt]:
        return [
            ProviderAttempt(
                model_id=model.id,
                provider=model.provider,
                model_name=completion.model,
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
                latency_ms=completion.provider_latency_ms,
            )
            for model, completion in executions
        ]

    async def _record_failure(
        self,
        *,
        request_id: str,
        profile,
        decision: RoutingDecision,
        initial_model: ModelDefinition,
        final_model: ModelDefinition,
        fallback_count: int,
        fallback_reasons: list[str],
        escalation_count: int,
        escalation_reason: str | None,
        executions: list[tuple[ModelDefinition, InternalChatCompletion]],
        total_latency_ms: int,
    ) -> None:
        accounting = self._cost_calculator.account_for(executions) if self._cost_calculator else None
        provider_attempts = self._provider_attempts(executions)
        await self._telemetry.record(
            RequestTrace(
                request_id=request_id,
                profile=profile,
                routing_decision=decision,
                completed_at=datetime.now(timezone.utc),
                initial_model=initial_model.id,
                final_model=final_model.id,
                escalated=escalation_count > 0,
                escalation_reason=escalation_reason,
                quality_escalation_count=escalation_count,
                infrastructure_fallback_count=fallback_count,
                infrastructure_fallback_reasons=fallback_reasons,
                fallback_used=fallback_count > 0,
                fallback_reason="; ".join(fallback_reasons) or None,
                total_latency_ms=total_latency_ms,
                provider_latency_ms=sum(attempt.latency_ms for attempt in provider_attempts),
                input_tokens=sum(attempt.input_tokens for attempt in provider_attempts),
                output_tokens=sum(attempt.output_tokens for attempt in provider_attempts),
                provider_attempts=provider_attempts,
                actual_cost=accounting.actual_cost if accounting else None,
                estimated_baseline_cost=accounting.estimated_baseline_cost if accounting else None,
                estimated_cost_saved=accounting.estimated_cost_saved if accounting else None,
                savings_percentage=accounting.savings_percentage if accounting else None,
                status="provider_failed",
            )
        )

    def _candidate_models(self, requested_model: str) -> list[ModelDefinition]:
        if requested_model == "nominal-auto":
            return self._catalog.enabled_models
        for model in self._catalog.enabled_models:
            if model.id == requested_model or model.model_name == requested_model:
                return [model]
        raise ModelNotAvailableError(f"Model '{requested_model}' is not available")
