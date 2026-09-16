import json
import time
from datetime import datetime, timezone
from decimal import Decimal
from math import ceil
from pathlib import Path
from statistics import mean
from uuid import uuid4

from backend.app.core.config import Settings
from backend.app.benchmark.quality import BenchmarkEvaluation, BenchmarkQualityEvaluator
from backend.app.benchmark.retry import BenchmarkOperationContext, BenchmarkOperationRetry
from backend.app.models import (
    BenchmarkRunResult,
    BenchmarkStrategy,
    BenchmarkStrategyMetrics,
    CandidateModel,
    ChatCompletionRequest,
    ChatMessage,
    InternalChatCompletion,
    ModelDefinition,
    RequestProfile,
    RoutingDecision,
)
from backend.app.providers.errors import ProviderError
from backend.app.providers.registry import ProviderRegistry
from backend.app.quality.base import QualityEvaluator
from backend.app.routing.base import RequestProfiler
from backend.app.routing.catalog import ModelCatalog
from backend.app.routing.gateway import ChatCompletionGateway
from backend.app.telemetry.costs import CostCalculator
from backend.app.telemetry.database import BenchmarkRepository
from backend.app.telemetry.base import TelemetryRepository


class BenchmarkRunner:
    """Runs a fixed prompt corpus through three real execution strategies, sequentially."""

    def __init__(
        self,
        *,
        settings: Settings,
        catalog: ModelCatalog,
        providers: ProviderRegistry,
        profiler: RequestProfiler,
        quality_evaluator: QualityEvaluator,
        benchmark_quality_evaluator: BenchmarkQualityEvaluator,
        gateway: ChatCompletionGateway,
        telemetry: TelemetryRepository,
        repository: BenchmarkRepository,
        cost_calculator: CostCalculator,
        operation_retry: BenchmarkOperationRetry | None = None,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._providers = providers
        self._profiler = profiler
        self._quality_evaluator = quality_evaluator
        self._benchmark_quality_evaluator = benchmark_quality_evaluator
        self._gateway = gateway
        self._telemetry = telemetry
        self._repository = repository
        self._cost_calculator = cost_calculator
        self._operation_retry = operation_retry or BenchmarkOperationRetry()

    async def run(self, max_items: int | None = None) -> BenchmarkRunResult:
        dataset_name, items = self._load_dataset(self._settings.benchmark_dataset_path)
        if max_items is not None:
            items = items[:max_items]
        strategies: list[BenchmarkStrategyMetrics] = []
        request_results: list[dict] = []
        for strategy in BenchmarkStrategy:
            measurements = []
            for item in items:
                measurement = await self._run_item(strategy, item)
                measurements.append(measurement)
                request_results.append(measurement)
            strategies.append(self._aggregate(strategy, measurements))

        evaluator_types = {item.get("evaluator_type", "unknown") for item in request_results}
        judge_models = {item.get("judge_model") for item in request_results if item.get("judge_model")}
        result = BenchmarkRunResult(
            id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            status="completed",
            dataset_name=dataset_name,
            evaluator_type=evaluator_types.pop() if len(evaluator_types) == 1 else "mixed",
            judge_model=judge_models.pop() if len(judge_models) == 1 else None,
            strategies=strategies,
            request_results=request_results,
        )
        await self._repository.save(result)
        return result

    async def list_runs(self) -> list[BenchmarkRunResult]:
        return await self._repository.list()

    async def get_run(self, run_id: str) -> BenchmarkRunResult | None:
        return await self._repository.get(run_id)

    async def _run_item(self, strategy: BenchmarkStrategy, item: dict) -> dict:
        request = ChatCompletionRequest(
            model="nominal-auto",
            messages=[ChatMessage(role="user", content=item["prompt"])],
            max_tokens=512,
        )
        if strategy is BenchmarkStrategy.NOMINAL:
            context = BenchmarkOperationContext(strategy=strategy.value, corpus_item_id=item["id"])
            return await self._run_nominal(item, request, time.perf_counter(), context)
        model = self._strategy_model(strategy)
        context = BenchmarkOperationContext(strategy=strategy.value, corpus_item_id=item["id"])
        profile = await self._profiler.profile(request)
        decision = self._direct_decision(profile, model, strategy)
        # Direct strategies have no control-plane execution to time. Their
        # comparable inference boundary begins immediately before the provider
        # call and ends as soon as its complete response is available.
        inference_started = time.perf_counter()
        try:
            completion = await self._operation_retry.execute_provider_chat(
                self._providers.get(model.provider), model, request, context
            )
            inference_latency_ms = int((time.perf_counter() - inference_started) * 1000)
            production_quality = await self._quality_evaluator.evaluate(profile, decision, completion)
            benchmark_quality = await self._benchmark_quality_evaluator.evaluate(item, completion, context)
            accounting = self._cost_calculator.account_for([(model, completion)])
            result = {
                "item_id": item["id"], "category": item["category"], "strategy": strategy.value,
                "selected_model": model.id, "final_model": model.id,
                "latency_ms": inference_latency_ms,
                "provider_latency_ms": completion.provider_latency_ms,
                "router_latency_ms": 0,
                "input_tokens": completion.usage.input_tokens, "output_tokens": completion.usage.output_tokens,
                "response_content": self._response_content(completion),
                "provider_attempts": [{
                    "model_id": model.id, "provider": model.provider, "model_name": completion.model,
                    "input_tokens": completion.usage.input_tokens, "output_tokens": completion.usage.output_tokens,
                    "latency_ms": completion.provider_latency_ms,
                }],
                "actual_cost": float(accounting.actual_cost),
                "production_quality_score": production_quality.score,
                "production_quality_passed": production_quality.passed,
                "production_quality_reasons": production_quality.reasons,
                "frontier_call": model.id == self._settings.benchmark_frontier_model,
                "escalated": False, "failure": False,
            }
            result.update(self._operation_fields(context))
            result.update(self._benchmark_evaluation_fields(benchmark_quality))
            return result
        except ProviderError as exc:
            result = {
                "item_id": item["id"], "category": item["category"], "strategy": strategy.value,
                "selected_model": model.id, "final_model": None,
                "latency_ms": int((time.perf_counter() - inference_started) * 1000), "provider_latency_ms": 0,
                "router_latency_ms": 0,
                "input_tokens": 0, "output_tokens": 0, "actual_cost": 0.0,
                "response_content": None,
                "provider_attempts": [],
                "quality_score": 0.0, "quality_passed": False, "evaluator_type": "heuristic",
                "frontier_call": model.id == self._settings.benchmark_frontier_model,
                "escalated": False, "failure": True, "failure_reason": exc.code,
            }
            result.update(self._operation_fields(context))
            return result

    async def _run_nominal(
        self,
        item: dict,
        request: ChatCompletionRequest,
        started: float,
        context: BenchmarkOperationContext,
    ) -> dict:
        try:
            request_id, completion, _ = await self._gateway.complete(request, operation_context=context)
            trace = await self._telemetry.get(request_id)
        except ProviderError as exc:
            trace = await self._telemetry.get(exc.request_id) if exc.request_id else None
            return await self._measurement_from_trace(
                item, trace, None, started, context, failure=True, failure_reason=exc.code
            )
        return await self._measurement_from_trace(item, trace, completion, started, context, failure=False)

    async def _measurement_from_trace(
        self,
        item: dict,
        trace,
        completion: InternalChatCompletion | None,
        started: float,
        context: BenchmarkOperationContext,
        *,
        failure: bool,
        failure_reason: str | None = None,
    ) -> dict:
        if trace is None:
            result = {
                "item_id": item["id"], "category": item["category"], "strategy": BenchmarkStrategy.NOMINAL.value,
                "selected_model": None, "final_model": None, "latency_ms": int((time.perf_counter() - started) * 1000),
                "provider_latency_ms": 0, "router_latency_ms": 0,
                "input_tokens": 0, "output_tokens": 0, "actual_cost": 0.0,
                "response_content": None,
                "provider_attempts": [],
                "quality_score": 0.0, "quality_passed": False, "evaluator_type": "heuristic",
                "frontier_call": False, "escalated": False, "failure": True,
                "failure_reason": failure_reason or "trace_unavailable",
            }
            result.update(self._operation_fields(context))
            return result
        benchmark_quality = (
            await self._benchmark_quality_evaluator.evaluate(item, completion, context) if completion else None
        )
        result = {
            "item_id": item["id"], "category": item["category"], "strategy": BenchmarkStrategy.NOMINAL.value,
            "selected_model": trace.initial_model, "final_model": trace.final_model,
            "latency_ms": trace.total_latency_ms or int((time.perf_counter() - started) * 1000),
            "provider_latency_ms": trace.provider_latency_ms or 0,
            "router_latency_ms": trace.routing_decision.router_latency_ms if trace.routing_decision else 0,
            "input_tokens": trace.input_tokens or 0, "output_tokens": trace.output_tokens or 0,
            "response_content": self._response_content(completion) if completion else None,
            "provider_attempts": [attempt.model_dump(mode="json") for attempt in trace.provider_attempts],
            "actual_cost": float(trace.actual_cost or Decimal("0")),
            "production_quality_score": trace.quality_result.score if trace.quality_result else None,
            "production_quality_passed": trace.quality_result.passed if trace.quality_result else None,
            "production_quality_reasons": trace.quality_result.reasons if trace.quality_result else [],
            "frontier_call": trace.final_model == self._settings.benchmark_frontier_model,
            "escalated": trace.escalated, "failure": failure or trace.status == "provider_failed",
            "failure_reason": failure_reason,
        }
        result.update(self._operation_fields(context))
        if benchmark_quality:
            result.update(self._benchmark_evaluation_fields(benchmark_quality))
        return result

    @staticmethod
    def _benchmark_evaluation_fields(evaluation: BenchmarkEvaluation) -> dict:
        operation = evaluation.operation
        return {
            "quality_score": evaluation.score,
            "quality_passed": evaluation.passed,
            "quality_reasons": evaluation.reasons,
            "criteria_results": [item.model_dump(mode="json") for item in evaluation.criteria_results],
            "evaluator_type": evaluation.evaluator_type,
            "judge_model": evaluation.judge_model,
            "benchmark_evaluation_cost": float(evaluation.evaluation_cost),
            "benchmark_evaluation_latency_ms": evaluation.evaluation_latency_ms,
            "benchmark_evaluation_input_tokens": evaluation.input_tokens,
            "benchmark_evaluation_output_tokens": evaluation.output_tokens,
            "evaluation_failed": evaluation.failed,
            "evaluation_failure_reason": evaluation.failure_reason,
            "judge_operation_attempt_count": operation.total_attempts if operation else 0,
            "judge_benchmark_retry_count": operation.benchmark_retry_count if operation else 0,
            "judge_http_status": operation.http_status if operation else None,
            "judge_exception_type": operation.exception_type if operation else None,
            "judge_failure_category": operation.failure_category if operation else None,
            "judge_retries_exhausted": operation.retries_exhausted if operation else False,
        }

    @staticmethod
    def _operation_fields(context: BenchmarkOperationContext) -> dict:
        return {
            "benchmark_operations": [operation.model_dump(mode="json") for operation in context.operations],
            "provider_operation_attempt_count": context.attempt_count("inference"),
            "benchmark_retry_count": context.retry_count(),
            "inference_benchmark_retry_count": context.retry_count("inference"),
        }

    @staticmethod
    def _response_content(completion: InternalChatCompletion) -> str:
        return "\n".join(
            choice.message.content.strip()
            for choice in completion.choices
            if isinstance(choice.message.content, str) and choice.message.content.strip()
        )

    def _strategy_model(self, strategy: BenchmarkStrategy) -> ModelDefinition:
        model_name = (
            self._settings.benchmark_frontier_model
            if strategy is BenchmarkStrategy.ALWAYS_FRONTIER
            else self._settings.benchmark_economy_model
        )
        model = self._catalog.get(model_name)
        if model is None or not model.enabled:
            raise ValueError(f"Benchmark model '{model_name}' is not enabled in the model catalog")
        return model

    @staticmethod
    def _direct_decision(profile: RequestProfile, model: ModelDefinition, strategy: BenchmarkStrategy) -> RoutingDecision:
        estimated_cost = (profile.estimated_input_tokens * model.input_cost_per_million_tokens + profile.estimated_output_tokens * model.output_cost_per_million_tokens) / 1_000_000
        candidate = CandidateModel(
            model_id=model.id, tier=model.tier, provider=model.provider, model_name=model.model_name,
            predicted_quality=max(model.reasoning_score, model.coding_score, model.general_score),
            estimated_cost=estimated_cost, estimated_latency=model.expected_latency_ms, fitness_score=0,
        )
        return RoutingDecision(
            request_id=profile.request_id, selected_model=model, candidate_models=[candidate], rejected_models=[],
            predicted_quality=candidate.predicted_quality, estimated_cost=estimated_cost,
            estimated_latency=model.expected_latency_ms,
            routing_reason=f"Benchmark {strategy.value}: direct route to {model.id}.",
            request_profile=profile, router_latency_ms=0,
        )

    @staticmethod
    def _aggregate(strategy: BenchmarkStrategy, measurements: list[dict]) -> BenchmarkStrategyMetrics:
        latencies = [item["latency_ms"] for item in measurements]
        qualities = [item["quality_score"] for item in measurements if item.get("quality_score") is not None]
        costs = [item["actual_cost"] for item in measurements]
        return BenchmarkStrategyMetrics(
            strategy=strategy, request_count=len(measurements), total_cost=round(sum(costs), 8),
            estimated_cost_per_request=round(sum(costs) / len(costs), 8) if costs else 0,
            total_tokens=sum(item["input_tokens"] + item["output_tokens"] for item in measurements),
            average_latency_ms=round(mean(latencies), 2) if latencies else 0,
            p50_latency_ms=BenchmarkRunner._percentile(latencies, 0.50),
            p95_latency_ms=BenchmarkRunner._percentile(latencies, 0.95),
            quality_score=round(mean(qualities), 3) if qualities else 0,
            quality_pass_rate=round(sum(item.get("quality_passed") is True for item in measurements) / len(qualities) * 100, 2) if qualities else 0,
            benchmark_evaluation_cost=round(sum(item.get("benchmark_evaluation_cost", 0.0) for item in measurements), 8),
            benchmark_evaluation_latency_ms=round(sum(item.get("benchmark_evaluation_latency_ms", 0.0) for item in measurements), 2),
            evaluation_failures=sum(item.get("evaluation_failed", False) for item in measurements),
            benchmark_retry_count=sum(item.get("benchmark_retry_count", 0) for item in measurements),
            provider_operation_attempt_count=sum(item.get("provider_operation_attempt_count", 0) for item in measurements),
            judge_operation_attempt_count=sum(item.get("judge_operation_attempt_count", 0) for item in measurements),
            frontier_calls=sum(item["frontier_call"] for item in measurements),
            escalations=sum(item["escalated"] for item in measurements),
            failures=sum(item["failure"] for item in measurements),
        )

    @staticmethod
    def _percentile(values: list[int], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return float(ordered[ceil(len(ordered) * percentile) - 1])

    @staticmethod
    def _load_dataset(path: Path) -> tuple[str, list[dict]]:
        document = json.loads(path.read_text(encoding="utf-8"))
        items = document.get("items", [])
        if not items:
            raise ValueError("Benchmark dataset contains no items")
        return document.get("name", path.stem), items
