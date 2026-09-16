from decimal import Decimal
from pathlib import Path

import pytest

from backend.app.benchmark.quality import BenchmarkEvaluation, BenchmarkQualityEvaluator, CriterionResult
from backend.app.benchmark.runner import BenchmarkRunner
from backend.app.benchmark.retry import BenchmarkOperationRetry
from backend.app.providers.errors import ProviderUnavailableError
from backend.app.core.config import Settings
from backend.app.models import (
    CandidateModel,
    BenchmarkStrategy,
    ChatCompletionRequest,
    ChatMessage,
    InternalChatChoice,
    InternalChatCompletion,
    ModelDefinition,
    ModelTier,
    QualityResult,
    RequestProfile,
    RequestTrace,
    RoutingDecision,
    TokenUsage,
)
from backend.app.providers import ModelProvider, ProviderRegistry
from backend.app.quality.base import QualityEvaluator
from backend.app.routing.base import RequestProfiler
from backend.app.routing.catalog import ModelCatalog
from backend.app.telemetry import InMemoryTelemetryRepository
from backend.app.telemetry.costs import CostCalculator
from backend.app.telemetry.database import BenchmarkRepository


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class StaticProfiler(RequestProfiler):
    async def profile(self, request: ChatCompletionRequest, request_id: str | None = None) -> RequestProfile:
        return RequestProfile(request_id=request_id or "benchmark", general_requirement=0.4, quality_target=0.8)


class StaticProductionQuality(QualityEvaluator):
    async def evaluate(
        self, profile: RequestProfile, decision: RoutingDecision, response: InternalChatCompletion
    ) -> QualityResult:
        return QualityResult(
            request_id=profile.request_id,
            model_id=decision.selected_model.id,
            score=0.95,
            passed=True,
            reasons=["deterministic checks passed"],
            evaluator_type="deterministic",
            evaluation_latency_ms=0,
        )


class TimedProvider(ModelProvider):
    def __init__(self, clock: FakeClock, events: list[str]) -> None:
        self.clock = clock
        self.events = events

    @property
    def name(self) -> str:
        return "mock"

    async def chat_completion(
        self, model: ModelDefinition, request: ChatCompletionRequest
    ) -> InternalChatCompletion:
        self.events.append("inference")
        self.clock.advance(0.100)
        return _completion(model.model_name)

    async def health_check(self) -> bool:
        return True


class FlakyTimedProvider(TimedProvider):
    def __init__(self, clock: FakeClock, events: list[str]) -> None:
        super().__init__(clock, events)
        self.calls = 0

    async def chat_completion(
        self, model: ModelDefinition, request: ChatCompletionRequest
    ) -> InternalChatCompletion:
        self.events.append("inference")
        self.clock.advance(0.100)
        self.calls += 1
        if self.calls == 1:
            raise ProviderUnavailableError("temporary")
        return _completion(model.model_name)


class TimedJudge(BenchmarkQualityEvaluator):
    def __init__(self, clock: FakeClock, events: list[str]) -> None:
        self.clock = clock
        self.events = events

    async def evaluate(self, item: dict, response: InternalChatCompletion, operation_context=None) -> BenchmarkEvaluation:
        self.events.append("judge")
        self.clock.advance(0.500)
        return BenchmarkEvaluation(
            score=0.9,
            passed=True,
            reasons=["criterion satisfied"],
            criteria_results=[CriterionResult(
                criterion=item["evaluation_criteria"], passed=True, score=0.9, reason="satisfied"
            )],
            evaluator_type="benchmark_judge",
            judge_model="judge",
            evaluation_cost=Decimal("0.00010000"),
            evaluation_latency_ms=500,
            input_tokens=40,
            output_tokens=20,
        )


class RecordedGateway:
    def __init__(self, telemetry: InMemoryTelemetryRepository, trace: RequestTrace, completion: InternalChatCompletion) -> None:
        self.telemetry = telemetry
        self.trace = trace
        self.completion = completion

    async def complete(self, request: ChatCompletionRequest, *, operation_context=None):
        await self.telemetry.record(self.trace)
        return self.trace.request_id, self.completion, self.trace.routing_decision


def _model(model_id: str, tier: ModelTier, price: float) -> ModelDefinition:
    return ModelDefinition(
        id=model_id, tier=tier, provider="mock", model_name=model_id,
        input_cost_per_million_tokens=price, output_cost_per_million_tokens=price,
        expected_latency_ms=100, reasoning_score=0.9, coding_score=0.9,
        general_score=0.9, context_window=100_000,
    )


def _completion(model: str) -> InternalChatCompletion:
    return InternalChatCompletion(
        model=model,
        provider_latency_ms=100,
        choices=[InternalChatChoice(message=ChatMessage(role="assistant", content="A complete answer."))],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _decision(profile: RequestProfile, model: ModelDefinition, router_latency_ms: int = 0) -> RoutingDecision:
    candidate = CandidateModel(
        model_id=model.id, tier=model.tier, provider=model.provider, model_name=model.model_name,
        predicted_quality=0.9, estimated_cost=0.0, estimated_latency=100, fitness_score=1.0,
    )
    return RoutingDecision(
        request_id=profile.request_id, selected_model=model, candidate_models=[candidate], rejected_models=[],
        predicted_quality=0.9, estimated_cost=0.0, estimated_latency=100,
        routing_reason="test", request_profile=profile, router_latency_ms=router_latency_ms,
    )


def _runner(
    tmp_path: Path, clock: FakeClock, events: list[str], gateway, telemetry: InMemoryTelemetryRepository,
    *, provider: ModelProvider | None = None, operation_retry: BenchmarkOperationRetry | None = None,
) -> BenchmarkRunner:
    economy = _model("economy", ModelTier.ECONOMY, 1.0)
    frontier = _model("frontier", ModelTier.FRONTIER, 2.0)
    settings = Settings(
        benchmark_economy_model="economy", benchmark_frontier_model="frontier", baseline_model="frontier"
    )
    provider = provider or TimedProvider(clock, events)
    return BenchmarkRunner(
        settings=settings, catalog=ModelCatalog(models=[economy, frontier]),
        providers=ProviderRegistry({"mock": provider}), profiler=StaticProfiler(),
        quality_evaluator=StaticProductionQuality(), benchmark_quality_evaluator=TimedJudge(clock, events),
        gateway=gateway, telemetry=telemetry,
        repository=BenchmarkRepository(f"sqlite:///{(tmp_path / 'timing.db').as_posix()}"),
        cost_calculator=CostCalculator(frontier),
        operation_retry=operation_retry,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", [BenchmarkStrategy.ALWAYS_FRONTIER, BenchmarkStrategy.ALWAYS_ECONOMY])
async def test_direct_inference_latency_stops_before_the_500ms_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strategy: BenchmarkStrategy
) -> None:
    clock = FakeClock()
    events: list[str] = []
    telemetry = InMemoryTelemetryRepository()
    runner = _runner(tmp_path, clock, events, gateway=object(), telemetry=telemetry)
    monkeypatch.setattr("backend.app.benchmark.runner.time.perf_counter", clock.now)
    item = {"id": "one", "category": "simple_qa", "prompt": "What is one plus one?", "evaluation_criteria": "Names two."}

    measurement = await runner._run_item(strategy, item)

    assert events == ["inference", "judge"]
    assert measurement["latency_ms"] == 100
    assert measurement["provider_latency_ms"] == 100
    assert measurement["router_latency_ms"] == 0
    assert measurement["benchmark_evaluation_latency_ms"] == 500
    assert measurement["actual_cost"] == pytest.approx(
        0.00003 if strategy is BenchmarkStrategy.ALWAYS_FRONTIER else 0.000015
    )
    assert measurement["benchmark_evaluation_cost"] == pytest.approx(0.0001)
    assert measurement["input_tokens"] == 10 and measurement["output_tokens"] == 5
    assert measurement["quality_score"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_nominal_uses_pre_judge_gateway_trace_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    events: list[str] = []
    telemetry = InMemoryTelemetryRepository()
    frontier = _model("frontier", ModelTier.FRONTIER, 2.0)
    profile = RequestProfile(request_id="nominal-timing", general_requirement=0.4, quality_target=0.8)
    decision = _decision(profile, frontier, router_latency_ms=12)
    completion = _completion("frontier")
    trace = RequestTrace(
        request_id="nominal-timing", profile=profile, routing_decision=decision,
        quality_result=QualityResult(
            request_id=profile.request_id, model_id="frontier", score=0.95, passed=True,
            reasons=["deterministic checks passed"], evaluator_type="deterministic", evaluation_latency_ms=4,
        ),
        initial_model="economy", final_model="frontier", total_latency_ms=125,
        provider_latency_ms=100, input_tokens=20, output_tokens=10,
        actual_cost=Decimal("0.00006000"), status="success",
    )
    runner = _runner(tmp_path, clock, events, RecordedGateway(telemetry, trace, completion), telemetry)
    monkeypatch.setattr("backend.app.benchmark.runner.time.perf_counter", clock.now)
    item = {"id": "one", "category": "simple_qa", "prompt": "What is one plus one?", "evaluation_criteria": "Names two."}

    measurement = await runner._run_item(BenchmarkStrategy.NOMINAL, item)

    assert events == ["judge"]
    assert measurement["latency_ms"] == 125
    assert measurement["provider_latency_ms"] == 100
    assert measurement["router_latency_ms"] == 12
    assert measurement["benchmark_evaluation_latency_ms"] == 500
    assert measurement["actual_cost"] == pytest.approx(0.00006)
    assert measurement["benchmark_evaluation_cost"] == pytest.approx(0.0001)
    assert measurement["input_tokens"] == 20 and measurement["output_tokens"] == 10
    assert measurement["quality_score"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_direct_retry_latency_includes_backoff_but_not_judge_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    events: list[str] = []
    telemetry = InMemoryTelemetryRepository()

    async def advance_backoff(delay: float) -> None:
        clock.advance(delay)

    retry = BenchmarkOperationRetry(
        sleeper=advance_backoff, random_source=lambda: 0.0,
    )
    runner = _runner(
        tmp_path,
        clock,
        events,
        gateway=object(),
        telemetry=telemetry,
        provider=FlakyTimedProvider(clock, events),
        operation_retry=retry,
    )
    monkeypatch.setattr("backend.app.benchmark.runner.time.perf_counter", clock.now)
    item = {"id": "one", "category": "simple_qa", "prompt": "What is one plus one?", "evaluation_criteria": "Names two."}

    measurement = await runner._run_item(BenchmarkStrategy.ALWAYS_ECONOMY, item)

    assert events == ["inference", "inference", "judge"]
    assert measurement["latency_ms"] == 700  # 100ms failure + 500ms backoff + 100ms success
    assert measurement["benchmark_evaluation_latency_ms"] == 500
    assert measurement["benchmark_retry_count"] == 1
    assert measurement["provider_operation_attempt_count"] == 2
