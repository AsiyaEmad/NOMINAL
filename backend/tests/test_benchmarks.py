import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.benchmark import BenchmarkRunner
from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.models import (
    ChatCompletionRequest,
    ChatMessage,
    InternalChatChoice,
    InternalChatCompletion,
    ModelDefinition,
    ModelTier,
    RequestProfile,
    TokenUsage,
)
from backend.app.providers import ModelProvider, ProviderRegistry
from backend.app.quality import LayeredQualityEvaluator
from backend.app.routing import ChatCompletionGateway, CostAwareRoutingEngine, ModelCatalog, RequestProfiler
from backend.app.telemetry import InMemoryTelemetryRepository
from backend.app.telemetry.costs import CostCalculator
from backend.app.telemetry.database import BenchmarkRepository


class BenchmarkProfiler(RequestProfiler):
    async def profile(
        self, request: ChatCompletionRequest, request_id: str | None = None
    ) -> RequestProfile:
        return RequestProfile(
            request_id=request_id or "benchmark",
            general_requirement=0.4,
            quality_target=0.8,
            estimated_input_tokens=100,
            estimated_output_tokens=50,
            required_context_tokens=200,
        )


class MockProvider(ModelProvider):
    @property
    def name(self) -> str:
        return "mock"

    async def chat_completion(
        self, model: ModelDefinition, request: ChatCompletionRequest
    ) -> InternalChatCompletion:
        return InternalChatCompletion(
            model=model.model_name,
            provider_latency_ms=12,
            choices=[InternalChatChoice(message=ChatMessage(role="assistant", content="A useful complete response."))],
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )

    async def health_check(self) -> bool:
        return True


def _model(model_id: str, tier: ModelTier, input_price: float, output_price: float) -> ModelDefinition:
    return ModelDefinition(
        id=model_id, tier=tier, provider="mock", model_name=model_id,
        input_cost_per_million_tokens=input_price, output_cost_per_million_tokens=output_price,
        expected_latency_ms=100, reasoning_score=0.9, coding_score=0.9,
        general_score=0.9, context_window=100_000,
    )


@pytest.mark.asyncio
async def test_mocked_benchmark_measures_and_persists_all_strategies(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({"name": "test-corpus", "items": [
        {"id": "one", "category": "simple_qa", "prompt": "What is one plus one?", "expected_difficulty": "low"},
        {"id": "two", "category": "translation", "prompt": "Translate hello to French.", "expected_difficulty": "low"},
    ]}), encoding="utf-8")
    economy = _model("economy", ModelTier.ECONOMY, 0.4, 1.6)
    frontier = _model("frontier", ModelTier.FRONTIER, 2.0, 8.0)
    catalog = ModelCatalog(models=[economy, frontier])
    telemetry = InMemoryTelemetryRepository()
    settings = Settings(
        benchmark_dataset_path=dataset_path,
        benchmark_economy_model="economy",
        benchmark_frontier_model="frontier",
        baseline_model="frontier",
    )
    provider = MockProvider()
    quality = LayeredQualityEvaluator()
    calculator = CostCalculator(frontier)
    gateway = ChatCompletionGateway(
        catalog=catalog, providers=ProviderRegistry({"mock": provider}), telemetry=telemetry,
        settings=settings, profiler=BenchmarkProfiler(), routing_engine=CostAwareRoutingEngine(),
        quality_evaluator=quality, cost_calculator=calculator,
    )
    repository = BenchmarkRepository(f"sqlite:///{(tmp_path / 'benchmarks.db').as_posix()}")
    repository.create_schema()
    runner = BenchmarkRunner(
        settings=settings, catalog=catalog, providers=ProviderRegistry({"mock": provider}),
        profiler=BenchmarkProfiler(), quality_evaluator=quality, gateway=gateway,
        telemetry=telemetry, repository=repository, cost_calculator=calculator,
    )

    result = await runner.run()
    persisted = await repository.get(result.id)
    by_strategy = {item.strategy.value: item for item in result.strategies}

    assert result.status == "completed"
    assert result.evaluator_type == "heuristic"
    assert len(result.request_results) == 6
    assert by_strategy["always_frontier"].total_cost == pytest.approx(0.0012)
    assert by_strategy["always_economy"].total_cost == pytest.approx(0.00024)
    assert by_strategy["nominal"].total_cost == pytest.approx(0.00024)
    assert by_strategy["always_frontier"].frontier_calls == 2
    assert by_strategy["nominal"].frontier_calls == 0
    assert all(item.quality_pass_rate == 100 for item in result.strategies)
    assert persisted is not None and len(persisted.request_results) == 6
    repository.dispose()


def test_benchmark_endpoints_return_persisted_comparison(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({"name": "endpoint-corpus", "items": [
        {"id": "translation-1", "category": "translation", "prompt": "Translate 'hello' into French.", "expected_difficulty": "low"},
    ]}), encoding="utf-8")
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'app.db').as_posix()}",
        benchmark_dataset_path=dataset_path,
    )
    app = create_app(
        settings=settings,
        provider_registry=ProviderRegistry({"openai": MockProvider()}),
    )

    with TestClient(app) as client:
        created = client.post("/api/benchmarks/run", json={"max_items": 1})
        assert created.status_code == 200
        run = created.json()

        listing = client.get("/api/benchmarks")
        detail = client.get(f"/api/benchmarks/{run['id']}")

    assert [item["strategy"] for item in run["strategies"]] == [
        "always_frontier", "always_economy", "nominal",
    ]
    assert listing.status_code == 200
    assert listing.json()[0]["id"] == run["id"]
    assert listing.json()[0]["request_results"] == []
    assert detail.status_code == 200
    assert len(detail.json()["request_results"]) == 3
