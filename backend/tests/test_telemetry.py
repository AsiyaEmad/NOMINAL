from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from backend.app.models import (
    ChatMessage,
    InternalChatChoice,
    InternalChatCompletion,
    ModelDefinition,
    ModelTier,
    RequestProfile,
    RequestTrace,
    TokenUsage,
)
from backend.app.telemetry.costs import CostCalculator
from backend.app.telemetry.database import SqliteTelemetryRepository


def _model(model_id: str, input_cost: float, output_cost: float) -> ModelDefinition:
    return ModelDefinition(
        id=model_id,
        tier=ModelTier.FRONTIER if model_id == "frontier" else ModelTier.ECONOMY,
        provider="test",
        model_name=model_id,
        input_cost_per_million_tokens=input_cost,
        output_cost_per_million_tokens=output_cost,
        expected_latency_ms=100,
        reasoning_score=0.9,
        coding_score=0.9,
        general_score=0.9,
        context_window=100_000,
    )


def _trace(index: int, *, latency: int, status: str = "success") -> RequestTrace:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index)
    return RequestTrace(
        request_id=f"trace-{index}",
        profile=RequestProfile(request_id=f"trace-{index}", intent="simple_qa"),
        started_at=timestamp,
        completed_at=timestamp,
        initial_model="economy",
        final_model="economy" if index != 4 else "frontier",
        input_tokens=100,
        output_tokens=50,
        provider_latency_ms=latency - 1,
        total_latency_ms=latency,
        actual_cost=Decimal("0.00100000"),
        estimated_baseline_cost=Decimal("0.00500000"),
        estimated_cost_saved=Decimal("0.00400000"),
        savings_percentage=80.0,
        escalated=index == 1,
        fallback_used=index == 2,
        fallback_reason="timeout" if index == 2 else None,
        quality_result=None,
        status=status,
    )


@pytest.fixture
def repository(tmp_path) -> SqliteTelemetryRepository:
    database_url = f"sqlite:///{(tmp_path / 'telemetry.db').as_posix()}"
    repo = SqliteTelemetryRepository(database_url)
    repo.create_schema()
    yield repo
    repo.dispose()


def test_cost_and_estimated_baseline_calculation_use_decimal_prices() -> None:
    economy = _model("economy", 0.4, 1.6)
    frontier = _model("frontier", 2.0, 8.0)
    completion = InternalChatCompletion(
        model="economy",
        provider_latency_ms=1,
        choices=[InternalChatChoice(message=ChatMessage(role="assistant", content="answer"))],
        usage=TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000),
    )

    accounting = CostCalculator(frontier).account_for([(economy, completion)])

    assert accounting.actual_cost == Decimal("2.00000000")
    assert accounting.estimated_baseline_cost == Decimal("10.00000000")
    assert accounting.estimated_cost_saved == Decimal("8.00000000")
    assert accounting.savings_percentage == 80.0


@pytest.mark.asyncio
async def test_aggregations_and_p95(repository: SqliteTelemetryRepository) -> None:
    for index, latency in enumerate([10, 20, 30, 40, 100]):
        await repository.record(_trace(index, latency=latency, status="provider_failed" if index == 3 else "success"))

    summary = await repository.summary()

    assert summary["total_requests"] == 5
    assert summary["successful_requests"] == 4
    assert summary["total_tokens"] == 750
    assert summary["actual_cost"] == pytest.approx(0.005)
    assert summary["estimated_baseline_cost"] == pytest.approx(0.025)
    assert summary["estimated_cost_saved"] == pytest.approx(0.02)
    assert summary["savings_percentage"] == 80.0
    assert summary["average_latency_ms"] == 40.0
    assert summary["p95_latency_ms"] == 100.0
    assert summary["escalation_rate"] == 20.0
    assert summary["fallback_rate"] == 20.0
    assert summary["frontier_calls_avoided"] == 3
    assert summary["routing_distribution"] == {"economy": 4, "frontier": 1}


@pytest.mark.asyncio
async def test_empty_database_returns_zero_metrics(repository: SqliteTelemetryRepository) -> None:
    summary = await repository.summary()

    assert summary["total_requests"] == 0
    assert summary["successful_requests"] == 0
    assert summary["total_tokens"] == 0
    assert summary["actual_cost"] == 0.0
    assert summary["p95_latency_ms"] == 0.0
    assert summary["routing_distribution"] == {}
    assert summary["average_quality_score"] == 0.0
