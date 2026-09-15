from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt


class BenchmarkStrategy(StrEnum):
    ALWAYS_FRONTIER = "always_frontier"
    ALWAYS_ECONOMY = "always_economy"
    NOMINAL = "nominal"


class BenchmarkRunRequest(BaseModel):
    max_items: int | None = Field(default=None, ge=1, le=40)


class BenchmarkStrategyMetrics(BaseModel):
    strategy: BenchmarkStrategy
    request_count: NonNegativeInt
    total_cost: NonNegativeFloat
    estimated_cost_per_request: NonNegativeFloat
    total_tokens: NonNegativeInt
    average_latency_ms: NonNegativeFloat
    p50_latency_ms: NonNegativeFloat
    p95_latency_ms: NonNegativeFloat
    quality_score: float = Field(ge=0, le=1)
    quality_pass_rate: float = Field(ge=0, le=100)
    frontier_calls: NonNegativeInt
    escalations: NonNegativeInt
    failures: NonNegativeInt


class BenchmarkRunResult(BaseModel):
    id: str
    created_at: datetime
    status: str
    dataset_name: str
    evaluator_type: str
    strategies: list[BenchmarkStrategyMetrics]
    request_results: list[dict[str, Any]] = Field(default_factory=list)
