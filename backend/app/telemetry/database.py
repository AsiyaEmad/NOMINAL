from datetime import datetime
from decimal import Decimal
from math import ceil
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, Numeric, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from backend.app.models import (
    BenchmarkRunResult,
    BenchmarkStrategyMetrics,
    QualityResult,
    RequestProfile,
    RequestTrace as DomainRequestTrace,
    RoutingDecision,
)
from backend.app.telemetry.base import TelemetryRepository


class Base(DeclarativeBase):
    pass


class RequestTrace(Base):
    """Persistent SQLite record for an inference request and its routing outcome."""

    __tablename__ = "request_traces"

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    intent: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    request_profile: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    selected_model: Mapped[str | None] = mapped_column(String(128))
    final_model: Mapped[str | None] = mapped_column(String(128), index=True)
    candidate_models: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    routing_reason: Mapped[str | None] = mapped_column(Text)
    router_latency_ms: Mapped[int | None] = mapped_column(Integer)
    provider_latency_ms: Mapped[int | None] = mapped_column(Integer)
    total_latency_ms: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    provider_attempts: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    quality_score: Mapped[float | None] = mapped_column(Float)
    quality_passed: Mapped[bool | None] = mapped_column(Boolean)
    actual_cost: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    baseline_cost: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    cost_saved: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    savings_percentage: Mapped[float | None] = mapped_column(Float)
    escalated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    escalation_reason: Mapped[str | None] = mapped_column(Text)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fallback_reason: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="success", index=True)
    routing_decision: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    quality_result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    provider: Mapped[str | None] = mapped_column(String(128))
    model_used: Mapped[str | None] = mapped_column(String(128))
    infrastructure_fallback_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    infrastructure_fallback_reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)


class BenchmarkRun(Base):
    """One reproducible execution of all benchmark strategies against the fixed dataset."""

    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluator_type: Mapped[str] = mapped_column(String(64), nullable=False)
    judge_model: Mapped[str | None] = mapped_column(String(128))
    strategies: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    request_results: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)


class MetricsSummary(dict):
    """Dictionary response with named numeric metrics, kept dependency-light for the MVP."""


class SqliteTelemetryRepository(TelemetryRepository):
    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self._engine = create_engine(database_url, connect_args=connect_args)
        self._sessions = sessionmaker(self._engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self._engine)
        existing_columns = {column["name"] for column in inspect(self._engine).get_columns("request_traces")}
        if "provider_attempts" not in existing_columns:
            with self._engine.begin() as connection:
                connection.execute(text("ALTER TABLE request_traces ADD COLUMN provider_attempts JSON"))

    def dispose(self) -> None:
        self._engine.dispose()

    async def record(self, trace: DomainRequestTrace) -> None:
        with self._sessions.begin() as session:
            record = self._to_record(trace)
            session.merge(record)

    async def get(self, request_id: str) -> DomainRequestTrace | None:
        with self._sessions() as session:
            record = session.get(RequestTrace, request_id)
            return self._to_domain(record) if record else None

    async def list(self, limit: int, offset: int) -> list[DomainRequestTrace]:
        with self._sessions() as session:
            records = session.scalars(
                select(RequestTrace).order_by(RequestTrace.timestamp.desc()).offset(offset).limit(limit)
            ).all()
            return [self._to_domain(record) for record in records]

    async def summary(self) -> MetricsSummary:
        with self._sessions() as session:
            records = session.scalars(select(RequestTrace)).all()
        total_requests = len(records)
        successful = [record for record in records if record.status == "success"]
        total_tokens = sum((record.input_tokens or 0) + (record.output_tokens or 0) for record in records)
        actual = sum((record.actual_cost or Decimal("0")) for record in records)
        baseline = sum((record.baseline_cost or Decimal("0")) for record in records)
        saved = sum((record.cost_saved or Decimal("0")) for record in records)
        latencies = [record.total_latency_ms for record in records if record.total_latency_ms is not None]
        router_latencies = [record.router_latency_ms for record in records if record.router_latency_ms is not None]
        quality_scores = [record.quality_score for record in records if record.quality_score is not None]
        distribution: dict[str, int] = {}
        for record in records:
            if record.final_model:
                distribution[record.final_model] = distribution.get(record.final_model, 0) + 1
        frontier_calls_avoided = sum(
            1 for record in successful if record.final_model and "frontier" not in record.final_model
        )
        return MetricsSummary(
            total_requests=total_requests,
            successful_requests=len(successful),
            total_tokens=total_tokens,
            actual_cost=float(actual),
            estimated_baseline_cost=float(baseline),
            estimated_cost_saved=float(saved),
            savings_percentage=round(float(saved / baseline * 100), 2) if baseline else 0.0,
            average_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            p95_latency_ms=self._p95(latencies),
            average_router_overhead_ms=round(sum(router_latencies) / len(router_latencies), 2) if router_latencies else 0.0,
            escalation_rate=round(sum(record.escalated for record in records) / total_requests * 100, 2) if total_requests else 0.0,
            fallback_rate=round(sum(record.fallback_used for record in records) / total_requests * 100, 2) if total_requests else 0.0,
            frontier_calls_avoided=frontier_calls_avoided,
            routing_distribution=distribution,
            average_quality_score=round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0,
        )

    @staticmethod
    def _p95(values: list[int]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return float(ordered[ceil(len(ordered) * 0.95) - 1])

    @staticmethod
    def _to_record(trace: DomainRequestTrace) -> RequestTrace:
        decision = trace.routing_decision
        quality = trace.quality_result
        return RequestTrace(
            request_id=trace.request_id,
            timestamp=trace.completed_at or trace.started_at,
            intent=trace.profile.intent.value,
            request_profile=trace.profile.model_dump(mode="json"),
            selected_model=trace.initial_model or (decision.selected_model.id if decision else None),
            final_model=trace.final_model,
            candidate_models=[item.model_dump(mode="json") for item in decision.candidate_models] if decision else [],
            routing_reason=decision.routing_reason if decision else None,
            router_latency_ms=decision.router_latency_ms if decision else None,
            provider_latency_ms=trace.provider_latency_ms or trace.actual_latency_ms,
            total_latency_ms=trace.total_latency_ms,
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            provider_attempts=[attempt.model_dump(mode="json") for attempt in trace.provider_attempts],
            quality_score=quality.score if quality else None,
            quality_passed=quality.passed if quality else None,
            actual_cost=trace.actual_cost,
            baseline_cost=trace.estimated_baseline_cost,
            cost_saved=trace.estimated_cost_saved,
            savings_percentage=trace.savings_percentage,
            escalated=trace.escalated,
            escalation_reason=trace.escalation_reason,
            fallback_used=trace.fallback_used,
            fallback_reason=trace.fallback_reason,
            status=trace.status,
            routing_decision=decision.model_dump(mode="json") if decision else None,
            quality_result=quality.model_dump(mode="json") if quality else None,
            provider=trace.provider,
            model_used=trace.model_used,
            infrastructure_fallback_count=trace.infrastructure_fallback_count,
            infrastructure_fallback_reasons=trace.infrastructure_fallback_reasons,
        )

    @staticmethod
    def _to_domain(record: RequestTrace) -> DomainRequestTrace:
        decision = RoutingDecision.model_validate(record.routing_decision) if record.routing_decision else None
        quality = QualityResult.model_validate(record.quality_result) if record.quality_result else None
        return DomainRequestTrace(
            request_id=record.request_id,
            profile=RequestProfile.model_validate(record.request_profile),
            routing_decision=decision,
            quality_result=quality,
            started_at=record.timestamp,
            completed_at=record.timestamp,
            actual_latency_ms=record.provider_latency_ms,
            provider_latency_ms=record.provider_latency_ms,
            total_latency_ms=record.total_latency_ms,
            provider=record.provider,
            model_used=record.model_used,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            provider_attempts=record.provider_attempts or [],
            initial_model=record.selected_model,
            final_model=record.final_model,
            escalated=record.escalated,
            escalation_reason=record.escalation_reason,
            infrastructure_fallback_count=record.infrastructure_fallback_count,
            infrastructure_fallback_reasons=record.infrastructure_fallback_reasons,
            fallback_used=record.fallback_used,
            fallback_reason=record.fallback_reason,
            actual_cost=record.actual_cost,
            estimated_baseline_cost=record.baseline_cost,
            estimated_cost_saved=record.cost_saved,
            savings_percentage=record.savings_percentage,
            status=record.status,
        )


class BenchmarkRepository:
    """Small SQLite persistence layer for benchmark results, sharing the app database."""

    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self._engine = create_engine(database_url, connect_args=connect_args)
        self._sessions = sessionmaker(self._engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self._engine)
        existing_columns = {column["name"] for column in inspect(self._engine).get_columns("benchmark_runs")}
        if "judge_model" not in existing_columns:
            with self._engine.begin() as connection:
                connection.execute(text("ALTER TABLE benchmark_runs ADD COLUMN judge_model VARCHAR(128)"))

    def dispose(self) -> None:
        self._engine.dispose()

    async def save(self, result: BenchmarkRunResult) -> None:
        with self._sessions.begin() as session:
            session.merge(
                BenchmarkRun(
                    id=result.id,
                    created_at=result.created_at,
                    status=result.status,
                    dataset_name=result.dataset_name,
                    evaluator_type=result.evaluator_type,
                    judge_model=result.judge_model,
                    strategies=[item.model_dump(mode="json") for item in result.strategies],
                    request_results=result.request_results,
                )
            )

    async def list(self, limit: int = 20) -> list[BenchmarkRunResult]:
        with self._sessions() as session:
            records = session.scalars(
                select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc()).limit(limit)
            ).all()
            return [self._to_result(record, include_requests=False) for record in records]

    async def get(self, run_id: str) -> BenchmarkRunResult | None:
        with self._sessions() as session:
            record = session.get(BenchmarkRun, run_id)
            return self._to_result(record, include_requests=True) if record else None

    @staticmethod
    def _to_result(record: BenchmarkRun, *, include_requests: bool) -> BenchmarkRunResult:
        return BenchmarkRunResult(
            id=record.id,
            created_at=record.created_at,
            status=record.status,
            dataset_name=record.dataset_name,
            evaluator_type=record.evaluator_type,
            judge_model=record.judge_model,
            strategies=[BenchmarkStrategyMetrics.model_validate(item) for item in record.strategies],
            request_results=record.request_results if include_requests else [],
        )
