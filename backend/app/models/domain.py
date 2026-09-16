from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt


class ModelTier(StrEnum):
    LOCAL = "local"
    ECONOMY = "economy"
    BALANCED = "balanced"
    FRONTIER = "frontier"


class TaskType(StrEnum):
    GENERAL = "general"
    CODING = "coding"
    REASONING = "reasoning"


class RequestIntent(StrEnum):
    SIMPLE_QA = "simple_qa"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    CODING = "coding"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    UNKNOWN = "unknown"


class ModelCapabilities(BaseModel):
    """Feature-level capabilities independent of scored model benchmarks."""

    model_config = ConfigDict(extra="forbid")

    supports_chat: bool = True
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False


class ModelDefinition(BaseModel):
    """Catalog entry used as routing input; prices are USD per million tokens."""

    model_config = ConfigDict(extra="forbid")

    id: str
    tier: ModelTier
    provider: str
    model_name: str
    input_cost_per_million_tokens: NonNegativeFloat
    output_cost_per_million_tokens: NonNegativeFloat
    expected_latency_ms: NonNegativeInt
    reasoning_score: float = Field(ge=0, le=1)
    coding_score: float = Field(ge=0, le=1)
    general_score: float = Field(ge=0, le=1)
    context_window: NonNegativeInt
    enabled: bool = True
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)


class RequestProfile(BaseModel):
    """Normalized routing constraints predicted from an incoming chat request."""

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    intent: RequestIntent = RequestIntent.UNKNOWN
    reasoning_requirement: float = Field(default=0, ge=0, le=1)
    coding_requirement: float = Field(default=0, ge=0, le=1)
    general_requirement: float = Field(default=0, ge=0, le=1)
    context_requirement: float = Field(default=0, ge=0, le=1)
    estimated_complexity: float = Field(default=0, ge=0, le=1)
    quality_target: float = Field(default=0.5, ge=0, le=1)
    latency_priority: float = Field(default=0.5, ge=0, le=1)
    task_type: TaskType = TaskType.GENERAL
    estimated_input_tokens: NonNegativeInt = 0
    estimated_output_tokens: NonNegativeInt = 0
    required_context_tokens: NonNegativeInt = 0
    max_latency_ms: NonNegativeInt | None = None
    requires_tools: bool = False
    requires_vision: bool = False
    requires_streaming: bool = False
    expects_json: bool = False


class CandidateModel(BaseModel):
    model_id: str
    tier: ModelTier
    provider: str
    model_name: str
    predicted_quality: float = Field(ge=0, le=1)
    estimated_cost: NonNegativeFloat
    estimated_latency: NonNegativeInt
    fitness_score: float


class RejectedModel(BaseModel):
    model_id: str
    tier: ModelTier
    reason: str


class RoutingDecision(BaseModel):
    """Auditable result of comparing a profile with enabled catalog models."""

    request_id: str
    selected_model: ModelDefinition
    candidate_models: list[CandidateModel]
    rejected_models: list[RejectedModel]
    predicted_quality: float = Field(ge=0, le=1)
    estimated_cost: NonNegativeFloat
    estimated_latency: NonNegativeInt
    routing_reason: str
    request_profile: RequestProfile
    router_latency_ms: NonNegativeInt
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QualityResult(BaseModel):
    request_id: str
    model_id: str
    score: float = Field(ge=0, le=1)
    passed: bool
    evaluator_type: str
    reasons: list[str] = Field(default_factory=list)
    evaluation_latency_ms: NonNegativeInt
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProviderAttempt(BaseModel):
    """One successful upstream completion within a logical NOMINAL request."""

    model_id: str
    provider: str
    model_name: str
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    latency_ms: NonNegativeInt


class RequestTrace(BaseModel):
    """Minimal end-to-end record for telemetry implementations to persist."""

    request_id: str
    profile: RequestProfile
    routing_decision: RoutingDecision | None = None
    quality_result: QualityResult | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    actual_latency_ms: NonNegativeInt | None = None
    provider: str | None = None
    model_used: str | None = None
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    provider_attempts: list[ProviderAttempt] = Field(default_factory=list)
    initial_model: str | None = None
    final_model: str | None = None
    escalated: bool = False
    escalation_reason: str | None = None
    quality_escalation_count: NonNegativeInt = 0
    infrastructure_fallback_count: NonNegativeInt = 0
    infrastructure_fallback_reasons: list[str] = Field(default_factory=list)
    provider_latency_ms: NonNegativeInt | None = None
    total_latency_ms: NonNegativeInt | None = None
    actual_cost: Decimal | None = None
    estimated_baseline_cost: Decimal | None = None
    estimated_cost_saved: Decimal | None = None
    savings_percentage: float | None = Field(default=None, ge=0, le=100)
    fallback_used: bool = False
    fallback_reason: str | None = None
    status: str = "success"
    metadata: dict[str, Any] = Field(default_factory=dict)
