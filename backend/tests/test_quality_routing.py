from collections.abc import Sequence

import pytest

from backend.app.core.config import Settings
from backend.app.models import (
    ChatCompletionRequest,
    ChatMessage,
    InternalChatChoice,
    InternalChatCompletion,
    ModelDefinition,
    ModelTier,
    RequestProfile,
)
from backend.app.providers import ModelProvider, ProviderRegistry
from backend.app.providers.errors import ProviderRateLimitError, ProviderTimeoutError, ProviderUnavailableError
from backend.app.quality import LayeredQualityEvaluator
from backend.app.routing import ChatCompletionGateway, CostAwareRoutingEngine, ModelCatalog, RequestProfiler
from backend.app.telemetry import InMemoryTelemetryRepository


class FixedProfiler(RequestProfiler):
    async def profile(
        self, request: ChatCompletionRequest, request_id: str | None = None
    ) -> RequestProfile:
        return RequestProfile(
            request_id=request_id or "test-request",
            general_requirement=0.40,
            quality_target=0.80,
            estimated_input_tokens=10,
            estimated_output_tokens=10,
            required_context_tokens=20,
        )


class ScriptedProvider(ModelProvider):
    def __init__(self, name: str, outcomes: list[str | Exception]) -> None:
        self._name = name
        self._outcomes = outcomes
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    async def chat_completion(
        self, model: ModelDefinition, request: ChatCompletionRequest
    ) -> InternalChatCompletion:
        self.calls.append(model.id)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return InternalChatCompletion(
            model=model.model_name,
            provider_latency_ms=10,
            choices=[InternalChatChoice(message=ChatMessage(role="assistant", content=outcome))],
        )

    async def health_check(self) -> bool:
        return True


def _model(model_id: str, provider: str, score: float, cost: float, tier: ModelTier) -> ModelDefinition:
    return ModelDefinition(
        id=model_id,
        tier=tier,
        provider=provider,
        model_name=model_id,
        input_cost_per_million_tokens=cost,
        output_cost_per_million_tokens=cost,
        expected_latency_ms=500,
        reasoning_score=score,
        coding_score=score,
        general_score=score,
        context_window=8_000,
    )


def _gateway(
    providers: dict[str, ScriptedProvider], *, max_quality_escalations: int = 1, max_provider_failovers: int = 1
) -> tuple[ChatCompletionGateway, InMemoryTelemetryRepository]:
    catalog = ModelCatalog(
        models=[
            _model("cheap", "cheap-provider", 0.85, 0.10, ModelTier.ECONOMY),
            _model("capable", "capable-provider", 0.95, 1.00, ModelTier.BALANCED),
        ]
    )
    telemetry = InMemoryTelemetryRepository()
    gateway = ChatCompletionGateway(
        catalog=catalog,
        providers=ProviderRegistry(providers),
        telemetry=telemetry,
        settings=Settings(
            max_quality_escalations=max_quality_escalations,
            max_provider_failovers=max_provider_failovers,
        ),
        profiler=FixedProfiler(),
        routing_engine=CostAwareRoutingEngine(),
        quality_evaluator=LayeredQualityEvaluator(),
    )
    return gateway, telemetry


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="nominal-auto", messages=[ChatMessage(role="user", content="Answer this request.")]
    )


@pytest.mark.asyncio
async def test_successful_cheap_model_response_does_not_escalate() -> None:
    cheap = ScriptedProvider("cheap-provider", ["A complete, useful answer from the cheap model."])
    capable = ScriptedProvider("capable-provider", ["unused"])
    gateway, telemetry = _gateway({cheap.name: cheap, capable.name: capable})

    request_id, completion, _ = await gateway.complete(_request())
    trace = await telemetry.get(request_id)

    assert completion.model == "cheap"
    assert cheap.calls == ["cheap"]
    assert capable.calls == []
    assert trace is not None and trace.escalated is False
    assert trace.quality_result is not None and trace.quality_result.passed is True


@pytest.mark.asyncio
async def test_quality_failure_escalates_to_higher_capability_model() -> None:
    cheap = ScriptedProvider("cheap-provider", ["no"])
    capable = ScriptedProvider("capable-provider", ["A complete, useful answer from the capable model."])
    gateway, telemetry = _gateway({cheap.name: cheap, capable.name: capable})

    request_id, completion, _ = await gateway.complete(_request())
    trace = await telemetry.get(request_id)

    assert completion.model == "capable"
    assert cheap.calls == ["cheap"]
    assert capable.calls == ["capable"]
    assert trace is not None and trace.escalated is True
    assert trace.initial_model == "cheap"
    assert trace.final_model == "capable"
    assert trace.escalation_reason == "response is suspiciously short"


@pytest.mark.asyncio
async def test_timeout_uses_independent_infrastructure_fallback() -> None:
    cheap = ScriptedProvider("cheap-provider", [ProviderTimeoutError("timed out")])
    capable = ScriptedProvider("capable-provider", ["A complete answer after fallback."])
    gateway, telemetry = _gateway({cheap.name: cheap, capable.name: capable})

    request_id, completion, _ = await gateway.complete(_request())
    trace = await telemetry.get(request_id)

    assert completion.model == "capable"
    assert trace is not None and trace.escalated is False
    assert trace.infrastructure_fallback_count == 1
    assert trace.infrastructure_fallback_reasons == ["cheap: upstream_timeout"]


@pytest.mark.asyncio
async def test_rate_limit_uses_independent_infrastructure_fallback() -> None:
    cheap = ScriptedProvider("cheap-provider", [ProviderRateLimitError("rate limited")])
    capable = ScriptedProvider("capable-provider", ["A complete answer after fallback."])
    gateway, telemetry = _gateway({cheap.name: cheap, capable.name: capable})

    request_id, completion, _ = await gateway.complete(_request())
    trace = await telemetry.get(request_id)

    assert completion.model == "capable"
    assert trace is not None and trace.infrastructure_fallback_count == 1
    assert trace.infrastructure_fallback_reasons == ["cheap: upstream_rate_limited"]


@pytest.mark.asyncio
async def test_all_providers_unavailable_records_failure_and_stops() -> None:
    cheap = ScriptedProvider("cheap-provider", [ProviderUnavailableError("cheap unavailable")])
    capable = ScriptedProvider("capable-provider", [ProviderUnavailableError("capable unavailable")])
    gateway, telemetry = _gateway(
        {cheap.name: cheap, capable.name: capable}, max_provider_failovers=3
    )

    with pytest.raises(ProviderUnavailableError) as error:
        await gateway.complete(_request())

    trace = await telemetry.get(error.value.request_id)
    assert cheap.calls == ["cheap"]
    assert capable.calls == ["capable"]
    assert trace is not None
    assert trace.infrastructure_fallback_count == 2
    assert trace.escalated is False
