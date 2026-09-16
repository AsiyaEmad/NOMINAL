from collections.abc import Awaitable, Callable

import pytest

from backend.app.benchmark.quality import JudgeModelBenchmarkQualityEvaluator
from backend.app.benchmark.retry import BenchmarkOperationContext, BenchmarkOperationRetry
from backend.app.models import (
    ChatCompletionRequest,
    ChatMessage,
    InternalChatChoice,
    InternalChatCompletion,
    ModelDefinition,
    ModelTier,
    TokenUsage,
)
from backend.app.providers import ModelProvider, ProviderRegistry
from backend.app.providers.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def _model(model_id: str = "economy") -> ModelDefinition:
    return ModelDefinition(
        id=model_id,
        tier=ModelTier.ECONOMY,
        provider="mock",
        model_name=model_id,
        input_cost_per_million_tokens=0.4,
        output_cost_per_million_tokens=1.6,
        expected_latency_ms=100,
        reasoning_score=0.8,
        coding_score=0.8,
        general_score=0.8,
        context_window=100_000,
    )


def _completion(model: str = "economy") -> InternalChatCompletion:
    return InternalChatCompletion(
        model=model,
        provider_latency_ms=10,
        choices=[InternalChatChoice(message=ChatMessage(role="assistant", content="Paris."))],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _context(strategy: str = "always_economy") -> BenchmarkOperationContext:
    return BenchmarkOperationContext(strategy=strategy, corpus_item_id="qa-01")


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="nominal-auto", messages=[ChatMessage(role="user", content="Reply briefly.")]
    )


class _SequencedProvider(ModelProvider):
    def __init__(self, outcomes: list[InternalChatCompletion | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    @property
    def name(self) -> str:
        return "mock"

    async def chat_completion(
        self, model: ModelDefinition, request: ChatCompletionRequest
    ) -> InternalChatCompletion:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def health_check(self) -> bool:
        return True


def _retry(delays: list[float]) -> BenchmarkOperationRetry:
    async def sleeper(delay: float) -> None:
        delays.append(delay)

    return BenchmarkOperationRetry(sleeper=sleeper, random_source=lambda: 0.0)


@pytest.mark.asyncio
async def test_transient_failure_then_success_is_retried_once() -> None:
    delays: list[float] = []
    provider = _SequencedProvider([ProviderUnavailableError("temporary"), _completion()])
    context = _context()

    result = await _retry(delays).execute_provider_chat(
        provider, _model(), _request(), context
    )

    operation = context.operation("inference")
    assert result.model == "economy"
    assert provider.calls == 2
    assert delays == [0.5]
    assert operation and operation.total_attempts == 2
    assert operation.benchmark_retry_count == 1
    assert operation.succeeded_after_retry is True


@pytest.mark.asyncio
async def test_three_transient_attempts_exhaust_neutral_retry_budget() -> None:
    delays: list[float] = []
    provider = _SequencedProvider([ProviderTimeoutError("timeout")] * 3)
    context = _context()

    with pytest.raises(ProviderTimeoutError):
        await _retry(delays).execute_provider_chat(
            provider, _model(), _request(), context
        )

    operation = context.operation("inference")
    assert provider.calls == 3
    assert delays == [0.5, 1.0]
    assert operation and operation.total_attempts == 3
    assert operation.benchmark_retry_count == 2
    assert operation.retries_exhausted is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ProviderError("bad request", upstream_status=400, failure_category="deterministic_http"),
        ProviderAuthenticationError("not authorized", upstream_status=401, failure_category="authentication"),
    ],
)
async def test_deterministic_or_auth_failures_are_not_retried(error: ProviderError) -> None:
    delays: list[float] = []
    provider = _SequencedProvider([error])
    context = _context()

    with pytest.raises(type(error)):
        await _retry(delays).execute_provider_chat(
            provider, _model(), _request(), context
        )

    operation = context.operation("inference")
    assert provider.calls == 1
    assert delays == []
    assert operation and operation.benchmark_retry_count == 0
    assert operation.retries_exhausted is False


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ProviderRateLimitError("limited"), ProviderTimeoutError("timeout")])
async def test_rate_limit_and_timeout_are_retryable(error: ProviderError) -> None:
    delays: list[float] = []
    provider = _SequencedProvider([error, _completion()])
    context = _context()

    await _retry(delays).execute_provider_chat(
        provider, _model(), _request(), context
    )

    assert provider.calls == 2
    assert context.retry_count("inference") == 1


@pytest.mark.asyncio
async def test_judge_retry_never_reexecutes_candidate_inference() -> None:
    judge_payload = (
        '{"score":0.9,"passed":true,"reasons":["correct"],'
        '"criteria_results":[{"criterion":"Names Paris","passed":true,"score":0.9,"reason":"yes"}]}'
    )
    provider = _SequencedProvider([
        ProviderUnavailableError("temporary", upstream_status=503, failure_category="http_5xx"),
        _completion("judge" ).model_copy(update={
            "choices": [InternalChatChoice(message=ChatMessage(role="assistant", content=judge_payload))]
        }),
    ])
    delays: list[float] = []
    evaluator = JudgeModelBenchmarkQualityEvaluator(
        providers=ProviderRegistry({"mock": provider}), judge_model=_model("judge"), max_tokens=128,
        operation_retry=_retry(delays),
    )
    context = _context()
    candidate = _completion("candidate")

    evaluation = await evaluator.evaluate(
        {"id": "qa-01", "category": "simple_qa", "prompt": "Capital?", "evaluation_criteria": "Names Paris"},
        candidate,
        context,
    )

    assert evaluation.failed is False
    assert provider.calls == 2
    assert context.attempt_count("inference") == 0
    assert context.retry_count("benchmark_judge") == 1
    assert evaluation.operation and evaluation.operation.operation_type == "benchmark_judge"
    assert evaluation.evaluation_cost > 0


def test_operation_telemetry_is_sanitized() -> None:
    context = _context()
    context.record(
        # This mirrors the persisted operation shape and protects against
        # accidentally serializing exception args or request headers.
        __import__("backend.app.benchmark.retry", fromlist=["BenchmarkOperationTelemetry"]).BenchmarkOperationTelemetry(
            operation_type="inference", strategy="always_economy", corpus_item_id="qa-01",
            provider="openai", model="economy", total_attempts=1, benchmark_retry_count=0,
            final_status="failed", elapsed_ms=1, exception_type="ProviderUnavailableError",
            failure_category="transport",
        )
    )
    serialized = str([operation.model_dump(mode="json") for operation in context.operations])
    assert "authorization" not in serialized.lower()
    assert "api_key" not in serialized.lower()
    assert "bearer" not in serialized.lower()
