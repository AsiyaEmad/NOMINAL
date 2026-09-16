from decimal import Decimal

import pytest

from backend.app.benchmark.quality import JudgeModelBenchmarkQualityEvaluator
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
from backend.app.providers.errors import ProviderUnavailableError


def _judge_model() -> ModelDefinition:
    return ModelDefinition(
        id="benchmark-judge",
        tier=ModelTier.ECONOMY,
        provider="judge",
        model_name="gpt-4.1-mini",
        input_cost_per_million_tokens=0.4,
        output_cost_per_million_tokens=1.6,
        expected_latency_ms=100,
        reasoning_score=0.8,
        coding_score=0.8,
        general_score=0.8,
        context_window=100_000,
    )


def _completion(content: str, *, model: str = "candidate", input_tokens: int = 9, output_tokens: int = 4) -> InternalChatCompletion:
    return InternalChatCompletion(
        model=model,
        provider_latency_ms=23,
        choices=[InternalChatChoice(message=ChatMessage(role="assistant", content=content))],
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class CapturingJudgeProvider(ModelProvider):
    def __init__(self, outcome: InternalChatCompletion | Exception) -> None:
        self.outcome = outcome
        self.requests: list[ChatCompletionRequest] = []

    @property
    def name(self) -> str:
        return "judge"

    async def chat_completion(self, model: ModelDefinition, request: ChatCompletionRequest) -> InternalChatCompletion:
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    async def health_check(self) -> bool:
        return True


def _item() -> dict:
    return {
        "id": "qa-01",
        "category": "simple_qa",
        "prompt": "What is the capital of France?",
        "expected_difficulty": "low",
        "evaluation_criteria": "Names Paris directly.",
    }


def _judge_completion(payload: str) -> InternalChatCompletion:
    return _completion(payload, model="gpt-4.1-mini", input_tokens=100, output_tokens=25)


@pytest.mark.asyncio
async def test_judge_consumes_criteria_is_strategy_blind_and_tracks_its_own_cost() -> None:
    provider = CapturingJudgeProvider(_judge_completion(
        '{"score":0.92,"passed":true,"reasons":["Correct."],'
        '"criteria_results":[{"criterion":"Names Paris directly.","passed":true,"score":0.92,"reason":"Paris is named."}]}'
    ))
    evaluator = JudgeModelBenchmarkQualityEvaluator(
        providers=ProviderRegistry({"judge": provider}), judge_model=_judge_model(), max_tokens=256
    )

    result = await evaluator.evaluate(_item(), _completion("Paris.", model="economy-response"))

    assert result.score == pytest.approx(0.92)
    assert result.passed is True
    assert result.criteria_results[0].criterion == "Names Paris directly."
    assert result.evaluator_type == "benchmark_judge"
    assert result.judge_model == "benchmark-judge"
    assert result.evaluation_cost == Decimal("0.00008000")
    assert result.evaluation_latency_ms == 23
    assert result.input_tokens == 100 and result.output_tokens == 25

    request = provider.requests[0]
    prompt = "\n".join(str(message.content) for message in request.messages)
    assert "Names Paris directly." in prompt
    assert "What is the capital of France?" in prompt
    assert "simple_qa" in prompt
    assert "always_frontier" not in prompt
    assert "always_economy" not in prompt
    assert "nominal" not in prompt.lower()
    assert "latency" not in prompt.lower()
    assert "cost" not in prompt.lower()
    assert request.temperature == 0
    assert request.model_extra["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_judge_uses_one_fixed_rubric_for_all_strategy_outputs() -> None:
    payload = (
        '{"score":0.8,"passed":true,"reasons":["Adequate."],'
        '"criteria_results":[{"criterion":"Names Paris directly.","passed":true,"score":0.8,"reason":"Adequate."}]}'
    )
    provider = CapturingJudgeProvider(_judge_completion(payload))
    evaluator = JudgeModelBenchmarkQualityEvaluator(
        providers=ProviderRegistry({"judge": provider}), judge_model=_judge_model(), max_tokens=256
    )

    for model_id in ("economy", "frontier", "nominal-final"):
        result = await evaluator.evaluate(_item(), _completion("Paris.", model=model_id))
        assert result.passed is True

    assert len(provider.requests) == 3
    prompts = [str(request.messages[1].content) for request in provider.requests]
    rubric = "0.90-1.00 fully satisfies; 0.75-0.89 substantially satisfies"
    assert all(rubric in prompt for prompt in prompts)
    assert all("strategy" not in prompt.lower() for prompt in prompts)


@pytest.mark.asyncio
async def test_malformed_judge_output_is_an_explicit_evaluation_failure() -> None:
    provider = CapturingJudgeProvider(_judge_completion("not valid json"))
    evaluator = JudgeModelBenchmarkQualityEvaluator(
        providers=ProviderRegistry({"judge": provider}), judge_model=_judge_model(), max_tokens=256
    )

    result = await evaluator.evaluate(_item(), _completion("Paris."))

    assert result.failed is True
    assert result.score is None
    assert result.passed is None
    assert result.failure_reason == "malformed_judge_response"
    # The malformed verdict was still a completed, billable judge call.
    assert result.evaluation_cost == Decimal("0.00008000")
    assert result.input_tokens == 100 and result.output_tokens == 25


@pytest.mark.asyncio
async def test_judge_provider_failure_is_explicit_and_not_scored() -> None:
    provider = CapturingJudgeProvider(ProviderUnavailableError("judge unavailable"))
    evaluator = JudgeModelBenchmarkQualityEvaluator(
        providers=ProviderRegistry({"judge": provider}), judge_model=_judge_model(), max_tokens=256
    )

    result = await evaluator.evaluate(_item(), _completion("Paris."))

    assert result.failed is True
    assert result.score is None
    assert result.passed is None
    assert result.failure_reason == "upstream_unavailable"
    assert result.evaluation_cost == Decimal("0")
