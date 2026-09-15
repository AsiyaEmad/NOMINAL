from pathlib import Path

import pytest

from backend.app.models import ChatCompletionRequest, ChatMessage
from backend.app.routing import CostAwareRoutingEngine, HeuristicRequestProfiler, ModelCatalog


@pytest.fixture
def catalog() -> ModelCatalog:
    return ModelCatalog.from_yaml(Path(__file__).resolve().parents[1] / "config" / "models.yaml")


@pytest.mark.asyncio
async def test_profiler_recognizes_capability_dimensions() -> None:
    profile = await HeuristicRequestProfiler().profile(
        ChatCompletionRequest(
            model="nominal-auto",
            messages=[ChatMessage(role="user", content="Implement and debug a Python function.")],
        )
    )

    assert profile.intent.value == "coding"
    assert profile.coding_requirement > profile.general_requirement
    assert profile.reasoning_requirement >= 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "expected_model"),
    [
        ("What is the capital of France?", "balanced-gpt-4.1"),
        ("Translate 'good morning' into French.", "economy-gpt-4.1-mini"),
        ("Extract all company names and dates from this paragraph.", "economy-gpt-4.1-mini"),
        ("Summarize this long document in detail.", "economy-gpt-4.1-mini"),
        ("Write a Python function to parse CSV input and handle edge cases.", "balanced-gpt-4.1"),
        ("Solve this dynamic programming algorithm and prove correctness.", "balanced-gpt-4.1"),
        ("Provide a comprehensive analysis of strategic tradeoffs and root causes.", "frontier-o3"),
    ],
)
async def test_different_prompts_produce_capability_aware_decisions(
    catalog: ModelCatalog, prompt: str, expected_model: str
) -> None:
    profiler = HeuristicRequestProfiler()
    engine = CostAwareRoutingEngine()
    request = ChatCompletionRequest(
        model="nominal-auto",
        messages=[ChatMessage(role="user", content=prompt)],
        max_tokens=300,
    )

    profile = await profiler.profile(request)
    decision = await engine.route(profile, catalog.enabled_models)

    assert decision.selected_model.id == expected_model
    assert decision.predicted_quality >= profile.quality_target or "fallback" in decision.routing_reason
    assert decision.request_profile.intent == profile.intent
    assert decision.candidate_models
    assert "Selected" in decision.routing_reason
