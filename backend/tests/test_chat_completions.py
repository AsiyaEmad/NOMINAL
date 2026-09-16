import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import ProviderConfig
from backend.app.main import create_app
from backend.app.models import ChatCompletionRequest, ModelDefinition, ModelTier
from backend.app.providers import OpenAICompatibleProvider, ProviderRegistry
from backend.app.providers.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)


def test_chat_completion_proxies_messages_and_normalizes_response() -> None:
    received_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received_request.update(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-secret"
        return httpx.Response(
            200,
            json={
                "id": "upstream-response-id",
                "created": 1_700_000_000,
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from NOMINAL."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        )

    provider = OpenAICompatibleProvider(
        "openai",
        ProviderConfig(base_url="https://provider.example/v1", api_key="test-secret"),
        transport=httpx.MockTransport(handler),
    )
    app = create_app(provider_registry=ProviderRegistry({"openai": provider}))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nominal-auto",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Summarize this text: Hello"},
                    {"role": "assistant", "content": "Earlier context"},
                ],
                "temperature": 0.7,
                "max_tokens": 500,
            },
        )
        trace_response = client.get(f"/api/traces/{response.headers['x-request-id']}")

    assert response.status_code == 200
    assert received_request["model"] == "gpt-4.1-mini"
    assert received_request["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Summarize this text: Hello"},
        {"role": "assistant", "content": "Earlier context"},
    ]
    assert response.json()["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }
    assert trace_response.status_code == 200
    trace = trace_response.json()
    assert trace["provider"] == "openai"
    assert trace["model_used"] == "gpt-4.1-mini"
    assert trace["input_tokens"] == 12
    assert trace["output_tokens"] == 4


def test_chat_completion_returns_safe_error_for_malformed_provider_response() -> None:
    provider = OpenAICompatibleProvider(
        "openai",
        ProviderConfig(base_url="https://provider.example/v1"),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []})),
    )
    app = create_app(provider_registry=ProviderRegistry({"openai": provider}))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "nominal-auto", "messages": [{"role": "user", "content": "Summarize: Hello"}]},
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "malformed_provider_response"


def test_malformed_chat_request_returns_openai_style_validation_error() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "nominal-auto", "messages": []},
        )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "message": "Request validation failed",
            "type": "invalid_request_error",
            "param": None,
            "code": "validation_error",
        }
    }


async def test_o3_uses_max_completion_tokens_for_openai_compatibility() -> None:
    received_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received_request.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Done."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    provider = OpenAICompatibleProvider(
        "openai",
        ProviderConfig(base_url="https://api.openai.com/v1"),
        transport=httpx.MockTransport(handler),
    )
    model = ModelDefinition(
        id="frontier-o3",
        tier=ModelTier.FRONTIER,
        provider="openai",
        model_name="o3",
        input_cost_per_million_tokens=2,
        output_cost_per_million_tokens=8,
        expected_latency_ms=3500,
        reasoning_score=0.96,
        coding_score=0.94,
        general_score=0.93,
        context_window=200000,
    )

    await provider.chat_completion(
        model,
        ChatCompletionRequest(
            model="frontier-o3",
            messages=[{"role": "user", "content": "Solve this."}],
            max_tokens=512,
        ),
    )

    assert received_request["model"] == "o3"
    assert received_request["max_completion_tokens"] == 512
    assert "max_tokens" not in received_request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error_type", "category"),
    [
        (401, ProviderAuthenticationError, "authentication"),
        (429, ProviderRateLimitError, "rate_limit"),
        (503, ProviderUnavailableError, "http_5xx"),
    ],
)
async def test_provider_errors_preserve_safe_status_and_category(
    status: int, error_type: type[Exception], category: str
) -> None:
    provider = OpenAICompatibleProvider(
        "openai",
        ProviderConfig(base_url="https://provider.example/v1"),
        transport=httpx.MockTransport(lambda request: httpx.Response(status, text="upstream body is ignored")),
    )
    model = ModelDefinition(
        id="economy", tier=ModelTier.ECONOMY, provider="openai", model_name="gpt-4.1-mini",
        input_cost_per_million_tokens=0.4, output_cost_per_million_tokens=1.6,
        expected_latency_ms=100, reasoning_score=0.8, coding_score=0.8, general_score=0.8,
        context_window=100_000,
    )
    request = ChatCompletionRequest(model="economy", messages=[{"role": "user", "content": "Hello"}])

    with pytest.raises(error_type) as raised:
        await provider.chat_completion(model, request)

    assert raised.value.upstream_status == status
    assert raised.value.failure_category == category
    assert "upstream body" not in str(raised.value)
