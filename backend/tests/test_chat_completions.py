import json

import httpx
from fastapi.testclient import TestClient

from backend.app.core.config import ProviderConfig
from backend.app.main import create_app
from backend.app.providers import OpenAICompatibleProvider, ProviderRegistry


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
