import time
from typing import Any

import httpx
from pydantic import ValidationError

from backend.app.core.config import ProviderConfig
from backend.app.models import (
    ChatCompletionRequest,
    ChatMessage,
    InternalChatChoice,
    InternalChatCompletion,
    ModelDefinition,
    TokenUsage,
)
from backend.app.providers.base import ModelProvider
from backend.app.providers.errors import (
    MalformedProviderResponseError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


class OllamaProvider(ModelProvider):
    """Adapter for Ollama's `/api/chat` endpoint."""

    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._name = name
        self._config = config
        self._transport = transport

    @property
    def name(self) -> str:
        return self._name

    async def chat_completion(
        self, model: ModelDefinition, request: ChatCompletionRequest
    ) -> InternalChatCompletion:
        payload: dict[str, Any] = {
            "model": model.model_name,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "stream": False,
        }
        if request.temperature is not None:
            payload["options"] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            payload.setdefault("options", {})["num_predict"] = request.max_tokens

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.timeout_seconds), transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._config.base_url.rstrip('/')}/api/chat", json=payload
                )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Ollama timed out") from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError("Ollama is unavailable") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code in (401, 403):
            raise ProviderAuthenticationError("Ollama rejected credentials")
        if response.status_code == 429:
            raise ProviderRateLimitError("Ollama rate limit exceeded")
        if response.status_code >= 500:
            raise ProviderUnavailableError("Ollama is unavailable")
        if response.is_error:
            raise ProviderError(f"Ollama returned HTTP {response.status_code}")
        try:
            raw_response: dict[str, Any] = response.json()
            return InternalChatCompletion(
                model=raw_response.get("model", model.model_name),
                choices=[
                    InternalChatChoice(
                        message=ChatMessage.model_validate(raw_response["message"]),
                        finish_reason=raw_response.get("done_reason"),
                    )
                ],
                usage=TokenUsage(
                    input_tokens=raw_response.get("prompt_eval_count", 0),
                    output_tokens=raw_response.get("eval_count", 0),
                ),
                provider_latency_ms=latency_ms,
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise MalformedProviderResponseError("Ollama returned an invalid completion") from exc

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.timeout_seconds), transport=self._transport
            ) as client:
                response = await client.get(f"{self._config.base_url.rstrip('/')}/api/tags")
            return response.is_success
        except httpx.HTTPError:
            return False
