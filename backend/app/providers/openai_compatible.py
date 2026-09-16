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


class OpenAICompatibleProvider(ModelProvider):
    """Adapter for providers exposing `POST /chat/completions` beneath a base URL."""

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
        headers = {"content-type": "application/json"}
        api_key = self._config.api_key.get_secret_value() if self._config.api_key else None
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"

        payload = request.to_provider_payload(model.model_name)
        # The o-series Chat Completions API requires the modern completion-token
        # parameter; NOMINAL's public compatibility surface still accepts max_tokens.
        if model.model_name.startswith("o") and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.timeout_seconds),
                transport=self._transport,
            ) as client:
                response = await client.post(
                    f"{self._config.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Upstream provider timed out", failure_category="timeout") from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError("Upstream provider is unavailable", failure_category="transport") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code in (401, 403):
            raise ProviderAuthenticationError(
                "Upstream provider rejected credentials",
                upstream_status=response.status_code,
                failure_category="authentication",
            )
        if response.status_code == 429:
            raise ProviderRateLimitError(
                "Upstream provider rate limit exceeded", upstream_status=429, failure_category="rate_limit"
            )
        if response.status_code >= 500:
            raise ProviderUnavailableError(
                "Upstream provider is unavailable",
                upstream_status=response.status_code,
                failure_category="http_5xx",
            )
        if response.is_error:
            raise ProviderError(
                f"Upstream provider returned HTTP {response.status_code}",
                upstream_status=response.status_code,
                failure_category="deterministic_http",
            )

        try:
            raw_response: dict[str, Any] = response.json()
        except ValueError as exc:
            raise MalformedProviderResponseError(
                "Upstream provider returned invalid JSON", failure_category="validation"
            ) from exc
        return self._normalize(raw_response, model.model_name, latency_ms)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.timeout_seconds), transport=self._transport
            ) as client:
                response = await client.get(f"{self._config.base_url.rstrip('/')}/models")
            return response.is_success
        except httpx.HTTPError:
            return False

    @staticmethod
    def _normalize(
        payload: dict[str, Any], fallback_model: str, latency_ms: int
    ) -> InternalChatCompletion:
        try:
            raw_choices = payload["choices"]
            if not isinstance(raw_choices, list) or not raw_choices:
                raise ValueError("choices must be a non-empty list")
            choices = [
                InternalChatChoice(
                    index=item.get("index", index),
                    message=ChatMessage.model_validate(item["message"]),
                    finish_reason=item.get("finish_reason"),
                )
                for index, item in enumerate(raw_choices)
            ]
            raw_usage = payload.get("usage", {})
            usage = TokenUsage(
                input_tokens=raw_usage.get("prompt_tokens", 0),
                output_tokens=raw_usage.get("completion_tokens", 0),
            )
            return InternalChatCompletion(
                id=payload.get("id") or f"chatcmpl-{int(time.time() * 1000)}",
                created=payload.get("created", int(time.time())),
                model=payload.get("model", fallback_model),
                choices=choices,
                usage=usage,
                provider_latency_ms=latency_ms,
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise MalformedProviderResponseError(
                "Upstream provider returned an invalid completion", failure_category="validation"
            ) from exc
