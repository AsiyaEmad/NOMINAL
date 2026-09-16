"""Neutral, benchmark-only retries for transient provider operations."""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.models import ChatCompletionRequest, InternalChatCompletion, ModelDefinition
from backend.app.providers.base import ModelProvider
from backend.app.providers.errors import (
    MalformedProviderResponseError,
    ModelNotAvailableError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

OperationType = Literal["inference", "benchmark_judge"]


class BenchmarkOperationTelemetry(BaseModel):
    """Sanitized operation-level record persisted with a benchmark result."""

    operation_type: OperationType
    strategy: str
    corpus_item_id: str
    provider: str
    model: str
    total_attempts: int = Field(ge=1)
    benchmark_retry_count: int = Field(ge=0)
    final_status: Literal["success", "failed"]
    elapsed_ms: int = Field(ge=0)
    http_status: int | None = None
    exception_type: str | None = None
    failure_category: str | None = None
    succeeded_after_retry: bool = False
    retries_exhausted: bool = False


@dataclass
class BenchmarkOperationContext:
    strategy: str
    corpus_item_id: str
    operations: list[BenchmarkOperationTelemetry] = field(default_factory=list)

    def record(self, telemetry: BenchmarkOperationTelemetry) -> None:
        self.operations.append(telemetry)

    def operation(self, operation_type: OperationType) -> BenchmarkOperationTelemetry | None:
        return next((item for item in reversed(self.operations) if item.operation_type == operation_type), None)

    def retry_count(self, operation_type: OperationType | None = None) -> int:
        return sum(
            item.benchmark_retry_count
            for item in self.operations
            if operation_type is None or item.operation_type == operation_type
        )

    def attempt_count(self, operation_type: OperationType | None = None) -> int:
        return sum(
            item.total_attempts
            for item in self.operations
            if operation_type is None or item.operation_type == operation_type
        )


class BenchmarkOperationRetry:
    """Retry the exact same benchmark operation, never select another model."""

    def __init__(
        self,
        *,
        max_total_attempts: int = 3,
        base_backoff_seconds: float = 0.5,
        jitter_seconds: float = 0.1,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        self.max_total_attempts = max_total_attempts
        self._base_backoff_seconds = base_backoff_seconds
        self._jitter_seconds = jitter_seconds
        self._sleeper = sleeper
        self._random_source = random_source

    async def execute_provider_chat(
        self,
        provider: ModelProvider,
        model: ModelDefinition,
        request: ChatCompletionRequest,
        context: BenchmarkOperationContext,
        *,
        operation_type: OperationType = "inference",
    ) -> InternalChatCompletion:
        return await self.execute(
            operation=lambda: provider.chat_completion(model, request),
            operation_type=operation_type,
            provider=model.provider,
            model=model.id,
            context=context,
        )

    async def execute(
        self,
        *,
        operation: Callable[[], Awaitable[InternalChatCompletion]],
        operation_type: OperationType,
        provider: str,
        model: str,
        context: BenchmarkOperationContext,
    ) -> InternalChatCompletion:
        started = time.perf_counter()
        attempts = 0
        last_error: ProviderError | None = None
        while attempts < self.max_total_attempts:
            attempts += 1
            try:
                completion = await operation()
            except ProviderError as exc:
                last_error = exc
                if not self.is_retryable(exc) or attempts >= self.max_total_attempts:
                    telemetry = self._telemetry(
                        operation_type, provider, model, context, attempts, started, exc, failed=True,
                        max_total_attempts=self.max_total_attempts,
                    )
                    context.record(telemetry)
                    exc.benchmark_operation_telemetry = telemetry
                    raise
                await self._sleeper(self._backoff_seconds(attempts))
                continue

            context.record(self._telemetry(
                operation_type, provider, model, context, attempts, started, None, failed=False,
                max_total_attempts=self.max_total_attempts,
            ))
            return completion

        # Defensive: the loop always returns or raises, but keeps static analysis honest.
        assert last_error is not None
        raise last_error

    @staticmethod
    def is_retryable(error: ProviderError) -> bool:
        return isinstance(error, (ProviderTimeoutError, ProviderRateLimitError, ProviderUnavailableError))

    def _backoff_seconds(self, attempt: int) -> float:
        return self._base_backoff_seconds * (2 ** (attempt - 1)) + self._random_source() * self._jitter_seconds

    @staticmethod
    def _telemetry(
        operation_type: OperationType,
        provider: str,
        model: str,
        context: BenchmarkOperationContext,
        attempts: int,
        started: float,
        error: ProviderError | None,
        *,
        failed: bool,
        max_total_attempts: int,
    ) -> BenchmarkOperationTelemetry:
        return BenchmarkOperationTelemetry(
            operation_type=operation_type,
            strategy=context.strategy,
            corpus_item_id=context.corpus_item_id,
            provider=provider,
            model=model,
            total_attempts=attempts,
            benchmark_retry_count=attempts - 1,
            final_status="failed" if failed else "success",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            http_status=getattr(error, "upstream_status", None),
            exception_type=type(error).__name__ if error else None,
            failure_category=getattr(error, "failure_category", None) if error else None,
            succeeded_after_retry=not failed and attempts > 1,
            retries_exhausted=(
                failed and attempts >= max_total_attempts and error is not None
                and BenchmarkOperationRetry.is_retryable(error)
            ),
        )
