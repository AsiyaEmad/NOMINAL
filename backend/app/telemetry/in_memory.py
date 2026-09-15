from backend.app.models import RequestTrace
from backend.app.telemetry.base import TelemetryRepository


class InMemoryTelemetryRepository(TelemetryRepository):
    """MVP trace store; replace it with the SQLite implementation without changing callers."""

    def __init__(self) -> None:
        self._traces: dict[str, RequestTrace] = {}

    async def record(self, trace: RequestTrace) -> None:
        self._traces[trace.request_id] = trace

    async def get(self, request_id: str) -> RequestTrace | None:
        return self._traces.get(request_id)

    async def list(self, limit: int, offset: int) -> list[RequestTrace]:
        traces = sorted(self._traces.values(), key=lambda trace: trace.started_at, reverse=True)
        return traces[offset : offset + limit]

    async def summary(self) -> dict:
        return {"total_requests": len(self._traces)}
