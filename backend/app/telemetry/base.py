from abc import ABC, abstractmethod

from backend.app.models import RequestTrace


class TelemetryRepository(ABC):
    """Persistence boundary; a SQLite implementation can be added without API changes."""

    @abstractmethod
    async def record(self, trace: RequestTrace) -> None: ...

    @abstractmethod
    async def get(self, request_id: str) -> RequestTrace | None: ...

    @abstractmethod
    async def list(self, limit: int, offset: int) -> list[RequestTrace]: ...

    @abstractmethod
    async def summary(self) -> dict: ...
