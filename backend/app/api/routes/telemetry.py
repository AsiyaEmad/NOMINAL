from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.core.dependencies import get_telemetry_repository
from backend.app.models import RequestTrace
from backend.app.telemetry.base import TelemetryRepository

router = APIRouter()


@router.get("/api/metrics/summary")
async def metrics_summary(
    repository: Annotated[TelemetryRepository, Depends(get_telemetry_repository)],
) -> dict[str, Any]:
    """Aggregate persisted requests; baseline cost fields are explicitly estimated."""
    return await repository.summary()


@router.get("/api/traces", response_model=list[RequestTrace])
async def list_traces(
    repository: Annotated[TelemetryRepository, Depends(get_telemetry_repository)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[RequestTrace]:
    return await repository.list(limit=limit, offset=offset)


@router.get("/api/traces/{request_id}", response_model=RequestTrace)
async def get_trace(
    request_id: str,
    repository: Annotated[TelemetryRepository, Depends(get_telemetry_repository)],
) -> RequestTrace:
    trace = await repository.get(request_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Request trace not found")
    return trace
