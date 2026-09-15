from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app.core.config import Settings
from backend.app.core.dependencies import get_app_settings, get_model_catalog
from backend.app.routing.catalog import ModelCatalog

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    enabled_models: int


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Annotated[Settings, Depends(get_app_settings)],
    catalog: Annotated[ModelCatalog, Depends(get_model_catalog)],
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_env,
        enabled_models=len(catalog.enabled_models),
    )
