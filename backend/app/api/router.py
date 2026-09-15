from fastapi import APIRouter

from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.chat_completions import router as chat_completions_router
from backend.app.api.routes.telemetry import router as telemetry_router
from backend.app.api.routes.benchmarks import router as benchmarks_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["operations"])
api_router.include_router(chat_completions_router, tags=["chat completions"])
api_router.include_router(telemetry_router, tags=["telemetry"])
api_router.include_router(benchmarks_router, tags=["benchmarks"])
