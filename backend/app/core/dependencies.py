from fastapi import Request

from backend.app.core.config import Settings
from backend.app.routing.catalog import ModelCatalog
from backend.app.routing.gateway import ChatCompletionGateway
from backend.app.telemetry.base import TelemetryRepository
from backend.app.benchmark.runner import BenchmarkRunner


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_model_catalog(request: Request) -> ModelCatalog:
    return request.app.state.model_catalog


def get_chat_gateway(request: Request) -> ChatCompletionGateway:
    return request.app.state.chat_gateway


def get_telemetry_repository(request: Request) -> TelemetryRepository:
    return request.app.state.telemetry


def get_benchmark_runner(request: Request) -> BenchmarkRunner:
    return request.app.state.benchmark_runner
