from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.router import api_router
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import configure_logging
from backend.app.providers.registry import ProviderRegistry
from backend.app.routing.catalog import ModelCatalog
from backend.app.routing.cost_aware import CostAwareRoutingEngine
from backend.app.routing.gateway import ChatCompletionGateway
from backend.app.routing.profiler import HeuristicRequestProfiler
from backend.app.quality.layered import LayeredQualityEvaluator
from backend.app.telemetry.costs import CostCalculator
from backend.app.telemetry.database import SqliteTelemetryRepository
from backend.app.telemetry.base import TelemetryRepository
from backend.app.telemetry.database import BenchmarkRepository
from backend.app.benchmark.runner import BenchmarkRunner
from backend.app.benchmark.quality import JudgeModelBenchmarkQualityEvaluator, UnavailableBenchmarkQualityEvaluator
from backend.app.benchmark.retry import BenchmarkOperationRetry

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    provider_registry: ProviderRegistry | None = None,
    telemetry_repository: TelemetryRepository | None = None,
) -> FastAPI:
    """Create the application with explicit, testable wiring."""
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        catalog = ModelCatalog.from_yaml(app_settings.model_catalog_path)
        providers = provider_registry or ProviderRegistry.from_settings(app_settings)
        telemetry = telemetry_repository or SqliteTelemetryRepository(app_settings.database_url)
        if isinstance(telemetry, SqliteTelemetryRepository):
            telemetry.create_schema()
        benchmark_repository = BenchmarkRepository(app_settings.database_url)
        benchmark_repository.create_schema()
        baseline_model = catalog.get(app_settings.baseline_model)
        if baseline_model is None:
            baseline_model = next(
                (model for model in catalog.enabled_models if model.tier.value == "frontier"),
                catalog.enabled_models[-1],
            )
        application.state.settings = app_settings
        application.state.model_catalog = catalog
        application.state.telemetry = telemetry
        quality_evaluator = LayeredQualityEvaluator(
            judge_enabled=app_settings.quality_judge_enabled,
        )
        cost_calculator = CostCalculator(baseline_model)
        application.state.chat_gateway = ChatCompletionGateway(
            catalog=catalog,
            providers=providers,
            telemetry=telemetry,
            settings=app_settings,
            profiler=HeuristicRequestProfiler(),
            routing_engine=CostAwareRoutingEngine(),
            quality_evaluator=quality_evaluator,
            cost_calculator=cost_calculator,
        )
        # Benchmarks use a distinct gateway so neutral same-model retries never
        # change the production API's retry/fallback semantics.
        benchmark_operation_retry = BenchmarkOperationRetry()
        benchmark_gateway = ChatCompletionGateway(
            catalog=catalog,
            providers=providers,
            telemetry=telemetry,
            settings=app_settings,
            profiler=HeuristicRequestProfiler(),
            routing_engine=CostAwareRoutingEngine(),
            quality_evaluator=quality_evaluator,
            cost_calculator=cost_calculator,
            provider_operation_executor=benchmark_operation_retry.execute_provider_chat,
        )
        judge_model = catalog.get(app_settings.benchmark_judge_model)
        benchmark_quality_evaluator = (
            JudgeModelBenchmarkQualityEvaluator(
                providers=providers,
                judge_model=judge_model,
                max_tokens=app_settings.benchmark_judge_max_tokens,
                operation_retry=benchmark_operation_retry,
            )
            if app_settings.benchmark_judge_enabled and judge_model is not None
            else UnavailableBenchmarkQualityEvaluator()
        )
        application.state.benchmark_runner = BenchmarkRunner(
            settings=app_settings,
            catalog=catalog,
            providers=providers,
            profiler=HeuristicRequestProfiler(),
            quality_evaluator=quality_evaluator,
            benchmark_quality_evaluator=benchmark_quality_evaluator,
            gateway=benchmark_gateway,
            telemetry=telemetry,
            repository=benchmark_repository,
            cost_calculator=cost_calculator,
            operation_retry=benchmark_operation_retry,
        )
        logger.info(
            "application_started",
            extra={"event": "application_started", "model_count": len(catalog.models)},
        )
        yield
        if isinstance(telemetry, SqliteTelemetryRepository):
            telemetry.dispose()
        benchmark_repository.dispose()
        logger.info("application_stopped", extra={"event": "application_stopped"})

    application = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Nominal-Debug"],
        expose_headers=["X-Request-Id"],
    )

    @application.exception_handler(RequestValidationError)
    async def request_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": "Request validation failed",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "validation_error",
                }
            },
        )

    @application.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_request_error",
            extra={"event": "unhandled_request_error", "request_id": request.headers.get("x-request-id")},
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "server_error",
                    "param": None,
                    "code": "internal_error",
                }
            },
        )
    application.include_router(api_router)
    return application


app = create_app()
