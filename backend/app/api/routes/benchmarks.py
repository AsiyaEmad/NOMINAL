from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.app.benchmark.runner import BenchmarkRunner
from backend.app.core.dependencies import get_benchmark_runner
from backend.app.models import BenchmarkRunRequest, BenchmarkRunResult

router = APIRouter()


@router.post("/api/benchmarks/run", response_model=BenchmarkRunResult)
async def run_benchmark(
    request: BenchmarkRunRequest,
    runner: Annotated[BenchmarkRunner, Depends(get_benchmark_runner)],
) -> BenchmarkRunResult:
    return await runner.run(max_items=request.max_items)


@router.get("/api/benchmarks", response_model=list[BenchmarkRunResult])
async def list_benchmarks(
    runner: Annotated[BenchmarkRunner, Depends(get_benchmark_runner)],
) -> list[BenchmarkRunResult]:
    return await runner.list_runs()


@router.get("/api/benchmarks/{run_id}", response_model=BenchmarkRunResult)
async def get_benchmark(
    run_id: str,
    runner: Annotated[BenchmarkRunner, Depends(get_benchmark_runner)],
) -> BenchmarkRunResult:
    result = await runner.get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return result
