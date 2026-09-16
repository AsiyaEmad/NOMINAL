# NOMINAL

> **The optimal inference path for every AI request.**

NOMINAL is an OpenAI-compatible inference optimization control plane. It profiles every conversation across capability, quality, context, and latency, then selects the lowest-cost enabled model predicted to satisfy the request. It records the decision, evaluates the answer, and can escalate or fail over when needed.

**NOMINAL is not an easy/hard prompt classifier.** A coding task, a proof, and a long-context summary require different combinations of reasoning, coding, general capability, context, quality, and latency. NOMINAL routes on those dimensions.

## 1. Problem

Sending every request to a premium model is safe but expensive and slow. Sending every request to the cheapest model is cheaper but makes quality unreliable. AI applications need an explicit, auditable way to minimize inference cost while respecting quality and latency needs.

## 2. Solution

For a standard POST /v1/chat/completions request with model set to nominal-auto, NOMINAL:

1. Profiles the whole conversation with deterministic signals.
2. Scores every enabled model against its capability and context requirements.
3. Selects the lowest-cost qualifying candidate.
4. Runs a layered quality check on the response.
5. Escalates to a more capable candidate if quality misses the target.
6. Separately falls back for timeouts, rate limits, and provider outages.

## 3. Why Nominal is different

- **Multi-dimensional, not binary:** reasoning, coding, general capability, context, quality target, and latency priority all matter.
- **Quality-bounded:** a low-cost initial route can fail safely; escalation is a distinct quality decision.
- **Failure-aware:** provider fallback is distinct from quality escalation, so traces explain why the model changed.
- **Auditable:** every trace persists the profile, candidates, rejections, model path, usage, configured-price cost accounting, quality, and latency.
- **Provider-flexible:** OpenAI-compatible and Ollama adapters share one provider interface.

## 4. Architecture

~~~mermaid
flowchart LR
    Client[OpenAI-compatible client] --> API[FastAPI gateway]
    API --> Profiler[Deterministic Request Profiler]
    Profiler --> Router[Capability-aware Routing Engine]
    Router --> Gateway[Execution Gateway]
    Gateway --> Quality[Layered Quality Evaluator]
    Gateway --> Providers[Provider adapters]
    Providers --> OpenAI[OpenAI-compatible APIs]
    Providers --> Ollama[Ollama]
    Quality --> Gateway
    Gateway --> Telemetry[(SQLite telemetry and benchmark runs)]
    Telemetry --> Dashboard[React control-plane dashboard]
~~~

## 5. How routing works

For every enabled model, NOMINAL:

1. Rejects models that cannot meet hard capability, context-window, or high latency-priority constraints.
2. Predicts quality from the request profile and configured model scores.
3. Estimates cost from catalog pricing and estimated token counts.
4. Evaluates latency and a fitness score.
5. Chooses the cheapest candidate predicted to meet the quality target. If none qualifies, it explicitly selects the highest-quality enabled fallback.

The result is a human-readable RoutingDecision, not merely a model name.

~~~mermaid
flowchart TD
    A[OpenAI chat request] --> B[Profile conversation]
    B --> C[Compare all enabled models]
    C --> D{Meets hard requirements and quality target?}
    D -- No --> E[Record rejected model and reason]
    E --> C
    D -- Yes --> F[Choose lowest-cost acceptable candidate]
    F --> G[Call provider]
    G --> H{Provider healthy?}
    H -- Timeout / 429 / 5xx --> I[Infrastructure fallback]
    I --> G
    H -- Yes --> J[Layered quality gate]
    J --> K{Score meets target?}
    K -- No --> L[Quality escalation]
    L --> G
    K -- Yes --> M[Return normalized OpenAI response]
    M --> N[Persist trace, costs, and latency]
~~~

## 6. Capability-aware routing

The profiler produces a RequestProfile containing:

- intent: simple_qa, translation, extraction, classification, summarization, coding, reasoning, analysis, creative, or unknown
- reasoning, coding, general, and context requirements from 0 to 1
- estimated complexity and token counts
- quality target and latency priority from 0 to 1

The MVP uses explainable keyword, conversation-length, structured-output, and urgency signals. It makes no routing-time LLM call by default, keeping control-plane overhead low and predictable.

## 7. Quality-bounded escalation

The layered evaluator checks for empty output, suspiciously short answers, provider-error-shaped content, missing code, and invalid requested JSON. When the score is below the request quality target, NOMINAL selects the next higher-capability candidate and retries.

NOMINAL_MAX_QUALITY_ESCALATIONS defaults to 1, preventing unbounded loops. An optional judge callback is available for benchmark/demo use but is disabled by default, so production traffic does not automatically incur judge-model cost.

## 8. Failure-aware fallback

Timeouts, HTTP 429, HTTP 5xx, and unavailable providers trigger infrastructure fallback. This is intentionally independent of quality escalation:

- **Escalation:** the answer did not meet the requested quality.
- **Fallback:** the selected provider could not reliably serve the request.

NOMINAL_MAX_PROVIDER_FAILOVERS defaults to 1. The fallback count and reason are stored in telemetry.

## 9. Cost and telemetry engine

SQLite stores a RequestTrace for each attempted request: profile JSON, candidates and rejections, initial/final model, router/provider/total latency, returned token use, quality, escalation/fallback, and status.

Cost is calculated from configured model pricing and provider-returned token counts. The baseline is clearly an **estimated** cost of processing the same token usage on the configured frontier model; it is not claimed as incurred spend.

Useful endpoints:

- GET /api/metrics/summary
- GET /api/traces?limit=50&offset=0
- GET /api/traces/{request_id}

## 10. Benchmark methodology

benchmarks/dataset.json contains 40 safe, fixed prompts: five each across simple Q&A, translation, extraction, classification, summarization, coding, reasoning, and analysis.

Each corpus item is run through three strategies using real configured-provider calls:

| Strategy | Behavior |
| --- | --- |
| always_frontier | Direct call to the configured frontier model. |
| always_economy | Direct call to the configured economy model. |
| nominal | Full NOMINAL routing, quality, escalation, and fallback pipeline. |

Runs persist configured-price inference cost, returned inference tokens, average/P50/P95 inference latency, benchmark quality score/pass rate, frontier calls, escalations, failures, and separate judge cost/latency. The production quality gate is not the benchmark evaluator.

### Canonical publication benchmark

The audited publication artifact is canonical run `ea187efb-9229-4ddd-949f-107c9bc7fc61`, recorded on **2026-09-16 05:16:20 UTC**. It evaluated the same frozen 40-item corpus for all three strategies using real provider calls:

- **Always Frontier**: direct configured frontier model.
- **Always Economy**: direct configured economy model.
- **NOMINAL**: capability-aware routing with the production quality gate, bounded escalation, and production fallback.

The benchmark used a strategy-blind configured judge: the judge received the prompt, category, evaluation criteria, and response, but not strategy, model, cost, or latency. A symmetric transient retry policy allowed at most **3 total attempts per operation**. Judge cost and judge latency are deliberately excluded from inference-cost and inference-latency comparisons.

| Metric | Always Frontier | Always Economy | NOMINAL |
| --- | ---: | ---: | ---: |
| Logical requests | 40 | 40 | 40 |
| Configured-price inference cost | $0.0715560 | $0.0095096 | $0.0564296 |
| Inference cost / request | $0.0018 | $0.0002 | $0.0014 |
| Average benchmark quality | 0.900 | 0.961 | 0.907 |
| Benchmark quality pass rate | 95.0% | 100.0% | 92.5% |
| Average inference latency | 2,260.72 ms | 2,124.05 ms | 2,237.10 ms |
| P95 inference latency | 4,444 ms | 5,424 ms | 5,122 ms |
| Frontier provider calls | 40 | 0 | 15 |
| Quality escalations | 0 | 0 | 1 |
| Production fallbacks | 0 | 0 | 0 |
| Provider/evaluation failures | 0 / 0 | 0 / 0 | 0 / 0 |

On this frozen corpus, NOMINAL used **21.14% lower configured-price inference cost** than Always Frontier ($0.0564296 versus $0.0715560). Its average benchmark quality score was **+0.007** higher (0.907 versus 0.900), while its quality pass rate was **2.5 percentage points lower** (92.5% versus 95.0%).

NOMINAL's average inference latency was **23.62 ms lower** than Frontier (1.04% faster), but its P95 inference latency was **678 ms higher** (15.26% slower). It completed **25 of 40 logical requests without using Frontier** (62.5% avoidance), with one quality escalation (2.5%) and no production fallbacks. These are controlled benchmark observations, not a claim about every workload or provider bill.

## 11. Dashboard

The React dashboard is a dark control-plane view built for a fast judge read:

- overview KPI cards for requests, estimated baseline, routed cost, latency, quality, and frontier calls avoided
- routing distribution, baseline-vs-NOMINAL cost, latency, and cumulative-savings charts
- auto-refreshing live request table
- request-level trace with candidates, rejections, quality gate, escalation, and final path
- a separately labeled **canonical publication benchmark** table and charts for Frontier, Economy, and NOMINAL, pinned to audited run `ea187efb-9229-4ddd-949f-107c9bc7fc61`

The top overview and live-request panels intentionally use accumulated operational telemetry. They may show different values from the controlled benchmark section.

When the backend is unavailable, the dashboard shows a clear reconnect message rather than an unhandled browser error.

## 12. Tech stack

| Layer | Technology |
| --- | --- |
| API and gateway | Python 3.12+, FastAPI, Uvicorn, Pydantic v2 |
| Configuration and providers | pydantic-settings, httpx, OpenAI-compatible adapter, Ollama adapter |
| Routing and quality | deterministic profiler, pluggable interfaces |
| Persistence | SQLAlchemy and SQLite |
| Dashboard | React, Vite, TypeScript, Tailwind CSS, Recharts |
| Validation | pytest, pytest-asyncio, ESLint, TypeScript build |
| Packaging | Docker, Docker Compose, Nginx |

## 13. Quick start

Prerequisites: Python 3.12+, Node.js 20+, and optionally Docker Desktop.

~~~powershell
Copy-Item .env.example .env
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
~~~

In another terminal, either run the dashboard commands from its directory:

~~~powershell
cd frontend
npm ci
npm run dev
~~~

Or run `npm run dev` from the repository root; root scripts forward to the frontend workspace. The same applies to `npm run lint`, `npm run typecheck`, and `npm run build`.

Open http://localhost:5173. Check backend readiness at http://127.0.0.1:8000/health.

For Docker:

~~~powershell
Copy-Item .env.example .env
docker compose up --build
~~~

Open http://localhost:8080; the backend is exposed at http://localhost:8000.

## 14. Environment variables

Copy .env.example to .env. It contains placeholders only; never commit real provider credentials.

| Variable | Purpose |
| --- | --- |
| NOMINAL_PROVIDER_CONFIGS | JSON map of named OpenAI-compatible/Ollama providers, URLs, optional key, and timeout. |
| NOMINAL_MODEL_CATALOG_PATH | Model capability/pricing YAML path. |
| NOMINAL_DEFAULT_MODEL | Direct default model when applicable. |
| NOMINAL_BASELINE_MODEL | Frontier-equivalent model for estimated baseline cost. |
| NOMINAL_DATABASE_URL | SQLite URL; default sqlite:///./nominal.db. |
| NOMINAL_CORS_ORIGINS | Comma-separated allowed browser origins. |
| NOMINAL_MAX_QUALITY_ESCALATIONS | Quality retry ceiling; default 1. |
| NOMINAL_MAX_PROVIDER_FAILOVERS | Infrastructure fallback ceiling; default 1. |
| NOMINAL_BENCHMARK_FRONTIER_MODEL / NOMINAL_BENCHMARK_ECONOMY_MODEL | Fixed comparator models. |
| NOMINAL_BENCHMARK_DATASET_PATH | Fixed corpus path. |
| NOMINAL_BENCHMARK_JUDGE_ENABLED / NOMINAL_BENCHMARK_JUDGE_MODEL | Enables the benchmark-only, strategy-blind judge; it does not enable judge calls in the production routing quality gate. |

## 15. API example

~~~bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Nominal-Debug: true" \
  -d '{
    "model": "nominal-auto",
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "Translate good morning into French."}
    ],
    "temperature": 0.2,
    "max_tokens": 100
  }'
~~~

The normal response remains OpenAI-compatible. X-Nominal-Debug: true adds a nominal object with the request ID and full routing decision for demo/debug use.

### Example model configuration

backend/config/models.yaml is the model registry. A representative entry:

~~~yaml
- id: economy-gpt-4.1-mini
  tier: economy
  provider: openai
  model_name: gpt-4.1-mini
  input_cost_per_million_tokens: 0.40
  output_cost_per_million_tokens: 1.60
  expected_latency_ms: 750
  reasoning_score: 0.68
  coding_score: 0.72
  general_score: 0.78
  context_window: 1047576
  enabled: true
~~~

## 16. Benchmark instructions

Configure providers first, then run the full corpus:

~~~bash
curl -X POST http://127.0.0.1:8000/api/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{"max_items": 40}'
~~~

Inspect summaries at GET /api/benchmarks, then fetch a complete run at GET /api/benchmarks/{id}. The dashboard publication comparison is intentionally pinned to canonical run `ea187efb-9229-4ddd-949f-107c9bc7fc61`, rather than whichever development run is newest. Treat a run with failures as a reliability result, not a cost-savings result.


## 17. Known MVP limitations

- SQLite is appropriate for one demo instance, not multi-replica production deployment.
- The default quality gate is deterministic; an expensive judge is intentionally not forced on every request.
- No authentication, per-user rate limiting, streaming, or background benchmark queue is implemented.
- Benchmark execution is synchronous; use a bounded corpus/provider budget for live demos.
- Capability scores and pricing are registry inputs and should be calibrated against real workloads before production use.

## 18. Future roadmap

1. Calibrate capability and quality predictions using labeled workload outcomes.
2. Add asynchronous benchmark execution and richer evaluation datasets.
3. Support streaming while preserving routing/telemetry semantics.
4. Add tenant-aware authentication, quotas, and cost allocation.
5. Move beyond SQLite when multi-instance operation is required.
6. Add policy-driven constraints for region, provider, tools, and data handling.

For concise submission copy, see [SUBMISSION.md](SUBMISSION.md).
