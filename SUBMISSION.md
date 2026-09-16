# NOMINAL — Hackathon Submission

## Project name

NOMINAL

## One-line tagline

**The optimal inference path for every AI request.**

## Problem

AI applications usually choose between two weak defaults: send every request to a premium model and accept unnecessary cost/latency, or send every request to a cheaper model and accept unpredictable quality. The challenge is not simply whether a prompt is easy or hard. Different tasks require different combinations of reasoning, coding, general capability, context, quality, and latency.

## Solution

NOMINAL is an OpenAI-compatible inference routing control plane. It profiles an incoming conversation, compares it with each enabled model’s configured capability and pricing profile, and selects the lowest-cost model predicted to satisfy the request. It evaluates the answer, escalates when quality misses the target, and falls back when provider infrastructure fails.

## Key innovation

NOMINAL separates two decisions that are often conflated:

- **Quality escalation:** move to a higher-capability model when the answer misses the requested quality target.
- **Infrastructure fallback:** move to a valid alternate candidate when a provider times out, rate-limits, or becomes unavailable.

That separation makes routing behavior explainable, bounded, and auditable.

## Infrastructure impact

NOMINAL makes model selection an observable control-plane concern. It persists the request profile, candidates, rejected models, routing reason, model path, token use, configured-price cost accounting, latency, quality score, escalation, fallback, and status. Teams can inspect why a request was routed, not merely which model answered it.

No unverified cost-savings percentage is claimed. Benchmark outputs are based on real configured-provider calls and returned token usage; a run with provider failures is treated as a reliability result, not a value claim.

## Features

- OpenAI-compatible non-streaming chat-completions gateway
- Deterministic, multi-dimensional request profiler
- Pluggable capability-aware cost routing engine
- OpenAI-compatible and Ollama provider adapters
- Layered deterministic quality checks with optional judge callback
- Bounded quality escalation and independent timeout/429/5xx fallback
- Persistent SQLite telemetry and estimated frontier-baseline cost accounting
- Reproducible 40-prompt benchmark corpus with Frontier, Economy, and NOMINAL strategies
- React control-plane dashboard with trace inspection and benchmark comparison
- Docker Compose startup, CORS, health checks, validated configuration, and safe error responses

## Tech stack

Python, FastAPI, Uvicorn, Pydantic v2, pydantic-settings, httpx, SQLAlchemy, SQLite, PyYAML, pytest, React, Vite, TypeScript, Tailwind CSS, Recharts, Docker, and Nginx.

## Challenges

- Balancing a credible control-plane design with a hackathon-sized implementation.
- Making routing decisions explainable without calling another LLM on every request.
- Keeping cost, quality, escalation, and provider-failure measurements distinct.
- Supporting OpenAI-compatible APIs and local Ollama through a common provider abstraction.

## Accomplishments

- Built an end-to-end routing gateway with swappable provider, profiler, router, evaluator, and telemetry interfaces.
- Added persisted request traces and benchmark runs rather than relying on demo-only console output.
- Added unit coverage for routing profiles, quality escalation, timeout fallback, rate-limit fallback, provider exhaustion, cost accounting, telemetry aggregation, benchmark aggregation, and API validation.
- Hardened the project for judging with Docker assets, README-driven startup, health checks, CORS, configuration validation, and safe error behavior.

## What I learned

Inference optimization needs more than a cheap-model-first rule. A practical system needs distinct capability signals, quality boundaries, reliable fallback behavior, and visibility into the consequence of every routing decision. It also needs to distinguish measured results from estimates.

## Future roadmap

- Calibrate capability and quality predictions using labeled production workloads.
- Add richer benchmark evaluation and optional judge-model scoring.
- Support streaming and policy constraints without losing traceability.
- Add tenant-aware authentication, quotas, and cost allocation.
- Move beyond SQLite when multi-instance operation is needed.
