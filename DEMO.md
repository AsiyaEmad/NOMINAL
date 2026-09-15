# NOMINAL Judge Demo (2–3 minutes)

## Before the judge arrives

1. Start the backend and dashboard using the commands in [README.md](README.md).
2. Configure the providers you intend to use and confirm GET /health returns ok.
3. Open the dashboard and keep a terminal ready for curl commands.
4. For the failure demonstrations, use the controlled mocked-provider tests below unless you have a safe local mock provider. Do not present mocked failures as live upstream incidents.

## 0:00–0:15 — Problem

Say:

> “Most AI products pay premium-model cost for every request because the safe alternative is unreliable. NOMINAL is the control plane between an application and its models: it picks the cheapest path predicted to meet that request’s quality and latency requirements.”

Point to the dashboard. Emphasize that NOMINAL is **not** an easy/hard classifier; it profiles reasoning, coding, general, context, quality, and latency needs separately.

## 0:15–0:35 — Simple request routes economically

Run:

~~~bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Nominal-Debug: true" \
  -d '{"model":"nominal-auto","messages":[{"role":"user","content":"Translate good morning into French."}],"max_tokens":100}'
~~~

Show the nominal.routing metadata and the new live trace. The default catalog/test profile makes this translation eligible for the economy tier. If an enabled local model qualifies in your configured catalog, NOMINAL may select it instead. The point is the auditable lowest-cost qualifying path, not a hard-coded tier.

## 0:35–0:55 — Difficult request routes to capability

Run:

~~~bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Nominal-Debug: true" \
  -d '{"model":"nominal-auto","messages":[{"role":"user","content":"Solve this dynamic programming algorithm and prove correctness."}],"max_tokens":500}'
~~~

Show that reasoning/coding requirements and the quality target eliminate weaker candidates. In the default routing tests, this profile selects balanced-gpt-4.1; a comprehensive analysis profile selects frontier-o3.

## 0:55–1:15 — Quality failure triggers escalation

Use the controlled, mocked-provider test to prove the behavior without pretending a real provider produced a bad answer:

~~~powershell
.\venv\Scripts\python.exe -m pytest backend/tests/test_quality_routing.py -k quality_failure_escalates -q
~~~

Say:

> “When the cheap candidate returns a deliberately short response, NOMINAL records the quality failure and retries the next higher-capability candidate. This is quality escalation, not provider failover.”

If you have a safe local mock provider wired for the demo, show its resulting trace in the dashboard instead.

## 1:15–1:30 — Provider failure triggers fallback

Run the distinct controlled test:

~~~powershell
.\venv\Scripts\python.exe -m pytest backend/tests/test_quality_routing.py -k timeout_uses_independent_infrastructure_fallback -q
~~~

Say:

> “A timeout or rate limit takes the infrastructure fallback path. NOMINAL records the provider reason and tries the next valid candidate without confusing it with answer-quality escalation.”

Mention that the suite also covers HTTP 429 fallback.

## 1:30–1:55 — Dashboard

Select the simple or difficult live request in the dashboard and walk top to bottom:

1. request and profile
2. candidate models and rejection reasons
3. highlighted Nominal Path
4. quality gate
5. escalation/fallback, when applicable
6. final response, cost, and latency telemetry

This is the proof that NOMINAL is a control plane rather than a black-box model switch.

## 1:55–2:25 — Benchmark comparison

If providers are configured, run the fixed corpus from the dashboard or:

~~~bash
curl -X POST http://127.0.0.1:8000/api/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{"max_items":40}'
~~~

Open the Benchmark section. Compare Always Frontier, Always Economy, and NOMINAL using only the displayed measured cost, quality, latency, failures, and escalation values. Do not quote a savings percentage unless the successful run produced it.

## 2:25–2:40 — Closing value proposition

Say:

> “NOMINAL makes inference spend a controllable engineering decision. It routes cheaply when the request allows it, protects quality with escalation, protects reliability with fallback, and leaves an auditable trace for every decision.”
