import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { nominalApi } from './api'
import { CANONICAL_BENCHMARK_RUN_ID } from './canonicalBenchmark'
import type { BenchmarkRun, BenchmarkStrategyMetrics, MetricsSummary, RequestTrace } from './types'
import { formatCurrency, formatMs, formatNumber, shortId, tierFromModel, titleCase } from './utils'

const TIER_COLORS: Record<string, string> = {
  local: '#7585a0',
  economy: '#9cf7c8',
  balanced: '#a99bff',
  frontier: '#ffca76',
}
const tooltipStyle = { background: '#101622', border: '1px solid #293247', borderRadius: 8, color: '#e9edf5', fontSize: 12 }

type DashboardState = { summary: MetricsSummary; traces: RequestTrace[] }

const emptySummary: MetricsSummary = {
  total_requests: 0, successful_requests: 0, total_tokens: 0, actual_cost: 0,
  estimated_baseline_cost: 0, estimated_cost_saved: 0, savings_percentage: 0,
  average_latency_ms: 0, p95_latency_ms: 0, average_router_overhead_ms: 0,
  escalation_rate: 0, fallback_rate: 0, frontier_calls_avoided: 0,
  routing_distribution: {}, average_quality_score: 0,
}
const emptyTraces: RequestTrace[] = []

function KpiCard({ label, value, detail, accent = 'text-slate-100' }: { label: string; value: string; detail: string; accent?: string }) {
  return (
    <div className="panel min-w-0 p-4">
      <p className="eyebrow">{label}</p>
      <p className={`metric-value ${accent}`}>{value}</p>
      <p className="mt-1 truncate text-xs text-slate-500">{detail}</p>
    </div>
  )
}

function ChartPanel({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <section className="panel p-4">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div><h2 className="text-sm font-medium text-slate-200">{title}</h2><p className="mt-1 text-xs text-slate-500">{subtitle}</p></div>
      </div>
      <div className="h-56">{children}</div>
    </section>
  )
}

function EmptyState() {
  return <div className="panel flex min-h-64 items-center justify-center p-8 text-center"><div><p className="text-lg text-slate-200">No inference traces yet</p><p className="mt-2 max-w-sm text-sm text-slate-500">Send a chat completion through NOMINAL and the control plane will populate with routing, quality, and savings telemetry.</p></div></div>
}

function strategyName(strategy: string) {
  return strategy.replace(/_/g, ' ').replace(/\b\w/g, (letter: string) => letter.toUpperCase())
}

function BenchmarkSection({ run, canonical = false, running, onRun }: { run: BenchmarkRun | undefined; canonical?: boolean; running: boolean; onRun: () => void }) {
  if (!run) {
    return (
      <section className="panel mt-6 p-5">
        <p className="eyebrow">Benchmark / canonical publication run</p><h2 className="mt-1 text-lg font-semibold">Controlled 40-item comparison unavailable</h2><p className="mt-1 max-w-2xl text-sm text-slate-500">The audited benchmark artifact could not be loaded. Live operational telemetry above remains available and is intentionally separate.</p>
      </section>
    )
  }

  const frontier = run.strategies.find((item) => item.strategy === 'always_frontier')
  const nominal = run.strategies.find((item) => item.strategy === 'nominal')
  const savings = frontier && nominal && frontier.total_cost > 0 ? (1 - nominal.total_cost / frontier.total_cost) * 100 : null
  const comparison = run.strategies.map((item) => ({
    name: strategyName(item.strategy), cost: item.total_cost, latency: item.average_latency_ms, quality: item.quality_score * 100,
  }))
  const qualityLabel = run.evaluator_type === 'benchmark_judge' ? 'Strategy-blind benchmark judge' : run.evaluator_type

  if (canonical) {
    return (
      <section className="panel mt-6 overflow-hidden">
        <div className="flex flex-col gap-4 border-b border-line px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="eyebrow">Benchmark / canonical publication run</p>
            <h2 className="mt-1 text-lg font-semibold">Controlled 40-item comparison</h2>
            <p className="mt-1 text-xs text-slate-500">Run {shortId(CANONICAL_BENCHMARK_RUN_ID)} - {qualityLabel} - {new Date(run.created_at).toLocaleString()}</p>
          </div>
          <div className="rounded border border-signal/30 bg-signal/[.06] px-3 py-2 text-right">
            <p className="eyebrow">NOMINAL cost eliminated</p>
            <p className="text-lg font-semibold text-signal">{savings === null ? 'N/A' : `${savings.toFixed(1)}%`}</p>
            <p className="mt-0.5 text-[10px] text-slate-500">vs Always Frontier; inference cost only</p>
          </div>
        </div>
        <div className="grid gap-4 p-5 xl:grid-cols-[1.25fr_.75fr]">
          <div className="overflow-x-auto"><table className="min-w-[760px] w-full text-left text-xs"><thead className="text-[10px] uppercase tracking-[.12em] text-slate-500"><tr>{['Strategy', 'Cost', 'Cost / request', 'Quality', 'Pass rate', 'Avg / P95 latency', 'Frontier', 'Escalations', 'Failures'].map((heading) => <th key={heading} className="border-b border-line px-2 py-2.5 font-medium">{heading}</th>)}</tr></thead><tbody>{run.strategies.map((item: BenchmarkStrategyMetrics) => <tr key={item.strategy} className={item.strategy === 'nominal' ? 'bg-signal/[.05] text-slate-100' : 'text-slate-400'}><td className="border-b border-line/70 px-2 py-3 font-medium">{strategyName(item.strategy)}{item.strategy === 'nominal' && <span className="ml-2 text-[10px] uppercase text-signal">NOMINAL path</span>}</td><td className="border-b border-line/70 px-2 py-3">{formatCurrency(item.total_cost)}</td><td className="border-b border-line/70 px-2 py-3">{formatCurrency(item.estimated_cost_per_request)}</td><td className="border-b border-line/70 px-2 py-3">{Math.round(item.quality_score * 100)}%</td><td className="border-b border-line/70 px-2 py-3">{item.quality_pass_rate.toFixed(1)}%</td><td className="border-b border-line/70 px-2 py-3">{formatMs(item.average_latency_ms)} / {formatMs(item.p95_latency_ms)}</td><td className="border-b border-line/70 px-2 py-3">{item.frontier_calls}</td><td className="border-b border-line/70 px-2 py-3">{item.escalations}</td><td className="border-b border-line/70 px-2 py-3">{item.failures}</td></tr>)}</tbody></table></div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 xl:grid-cols-1"><ChartPanel title="Configured-price inference cost" subtitle="Judge cost is tracked separately"><ResponsiveContainer><BarChart data={comparison}><XAxis dataKey="name" tick={{ fill: '#8490a5', fontSize: 10 }} /><YAxis tick={{ fill: '#8490a5', fontSize: 10 }} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => formatCurrency(Number(value))} /><Bar dataKey="cost" fill="#9cf7c8" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></ChartPanel><ChartPanel title="Benchmark quality score" subtitle={qualityLabel}><ResponsiveContainer><BarChart data={comparison}><XAxis dataKey="name" tick={{ fill: '#8490a5', fontSize: 10 }} /><YAxis domain={[0, 100]} tick={{ fill: '#8490a5', fontSize: 10 }} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => `${Number(value).toFixed(1)}%`} /><Bar dataKey="quality" fill="#ffca76" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></ChartPanel><ChartPanel title="Average inference latency" subtitle="Benchmark judge time excluded"><ResponsiveContainer><BarChart data={comparison}><XAxis dataKey="name" tick={{ fill: '#8490a5', fontSize: 10 }} /><YAxis tick={{ fill: '#8490a5', fontSize: 10 }} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => formatMs(Number(value))} /><Bar dataKey="latency" fill="#a99bff" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></ChartPanel></div>
        </div>
      </section>
    )
  }

  return (
    <section className="panel mt-6 overflow-hidden">
      <div className="flex flex-col gap-4 border-b border-line px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
        <div><p className="eyebrow">Benchmark / saved run</p><h2 className="mt-1 text-lg font-semibold">Measured routing comparison</h2><p className="mt-1 text-xs text-slate-500">{run.dataset_name} · {qualityLabel} · {new Date(run.created_at).toLocaleString()}</p></div>
        <div className="flex items-center gap-3"><div className="rounded border border-signal/30 bg-signal/[.06] px-3 py-2 text-right"><p className="eyebrow">Nominal cost eliminated</p><p className="text-lg font-semibold text-signal">{savings === null ? '—' : `${savings.toFixed(1)}%`}</p></div><button onClick={onRun} disabled={running} className="rounded border border-slate-600 px-3 py-2 text-sm text-slate-200 hover:border-slate-400 disabled:opacity-50">{running ? 'Running...' : 'Run again'}</button></div>
      </div>
      <div className="grid gap-4 p-5 xl:grid-cols-[1.25fr_.75fr]">
        <div className="overflow-x-auto"><table className="min-w-[760px] w-full text-left text-xs"><thead className="text-[10px] uppercase tracking-[.12em] text-slate-500"><tr>{['Strategy', 'Cost', 'Cost / request', 'Quality', 'Pass rate', 'Avg / P95 latency', 'Frontier', 'Escalations', 'Failures'].map((heading) => <th key={heading} className="border-b border-line px-2 py-2.5 font-medium">{heading}</th>)}</tr></thead><tbody>{run.strategies.map((item: BenchmarkStrategyMetrics) => <tr key={item.strategy} className={item.strategy === 'nominal' ? 'bg-signal/[.05] text-slate-100' : 'text-slate-400'}><td className="border-b border-line/70 px-2 py-3 font-medium">{strategyName(item.strategy)}{item.strategy === 'nominal' && <span className="ml-2 text-[10px] uppercase text-signal">Nominal path</span>}</td><td className="border-b border-line/70 px-2 py-3">{formatCurrency(item.total_cost)}</td><td className="border-b border-line/70 px-2 py-3">{formatCurrency(item.estimated_cost_per_request)}</td><td className="border-b border-line/70 px-2 py-3">{Math.round(item.quality_score * 100)}%</td><td className="border-b border-line/70 px-2 py-3">{item.quality_pass_rate.toFixed(1)}%</td><td className="border-b border-line/70 px-2 py-3">{formatMs(item.average_latency_ms)} / {formatMs(item.p95_latency_ms)}</td><td className="border-b border-line/70 px-2 py-3">{item.frontier_calls}</td><td className="border-b border-line/70 px-2 py-3">{item.escalations}</td><td className="border-b border-line/70 px-2 py-3">{item.failures}</td></tr>)}</tbody></table></div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 xl:grid-cols-1"><ChartPanel title="Measured cost" subtitle="Actual provider cost"><ResponsiveContainer><BarChart data={comparison}><XAxis dataKey="name" tick={{ fill: '#8490a5', fontSize: 10 }} /><YAxis tick={{ fill: '#8490a5', fontSize: 10 }} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => formatCurrency(Number(value))} /><Bar dataKey="cost" fill="#9cf7c8" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></ChartPanel><ChartPanel title="Quality score" subtitle={qualityLabel}><ResponsiveContainer><BarChart data={comparison}><XAxis dataKey="name" tick={{ fill: '#8490a5', fontSize: 10 }} /><YAxis domain={[0, 100]} tick={{ fill: '#8490a5', fontSize: 10 }} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => `${Number(value).toFixed(1)}%`} /><Bar dataKey="quality" fill="#ffca76" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></ChartPanel><ChartPanel title="Average latency" subtitle="End-to-end provider measurement"><ResponsiveContainer><BarChart data={comparison}><XAxis dataKey="name" tick={{ fill: '#8490a5', fontSize: 10 }} /><YAxis tick={{ fill: '#8490a5', fontSize: 10 }} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => formatMs(Number(value))} /><Bar dataKey="latency" fill="#a99bff" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></ChartPanel></div>
      </div>
    </section>
  )
}

function TraceFlow({ trace }: { trace: RequestTrace }) {
  const decision = trace.routing_decision
  if (!decision) return <EmptyState />
  const selectedId = trace.initial_model ?? decision.selected_model.id
  const finalId = trace.final_model ?? trace.model_used ?? selectedId
  const stage = (label: string, content: React.ReactNode, highlighted = false) => (
    <div className={`relative rounded-lg border p-3 ${highlighted ? 'border-signal/60 bg-signal/[.06]' : 'border-line bg-black/10'}`}>
      <p className="eyebrow">{label}</p><div className="mt-1 text-sm text-slate-200">{content}</div>
    </div>
  )
  return (
    <div className="space-y-2">
      {stage('Request', <><span className="font-mono text-xs text-slate-300">{trace.request_id}</span><span className="ml-2 tag">{titleCase(trace.profile.intent)}</span></>)}
      <div className="ml-5 h-3 border-l border-slate-700" />
      {stage('Profile', <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-400"><span>Quality target <b className="text-slate-200">{Math.round(trace.profile.quality_target * 100)}%</b></span><span>Complexity <b className="text-slate-200">{Math.round(trace.profile.estimated_complexity * 100)}%</b></span></div>)}
      <div className="ml-5 h-3 border-l border-slate-700" />
      {stage('Candidate models', <div className="mt-2 space-y-1.5">{decision.candidate_models.map((candidate) => <div key={candidate.model_id} className={`flex items-center justify-between rounded px-2 py-1 text-xs ${candidate.model_id === selectedId ? 'bg-signal/10 text-signal' : 'bg-slate-800/60 text-slate-400'}`}><span>{candidate.model_id}</span><span>Q {Math.round(candidate.predicted_quality * 100)}% · {formatCurrency(candidate.estimated_cost)}</span></div>)}</div>)}
      {decision.rejected_models.length > 0 && <div className="rounded-lg border border-dashed border-slate-700 p-3"><p className="eyebrow">Rejected models</p>{decision.rejected_models.map((model) => <p key={model.model_id} className="mt-1 text-xs text-slate-500"><span className="text-slate-300">{model.model_id}</span> — {model.reason}</p>)}</div>}
      <div className="ml-5 h-3 border-l border-signal/60" />
      {stage('Selected model · Nominal Path', <><span className="font-medium text-signal">{selectedId}</span><span className="ml-2 text-xs text-slate-500">{decision.routing_reason}</span></>, true)}
      <div className="ml-5 h-3 border-l border-signal/60" />
      {stage('Quality gate', <><span className={trace.quality_result?.passed ? 'text-signal' : 'text-amber'}>{trace.quality_result ? `${Math.round(trace.quality_result.score * 100)}% ${trace.quality_result.passed ? 'passed' : 'failed'}` : 'Not evaluated'}</span>{trace.quality_result?.reasons?.[0] && <span className="ml-2 text-xs text-slate-500">{trace.quality_result.reasons[0]}</span>}</>)}
      {trace.escalated && <><div className="ml-5 h-3 border-l border-amber/70" />{stage('Escalation', <span className="text-amber">{trace.escalation_reason ?? 'Quality-bound escalation applied'}</span>)}</>}
      <div className="ml-5 h-3 border-l border-signal/60" />
      {stage('Final response', <><span className="font-medium text-slate-100">{finalId}</span><span className="ml-2 tag">{trace.status}</span></>, true)}
    </div>
  )
}

export default function App() {
  const [state, setState] = useState<DashboardState | null>(null)
  const [canonicalBenchmark, setCanonicalBenchmark] = useState<BenchmarkRun | undefined>(undefined)
  const [selected, setSelected] = useState<RequestTrace | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)

  const load = useCallback(async () => {
    try {
      const [summary, traces] = await Promise.all([nominalApi.getSummary(), nominalApi.getTraces()])
      setState({ summary, traces })
      setSelected((current) => current ? traces.find((trace) => trace.request_id === current.request_id) ?? current : traces[0] ?? null)
      setUpdatedAt(new Date())
      setError(null)
      try {
        setCanonicalBenchmark(await nominalApi.getBenchmark(CANONICAL_BENCHMARK_RUN_ID))
      } catch {
        // The controlled publication artifact is intentionally independent
        // from live operational telemetry and may be unavailable by itself.
        setCanonicalBenchmark(undefined)
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to reach NOMINAL telemetry')
    } finally { setLoading(false) }
  }, [])

  useEffect(() => {
    void load()
    const timer = window.setInterval(() => void load(), 10_000)
    return () => window.clearInterval(timer)
  }, [load])

  const selectTrace = async (trace: RequestTrace) => {
    setSelected(trace)
    try { setSelected(await nominalApi.getTrace(trace.request_id)) } catch { /* list payload remains useful */ }
  }

  const summary = state?.summary ?? emptySummary
  const traces = state?.traces ?? emptyTraces
  const tierDistribution = useMemo(() => Object.entries(summary.routing_distribution).reduce<Record<string, number>>((tiers, [model, count]) => {
    const tier = tierFromModel(model); tiers[tier] = (tiers[tier] ?? 0) + count; return tiers
  }, {}), [summary.routing_distribution])
  const distributionData = Object.entries(tierDistribution).map(([name, value]) => ({ name, value }))
  const orderedTraces = useMemo(() => [...traces].sort((a, b) => new Date(a.started_at).getTime() - new Date(b.started_at).getTime()), [traces])
  const latencyData = orderedTraces.map((trace, index) => ({ name: `${index + 1}`, latency: trace.total_latency_ms ?? trace.provider_latency_ms ?? 0 }))
  let cumulative = 0
  const savingsData = orderedTraces.map((trace, index) => { cumulative += Number(trace.estimated_cost_saved ?? 0); return { name: `${index + 1}`, savings: cumulative } })
  return (
    <main className="min-h-screen bg-ink text-slate-100"><div className="grid-noise min-h-screen">
      <header className="border-b border-line bg-ink/80 px-5 py-4 backdrop-blur lg:px-8"><div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4">
        <div className="flex items-center gap-3"><div className="flex h-8 w-8 items-center justify-center rounded bg-signal text-xs font-black text-ink">N</div><div><h1 className="text-sm font-semibold tracking-wide">NOMINAL <span className="font-normal text-slate-500">/ CONTROL PLANE</span></h1><p className="text-[10px] uppercase tracking-[.16em] text-slate-500">Adaptive inference routing</p></div></div>
        <div className="flex items-center gap-3 text-xs text-slate-500"><span className="hidden sm:inline">{updatedAt ? `Updated ${updatedAt.toLocaleTimeString()}` : 'Connecting…'}</span><span className="flex items-center gap-2 rounded-full border border-signal/20 bg-signal/[.06] px-2.5 py-1 text-signal"><i className="status-dot" /> Live telemetry</span><button onClick={() => void load()} className="rounded border border-slate-700 px-2 py-1 text-slate-300 hover:border-slate-500">Refresh</button></div>
      </div></header>
      <div className="mx-auto max-w-[1600px] px-5 py-6 lg:px-8">
        {error && <div className="mb-5 flex items-center justify-between rounded-lg border border-red-400/30 bg-red-400/10 px-4 py-3 text-sm text-red-200"><span>{error}</span><button onClick={() => void load()} className="underline">Retry</button></div>}
        <section><div className="mb-4 flex items-end justify-between"><div><p className="eyebrow">Overview / live operational telemetry</p><h2 className="mt-1 text-xl font-semibold">Optimization at a glance</h2></div><p className="hidden text-xs text-slate-500 md:block">Baseline is an estimated premium-model equivalent, not incurred spend.</p></div>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
            <KpiCard label="Total requests" value={formatNumber(summary.total_requests)} detail={`${formatNumber(summary.total_tokens)} tokens`} />
            <KpiCard label="Est. baseline" value={formatCurrency(summary.estimated_baseline_cost)} detail="Premium-model equivalent" />
            <KpiCard label="Nominal cost" value={formatCurrency(summary.actual_cost)} detail="Actual routed spend" accent="text-signal" />
            <KpiCard label="Cost saved" value={`${summary.savings_percentage.toFixed(1)}%`} detail={formatCurrency(summary.estimated_cost_saved)} accent="text-signal" />
            <KpiCard label="Avg latency" value={formatMs(summary.average_latency_ms)} detail={`Router ${formatMs(summary.average_router_overhead_ms)}`} />
            <KpiCard label="P95 latency" value={formatMs(summary.p95_latency_ms)} detail="End-to-end request time" />
            <KpiCard label="Frontier avoided" value={formatNumber(summary.frontier_calls_avoided)} detail={`${summary.fallback_rate.toFixed(1)}% fallback rate`} accent="text-violet" />
            <KpiCard label="Quality score" value={`${Math.round(summary.average_quality_score * 100)}%`} detail={`${summary.escalation_rate.toFixed(1)}% escalated`} accent="text-amber" />
          </div>
        </section>
        <BenchmarkSection run={canonicalBenchmark} canonical running={false} onRun={() => undefined} />
        {loading && !state ? <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2"><div className="panel h-64 animate-pulse" /><div className="panel h-64 animate-pulse" /></div> : traces.length === 0 && !error ? <div className="mt-6"><EmptyState /></div> : <>
          <section className="mt-6 grid grid-cols-1 gap-4 xl:grid-cols-2">
            <ChartPanel title="Routing distribution" subtitle="Requests by model tier"><ResponsiveContainer><PieChart><Pie data={distributionData} dataKey="value" nameKey="name" innerRadius={54} outerRadius={84} paddingAngle={4}>{distributionData.map((entry) => <Cell key={entry.name} fill={TIER_COLORS[entry.name] ?? '#7585a0'} />)}</Pie><Tooltip contentStyle={tooltipStyle} /><Legend /></PieChart></ResponsiveContainer></ChartPanel>
            <ChartPanel title="Cost compression" subtitle="Actual spend against estimated premium baseline"><ResponsiveContainer><BarChart data={[{ name: 'Cost', Baseline: summary.estimated_baseline_cost, Nominal: summary.actual_cost }]}><XAxis dataKey="name" tick={{ fill: '#8490a5', fontSize: 12 }} /><YAxis tick={{ fill: '#8490a5', fontSize: 12 }} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => formatCurrency(Number(value))} /><Legend /><Bar dataKey="Baseline" fill="#7585a0" radius={[4, 4, 0, 0]} /><Bar dataKey="Nominal" fill="#9cf7c8" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></ChartPanel>
            <ChartPanel title="Request latency" subtitle="End-to-end telemetry across recent requests"><ResponsiveContainer><LineChart data={latencyData}><XAxis dataKey="name" tick={{ fill: '#8490a5', fontSize: 12 }} /><YAxis tick={{ fill: '#8490a5', fontSize: 12 }} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => formatMs(Number(value))} /><Line type="monotone" dataKey="latency" stroke="#a99bff" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></ChartPanel>
            <ChartPanel title="Cumulative savings" subtitle="Estimated dollars retained by capability-aware routing"><ResponsiveContainer><AreaChart data={savingsData}><defs><linearGradient id="savings" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#9cf7c8" stopOpacity={.35} /><stop offset="100%" stopColor="#9cf7c8" stopOpacity={0} /></linearGradient></defs><XAxis dataKey="name" tick={{ fill: '#8490a5', fontSize: 12 }} /><YAxis tick={{ fill: '#8490a5', fontSize: 12 }} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => formatCurrency(Number(value))} /><Area type="monotone" dataKey="savings" stroke="#9cf7c8" strokeWidth={2} fill="url(#savings)" /></AreaChart></ResponsiveContainer></ChartPanel>
          </section>
          <section className="mt-6 grid grid-cols-1 gap-4 2xl:grid-cols-[1.3fr_.7fr]">
            <div className="panel overflow-hidden"><div className="flex items-center justify-between border-b border-line px-4 py-3"><div><p className="eyebrow">Live requests</p><h2 className="mt-1 text-sm font-medium">Recent routing decisions</h2></div><span className="text-xs text-slate-500">Auto-refreshes every 10s</span></div><div className="overflow-x-auto"><table className="min-w-[850px] w-full text-left text-xs"><thead className="bg-black/10 text-[10px] uppercase tracking-[.12em] text-slate-500"><tr>{['Request', 'Intent', 'Selected → Final', 'Quality', 'Latency', 'Cost / saved', 'Status'].map((heading) => <th key={heading} className="px-4 py-3 font-medium">{heading}</th>)}</tr></thead><tbody>{traces.map((trace) => <tr key={trace.request_id} onClick={() => void selectTrace(trace)} className={`cursor-pointer border-t border-line/70 transition hover:bg-slate-800/40 ${selected?.request_id === trace.request_id ? 'bg-signal/[.05]' : ''}`}><td className="px-4 py-3 font-mono text-slate-300">{shortId(trace.request_id)}</td><td className="px-4 py-3"><span className="tag">{titleCase(trace.profile.intent)}</span></td><td className="px-4 py-3 text-slate-300"><span>{trace.initial_model ?? '—'}</span><span className="mx-1 text-slate-600">→</span><span className={trace.escalated ? 'text-amber' : 'text-signal'}>{trace.final_model ?? trace.model_used ?? '—'}</span></td><td className="px-4 py-3"><span className={trace.quality_result?.passed ? 'text-signal' : 'text-amber'}>{trace.quality_result ? `${Math.round(trace.quality_result.score * 100)}%` : '—'}</span></td><td className="px-4 py-3 text-slate-300">{formatMs(trace.total_latency_ms ?? trace.provider_latency_ms)}</td><td className="px-4 py-3"><span className="text-slate-300">{formatCurrency(trace.actual_cost)}</span><span className="ml-1 text-signal">+{formatCurrency(trace.estimated_cost_saved)}</span></td><td className="px-4 py-3">{trace.escalated ? <span className="text-amber">Escalated</span> : trace.fallback_used ? <span className="text-violet">Fallback</span> : <span className="text-slate-500">Routed</span>}</td></tr>)}</tbody></table></div></div>
            <aside className="panel max-h-[720px] overflow-y-auto p-4"><div className="mb-4"><p className="eyebrow">Routing decision trace</p><h2 className="mt-1 text-sm font-medium">{selected ? shortId(selected.request_id) : 'Select a request'}</h2></div>{selected ? <TraceFlow trace={selected} /> : <p className="text-sm text-slate-500">Choose a request to inspect its Nominal Path.</p>}</aside>
          </section>
        </>}
      </div>
    </div></main>
  )
}
