export type CandidateModel = {
  model_id: string
  tier: 'local' | 'economy' | 'balanced' | 'frontier'
  provider: string
  model_name: string
  predicted_quality: number
  estimated_cost: number
  estimated_latency: number
  fitness_score: number
}

export type RejectedModel = {
  model_id: string
  tier: string
  reason: string
}

export type RoutingDecision = {
  selected_model: { id: string; tier: string; model_name: string; provider: string }
  candidate_models: CandidateModel[]
  rejected_models: RejectedModel[]
  predicted_quality: number
  estimated_cost: number
  estimated_latency: number
  routing_reason: string
  router_latency_ms: number
}

export type QualityResult = {
  score: number
  passed: boolean
  reasons: string[]
  evaluator_type: string
  evaluation_latency_ms: number
}

export type RequestTrace = {
  request_id: string
  started_at: string
  completed_at?: string
  profile: { intent: string; quality_target: number; estimated_complexity: number }
  routing_decision?: RoutingDecision
  quality_result?: QualityResult
  provider?: string
  model_used?: string
  initial_model?: string
  final_model?: string
  input_tokens?: number
  output_tokens?: number
  provider_latency_ms?: number
  total_latency_ms?: number
  actual_cost?: string | number
  estimated_baseline_cost?: string | number
  estimated_cost_saved?: string | number
  savings_percentage?: number
  escalated: boolean
  escalation_reason?: string
  fallback_used: boolean
  fallback_reason?: string
  status: string
}

export type MetricsSummary = {
  total_requests: number
  successful_requests: number
  total_tokens: number
  actual_cost: number
  estimated_baseline_cost: number
  estimated_cost_saved: number
  savings_percentage: number
  average_latency_ms: number
  p95_latency_ms: number
  average_router_overhead_ms: number
  escalation_rate: number
  fallback_rate: number
  frontier_calls_avoided: number
  routing_distribution: Record<string, number>
  average_quality_score: number
}

export type BenchmarkStrategyMetrics = {
  strategy: 'always_frontier' | 'always_economy' | 'nominal'
  request_count: number
  total_cost: number
  estimated_cost_per_request: number
  total_tokens: number
  average_latency_ms: number
  p50_latency_ms: number
  p95_latency_ms: number
  quality_score: number
  quality_pass_rate: number
  frontier_calls: number
  escalations: number
  failures: number
}

export type BenchmarkRun = {
  id: string
  created_at: string
  status: string
  dataset_name: string
  evaluator_type: 'heuristic' | 'judge_model'
  strategies: BenchmarkStrategyMetrics[]
  request_results: Record<string, unknown>[]
}
