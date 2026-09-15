import type { BenchmarkRun, MetricsSummary, RequestTrace } from './types'

const baseUrl = import.meta.env.VITE_API_BASE_URL ?? ''

async function get<T>(path: string): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${baseUrl}${path}`)
  } catch {
    throw new Error('NOMINAL backend is unavailable. Start the backend and retry.')
  }
  if ([502, 503, 504].includes(response.status)) {
    throw new Error('NOMINAL backend is unavailable. Start the backend and retry.')
  }
  if (!response.ok) {
    throw new Error(`API request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

async function post<T>(path: string, body: unknown): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${baseUrl}${path}`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
    })
  } catch {
    throw new Error('NOMINAL backend is unavailable. Start the backend and retry.')
  }
  if ([502, 503, 504].includes(response.status)) {
    throw new Error('NOMINAL backend is unavailable. Start the backend and retry.')
  }
  if (!response.ok) throw new Error(`API request failed (${response.status})`)
  return response.json() as Promise<T>
}

export const nominalApi = {
  getSummary: () => get<MetricsSummary>('/api/metrics/summary'),
  getTraces: (limit = 100, offset = 0) => get<RequestTrace[]>(`/api/traces?limit=${limit}&offset=${offset}`),
  getTrace: (requestId: string) => get<RequestTrace>(`/api/traces/${encodeURIComponent(requestId)}`),
  getBenchmarks: () => get<BenchmarkRun[]>('/api/benchmarks'),
  runBenchmark: () => post<BenchmarkRun>('/api/benchmarks/run', {}),
}
