export const formatCurrency = (value: number | string | undefined) => {
  const amount = Number(value ?? 0)
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: amount < 1 ? 4 : 2 }).format(amount)
}

export const formatNumber = (value: number | undefined) => new Intl.NumberFormat('en-US').format(value ?? 0)

export const formatMs = (value: number | undefined) => `${Math.round(value ?? 0).toLocaleString()} ms`

export const shortId = (value: string) => `${value.slice(0, 8)}…${value.slice(-4)}`

export const titleCase = (value: string | undefined) => (value ?? 'unknown').replace(/_/g, ' ')

export const tierFromModel = (model: string | undefined) => {
  const normalized = (model ?? '').toLowerCase()
  if (normalized.includes('frontier') || normalized === 'o3') return 'frontier'
  if (normalized.includes('balanced') || normalized.includes('gpt-4.1')) return 'balanced'
  if (normalized.includes('economy') || normalized.includes('mini')) return 'economy'
  return 'local'
}
