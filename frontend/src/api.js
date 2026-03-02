const API_BASE = '/api'

export async function getDefaultConfig() {
  const res = await fetch(`${API_BASE}/config/default`)
  if (!res.ok) throw new Error('Failed to load default configuration')
  return res.json()
}

export async function runSimulation(config, workingMonthsOverride) {
  const body = { config }
  if (workingMonthsOverride != null && workingMonthsOverride !== '') {
    body.working_months_override = parseInt(workingMonthsOverride, 10)
  }

  const res = await fetch(`${API_BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Simulation failed' }))
    throw new Error(err.detail || 'Simulation failed')
  }

  return res.json()
}
