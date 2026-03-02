const API_BASE = '/api'

export async function getDefaultConfig() {
  const res = await fetch(`${API_BASE}/config/default`)
  if (!res.ok) throw new Error('Failed to load default configuration')
  return res.json()
}

export async function runSimulationStream(
  config,
  workingMonthsOverride,
  { onProgress, onResult, onError },
) {
  const body = { config }
  if (workingMonthsOverride != null && workingMonthsOverride !== '') {
    body.working_months_override = parseInt(workingMonthsOverride, 10)
  }

  const res = await fetch(`${API_BASE}/simulate/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Simulation failed' }))
    throw new Error(err.detail || 'Simulation failed')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop()

    for (const part of parts) {
      const line = part.replace(/^data: /, '').trim()
      if (!line) continue
      try {
        const event = JSON.parse(line)
        if (event.type === 'result') {
          onResult(event.data)
        } else if (event.type === 'error') {
          onError(event.message)
        } else {
          onProgress(event)
        }
      } catch {
        /* ignore malformed chunks */
      }
    }
  }
}
