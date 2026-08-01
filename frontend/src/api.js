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
    const override = Number(workingMonthsOverride)
    if (!Number.isInteger(override) || override < 0) {
      throw new Error('Working months override must be a nonnegative whole number')
    }
    body.working_months_override = override
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

  if (!res.body) throw new Error('Simulation response stream is unavailable')
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let receivedTerminalEvent = false

  const handlePart = (part) => {
    const line = part.replace(/^data: /, '').trim()
    if (!line) return
    let event
    try {
      event = JSON.parse(line)
    } catch {
      throw new Error('Received malformed simulation progress data')
    }
    if (event.type === 'result') {
      receivedTerminalEvent = true
      onResult(event.data)
    } else if (event.type === 'error') {
      receivedTerminalEvent = true
      onError(event.message)
    } else {
      onProgress(event)
    }
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop()

    for (const part of parts) {
      handlePart(part)
    }
  }

  buffer += decoder.decode()
  if (buffer.trim()) handlePart(buffer)
  if (!receivedTerminalEvent) {
    throw new Error('Simulation stream ended without a result')
  }
}
