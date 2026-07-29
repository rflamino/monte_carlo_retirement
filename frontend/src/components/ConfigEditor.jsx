import { useState, useEffect, useRef } from 'react'
import { getDefaultConfig } from '../api'

const EMPTY_STREAM = {
  name: '',
  monthly_amount_today: 0,
  start_after_retirement_years: 0,
  duration_years: null,
  inflation_indexed: true,
  tax_rate: 0,
}

function cloneConfig(cfg) {
  return JSON.parse(JSON.stringify(cfg))
}

function Field({ label, hint, children, wide }) {
  return (
    <div className={'cfg-field' + (wide ? ' cfg-field-wide' : '')}>
      <label className="cfg-label">
        {label}
        {hint && <span className="cfg-hint">{hint}</span>}
      </label>
      {children}
    </div>
  )
}

function NumberInput({ value, onChange, step = 1, min, max, placeholder }) {
  const display =
    value === null || value === undefined || Number.isNaN(value) ? '' : value
  return (
    <input
      type="number"
      className="cfg-input"
      value={display}
      step={step}
      min={min}
      max={max}
      placeholder={placeholder}
      onChange={(e) => {
        const raw = e.target.value
        if (raw === '') {
          onChange(null)
          return
        }
        const n = Number(raw)
        onChange(Number.isFinite(n) ? n : null)
      }}
    />
  )
}

/** Rates stored as fractions (0.12) but edited as percents (12). */
function PercentInput({ value, onChange, step = 0.1, min = 0, max }) {
  const pct =
    value === null || value === undefined || Number.isNaN(value)
      ? ''
      : Math.round(value * 10000) / 100
  return (
    <div className="cfg-input-affix">
      <input
        type="number"
        className="cfg-input"
        value={pct}
        step={step}
        min={min}
        max={max}
        onChange={(e) => {
          const raw = e.target.value
          if (raw === '') {
            onChange(null)
            return
          }
          const n = Number(raw)
          onChange(Number.isFinite(n) ? n / 100 : null)
        }}
      />
      <span className="cfg-affix">%</span>
    </div>
  )
}

function MoneyInput({ value, onChange, step = 100 }) {
  return (
    <div className="cfg-input-affix">
      <span className="cfg-affix cfg-affix-left">$</span>
      <input
        type="number"
        className="cfg-input cfg-input-money"
        value={value ?? ''}
        step={step}
        min={0}
        onChange={(e) => {
          const raw = e.target.value
          if (raw === '') {
            onChange(null)
            return
          }
          const n = Number(raw)
          onChange(Number.isFinite(n) ? n : null)
        }}
      />
    </div>
  )
}

function Toggle({ checked, onChange, label }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      className={'cfg-toggle' + (checked ? ' on' : '')}
      onClick={() => onChange(!checked)}
    >
      <span className="cfg-toggle-track">
        <span className="cfg-toggle-thumb" />
      </span>
      {label && <span className="cfg-toggle-label">{label}</span>}
    </button>
  )
}

function Section({ id, title, open, onToggle, children, badge }) {
  return (
    <div className={'cfg-section' + (open ? ' open' : '')}>
      <button type="button" className="cfg-section-head" onClick={() => onToggle(id)}>
        <span className="cfg-section-chevron" aria-hidden>
          {open ? '▾' : '▸'}
        </span>
        <span className="cfg-section-title">{title}</span>
        {badge != null && <span className="cfg-section-badge">{badge}</span>}
      </button>
      {open && <div className="cfg-section-body">{children}</div>}
    </div>
  )
}

export default function ConfigEditor({ onSimulate, loading }) {
  const [config, setConfig] = useState(null)
  const [mode, setMode] = useState('form') // 'form' | 'json'
  const [jsonText, setJsonText] = useState('')
  const [jsonError, setJsonError] = useState(null)
  const [workingMonths, setWorkingMonths] = useState('')
  const [statusMsg, setStatusMsg] = useState(null)
  const [openSections, setOpenSections] = useState({
    scenario: true,
    finances: true,
    assets: false,
    inflation: false,
    income: true,
    sim: false,
  })
  const fileInput = useRef(null)

  useEffect(() => {
    getDefaultConfig()
      .then((cfg) => {
        setConfig(cfg)
        setJsonText(JSON.stringify(cfg, null, 2))
      })
      .catch(() => {
        setConfig(null)
        setJsonError('Could not load default configuration')
      })
  }, [])

  const flash = (msg) => {
    setStatusMsg(msg)
    window.setTimeout(() => setStatusMsg(null), 2200)
  }

  const update = (key, value) => {
    setConfig((prev) => (prev ? { ...prev, [key]: value } : prev))
  }

  const toggleSection = (id) => {
    setOpenSections((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  const switchToJson = () => {
    if (!config) return
    setJsonText(JSON.stringify(config, null, 2))
    setJsonError(null)
    setMode('json')
  }

  const switchToForm = () => {
    try {
      const parsed = JSON.parse(jsonText)
      setConfig(parsed)
      setJsonError(null)
      setMode('form')
    } catch (e) {
      setJsonError('Fix JSON before switching to form view: ' + e.message)
    }
  }

  const handleLoadFile = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const parsed = JSON.parse(ev.target.result)
        setConfig(parsed)
        setJsonText(JSON.stringify(parsed, null, 2))
        setJsonError(null)
        flash(`Loaded ${file.name}`)
      } catch (err) {
        setJsonError('Invalid JSON file: ' + err.message)
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  const handleSaveFile = () => {
    let payload
    try {
      payload = mode === 'json' ? JSON.parse(jsonText) : config
    } catch (e) {
      setJsonError('Invalid JSON: ' + e.message)
      return
    }
    if (!payload) return
    const name = (payload.scenario || 'config')
      .replace(/[^\w\-]+/g, '_')
      .replace(/^_|_$/g, '')
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${name || 'config'}.json`
    a.click()
    URL.revokeObjectURL(url)
    flash('Saved JSON file')
  }

  const handleReset = async () => {
    try {
      const cfg = await getDefaultConfig()
      setConfig(cfg)
      setJsonText(JSON.stringify(cfg, null, 2))
      setJsonError(null)
      flash('Reset to default')
    } catch {
      setJsonError('Could not reload default configuration')
    }
  }

  const handleSimulate = () => {
    let payload
    try {
      payload = mode === 'json' ? JSON.parse(jsonText) : cloneConfig(config)
      setJsonError(null)
    } catch (e) {
      setJsonError('Invalid JSON: ' + e.message)
      return
    }
    if (!payload) {
      setJsonError('No configuration loaded')
      return
    }
    onSimulate(payload, workingMonths || undefined)
  }

  const streams = config?.other_income_streams || []

  const updateStream = (index, key, value) => {
    setConfig((prev) => {
      if (!prev) return prev
      const next = cloneConfig(prev)
      next.other_income_streams = [...(next.other_income_streams || [])]
      next.other_income_streams[index] = {
        ...next.other_income_streams[index],
        [key]: value,
      }
      return next
    })
  }

  const addStream = () => {
    setConfig((prev) => {
      if (!prev) return prev
      return {
        ...prev,
        other_income_streams: [...(prev.other_income_streams || []), { ...EMPTY_STREAM }],
      }
    })
    setOpenSections((s) => ({ ...s, income: true }))
  }

  const removeStream = (index) => {
    setConfig((prev) => {
      if (!prev) return prev
      return {
        ...prev,
        other_income_streams: (prev.other_income_streams || []).filter((_, i) => i !== index),
      }
    })
  }

  if (!config && mode === 'form') {
    return (
      <div className="config-editor">
        <div className="cfg-loading">Loading configuration…</div>
      </div>
    )
  }

  return (
    <div className="config-editor">
      <div className="cfg-toolbar">
        <div className="cfg-mode-toggle" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'form'}
            className={'cfg-mode-btn' + (mode === 'form' ? ' active' : '')}
            onClick={() => (mode === 'json' ? switchToForm() : null)}
          >
            Form
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'json'}
            className={'cfg-mode-btn' + (mode === 'json' ? ' active' : '')}
            onClick={switchToJson}
          >
            JSON
          </button>
        </div>
        <div className="cfg-toolbar-actions">
          <button type="button" className="btn btn-sm" onClick={() => fileInput.current?.click()}>
            Load
          </button>
          <button type="button" className="btn btn-sm" onClick={handleSaveFile}>
            Save
          </button>
          <button type="button" className="btn btn-sm" onClick={handleReset}>
            Reset
          </button>
          <input
            ref={fileInput}
            type="file"
            accept=".json,application/json"
            className="cfg-file-input"
            onChange={handleLoadFile}
          />
        </div>
      </div>

      {statusMsg && <div className="cfg-status">{statusMsg}</div>}
      {jsonError && <div className="parse-error">{jsonError}</div>}

      {mode === 'json' ? (
        <textarea
          className="config-textarea"
          value={jsonText}
          onChange={(e) => {
            setJsonText(e.target.value)
            setJsonError(null)
          }}
          spellCheck={false}
        />
      ) : (
        <div className="cfg-form">
          <Section
            id="scenario"
            title="Scenario"
            open={openSections.scenario}
            onToggle={toggleSection}
          >
            <div className="cfg-grid">
              <Field label="Name" wide>
                <input
                  className="cfg-input"
                  type="text"
                  value={config.scenario ?? ''}
                  onChange={(e) => update('scenario', e.target.value)}
                />
              </Field>
              <Field label="Retirement years">
                <NumberInput
                  value={config.retirement_years}
                  min={1}
                  onChange={(v) => update('retirement_years', v ?? 1)}
                />
              </Field>
              <Field label="Target success" hint="probability">
                <PercentInput
                  value={(config.target_probability ?? 0) / 100}
                  max={100}
                  onChange={(v) => update('target_probability', (v ?? 0) * 100)}
                />
              </Field>
            </div>
          </Section>

          <Section
            id="finances"
            title="Finances"
            open={openSections.finances}
            onToggle={toggleSection}
          >
            <div className="cfg-grid">
              <Field label="Initial balance">
                <MoneyInput
                  value={config.initial_balance}
                  onChange={(v) => update('initial_balance', v ?? 0)}
                />
              </Field>
              <Field label="Monthly contribution">
                <MoneyInput
                  value={config.monthly_contribution}
                  onChange={(v) => update('monthly_contribution', v ?? 0)}
                />
              </Field>
              <Field label="Contribution growth" hint="annual">
                <PercentInput
                  value={config.contribution_growth_rate_annual}
                  onChange={(v) => update('contribution_growth_rate_annual', v ?? 0)}
                />
              </Field>
              <Field label="Monthly expenses" hint="today’s $">
                <MoneyInput
                  value={config.monthly_expenses}
                  onChange={(v) => update('monthly_expenses', v ?? 0)}
                />
              </Field>
            </div>
          </Section>

          <Section
            id="assets"
            title="Portfolio & taxes"
            open={openSections.assets}
            onToggle={toggleSection}
          >
            <div className="cfg-asset-block">
              <h4 className="cfg-asset-title">
                Asset 1 · Equity
                <span className="cfg-asset-pct">
                  {Math.round((config.allocation_inv1_pct ?? 0) * 100)}%
                </span>
              </h4>
              <div className="cfg-grid">
                <Field label="Allocation">
                  <PercentInput
                    value={config.allocation_inv1_pct}
                    max={100}
                    onChange={(v) => update('allocation_inv1_pct', v ?? 0)}
                  />
                </Field>
                <Field label="Expected return">
                  <PercentInput
                    value={config.inv1_returns_mean}
                    onChange={(v) => update('inv1_returns_mean', v ?? 0)}
                  />
                </Field>
                <Field label="Volatility">
                  <PercentInput
                    value={config.inv1_returns_volatility}
                    onChange={(v) => update('inv1_returns_volatility', v ?? 0)}
                  />
                </Field>
                <Field label="Tax model" wide>
                  <Toggle
                    checked={!!config.inv1_use_realized_gains_tax_system}
                    onChange={(v) => update('inv1_use_realized_gains_tax_system', v)}
                    label={
                      config.inv1_use_realized_gains_tax_system
                        ? 'Realized gains on sale'
                        : 'Annual tax on gains'
                    }
                  />
                </Field>
                {config.inv1_use_realized_gains_tax_system ? (
                  <Field label="Realized gains tax">
                    <PercentInput
                      value={config.inv1_realized_gains_tax_rate}
                      max={100}
                      onChange={(v) => update('inv1_realized_gains_tax_rate', v ?? 0)}
                    />
                  </Field>
                ) : (
                  <Field label="Annual gains tax">
                    <PercentInput
                      value={config.inv1_annual_tax_on_gains_rate}
                      max={100}
                      onChange={(v) => update('inv1_annual_tax_on_gains_rate', v ?? 0)}
                    />
                  </Field>
                )}
              </div>
            </div>

            <div className="cfg-asset-block">
              <h4 className="cfg-asset-title">
                Asset 2 · Safer
                <span className="cfg-asset-pct">
                  {Math.round((1 - (config.allocation_inv1_pct ?? 0)) * 100)}%
                </span>
              </h4>
              <div className="cfg-grid">
                <Field label="Premium over inflation">
                  <PercentInput
                    value={config.inv2_premium_over_inflation_mean}
                    onChange={(v) => update('inv2_premium_over_inflation_mean', v ?? 0)}
                  />
                </Field>
                <Field label="Premium volatility">
                  <PercentInput
                    value={config.inv2_premium_over_inflation_volatility}
                    onChange={(v) => update('inv2_premium_over_inflation_volatility', v ?? 0)}
                  />
                </Field>
                <Field label="Tax model" wide>
                  <Toggle
                    checked={!!config.inv2_use_realized_gains_tax_system}
                    onChange={(v) => update('inv2_use_realized_gains_tax_system', v)}
                    label={
                      config.inv2_use_realized_gains_tax_system
                        ? 'Realized gains on sale'
                        : 'Annual tax on gains'
                    }
                  />
                </Field>
                {config.inv2_use_realized_gains_tax_system ? (
                  <Field label="Realized gains tax">
                    <PercentInput
                      value={config.inv2_realized_gains_tax_rate}
                      max={100}
                      onChange={(v) => update('inv2_realized_gains_tax_rate', v ?? 0)}
                    />
                  </Field>
                ) : (
                  <Field label="Annual gains tax">
                    <PercentInput
                      value={config.inv2_annual_tax_on_gains_rate}
                      max={100}
                      onChange={(v) => update('inv2_annual_tax_on_gains_rate', v ?? 0)}
                    />
                  </Field>
                )}
              </div>
            </div>
          </Section>

          <Section
            id="inflation"
            title="Inflation"
            open={openSections.inflation}
            onToggle={toggleSection}
          >
            <div className="cfg-grid">
              <Field label="Mean">
                <PercentInput
                  value={config.inflation_rate_mean}
                  onChange={(v) => update('inflation_rate_mean', v ?? 0)}
                />
              </Field>
              <Field label="Volatility">
                <PercentInput
                  value={config.inflation_rate_volatility}
                  onChange={(v) => update('inflation_rate_volatility', v ?? 0)}
                />
              </Field>
              <Field label="Equity–inflation corr." hint="−1 to 1">
                <NumberInput
                  value={config.equity_inflation_correlation ?? 0}
                  step={0.05}
                  min={-1}
                  max={1}
                  onChange={(v) => update('equity_inflation_correlation', v ?? 0)}
                />
              </Field>
            </div>
          </Section>

          <Section
            id="income"
            title="Other income"
            open={openSections.income}
            onToggle={toggleSection}
            badge={streams.length}
          >
            {streams.length === 0 && (
              <p className="cfg-empty-streams">No income streams yet.</p>
            )}
            {streams.map((stream, i) => (
              <div key={i} className="cfg-stream">
                <div className="cfg-stream-head">
                  <input
                    className="cfg-input cfg-stream-name"
                    type="text"
                    placeholder="Stream name"
                    value={stream.name ?? ''}
                    onChange={(e) => updateStream(i, 'name', e.target.value)}
                  />
                  <button
                    type="button"
                    className="btn btn-sm cfg-stream-remove"
                    onClick={() => removeStream(i)}
                    title="Remove stream"
                  >
                    Remove
                  </button>
                </div>
                <div className="cfg-grid">
                  <Field label="Monthly amount" hint="today’s $">
                    <MoneyInput
                      value={stream.monthly_amount_today}
                      onChange={(v) => updateStream(i, 'monthly_amount_today', v ?? 0)}
                    />
                  </Field>
                  <Field label="Starts after" hint="ret. years">
                    <NumberInput
                      value={stream.start_after_retirement_years}
                      min={0}
                      onChange={(v) =>
                        updateStream(i, 'start_after_retirement_years', v ?? 0)
                      }
                    />
                  </Field>
                  <Field label="Duration" hint="years · blank = forever">
                    <NumberInput
                      value={stream.duration_years}
                      min={0}
                      placeholder="∞"
                      onChange={(v) => updateStream(i, 'duration_years', v)}
                    />
                  </Field>
                  <Field label="Tax rate">
                    <PercentInput
                      value={stream.tax_rate}
                      max={100}
                      onChange={(v) => updateStream(i, 'tax_rate', v ?? 0)}
                    />
                  </Field>
                  <Field label="Inflation indexed" wide>
                    <Toggle
                      checked={!!stream.inflation_indexed}
                      onChange={(v) => updateStream(i, 'inflation_indexed', v)}
                      label={stream.inflation_indexed ? 'Keeps pace with inflation' : 'Fixed nominal at start'}
                    />
                  </Field>
                </div>
              </div>
            ))}
            <button type="button" className="btn btn-sm cfg-add-stream" onClick={addStream}>
              + Add income stream
            </button>
          </Section>

          <Section
            id="sim"
            title="Simulation"
            open={openSections.sim}
            onToggle={toggleSection}
          >
            <div className="cfg-grid">
              <Field label="Main simulations">
                <NumberInput
                  value={config.num_simulations_main}
                  min={1}
                  step={100}
                  onChange={(v) => update('num_simulations_main', v ?? 1)}
                />
              </Field>
              <Field label="Search simulations">
                <NumberInput
                  value={config.num_simulations_search}
                  min={1}
                  step={50}
                  onChange={(v) => update('num_simulations_search', v ?? 1)}
                />
              </Field>
              <Field label="Start search at" hint="months">
                <NumberInput
                  value={config.starting_working_months_search}
                  min={0}
                  onChange={(v) => update('starting_working_months_search', v ?? 0)}
                />
              </Field>
              <Field label="Processes" hint="cores">
                <NumberInput
                  value={config.num_processes}
                  min={1}
                  onChange={(v) => update('num_processes', v)}
                />
              </Field>
              <Field label="Seed" hint="blank = random">
                <NumberInput
                  value={config.seed}
                  min={0}
                  placeholder="random"
                  onChange={(v) => update('seed', v)}
                />
              </Field>
            </div>
          </Section>
        </div>
      )}

      <div className="cfg-run">
        <Field label="Working months override" hint="leave blank to search">
          <NumberInput
            value={workingMonths === '' ? null : Number(workingMonths)}
            min={0}
            placeholder="Search automatically"
            onChange={(v) => setWorkingMonths(v === null ? '' : String(v))}
          />
        </Field>
        <button
          type="button"
          className="btn btn-primary btn-simulate"
          onClick={handleSimulate}
          disabled={loading}
        >
          {loading ? 'Simulating…' : 'Run Simulation'}
        </button>
      </div>
    </div>
  )
}
