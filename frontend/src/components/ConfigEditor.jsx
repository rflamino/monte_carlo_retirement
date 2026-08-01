import { useState, useEffect, useRef, useId } from 'react'
import { createPortal } from 'react-dom'
import { getDefaultConfig } from '../api'

const EMPTY_STREAM = {
  name: '',
  monthly_amount_today: 0,
  start_at_age: 65,
  duration_years: null,
  inflation_indexed: true,
  tax_rate: 0,
}

function cloneConfig(cfg) {
  return JSON.parse(JSON.stringify(cfg))
}

function TipBalloon({ text }) {
  const [open, setOpen] = useState(false)
  const [pinned, setPinned] = useState(false)
  const [coords, setCoords] = useState(null)
  const btnRef = useRef(null)
  const tipId = useId()

  const place = () => {
    const el = btnRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const pad = 10
    const maxW = Math.min(280, window.innerWidth - pad * 2)
    let left = r.left + r.width / 2
    left = Math.max(pad + maxW / 2, Math.min(left, window.innerWidth - pad - maxW / 2))
    const spaceAbove = r.top
    const placeBelow = spaceAbove < 96
    setCoords({
      left,
      top: placeBelow ? r.bottom + 8 : r.top - 8,
      below: placeBelow,
      maxW,
    })
  }

  const show = () => {
    place()
    setOpen(true)
  }

  const hide = () => {
    if (!pinned) setOpen(false)
  }

  const togglePin = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (pinned) {
      setPinned(false)
      setOpen(false)
    } else {
      place()
      setPinned(true)
      setOpen(true)
    }
  }

  useEffect(() => {
    if (!open) return undefined
    const onScroll = () => place()
    const onKey = (ev) => {
      if (ev.key === 'Escape') {
        setPinned(false)
        setOpen(false)
      }
    }
    const onDoc = (ev) => {
      if (
        pinned &&
        btnRef.current &&
        !btnRef.current.contains(ev.target) &&
        !(ev.target instanceof Element && ev.target.closest('.cfg-tip-balloon'))
      ) {
        setPinned(false)
        setOpen(false)
      }
    }
    window.addEventListener('scroll', onScroll, true)
    window.addEventListener('resize', onScroll)
    window.addEventListener('keydown', onKey)
    document.addEventListener('mousedown', onDoc)
    return () => {
      window.removeEventListener('scroll', onScroll, true)
      window.removeEventListener('resize', onScroll)
      window.removeEventListener('keydown', onKey)
      document.removeEventListener('mousedown', onDoc)
    }
  }, [open, pinned])

  return (
    <span className="cfg-tip">
      <button
        ref={btnRef}
        type="button"
        className={'cfg-tip-btn' + (open ? ' active' : '')}
        aria-label="Field help"
        aria-describedby={open ? tipId : undefined}
        aria-expanded={open}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        onClick={togglePin}
      >
        ?
      </button>
      {open &&
        coords &&
        createPortal(
          <div
            id={tipId}
            role="tooltip"
            className={'cfg-tip-balloon' + (coords.below ? ' below' : ' above')}
            style={{
              left: coords.left,
              top: coords.top,
              maxWidth: coords.maxW,
              transform: coords.below
                ? 'translate(-50%, 0)'
                : 'translate(-50%, -100%)',
            }}
            onMouseEnter={() => setOpen(true)}
            onMouseLeave={hide}
          >
            {text}
          </div>,
          document.body,
        )}
    </span>
  )
}

function Field({ label, hint, tip, children, wide }) {
  return (
    <div className={'cfg-field' + (wide ? ' cfg-field-wide' : '')}>
      <div className="cfg-label-row">
        <label className="cfg-label">
          {label}
          {hint && <span className="cfg-hint">{hint}</span>}
        </label>
        {tip && <TipBalloon text={tip} />}
      </div>
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
              <Field
                label="Name"
                wide
                tip="Nickname for this scenario. Used in charts, logs, and the saved JSON filename."
              >
                <input
                  className="cfg-input"
                  type="text"
                  value={config.scenario ?? ''}
                  onChange={(e) => update('scenario', e.target.value)}
                />
              </Field>
              <Field
                label="Current age"
                tip="Your age at simulation start (today). Retirement age = current age + working years. Income streams use start-at-age against this timeline."
              >
                <NumberInput
                  value={config.current_age}
                  min={0}
                  max={120}
                  step={0.1}
                  onChange={(v) => update('current_age', v ?? 0)}
                />
              </Field>
              <Field
                label="Retirement years"
                tip="How many years of retirement (decumulation) to simulate after you stop working."
              >
                <NumberInput
                  value={config.retirement_years}
                  min={1}
                  onChange={(v) => update('retirement_years', v ?? 1)}
                />
              </Field>
              <Field
                label="Target success"
                hint="probability"
                tip="Target probability of funding all retirement spending. The search estimates the earliest working month that reaches this rate."
              >
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
              <Field
                label="Initial balance"
                tip="Total portfolio value today (time = 0), split across Asset 1 and Asset 2 by allocation."
              >
                <MoneyInput
                  value={config.initial_balance}
                  onChange={(v) => update('initial_balance', v ?? 0)}
                />
              </Field>
              <Field
                label="Monthly contribution"
                tip="Amount saved into the portfolio each month while working. Starts at this value and can grow annually."
              >
                <MoneyInput
                  value={config.monthly_contribution}
                  onChange={(v) => update('monthly_contribution', v ?? 0)}
                />
              </Field>
              <Field
                label="Contribution growth"
                hint="annual"
                tip="Annual percentage increase applied to your monthly contribution at the start of each working year (e.g. raises)."
              >
                <PercentInput
                  value={config.contribution_growth_rate_annual}
                  onChange={(v) => update('contribution_growth_rate_annual', v ?? 0)}
                />
              </Field>
              <Field
                label="Monthly expenses"
                hint="today’s $"
                tip="Retirement spending in today’s purchasing power. Inflated to future dollars during the simulation."
              >
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
                <Field
                  label="Allocation"
                  tip="Share of the portfolio in Asset 1 (equities). Asset 2 automatically gets the remainder."
                >
                  <PercentInput
                    value={config.allocation_inv1_pct}
                    max={100}
                    onChange={(v) => update('allocation_inv1_pct', v ?? 0)}
                  />
                </Field>
                <Field
                  label="Expected return"
                  tip="Arithmetic expected annual return for equities. Simulated as lognormal so the mean annual gross return matches this value."
                >
                  <PercentInput
                    value={config.inv1_returns_mean}
                    onChange={(v) => update('inv1_returns_mean', v ?? 0)}
                  />
                </Field>
                <Field
                  label="Volatility"
                  tip="Annual standard deviation of equity returns. Higher values increase sequence-of-returns risk. Typical equities are around 15%."
                >
                  <PercentInput
                    value={config.inv1_returns_volatility}
                    onChange={(v) => update('inv1_returns_volatility', v ?? 0)}
                  />
                </Field>
                <Field
                  label="Tax model"
                  wide
                  tip="Realized gains: tax paid only when selling. Annual tax: tax deducted each year on that year’s gains (e.g. come-cotas style)."
                >
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
                  <Field
                    label="Realized gains tax"
                    tip="Tax rate applied to the gain portion of withdrawals and rebalancing sales for Asset 1."
                  >
                    <PercentInput
                      value={config.inv1_realized_gains_tax_rate}
                      max={100}
                      onChange={(v) => update('inv1_realized_gains_tax_rate', v ?? 0)}
                    />
                  </Field>
                ) : (
                  <Field
                    label="Annual gains tax"
                    tip="Tax rate applied each year to Asset 1 gains (excluding new contributions)."
                  >
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
                <Field
                  label="Premium over inflation"
                  tip="Expected annual real return of Asset 2 above inflation (e.g. bonds). Combined with inflation each month."
                >
                  <PercentInput
                    value={config.inv2_premium_over_inflation_mean}
                    onChange={(v) => update('inv2_premium_over_inflation_mean', v ?? 0)}
                  />
                </Field>
                <Field
                  label="Premium volatility"
                  tip="Annual volatility of Asset 2’s premium over inflation."
                >
                  <PercentInput
                    value={config.inv2_premium_over_inflation_volatility}
                    onChange={(v) => update('inv2_premium_over_inflation_volatility', v ?? 0)}
                  />
                </Field>
                <Field
                  label="Tax model"
                  wide
                  tip="Same options as Asset 1: tax only on sale (realized) or annual tax on gains."
                >
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
                  <Field
                    label="Realized gains tax"
                    tip="Tax rate on the gain portion of Asset 2 withdrawals and rebalancing sales."
                  >
                    <PercentInput
                      value={config.inv2_realized_gains_tax_rate}
                      max={100}
                      onChange={(v) => update('inv2_realized_gains_tax_rate', v ?? 0)}
                    />
                  </Field>
                ) : (
                  <Field
                    label="Annual gains tax"
                    tip="Tax rate applied each year to Asset 2 gains."
                  >
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
              <Field
                label="Mean"
                tip="Expected annual inflation rate. Expenses and inflation-linked income grow with the simulated price level."
              >
                <PercentInput
                  value={config.inflation_rate_mean}
                  onChange={(v) => update('inflation_rate_mean', v ?? 0)}
                />
              </Field>
              <Field
                label="Volatility"
                tip="Uncertainty in annual inflation. Higher values create more variable expense and Asset 2 paths."
              >
                <PercentInput
                  value={config.inflation_rate_volatility}
                  onChange={(v) => update('inflation_rate_volatility', v ?? 0)}
                />
              </Field>
              <Field
                label="Equity–inflation corr."
                hint="−1 to 1"
                tip="Correlation between equity log-returns and inflation. 0 = independent; positive means stocks and inflation tend to move together."
              >
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
                  <Field
                    label="Monthly amount"
                    hint="today’s $"
                    tip="Income amount in today’s dollars. Inflated if indexed; otherwise fixed in nominal terms from when the stream starts."
                  >
                    <MoneyInput
                      value={stream.monthly_amount_today}
                      onChange={(v) => updateStream(i, 'monthly_amount_today', v ?? 0)}
                    />
                  </Field>
                  <Field
                    label="Starts at age"
                    tip="Age when this income becomes eligible. Payments begin at max(retirement age, start-at-age). Retirement age = current age + working years."
                  >
                    <NumberInput
                      value={stream.start_at_age}
                      min={0}
                      max={120}
                      step={0.1}
                      onChange={(v) => updateStream(i, 'start_at_age', v ?? 0)}
                    />
                  </Field>
                  <Field
                    label="Duration"
                    hint="years · blank = forever"
                    tip="How many years payments last after they begin (from max(retirement age, start-at-age)). Leave blank for indefinite."
                  >
                    <NumberInput
                      value={stream.duration_years}
                      min={0}
                      placeholder="∞"
                      onChange={(v) => updateStream(i, 'duration_years', v)}
                    />
                  </Field>
                  <Field
                    label="Tax rate"
                    tip="Income tax applied to this stream before it offsets portfolio withdrawals."
                  >
                    <PercentInput
                      value={stream.tax_rate}
                      max={100}
                      onChange={(v) => updateStream(i, 'tax_rate', v ?? 0)}
                    />
                  </Field>
                  <Field
                    label="Inflation indexed"
                    wide
                    tip="On: amount keeps real purchasing power. Off: nominal amount is locked using the price level at the stream’s start date."
                  >
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
              <Field
                label="Main simulations"
                tip="Number of Monte Carlo paths in the final run (after the search). More paths = smoother estimates, slower runtime. Prefer 1,000+."
              >
                <NumberInput
                  value={config.num_simulations_main}
                  min={1}
                  step={100}
                  onChange={(v) => update('num_simulations_main', v ?? 1)}
                />
              </Field>
              <Field
                label="Search simulations"
                tip="Paths used at each search probe. More paths reduce noise in the estimated working period."
              >
                <NumberInput
                  value={config.num_simulations_search}
                  min={1}
                  step={50}
                  onChange={(v) => update('num_simulations_search', v ?? 1)}
                />
              </Field>
              <Field
                label="Start search at"
                hint="months"
                tip="Working months where the search begins. 0 means “retire today” is tested first."
              >
                <NumberInput
                  value={config.starting_working_months_search}
                  min={0}
                  onChange={(v) => update('starting_working_months_search', v ?? 0)}
                />
              </Field>
              <Field
                label="Processes"
                hint="cores"
                tip="CPU processes for parallel path evaluation. 1 or blank-equivalent runs sequentially. Higher uses more CPU."
              >
                <NumberInput
                  value={config.num_processes}
                  min={1}
                  onChange={(v) => update('num_processes', v)}
                />
              </Field>
              <Field
                label="Seed"
                hint="blank = random"
                tip="Fixes randomness for reproducible results. Leave blank for a new random seed each run. Search and final use independent streams from this seed."
              >
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
        <Field
          label="Working months override"
          hint="leave blank to search"
          tip="Skip the search and run the final simulation with this exact whole number of working months. Leave blank to estimate the earliest month that hits the target."
        >
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
