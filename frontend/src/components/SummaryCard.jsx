const fmt = (v) =>
  v == null
    ? '—'
    : v.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })

function Metric({ label, value, accent }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className={'metric-value' + (accent ? ` accent-${accent}` : '')}>{value}</span>
    </div>
  )
}

export default function SummaryCard({ summary }) {
  if (!summary) return null

  const s = summary
  const probColor = s.success_probability >= s.target_probability ? 'green' : 'red'

  return (
    <div className="card summary-card">
      <h3>Simulation Summary</h3>
      <div className="metrics-grid">
        <Metric
          label="Working Period"
          value={`${s.required_working_months} mo (${s.required_working_years} yr)`}
        />
        <Metric
          label="Success Probability"
          value={`${s.success_probability}%`}
          accent={probColor}
        />
        <Metric label="Target Probability" value={`${s.target_probability}%`} />
        <Metric label="SWR" value={s.swr != null ? `${s.swr}%` : '—'} />
        <Metric label="Median Start Balance" value={fmt(s.median_start_balance)} />
        <Metric
          label="Median Final Balance (succ.)"
          value={fmt(s.median_final_balance_successful)}
        />
      </div>

      <details className="percentiles-details">
        <summary>Final Balance Percentiles</summary>
        <div className="percentiles-grid">
          {Object.entries(s.final_balance_percentiles).map(([key, val]) => (
            <div key={key} className="percentile-item">
              <span className="percentile-key">{key.toUpperCase()}</span>
              <span className="percentile-val">{fmt(val)}</span>
            </div>
          ))}
        </div>
      </details>
    </div>
  )
}
