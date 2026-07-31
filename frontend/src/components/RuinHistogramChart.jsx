import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { cssVar } from '../theme'

function binRuinYears(years, maxYearHint) {
  if (!years?.length) return []
  const maxY = Math.max(...years, maxYearHint || 0)
  const bins = Array.from({ length: Math.ceil(maxY) }, (_, i) => ({
    year: i + 1,
    label: `${i + 1}`,
    count: 0,
  }))
  years.forEach((y) => {
    const idx = Math.min(Math.max(Math.ceil(y) - 1, 0), bins.length - 1)
    bins[idx].count++
  })
  // Drop trailing empty bins for readability
  let last = bins.length - 1
  while (last > 0 && bins[last].count === 0) last--
  return bins.slice(0, last + 1)
}

function RuinTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  return (
    <div className="chart-tooltip">
      <p className="tooltip-title">Year {d.year} of retirement</p>
      <p>Failed paths: {d.count}</p>
    </div>
  )
}

export default function RuinHistogramChart({ ruinHistogram, theme = 'light' }) {
  if (!ruinHistogram) return null

  const { years_to_ruin: years, failure_count: failures, total_paths: total } = ruinHistogram
  const failPct = total > 0 ? ((failures / total) * 100).toFixed(1) : '0.0'
  const grid = cssVar('--chart-grid', '#e2e8f0')
  const tick = cssVar('--chart-tick', '#64748b')
  const bar = theme === 'dark' ? '#fb7185' : '#e11d48'

  if (failures === 0) {
    return (
      <div className="card chart-card">
        <h3>Years to Ruin (Failed Paths)</h3>
        <p className="chart-subtitle">
          No failed paths in the final run ({total} simulations) — every path funded
          spending for the full horizon.
        </p>
      </div>
    )
  }

  const bins = binRuinYears(years)

  return (
    <div className="card chart-card">
      <h3>Years to Ruin (Failed Paths)</h3>
      <p className="chart-subtitle">
        Among the {failures} failures ({failPct}% of {total} paths), when the portfolio
        could no longer cover spending (years into retirement).
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={bins} margin={{ top: 12, right: 16, bottom: 36, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={grid} />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: tick }}
            label={{
              value: 'Years into retirement',
              position: 'insideBottom',
              offset: -4,
              fontSize: 12,
              fill: tick,
            }}
          />
          <YAxis
            allowDecimals={false}
            tick={{ fontSize: 11, fill: tick }}
            label={{
              value: 'Failed paths',
              angle: -90,
              position: 'insideLeft',
              offset: 4,
              fontSize: 12,
              fill: tick,
            }}
          />
          <Tooltip content={<RuinTooltip />} />
          <Bar dataKey="count" fill={bar} name="Failures" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
