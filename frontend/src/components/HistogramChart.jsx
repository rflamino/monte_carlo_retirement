import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { cssVar } from '../theme'

function binData(values, numBins = 60) {
  const positive = values.filter((v) => v > 1)
  if (positive.length === 0) return { bins: [], median: 0, successRate: 0 }

  const sorted = [...positive].sort((a, b) => a - b)
  const median = sorted[Math.floor(sorted.length / 2)]
  const min = sorted[0]
  const max = sorted[sorted.length - 1]
  const successRate = ((positive.length / values.length) * 100).toFixed(1)

  if (max <= min) {
    return {
      bins: [{ label: `$${(min / 1e6).toFixed(1)}M`, count: positive.length, mid: min / 1e6 }],
      median: median / 1e6,
      successRate,
    }
  }

  const width = (max - min) / numBins
  const bins = Array.from({ length: numBins }, (_, i) => ({
    start: min + i * width,
    end: min + (i + 1) * width,
    count: 0,
  }))

  positive.forEach((v) => {
    const idx = Math.min(Math.floor((v - min) / width), numBins - 1)
    bins[idx].count++
  })

  return {
    bins: bins.map((b) => ({
      label: `$${((b.start + b.end) / 2 / 1e6).toFixed(1)}M`,
      count: b.count,
      mid: (b.start + b.end) / 2 / 1e6,
    })),
    median: median / 1e6,
    successRate,
  }
}

function HistTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  return (
    <div className="chart-tooltip">
      <p className="tooltip-title">{d.label}</p>
      <p>Count: {d.count}</p>
    </div>
  )
}

export default function HistogramChart({ finalBalances, successProbability, theme = 'light' }) {
  if (!finalBalances?.length) return null

  const { bins, median, successRate } = binData(finalBalances)
  const grid = cssVar('--chart-grid', '#e2e8f0')
  const tick = cssVar('--chart-tick', '#64748b')
  const bar = cssVar('--chart-bar', '#60a5fa')
  const medianColor = cssVar('--chart-median', '#2563eb')

  if (bins.length === 0) {
    return (
      <div className="card chart-card">
        <h3>Final Balance Distribution</h3>
        <p className="empty-chart">All simulations resulted in zero balance.</p>
      </div>
    )
  }

  return (
    <div className="card chart-card" data-theme-chart={theme}>
      <h3>
        Final Balance Distribution{' '}
        <span className="chart-subtitle">
          Success rate: {successProbability ?? successRate}%
        </span>
      </h3>
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={bins} margin={{ top: 24, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={grid} />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 10, fill: tick }}
            interval={Math.max(0, Math.floor(bins.length / 10) - 1)}
            label={{
              value: 'Final Balance',
              position: 'insideBottom',
              offset: -10,
              fontSize: 12,
              fill: tick,
            }}
          />
          <YAxis
            tick={{ fontSize: 11, fill: tick }}
            label={{
              value: 'Frequency',
              angle: -90,
              position: 'insideLeft',
              offset: 0,
              fontSize: 12,
              fill: tick,
            }}
          />
          <Tooltip content={<HistTooltip />} />
          <Bar dataKey="count" fill={bar} radius={[2, 2, 0, 0]} />
          <ReferenceLine
            x={bins.reduce(
              (best, b) => (Math.abs(b.mid - median) < Math.abs(best.mid - median) ? b : best),
              bins[0],
            ).label}
            stroke={medianColor}
            strokeDasharray="6 3"
            strokeWidth={2}
            label={{
              value: `Median: $${median.toFixed(1)}M`,
              position: 'top',
              fontSize: 11,
              fill: medianColor,
            }}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
