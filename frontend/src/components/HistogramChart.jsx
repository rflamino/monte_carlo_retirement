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

function binData(values, successFlags, numBins = 60) {
  const successful = values.filter((v, i) =>
    successFlags?.length === values.length ? successFlags[i] : v > 1,
  )
  const successfulCount = successful.length
  const successRate = values.length
    ? ((successfulCount / values.length) * 100).toFixed(1)
    : '0.0'
  if (successfulCount === 0) return { bins: [], median: 0, successRate }

  const sorted = [...successful].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  const median =
    sorted.length % 2
      ? sorted[middle]
      : (sorted[middle - 1] + sorted[middle]) / 2
  const min = sorted[0]
  const max = sorted[sorted.length - 1]

  if (max <= min) {
    return {
      bins: [{ label: `$${(min / 1e6).toFixed(1)}M`, count: successfulCount, mid: min / 1e6 }],
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

  successful.forEach((v) => {
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

export default function HistogramChart({
  finalBalances,
  successFlags,
  successProbability,
  theme = 'light',
}) {
  if (!finalBalances?.length) return null

  const { bins, median, successRate } = binData(finalBalances, successFlags)
  const grid = cssVar('--chart-grid', '#e2e8f0')
  const tick = cssVar('--chart-tick', '#64748b')
  const bar = cssVar('--chart-bar', '#60a5fa')
  const medianColor = cssVar('--chart-median', '#2563eb')

  if (bins.length === 0) {
    return (
      <div className="card chart-card">
        <h3>Final Balance Distribution</h3>
        <p className="empty-chart">No successful paths to display.</p>
      </div>
    )
  }

  return (
    <div className="card chart-card" data-theme-chart={theme}>
      <h3>
        Final Balance Distribution{' '}
        <span className="chart-subtitle">
          Successful paths (including $0) · Success rate: {successProbability ?? successRate}%
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
