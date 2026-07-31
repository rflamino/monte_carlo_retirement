import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { cssVar } from '../theme'

function SearchTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  return (
    <div className="chart-tooltip">
      <p className="tooltip-title">{d.working_years} years ({d.working_months} mo)</p>
      <p>Success: {d.probability.toFixed(1)}%</p>
    </div>
  )
}

export default function SearchCurveChart({ searchCurve, theme = 'light' }) {
  if (!searchCurve?.points?.length) return null

  const data = [...searchCurve.points].sort(
    (a, b) => a.working_months - b.working_months
  )
  const grid = cssVar('--chart-grid', '#e2e8f0')
  const tick = cssVar('--chart-tick', '#64748b')
  const line = cssVar('--chart-median', '#2563eb')
  const targetColor = theme === 'dark' ? '#fbbf24' : '#b45309'
  const selectedColor = theme === 'dark' ? '#4ade80' : '#15803d'

  return (
    <div className="card chart-card">
      <h3>Success Probability vs Working Period</h3>
      <p className="chart-subtitle">
        Search probes: how success rate rises with more months of saving. Dashed line is
        the target; green marks the selected minimum.
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 16, right: 24, bottom: 36, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={grid} />
          <XAxis
            dataKey="working_years"
            type="number"
            label={{
              value: 'Working years',
              position: 'insideBottom',
              offset: -4,
              fontSize: 12,
              fill: tick,
            }}
            tick={{ fontSize: 11, fill: tick }}
          />
          <YAxis
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 11, fill: tick }}
            label={{
              value: 'Success probability',
              angle: -90,
              position: 'insideLeft',
              offset: 4,
              fontSize: 12,
              fill: tick,
            }}
          />
          <Tooltip content={<SearchTooltip />} />
          <Line
            type="monotone"
            dataKey="probability"
            stroke={line}
            strokeWidth={2.5}
            dot={{ r: 3, fill: line }}
            name="Search success %"
          />
          <ReferenceLine
            y={searchCurve.target_probability}
            stroke={targetColor}
            strokeDasharray="8 4"
            strokeWidth={1.5}
            label={{
              value: `Target ${searchCurve.target_probability}%`,
              position: 'insideTopRight',
              fill: targetColor,
              fontSize: 11,
            }}
          />
          <ReferenceLine
            x={Math.round((searchCurve.selected_working_months / 12) * 10) / 10}
            stroke={selectedColor}
            strokeDasharray="4 2"
            strokeWidth={1.5}
            label={{
              value: 'Selected',
              position: 'insideTopLeft',
              fill: selectedColor,
              fontSize: 11,
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, paddingTop: 8, color: tick }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
