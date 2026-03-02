import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'

function buildChartData(trajectory) {
  const { years, percentiles, sample_paths } = trajectory
  const p5 = percentiles.p5 || []
  const p25 = percentiles.p25 || []
  const p50 = percentiles.p50 || []
  const p75 = percentiles.p75 || []
  const p95 = percentiles.p95 || []

  return years.map((year, i) => {
    const entry = {
      year,
      base: (p5[i] || 0) / 1e6,
      lower_band: ((p25[i] || 0) - (p5[i] || 0)) / 1e6,
      inner_band: ((p75[i] || 0) - (p25[i] || 0)) / 1e6,
      upper_band: ((p95[i] || 0) - (p75[i] || 0)) / 1e6,
      median: (p50[i] || 0) / 1e6,
      _p5: (p5[i] || 0) / 1e6,
      _p25: (p25[i] || 0) / 1e6,
      _p50: (p50[i] || 0) / 1e6,
      _p75: (p75[i] || 0) / 1e6,
      _p95: (p95[i] || 0) / 1e6,
    }
    if (sample_paths) {
      sample_paths.forEach((path, j) => {
        entry[`s${j}`] = (path[i] || 0) / 1e6
      })
    }
    return entry
  })
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null

  return (
    <div className="chart-tooltip">
      <p className="tooltip-title">Year {d.year}</p>
      <p style={{ color: '#ef4444' }}>P95: ${d._p95?.toFixed(2)}M</p>
      <p style={{ color: '#3b82f6' }}>P75: ${d._p75?.toFixed(2)}M</p>
      <p style={{ color: '#1d4ed8', fontWeight: 600 }}>Median: ${d._p50?.toFixed(2)}M</p>
      <p style={{ color: '#3b82f6' }}>P25: ${d._p25?.toFixed(2)}M</p>
      <p style={{ color: '#ef4444' }}>P5: ${d._p5?.toFixed(2)}M</p>
    </div>
  )
}

export default function TrajectoryChart({ trajectory, workingMonths }) {
  if (!trajectory) return null

  const data = buildChartData(trajectory)
  const retirementYear = workingMonths / 12
  const sampleKeys = trajectory.sample_paths
    ? trajectory.sample_paths.map((_, i) => `s${i}`)
    : []

  return (
    <div className="card chart-card">
      <h3>Portfolio Trajectories</h3>
      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart data={data} margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="year"
            label={{ value: 'Years from Start', position: 'insideBottom', offset: -10, fontSize: 12 }}
            tick={{ fontSize: 11 }}
          />
          <YAxis
            tickFormatter={(v) => `$${v.toFixed(0)}M`}
            tick={{ fontSize: 11 }}
            label={{ value: 'Balance ($M)', angle: -90, position: 'insideLeft', offset: 0, fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />

          <Area
            stackId="bands"
            type="monotone"
            dataKey="base"
            stroke="none"
            fill="transparent"
            fillOpacity={0}
            activeDot={false}
            legendType="none"
          />
          <Area
            stackId="bands"
            type="monotone"
            dataKey="lower_band"
            stroke="none"
            fill="#fca5a5"
            fillOpacity={0.3}
            activeDot={false}
            name="P5–P25 / P75–P95"
          />
          <Area
            stackId="bands"
            type="monotone"
            dataKey="inner_band"
            stroke="none"
            fill="#93c5fd"
            fillOpacity={0.35}
            activeDot={false}
            name="P25–P75"
          />
          <Area
            stackId="bands"
            type="monotone"
            dataKey="upper_band"
            stroke="none"
            fill="#fca5a5"
            fillOpacity={0.3}
            activeDot={false}
            legendType="none"
          />

          {sampleKeys.map((key) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke="#9ca3af"
              strokeWidth={0.7}
              dot={false}
              activeDot={false}
              legendType="none"
              opacity={0.45}
            />
          ))}

          <Line
            type="monotone"
            dataKey="median"
            stroke="#2563eb"
            strokeWidth={2.5}
            dot={false}
            name="Median (P50)"
          />

          {retirementYear > 0 && retirementYear < (data.at(-1)?.year ?? 0) && (
            <ReferenceLine
              x={retirementYear}
              stroke="#0f172a"
              strokeDasharray="6 3"
              strokeWidth={1.5}
              label={{
                value: `Retirement (yr ${retirementYear.toFixed(1)})`,
                position: 'top',
                fontSize: 11,
                fill: '#0f172a',
              }}
            />
          )}

          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
