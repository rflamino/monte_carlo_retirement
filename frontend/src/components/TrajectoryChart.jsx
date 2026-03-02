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

const REF_LINE_COLORS = [
  '#16a34a', '#9333ea', '#ea580c', '#0891b2', '#be123c', '#4f46e5',
]

const YEAR_NEAR_THRESHOLD = 0.6

function groupReferenceLinesByYear(lines) {
  const groups = []
  const sorted = [...lines].sort((a, b) => a.year - b.year)
  for (const rl of sorted) {
    const existing = groups.find((g) => Math.abs(g.year - rl.year) < YEAR_NEAR_THRESHOLD)
    if (existing) {
      existing.names.push(rl.name)
    } else {
      groups.push({ year: rl.year, names: [rl.name] })
    }
  }
  return groups.map((g) => ({
    year: Math.round(g.year * 10) / 10,
    label: g.names.length > 1 ? `${g.names.join(', ')} (yr ${g.year.toFixed(1)})` : `${g.names[0]} (yr ${g.year.toFixed(1)})`,
    isRetirement: g.names.some((n) => n === 'Retirement Starts'),
    index: groups.indexOf(g),
  }))
}

export default function TrajectoryChart({ trajectory, referenceLines = [] }) {
  if (!trajectory) return null

  const data = buildChartData(trajectory)
  const maxYear = data.at(-1)?.year ?? 0
  const groupedLines = groupReferenceLinesByYear(referenceLines)
  const sampleKeys = trajectory.sample_paths
    ? trajectory.sample_paths.map((_, i) => `s${i}`)
    : []

  return (
    <div className="card chart-card trajectory-chart-card">
      <h3>Portfolio Trajectories</h3>
      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart data={data} margin={{ top: 32, right: 24, bottom: 48, left: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="year"
            type="number"
            domain={[0, maxYear]}
            label={{ value: 'Years from Start', position: 'insideBottom', offset: -8, fontSize: 12 }}
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

          {groupedLines.map((g, i) => {
            if (g.year <= 0 || g.year >= maxYear) return null
            const stroke = g.isRetirement ? '#0f172a' : REF_LINE_COLORS[i % REF_LINE_COLORS.length]
            return (
              <ReferenceLine
                key={`ref-${g.year}-${i}`}
                x={g.year}
                stroke={stroke}
                strokeDasharray={g.isRetirement ? '6 3' : '4 2'}
                strokeWidth={1.5}
                ifOverflow="extendDomain"
                label={{
                  value: g.label,
                  position: 'top',
                  fontSize: 10,
                  fill: stroke,
                }}
              />
            )
          })}

          <Legend
            wrapperStyle={{ fontSize: 11, paddingTop: 20, paddingBottom: 4 }}
            layout="horizontal"
            align="center"
            verticalAlign="bottom"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
