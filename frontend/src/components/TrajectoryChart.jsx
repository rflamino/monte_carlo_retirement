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
import { cssVar } from '../theme'

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
      <p style={{ color: cssVar('--chart-p95', '#ef4444') }}>P95: ${d._p95?.toFixed(2)}M</p>
      <p style={{ color: cssVar('--chart-p75', '#3b82f6') }}>P75: ${d._p75?.toFixed(2)}M</p>
      <p style={{ color: cssVar('--chart-p50', '#1d4ed8'), fontWeight: 600 }}>
        Median: ${d._p50?.toFixed(2)}M
      </p>
      <p style={{ color: cssVar('--chart-p75', '#3b82f6') }}>P25: ${d._p25?.toFixed(2)}M</p>
      <p style={{ color: cssVar('--chart-p95', '#ef4444') }}>P5: ${d._p5?.toFixed(2)}M</p>
    </div>
  )
}

const REF_LINE_COLORS_LIGHT = [
  '#15803d', '#7c3aed', '#c2410c', '#0e7490', '#be123c', '#4338ca',
]
const REF_LINE_COLORS_DARK = [
  '#4ade80', '#c4b5fd', '#fb923c', '#22d3ee', '#fb7185', '#a5b4fc',
]

/** Merge events that land within this many years so one line serves them. */
const YEAR_NEAR_THRESHOLD = 1.25

function prepareReferenceMarkers(lines, maxYear, refColors, retirementColor) {
  const visible = [...lines]
    .filter((rl) => rl.year > 0 && rl.year < maxYear)
    .sort((a, b) => a.year - b.year)

  const groups = []
  for (const rl of visible) {
    const existing = groups.find((g) => Math.abs(g.year - rl.year) < YEAR_NEAR_THRESHOLD)
    if (existing) {
      existing.items.push(rl)
      // Keep mean year so a cluster sits between nearby events
      const n = existing.items.length
      existing.year = existing.items.reduce((s, x) => s + x.year, 0) / n
    } else {
      groups.push({ year: rl.year, items: [rl] })
    }
  }

  return groups.map((g, i) => {
    const isRetirement = g.items.some((x) => x.name === 'Retirement Starts')
    const stroke = isRetirement ? retirementColor : refColors[i % refColors.length]
    const yearLabel = Math.round(g.year * 10) / 10
    return {
      year: yearLabel,
      stroke,
      isRetirement,
      marker: String(i + 1),
      items: g.items.map((x) => ({
        name: x.name,
        year: Math.round(x.year * 10) / 10,
        stroke,
      })),
    }
  })
}

/** Short numeric badge on the line — full names live in the legend chips. */
function RefMarkerLabel({ viewBox, marker, fill }) {
  const x = viewBox?.x ?? 0
  const y = viewBox?.y ?? 0
  return (
    <g transform={`translate(${x}, ${y - 8})`}>
      <circle r={9} fill={fill} opacity={0.95} />
      <text
        textAnchor="middle"
        dominantBaseline="central"
        fill="#fff"
        fontSize={10}
        fontWeight={700}
      >
        {marker}
      </text>
    </g>
  )
}

export default function TrajectoryChart({ trajectory, referenceLines = [], theme = 'light' }) {
  if (!trajectory) return null

  const data = buildChartData(trajectory)
  const maxYear = data.at(-1)?.year ?? 0
  const sampleKeys = trajectory.sample_paths
    ? trajectory.sample_paths.map((_, i) => `s${i}`)
    : []

  const grid = cssVar('--chart-grid', '#e2e8f0')
  const tick = cssVar('--chart-tick', '#64748b')
  const median = cssVar('--chart-median', '#2563eb')
  const bandOuter = cssVar('--chart-band-outer', '#fca5a5')
  const bandInner = cssVar('--chart-band-inner', '#93c5fd')
  const sample = cssVar('--chart-sample', '#9ca3af')
  const refDefault = cssVar('--chart-ref', '#0f172a')
  const refColors = theme === 'dark' ? REF_LINE_COLORS_DARK : REF_LINE_COLORS_LIGHT
  const markers = prepareReferenceMarkers(referenceLines, maxYear, refColors, refDefault)

  // Flatten for the chip legend (one chip per original event, shared color if clustered)
  const legendChips = markers.flatMap((m) =>
    m.items.map((item) => ({
      ...item,
      marker: m.marker,
    }))
  )

  return (
    <div className="card chart-card trajectory-chart-card">
      <h3>Portfolio Trajectories</h3>
      {legendChips.length > 0 && (
        <ul className="traj-ref-legend" aria-label="Timeline markers">
          {legendChips.map((chip, i) => (
            <li key={`${chip.name}-${chip.year}-${i}`} className="traj-ref-chip">
              <span className="traj-ref-badge" style={{ background: chip.stroke }}>
                {chip.marker}
              </span>
              <span className="traj-ref-name">{chip.name}</span>
              <span className="traj-ref-year">yr {chip.year}</span>
            </li>
          ))}
        </ul>
      )}
      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart data={data} margin={{ top: 20, right: 24, bottom: 48, left: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={grid} />
          <XAxis
            dataKey="year"
            type="number"
            domain={[0, maxYear]}
            label={{
              value: 'Years from Start',
              position: 'insideBottom',
              offset: -8,
              fontSize: 12,
              fill: tick,
            }}
            tick={{ fontSize: 11, fill: tick }}
          />
          <YAxis
            tickFormatter={(v) => `$${v.toFixed(0)}M`}
            tick={{ fontSize: 11, fill: tick }}
            label={{
              value: 'Balance ($M)',
              angle: -90,
              position: 'insideLeft',
              offset: 0,
              fontSize: 12,
              fill: tick,
            }}
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
            fill={bandOuter}
            fillOpacity={theme === 'dark' ? 0.25 : 0.3}
            activeDot={false}
            name="P5–P25 / P75–P95"
          />
          <Area
            stackId="bands"
            type="monotone"
            dataKey="inner_band"
            stroke="none"
            fill={bandInner}
            fillOpacity={theme === 'dark' ? 0.3 : 0.35}
            activeDot={false}
            name="P25–P75"
          />
          <Area
            stackId="bands"
            type="monotone"
            dataKey="upper_band"
            stroke="none"
            fill={bandOuter}
            fillOpacity={theme === 'dark' ? 0.25 : 0.3}
            activeDot={false}
            legendType="none"
          />

          {sampleKeys.map((key) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={sample}
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
            stroke={median}
            strokeWidth={2.5}
            dot={false}
            name="Median (P50)"
          />

          {markers.map((m) => (
            <ReferenceLine
              key={`ref-${m.marker}-${m.year}`}
              x={m.year}
              stroke={m.stroke}
              strokeDasharray={m.isRetirement ? '6 3' : '4 2'}
              strokeWidth={1.5}
              ifOverflow="extendDomain"
              label={<RefMarkerLabel marker={m.marker} fill={m.stroke} />}
            />
          ))}

          <Legend
            wrapperStyle={{ fontSize: 11, paddingTop: 20, paddingBottom: 4, color: tick }}
            layout="horizontal"
            align="center"
            verticalAlign="bottom"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
