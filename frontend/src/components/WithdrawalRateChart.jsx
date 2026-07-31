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

const REF_LINE_COLORS_LIGHT = [
  '#15803d', '#7c3aed', '#c2410c', '#0e7490', '#be123c', '#4338ca',
]
const REF_LINE_COLORS_DARK = [
  '#4ade80', '#c4b5fd', '#fb923c', '#22d3ee', '#fb7185', '#a5b4fc',
]

function buildChartData(withdrawalRate) {
  const { years, percentiles } = withdrawalRate
  const p5 = percentiles.p5 || []
  const p25 = percentiles.p25 || []
  const p50 = percentiles.p50 || []
  const p75 = percentiles.p75 || []
  const p95 = percentiles.p95 || []

  return years.map((year, i) => {
    const lo = p5[i]
    const midLo = p25[i]
    const med = p50[i]
    const midHi = p75[i]
    const hi = p95[i]
    if (med == null || Number.isNaN(med)) {
      return { year, median: null }
    }
    return {
      year,
      base: lo ?? 0,
      lower_band: Math.max(0, (midLo ?? lo ?? 0) - (lo ?? 0)),
      inner_band: Math.max(0, (midHi ?? med) - (midLo ?? med)),
      upper_band: Math.max(0, (hi ?? midHi ?? med) - (midHi ?? med)),
      median: med,
      _p5: lo,
      _p25: midLo,
      _p50: med,
      _p75: midHi,
      _p95: hi,
    }
  })
}

function WrTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d || d._p50 == null) return null
  return (
    <div className="chart-tooltip">
      <p className="tooltip-title">Year {d.year}</p>
      <p style={{ color: cssVar('--chart-p95', '#ef4444') }}>
        P95: {d._p95?.toFixed(2)}%
      </p>
      <p style={{ color: cssVar('--chart-p75', '#3b82f6') }}>
        P75: {d._p75?.toFixed(2)}%
      </p>
      <p style={{ color: cssVar('--chart-p50', '#1d4ed8'), fontWeight: 600 }}>
        Median: {d._p50?.toFixed(2)}%
      </p>
      <p style={{ color: cssVar('--chart-p75', '#3b82f6') }}>
        P25: {d._p25?.toFixed(2)}%
      </p>
      <p style={{ color: cssVar('--chart-p95', '#ef4444') }}>
        P5: {d._p5?.toFixed(2)}%
      </p>
    </div>
  )
}

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

function prepareMarkers(lines, minYear, maxYear, refColors, retirementColor) {
  const visible = [...lines]
    .filter((rl) => rl.year >= minYear && rl.year <= maxYear)
    .sort((a, b) => a.year - b.year)

  return visible.map((rl, i) => {
    const isRetirement = rl.name === 'Retirement Starts'
    return {
      year: rl.year,
      name: rl.name,
      marker: String(i + 1),
      stroke: isRetirement ? retirementColor : refColors[i % refColors.length],
      isRetirement,
    }
  })
}

export default function WithdrawalRateChart({
  withdrawalRate,
  referenceLines = [],
  theme = 'light',
}) {
  if (!withdrawalRate?.years?.length) return null

  const data = buildChartData(withdrawalRate).filter((d) => d.median != null)
  if (data.length === 0) return null

  const minYear = data[0].year
  const maxYear = data.at(-1).year
  const grid = cssVar('--chart-grid', '#e2e8f0')
  const tick = cssVar('--chart-tick', '#64748b')
  const median = cssVar('--chart-median', '#2563eb')
  const bandOuter = cssVar('--chart-band-outer', '#fca5a5')
  const bandInner = cssVar('--chart-band-inner', '#93c5fd')
  const refDefault = cssVar('--chart-ref', '#0f172a')
  const fourPct = theme === 'dark' ? '#fbbf24' : '#b45309'
  const refColors = theme === 'dark' ? REF_LINE_COLORS_DARK : REF_LINE_COLORS_LIGHT
  const markers = prepareMarkers(referenceLines, minYear, maxYear, refColors, refDefault)

  return (
    <div className="card chart-card">
      <h3>Withdrawal Rate Over Time</h3>
      <p className="chart-subtitle">
        Inflation-adjusted portfolio withdrawal ÷ balance at retirement start (Trinity /
        Bengen basis). A constant-real draw is flat; the dashed line marks the classic 4%.
        Rate falls when other income offsets spending.
      </p>
      {markers.length > 0 && (
        <ul className="traj-ref-legend" aria-label="Timeline markers">
          {markers.map((m) => (
            <li key={`${m.name}-${m.year}`} className="traj-ref-chip">
              <span className="traj-ref-badge" style={{ background: m.stroke }}>
                {m.marker}
              </span>
              <span className="traj-ref-name">{m.name}</span>
              <span className="traj-ref-year">yr {m.year}</span>
            </li>
          ))}
        </ul>
      )}
      <ResponsiveContainer width="100%" height={360}>
        <ComposedChart data={data} margin={{ top: 20, right: 24, bottom: 40, left: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={grid} />
          <XAxis
            dataKey="year"
            type="number"
            domain={[minYear, maxYear]}
            label={{
              value: 'Years from Start',
              position: 'insideBottom',
              offset: -4,
              fontSize: 12,
              fill: tick,
            }}
            tick={{ fontSize: 11, fill: tick }}
          />
          <YAxis
            tickFormatter={(v) => `${v.toFixed(0)}%`}
            tick={{ fontSize: 11, fill: tick }}
            label={{
              value: 'Real withdrawal rate',
              angle: -90,
              position: 'insideLeft',
              offset: 4,
              fontSize: 12,
              fill: tick,
            }}
          />
          <Tooltip content={<WrTooltip />} />

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

          <Line
            type="monotone"
            dataKey="median"
            stroke={median}
            strokeWidth={2.5}
            dot={false}
            connectNulls={false}
            name="Median (real)"
          />

          <ReferenceLine
            y={4}
            stroke={fourPct}
            strokeDasharray="8 4"
            strokeWidth={1.5}
            ifOverflow="extendDomain"
            label={{
              value: '4% rule',
              position: 'insideTopRight',
              fill: fourPct,
              fontSize: 11,
            }}
          />

          {markers.map((m) => (
            <ReferenceLine
              key={`wr-ref-${m.marker}-${m.year}`}
              x={m.year}
              stroke={m.stroke}
              strokeDasharray={m.isRetirement ? '6 3' : '4 2'}
              strokeWidth={1.5}
              ifOverflow="extendDomain"
              label={<RefMarkerLabel marker={m.marker} fill={m.stroke} />}
            />
          ))}

          <Legend
            wrapperStyle={{ fontSize: 11, paddingTop: 12, color: tick }}
            layout="horizontal"
            align="center"
            verticalAlign="bottom"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
