export default function SimulationProgress({ phase, iterations }) {
  const latest = iterations.at(-1)
  const pct = latest ? Math.min(100, (latest.probability / latest.target) * 100) : 0

  return (
    <div className="progress-panel">
      <div className="spinner" />

      <h2>{phase === 'search' ? 'Searching for retirement date\u2026' : 'Running final simulation\u2026'}</h2>

      {phase === 'search' && latest && (
        <>
          <div className="progress-current">
            <span className="progress-iter">Iteration {latest.iteration}</span>
            <span className="progress-months">
              Testing <strong>{latest.working_months}</strong> months ({latest.working_years} yr)
            </span>
          </div>

          <div className="progress-bar-container">
            <div className="progress-bar-track">
              <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
              <div className="progress-bar-target" />
            </div>
            <div className="progress-bar-labels">
              <span>0%</span>
              <span className="progress-achieved">
                {latest.probability}%
              </span>
              <span className="progress-target-label">
                Target {latest.target}%
              </span>
            </div>
          </div>

          {iterations.length > 0 && (
            <div className="progress-history">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Months</th>
                    <th>Years</th>
                    <th>Probability</th>
                    <th>Sims</th>
                  </tr>
                </thead>
                <tbody>
                  {iterations.map((it) => (
                    <tr key={`${it.iteration}-${it.working_months}`}>
                      <td>{it.iteration}</td>
                      <td>{it.working_months}</td>
                      <td>{it.working_years}</td>
                      <td className={it.probability >= it.target ? 'prob-met' : 'prob-miss'}>
                        {it.probability}%
                      </td>
                      <td>{it.sim_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {phase === 'final_sim' && (
        <p className="progress-sub">This usually takes a few seconds&hellip;</p>
      )}
    </div>
  )
}
