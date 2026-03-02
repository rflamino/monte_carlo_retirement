import { useState, useCallback } from 'react'
import ConfigEditor from './components/ConfigEditor'
import SummaryCard from './components/SummaryCard'
import TrajectoryChart from './components/TrajectoryChart'
import HistogramChart from './components/HistogramChart'
import { runSimulation } from './api'

export default function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [elapsed, setElapsed] = useState(null)

  const handleSimulate = useCallback(async (config, workingMonthsOverride) => {
    setLoading(true)
    setError(null)
    setResults(null)
    setElapsed(null)
    const t0 = performance.now()
    try {
      const data = await runSimulation(config, workingMonthsOverride)
      setResults(data)
      setElapsed(((performance.now() - t0) / 1000).toFixed(1))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>MC Retirement</h1>
          <span className="subtitle">Monte Carlo Simulator</span>
        </div>
        <ConfigEditor onSimulate={handleSimulate} loading={loading} />
        {error && <div className="error-banner">{error}</div>}
      </aside>

      <main className="content">
        {!results && !loading && (
          <div className="empty-state">
            <div className="empty-icon">&#x1F4CA;</div>
            <h2>No simulation results yet</h2>
            <p>Configure your scenario and click &ldquo;Run Simulation&rdquo; to begin.</p>
          </div>
        )}

        {loading && (
          <div className="loading-state">
            <div className="spinner" />
            <h2>Running simulation&hellip;</h2>
            <p>This may take a while depending on the number of simulations and search range.</p>
          </div>
        )}

        {results && (
          <div className="results">
            <div className="results-header">
              <h2>{results.scenario}</h2>
              {elapsed && <span className="elapsed">Completed in {elapsed}s</span>}
            </div>
            <SummaryCard summary={results.summary} />
            {results.trajectory && (
              <TrajectoryChart
                trajectory={results.trajectory}
                workingMonths={results.summary.required_working_months}
              />
            )}
            <HistogramChart
              finalBalances={results.histogram.final_balances}
              successProbability={results.summary.success_probability}
            />
          </div>
        )}
      </main>
    </div>
  )
}
