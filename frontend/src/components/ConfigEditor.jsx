import { useState, useEffect, useRef } from 'react'
import { getDefaultConfig } from '../api'

export default function ConfigEditor({ onSimulate, loading }) {
  const [configText, setConfigText] = useState('')
  const [workingMonths, setWorkingMonths] = useState('')
  const [parseError, setParseError] = useState(null)
  const fileInput = useRef(null)

  useEffect(() => {
    getDefaultConfig()
      .then((cfg) => setConfigText(JSON.stringify(cfg, null, 2)))
      .catch(() => setConfigText('{\n  \n}'))
  }, [])

  const handleFormat = () => {
    try {
      const parsed = JSON.parse(configText)
      setConfigText(JSON.stringify(parsed, null, 2))
      setParseError(null)
    } catch (e) {
      setParseError('Invalid JSON: ' + e.message)
    }
  }

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      setConfigText(ev.target.result)
      setParseError(null)
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  const handleSimulate = () => {
    try {
      const config = JSON.parse(configText)
      setParseError(null)
      onSimulate(config, workingMonths || undefined)
    } catch (e) {
      setParseError('Invalid JSON: ' + e.message)
    }
  }

  return (
    <div className="config-editor">
      <div className="config-actions">
        <button className="btn btn-sm" onClick={handleFormat}>
          Format
        </button>
        <button className="btn btn-sm" onClick={() => fileInput.current?.click()}>
          Upload JSON
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".json"
          style={{ display: 'none' }}
          onChange={handleFileUpload}
        />
      </div>

      <textarea
        className="config-textarea"
        value={configText}
        onChange={(e) => {
          setConfigText(e.target.value)
          setParseError(null)
        }}
        spellCheck={false}
      />

      {parseError && <div className="parse-error">{parseError}</div>}

      <div className="override-row">
        <label htmlFor="wm-override">Working months override</label>
        <input
          id="wm-override"
          type="number"
          min="0"
          placeholder="Leave empty to search"
          value={workingMonths}
          onChange={(e) => setWorkingMonths(e.target.value)}
        />
      </div>

      <button
        className="btn btn-primary btn-simulate"
        onClick={handleSimulate}
        disabled={loading}
      >
        {loading ? 'Simulating\u2026' : 'Run Simulation'}
      </button>
    </div>
  )
}
