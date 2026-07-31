# Retirement Monte Carlo Simulator

A Python Monte Carlo tool that projects portfolio longevity in retirement. It finds the minimum working months needed to hit a target success rate under stochastic returns, inflation, and tax rules.

Includes a **FastAPI backend** and **React frontend** for interactive simulation and charts, plus a CLI that writes logs and PNG plots.

## Features

* **Monte Carlo simulation** — Thousands of paths for market volatility and sequence-of-returns risk.
* **JSON configuration** — Scenarios are fully driven by config files (validated with Pydantic).
* **Dual-asset portfolio** — Equity (Inv1) + inflation-linked safer asset (Inv2), with rebalancing and tax on sales/gains.
* **Lognormal returns** — Arithmetic mean/vol in config; converted so \(E[\text{annual gross}] = 1 + \text{mean}\). Optional equity–inflation correlation (Cholesky).
* **Monthly inflation accrual** — Price level compounds monthly (no full-year bump on partial years).
* **Age-based other income** — Pensions, Social Security, rent, etc. via `current_age` + `start_at_age`.
* **Success = funded spending** — A path succeeds if every retirement year’s expenses are covered by portfolio withdrawals and/or after-tax other income. Ending at \$0 is allowed when income alone covers spending.
* **Bracket + bisection search** — Finds minimum working months; search and final runs use independent seed streams (no selection bias). Common random numbers across candidates.
* **Web UI** — Form + JSON config editor, dark mode, field tip balloons, live SSE progress, trajectory bands with numbered timeline markers, final-balance histogram.
* **REST API** — FastAPI with Swagger at `/docs`.
* **CLI** — PNG plots and detailed logs.

## Prerequisites

* Python 3.13+
* [`uv`](https://docs.astral.sh/uv/) (recommended)
* Node.js 18+ and npm (frontend only)

## Project Structure

```
monte_carlo_retirement/
├── config.json            # Default scenario
├── jorge.json             # Example scenario
├── pyproject.toml
├── tests/                 # Regression tests (pytest)
├── backend/
│   ├── main.py            # CLI entry point
│   ├── server.py          # FastAPI entry point
│   ├── config.py          # Pydantic models & validation
│   ├── simulation.py      # Core Monte Carlo engine
│   ├── plotting.py        # Matplotlib (CLI)
│   ├── utils.py
│   └── constants.py
└── frontend/
    ├── package.json
    ├── vite.config.js     # Dev proxy → :8080
    └── src/
        ├── App.jsx
        ├── theme.js       # Light/dark theme
        ├── api.js
        └── components/
            ├── ConfigEditor.jsx
            ├── SummaryCard.jsx
            ├── TrajectoryChart.jsx
            ├── HistogramChart.jsx
            └── SimulationProgress.jsx
```

## Quick Start

### Option A: Web UI (backend + frontend)

```bash
# 1. Install Python dependencies
uv sync

# 2. Start the API server (port 8080)
uv run python backend/server.py

# 3. In a second terminal, install and start the frontend (port 3000)
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**. The default config loads automatically—edit via the form (or JSON), then **Run Simulation**. Progress streams live during search and the final run.

### Option B: CLI only

```bash
uv run python backend/main.py
uv run python backend/main.py my_scenario.json
```

### Tests

```bash
uv sync --group dev
uv run pytest tests/ -v
```

## How the model works

### Timeline

1. **Accumulation** — `working_months` of contributions, returns, inflation, and tax.
2. **Retirement** — `retirement_years` of spending. Net portfolio withdrawal each year =  
   \(\max(0,\ \text{expenses} - \text{after-tax other income})\).

**Retirement age** = `current_age + working_months / 12`.

### Success definition

A path is **successful** if spending is fully funded in every retirement year (from the portfolio and/or other income). It may end with a \$0 balance if later income (e.g. a pension) covers expenses. Failure means a year where a withdrawal was still needed and the portfolio could not provide it.

Reported success probability uses this flag (not “final balance > 0”).

### First-year withdrawal rate

Median over paths of  
`(first-year gross portfolio withdrawal) / (balance at retirement start)`,  
both nominal at the retirement date. Other income that offsets spending reduces this rate.

### Other income timing

For each stream, payments begin at:

\[
\max(\text{retirement age},\ \texttt{start\_at\_age})
\]

Duration is counted from that payment-start age. Indexed streams track the simulated price level from T=0; non-indexed streams lock a nominal amount at payment start.

**Breaking change:** `start_after_retirement_years` was replaced by `start_at_age`. Old configs must be updated (set `current_age` and each stream’s `start_at_age`).

### Search algorithm

1. Coarse bracket (stepping working months until target probability is met).
2. Bisection for the minimum months that still hit the target.
3. Final detailed run with `num_simulations_main` on an independent seed stream.

-----

## API Endpoints

Interactive docs: `http://localhost:8080/docs`.

| Method | Path | Description |
| :----- | :--- | :---------- |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/config/default` | Bundled `config.json` template |
| `POST` | `/api/validate` | Validate config without simulating |
| `POST` | `/api/simulate` | Full simulation → plot-ready JSON |
| `POST` | `/api/simulate/stream` | Same, with SSE progress then result |

### `POST /api/simulate` request body

```json
{
  "config": { /* same schema as config.json */ },
  "working_months_override": 180
}
```

* `config` — required.
* `working_months_override` — optional; skips search and runs with this many working months.

### Response

| Field | Content |
| :---- | :------ |
| `summary` | Working months/years, **retirement age**, success probability, first-year withdrawal rate, median balances, percentiles (P1–P99) |
| `trajectory` | Year-indexed percentiles (P5–P95) and sample paths |
| `withdrawal_rate` | Real annual withdrawal rate (% of start-of-retirement balance, inflation-adjusted) percentiles by year from T=0 |
| `histogram` | `final_balances` / `start_balances` for client binning |
| `reference_lines` | `{ name, year }` markers (retirement start, income streams); years are from T=0 on the trajectory grid |

### SSE events (`/api/simulate/stream`)

| Event type | Content |
| :--------- | :------ |
| `phase` | `{ phase: "search" \| "final_sim", message }` |
| `search_iter` | `{ iteration, working_months, probability, target, sim_count }` |
| `search_complete` | `{ working_months, working_years, probability }` |
| `result` | `{ data: <full response> }` |
| `error` | `{ message }` |

-----

## Configuration Reference (`config.json`)

### 1. General scenario settings

| Key | Type | Description |
| :--- | :--- | :--- |
| `scenario` | String | Nickname for plots and logs. |
| `current_age` | Float | Age at T=0 (required). Used with income `start_at_age`. |
| `retirement_years` | Integer | Years of retirement to simulate. |
| `target_probability` | Float | Target success rate % (e.g. `97.0`). |

### 2. Current financials (T = 0)

| Key | Type | Description |
| :--- | :--- | :--- |
| `initial_balance` | Float | Portfolio value today. |
| `monthly_contribution` | Float | Monthly savings while working. |
| `contribution_growth_rate_annual` | Float | Annual growth of contributions (e.g. `0.04`). |
| `monthly_expenses` | Float | Retirement spending in today’s purchasing power. |

### 3. Investment allocation & returns

Asset 2 weight = `1 - allocation_inv1_pct`.

**Asset 1 (equities / risk)** — Lognormal. Config mean/vol are **arithmetic** annual; the engine converts so \(E[\text{gross}] = 1 + \text{mean}\).

| Key | Type | Description |
| :--- | :--- | :--- |
| `allocation_inv1_pct` | Float | Weight in Asset 1 (e.g. `0.60`). |
| `inv1_returns_mean` | Float | Expected annual arithmetic return. |
| `inv1_returns_volatility` | Float | Annual std. dev. (typical equities ~0.15; very low values understate risk). |

**Asset 2 (safer)** — Inflation × real premium (both lognormal, multiplied monthly).

| Key | Type | Description |
| :--- | :--- | :--- |
| `inv2_premium_over_inflation_mean` | Float | Expected real premium. |
| `inv2_premium_over_inflation_volatility` | Float | Premium volatility. |

### 4. Taxation

**Realized gains** (tax on sale / withdrawal of gains):

* `invX_use_realized_gains_tax_system`: `true`
* `invX_realized_gains_tax_rate`: rate on gains (e.g. `0.10`)

**Annual tax on gains** (e.g. come-cotas style):

* `invX_use_realized_gains_tax_system`: `false`
* `invX_annual_tax_on_gains_rate`: annual rate on gains

### 5. Inflation

Accrues **monthly**. Expenses and income are priced at the **start** of each retirement year.

| Key | Type | Description |
| :--- | :--- | :--- |
| `inflation_rate_mean` | Float | Average annual inflation. |
| `inflation_rate_volatility` | Float | Inflation volatility. |
| `equity_inflation_correlation` | Float | Corr. of equity and inflation log-shocks (default `0.0`). |

### 6. Simulation technicals

| Key | Type | Description |
| :--- | :--- | :--- |
| `num_simulations_main` | Integer | Paths in the final run (e.g. 1000+; 10000+ for production). |
| `num_simulations_search` | Integer | Paths per search probe (bracket + bisection). Small values make the chosen months noisier vs the final run. |
| `starting_working_months_search` | Integer | Where the search starts (`0` = today). |
| `seed` | Integer / `null` | Reproducibility; `null` = random. Search and final use separate streams. |
| `num_processes` | Integer / `null` | Parallel workers; `null` or `1` = sequential. |

### 7. Other income streams

List of objects. Example:

```json
{
  "name": "State Pension",
  "monthly_amount_today": 4000.0,
  "start_at_age": 65.0,
  "duration_years": null,
  "inflation_indexed": true,
  "tax_rate": 0.275
}
```

| Key | Type | Description |
| :--- | :--- | :--- |
| `name` | String | Label (charts, logs, reference markers). |
| `monthly_amount_today` | Float | Monthly amount in T=0 real terms. |
| `start_at_age` | Float | Age when the stream becomes eligible. |
| `duration_years` | Integer / `null` | Years of payments after they begin; `null` = indefinite. |
| `inflation_indexed` | Boolean | `true`: tracks inflation from T=0. `false`: fixed nominal at payment start. |
| `tax_rate` | Float | Flat tax on this stream (`0`–`1`). |

-----

## Outputs

### CLI

* **Log:** `ret_proj_log_YYYYMMDD_HHMMSS.log`
* **Histogram:** `ret_proj_[Scenario]_[Timestamp]_HIST.png`
* **Trajectories:** `ret_proj_[Scenario]_[Timestamp]_TRAJ.png`

### Web UI

* **Config editor** — Form sections with tips, Load/Save/Reset, Form ↔ JSON, dark mode toggle.
* **Summary card** — Working period, retirement age, success %, target, first-year withdrawal rate, median balances, percentiles.
* **Portfolio trajectories** — Percentile bands, median, sample paths; numbered reference markers with a chip legend (retirement, income streams).
* **Withdrawal rate over time** — Inflation-adjusted portfolio withdrawal as % of the balance at retirement start (Trinity/Bengen basis), with a **4% reference line**. Constant-real spending is flat; the series falls when pensions or other income begin.
* **Final balance histogram** — Outcome distribution with median line.
* **Live progress** — Search iterations and probability vs target.

## Notes & caveats

* Equity volatility in sample configs may be intentionally low for experimentation; raise toward ~15% for more realistic sequence risk.
* Spending is modeled as constant real `monthly_expenses` (no healthcare ramp or spending smile).
* Other-income tax is a flat rate, not progressive brackets.
* Prefer a larger `num_simulations_search` if the final-run success rate often undershoots the search result.
