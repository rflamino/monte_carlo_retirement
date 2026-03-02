# Retirement Monte Carlo Simulator

A robust Python-based Monte Carlo simulation tool designed to project portfolio longevity in retirement. This tool calculates the success probability of a retirement plan based on stochastic market returns, inflation, and specific tax regimes, helping determine the minimum working months required to achieve a target success rate.

Includes a **FastAPI backend** and **React frontend** for interactive browser-based simulation and visualization.

## Features

* **Monte Carlo Simulation:** Runs thousands of scenarios (paths) to model market volatility and sequence of returns risk.
* **JSON Configuration:** Fully configurable via an external JSON file—no code changes required to test different scenarios.
* **Dual-Asset Portfolio:** Models a split between "Riskier/Equity" (Asset 1) and "Safer/Bonds" (Asset 2) with rebalancing logic.
* **Advanced Tax Modeling:** Supports both **Realized Gains** (tax on sale) and **Annual Taxation** (e.g., "Come-Cotas") systems.
* **Inflation Modeling:** Simulates inflation as a stochastic variable, affecting both expenses and asset returns.
* **Additional Income Streams:** Handles complex income scenarios like Pensions, Social Security, or Rental income with specific start dates, durations, and tax rates.
* **Web UI:** React frontend with interactive charts (trajectory percentile bands, final balance histogram) and a JSON config editor.
* **REST API:** FastAPI backend exposing simulation as a service, with Swagger docs at `/docs`.
* **CLI Mode:** Standalone CLI that generates PNG plots and log files.

## Prerequisites

* Python 3.13+
* [`uv`](https://docs.astral.sh/uv/) (recommended for dependency management and execution)
* Node.js 18+ and npm (for the frontend only)

## Project Structure

```
monte_carlo_retirement/
├── main.py            # CLI entry point
├── server.py          # FastAPI backend entry point
├── config.py          # Configuration loading & Pydantic validation
├── simulation.py      # Core Monte Carlo simulation logic
├── plotting.py        # Matplotlib plot generation (CLI mode)
├── utils.py           # Utility functions (logging, seeding)
├── constants.py       # Shared constants
├── config.json        # Default scenario configuration
├── pyproject.toml     # Python dependencies
└── frontend/          # React single-page app
    ├── package.json
    ├── vite.config.js # Dev proxy → backend on :8080
    ├── index.html
    └── src/
        ├── App.jsx
        ├── App.css
        ├── api.js
        └── components/
            ├── ConfigEditor.jsx
            ├── SummaryCard.jsx
            ├── TrajectoryChart.jsx
            └── HistogramChart.jsx
```

## Quick Start

### Option A: Web UI (backend + frontend)

```bash
# 1. Install Python dependencies
uv sync

# 2. Start the API server (port 8080)
uv run python server.py

# 3. In a second terminal, install and start the frontend (port 3000)
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** in your browser. The default config loads automatically—edit it, then click **Run Simulation**.

### Option B: CLI only

```bash
# Default config (config.json)
uv run python main.py

# Custom config file
uv run python main.py my_scenario.json
```

## API Endpoints

The FastAPI server (`server.py`) exposes the following endpoints. Interactive Swagger docs are available at `http://localhost:8080/docs`.

| Method | Path | Description |
| :----- | :--- | :---------- |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/config/default` | Returns the bundled `config.json` as a template |
| `POST` | `/api/validate` | Validates a configuration without running a simulation |
| `POST` | `/api/simulate` | Runs the full simulation and returns data for plotting |

### `POST /api/simulate` request body

```json
{
  "config": { /* same schema as config.json */ },
  "working_months_override": 180
}
```

* `config` — required. Same structure as `config.json`.
* `working_months_override` — optional. If provided, skips the search phase and runs directly with this many working months.

### `POST /api/simulate` response

| Field | Content |
| :---- | :------ |
| `summary` | Working months/years, success probability, SWR, median balances, percentiles (P1–P99) |
| `trajectory` | Year-indexed percentile arrays (P5–P95) and sample paths for time-series charts |
| `histogram` | Raw `final_balances` and `start_balances` arrays for client-side binning |

-----

## Configuration Reference (config.json)

The simulation is controlled entirely by the JSON configuration file. Below is a reference for all available parameters.

### 1\. General Scenario Settings

| Key | Type | Description |
| :--- | :--- | :--- |
| `scenario` | String | A nickname for this simulation run (e.g., "Conservative"). Used in plots and logs. |
| `retirement_years` | Integer | The duration of retirement to simulate in years (e.g., 50). |
| `target_probability` | Float | The target success rate percentage (e.g., 99.0 for 99%). |

### 2\. Current Financials (Time = 0)

| Key | Type | Description |
| :--- | :--- | :--- |
| `initial_balance` | Float | Total portfolio value today. |
| `monthly_contribution` | Float | Amount saved/invested per month while working. |
| `contribution_growth_rate_annual` | Float | Annual percentage increase in contribution (e.g., 0.04). |
| `monthly_expenses` | Float | Estimated monthly spending in retirement in today's purchasing power. |

### 3\. Investment Allocation & Returns

> **Note:** Asset 2 allocation is calculated automatically as `1 - allocation_inv1_pct`.

**Asset 1 (Equities/Risk)**

| Key | Type | Description |
| :--- | :--- | :--- |
| `allocation_inv1_pct` | Float | % of portfolio in Asset 1 (e.g., 0.60). |
| `inv1_returns_mean` | Float | Expected annual return arithmetic mean (e.g., 0.12). |
| `inv1_returns_volatility` | Float | Standard deviation of returns (e.g., 0.15). |

**Asset 2 (Fixed Income/Safe)**
*Modeled as a premium over inflation.*

| Key | Type | Description |
| :--- | :--- | :--- |
| `inv2_premium_over_inflation_mean` | Float | Return above inflation (e.g., 0.05). |
| `inv2_premium_over_inflation_volatility` | Float | Volatility of this premium. |

### 4\. Taxation Settings

**System A: Realized Gains (Tax paid only upon sale)**

  * `invX_use_realized_gains_tax_system`: **true**
  * `invX_realized_gains_tax_rate`: Tax rate on gains (e.g., 0.15).

**System B: Annual Tax (Tax deducted yearly on gains)**

  * `invX_use_realized_gains_tax_system`: **false**
  * `invX_annual_tax_on_gains_rate`: Tax rate on annual gains.

### 5\. Inflation Assumptions

| Key | Type | Description |
| :--- | :--- | :--- |
| `inflation_rate_mean` | Float | Average annual inflation (e.g., 0.062). |
| `inflation_rate_volatility` | Float | Volatility of inflation. |

### 6\. Simulation Technicals

| Key | Type | Description |
| :--- | :--- | :--- |
| `num_simulations_main` | Integer | Paths for the final run (Rec: 10000+). |
| `num_simulations_search` | Integer | Paths for the "working months" search phase. |
| `starting_working_months_search` | Integer | Start searching for retirement date from this month (0 = today). |
| `seed` | Integer/Null | Fix the random seed for reproducibility. `null` for random. |
| `num_processes` | Integer/Null | CPU cores to use. `null` for auto-detect. |

### 7\. Other Income Streams

A list of objects defining extra income (Social Security, Rental, etc.).

| Key | Type | Description |
| :--- | :--- | :--- |
| `name` | String | Label for the income. |
| `monthly_amount_today` | Float | Monthly value in today's money. |
| `start_after_retirement_years` | Integer | Years after retirement starts that income begins. |
| `duration_years` | Integer/Null | How long it lasts. `null` = indefinitely. |
| `inflation_indexed` | Boolean | `true`: Grows with inflation. `false`: Fixed nominal value. |
| `tax_rate` | Float | Income tax applied to this stream. |

-----

## Outputs

### CLI mode

After a successful run, the CLI generates:

  * **Log File:** `ret_proj_log_YYYYMMDD_HHMMSS.log` containing detailed run statistics.
  * **Histogram:** `ret_proj_[ScenarioName]_[Timestamp]_HIST.png` showing the distribution of final balances.
  * **Trajectories:** `ret_proj_[ScenarioName]_[Timestamp]_TRAJ.png` showing the median, 5th, and 95th percentile portfolio values over time.

### Web UI

The React frontend displays interactive versions of the same charts, plus a summary card with key metrics, all rendered in the browser after each simulation run.
