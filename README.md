# Retirement Monte Carlo Simulator

A robust Python-based Monte Carlo simulation tool designed to project portfolio longevity in retirement. This tool calculates the success probability of a retirement plan based on stochastic market returns, inflation, and specific tax regimes, helping determine the minimum working months required to achieve a target success rate.

## Features

* **Monte Carlo Simulation:** Runs thousands of scenarios (paths) to model market volatility and sequence of returns risk.
* **JSON Configuration:** Fully configurable via an external JSON file—no code changes required to test different scenarios.
* **Dual-Asset Portfolio:** Models a split between "Riskier/Equity" (Asset 1) and "Safer/Bonds" (Asset 2) with rebalancing logic.
* **Advanced Tax Modeling:** Supports both **Realized Gains** (tax on sale) and **Annual Taxation** (e.g., "Come-Cotas") systems.
* **Inflation Modeling:** Simulates inflation as a stochastic variable, affecting both expenses and asset returns.
* **Additional Income Streams:** Handles complex income scenarios like Pensions, Social Security, or Rental income with specific start dates, durations, and tax rates.
* **Visualization:** Automatically generates:
    * **Histogram:** Distribution of final portfolio balances.
    * **Trajectory Plot:** Percentile bands (5th-95th) showing portfolio value over time.

## Prerequisites

* The following Python libraries:
    * `numpy`
    * `pandas`
    * `matplotlib`
    * `pydantic`
    * `loguru`

## Project Structure

The project is modularized into the following components:

*   `main.py`: Entry point for the application.
*   `config.py`: Configuration loading and validation.
*   `simulation.py`: Core Monte Carlo simulation logic.
*   `plotting.py`: Visualization generation.
*   `utils.py`: Utility functions (logging, seeding).
*   `constants.py`: Project constants.

## Usage

### 1. Default Execution
By default, the script looks for a file named `monte_carlo_retirement.json` in the same directory.

```bash
uv run python main.py
```

## 2. Custom Configuration

You can pass a specific JSON configuration file as an argument. This is useful for comparing different scenarios (e.g., `aggressive.json` vs `conservative.json`).

```bash
uv run python main.py my_scenario_config.json
````

## Configuration Manual (config.json)

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

After a successful run, the script generates:

  * **Log File:** `ret_proj_log_YYYYMMDD_HHMMSS.log` containing detailed run statistics.
  * **Histogram:** `ret_proj_[ScenarioName]_[Timestamp]_HIST.png` showing the distribution of final balances.
  * **Trajectories:** `ret_proj_[ScenarioName]_[Timestamp]_TRAJ.png` showing the median, 5th, and 95th percentile portfolio values over time.

-----

