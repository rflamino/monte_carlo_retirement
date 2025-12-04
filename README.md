# Retirement Monte Carlo Simulator

A robust Python-based Monte Carlo simulation tool designed to project portfolio longevity in retirement. This tool calculates the success probability of a retirement plan based on stochastic market returns, inflation, and specific tax regimes, helping determine the minimum working months required to achieve a target success rate.

## 🚀 Features

* **Monte Carlo Simulation:** Runs thousands of scenarios (paths) to model market volatility and sequence of returns risk.
* **JSON Configuration:** Fully configurable via an external JSON file—no code changes required to test different scenarios.
* **Dual-Asset Portfolio:** Models a split between "Riskier/Equity" (Asset 1) and "Safer/Bonds" (Asset 2) with rebalancing logic.
* **Advanced Tax Modeling:** Supports both **Realized Gains** (tax on sale) and **Annual Taxation** (e.g., "Come-Cotas") systems.
* **Inflation Modeling:** Simulates inflation as a stochastic variable, affecting both expenses and asset returns.
* **Additional Income Streams:** Handles complex income scenarios like Pensions, Social Security, or Rental income with specific start dates, durations, and tax rates.
* **Visualization:** Automatically generates:
    * **Histogram:** Distribution of final portfolio balances.
    * **Trajectory Plot:** Percentile bands (5th-95th) showing portfolio value over time.

## 📋 Prerequisites

* Python 3.8 or higher.
* The following Python libraries:
    * `numpy`
    * `pandas`
    * `matplotlib`
    * `pydantic`

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/rflamino/monte_carlo_retirement.git](https://github.com/rflamino/monte_carlo_retirement.git)
    cd monte_carlo_retirement
    ```

2.  **Install dependencies:**
    You can install the required packages directly via pip:
    ```bash
    pip install numpy pandas matplotlib pydantic
    ```

## 💻 Usage

### 1. Default Execution
By default, the script looks for a file named `monte_carlo_retirement.json` in the same directory.

```bash
python monte_carlo_retirement.py
