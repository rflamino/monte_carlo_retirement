"""Regression tests for simulation correctness fixes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import Config
from constants import MONTHS_PER_YEAR, SMALL_EPSILON
from simulation import (
    RetirementMonteCarloSimulator,
    arithmetic_to_log_params,
    median_first_year_withdrawal_rate,
)


def _base_config(**overrides) -> Config:
    data = {
        "scenario": "test",
        "initial_balance": 500_000.0,
        "monthly_contribution": 0.0,
        "contribution_growth_rate_annual": 0.0,
        "monthly_expenses": 2_000.0,
        "current_age": 40.0,
        "retirement_years": 10,
        "allocation_inv1_pct": 0.6,
        "inv1_returns_mean": 0.08,
        "inv1_returns_volatility": 0.15,
        "inv1_annual_tax_on_gains_rate": 0.0,
        "inv1_realized_gains_tax_rate": 0.0,
        "inv1_use_realized_gains_tax_system": False,
        "inv2_premium_over_inflation_mean": 0.02,
        "inv2_premium_over_inflation_volatility": 0.01,
        "inv2_annual_tax_on_gains_rate": 0.0,
        "inv2_realized_gains_tax_rate": 0.0,
        "inv2_use_realized_gains_tax_system": False,
        "inflation_rate_mean": 0.03,
        "inflation_rate_volatility": 0.01,
        "equity_inflation_correlation": 0.0,
        "num_simulations_main": 50,
        "num_simulations_search": 40,
        "target_probability": 80.0,
        "starting_working_months_search": 0,
        "seed": 42,
        "num_processes": 1,
        "other_income_streams": [],
    }
    data.update(overrides)
    return Config(**data)


def test_success_probability_non_decreasing_in_working_months():
    """More working months must not reduce success probability (common random numbers)."""
    config = _base_config(
        initial_balance=100_000.0,
        monthly_contribution=3_000.0,
        monthly_expenses=5_000.0,
        retirement_years=30,
        inv1_returns_mean=0.10,
        inv1_returns_volatility=0.12,
        inflation_rate_mean=0.04,
        inflation_rate_volatility=0.015,
        num_simulations_main=80,
        seed=123,
    )
    sim = RetirementMonteCarloSimulator(config)
    sim.use_search_seeds()

    probs = []
    for months in range(0, 61, 6):
        summary, _, _, _ = sim.run_monte_carlo_simulations(months, 80)
        probs.append(sim._success_probability(summary))

    for i in range(1, len(probs)):
        assert probs[i] + 1e-9 >= probs[i - 1], (
            f"Probability fell from {probs[i - 1]:.2f}% at "
            f"{(i - 1) * 6} months to {probs[i]:.2f}% at {i * 6} months: {probs}"
        )


def test_partial_year_inflation_accrual():
    """With zero vol, inflation at retirement equals (1+mean)^(months/12)."""
    mean = 0.06
    config = _base_config(
        inflation_rate_mean=mean,
        inflation_rate_volatility=0.0,
        inv1_returns_volatility=0.0,
        inv2_premium_over_inflation_volatility=0.0,
        inv1_returns_mean=0.0,
        inv2_premium_over_inflation_mean=0.0,
        monthly_expenses=0.0,
        retirement_years=1,
        seed=7,
    )
    sim = RetirementMonteCarloSimulator(config)
    working_months = 13  # partial year — previously over-applied a full second year
    result = sim._run_single_simulation_path(working_months, path_seed=99)
    expected = (1.0 + mean) ** (working_months / MONTHS_PER_YEAR)
    actual = result["Inflation At Retirement"]
    assert abs(actual - expected) < 1e-9, f"expected {expected}, got {actual}"


def test_mean_realised_annual_return_matches_config():
    """Over a large sample, mean compounded annual equity return ≈ inv1_returns_mean."""
    mean = 0.12
    vol = 0.15
    mu_log, sigma_log = arithmetic_to_log_params(mean, vol)

    rng = np.random.default_rng(0)
    n_years = 50_000
    # One annual gross return per draw: exp(mu + sigma * z)
    z = rng.standard_normal(n_years)
    annual_gross = np.exp(mu_log + sigma_log * z)
    realised_mean = float(annual_gross.mean() - 1.0)
    assert abs(realised_mean - mean) < 0.005, (
        f"realised mean {realised_mean:.4f} vs config {mean}"
    )

    # Also check monthly compounding of the simulator's monthly factor
    n_months = 12 * 20_000
    z_m = rng.standard_normal(n_months)
    monthly_gross = np.exp(
        mu_log / MONTHS_PER_YEAR + sigma_log / math.sqrt(MONTHS_PER_YEAR) * z_m
    )
    # Compound each year of 12 months
    yearly = monthly_gross.reshape(-1, 12).prod(axis=1)
    realised_monthly_compound = float(yearly.mean() - 1.0)
    assert abs(realised_monthly_compound - mean) < 0.01, (
        f"monthly-compounded mean {realised_monthly_compound:.4f} vs config {mean}"
    )


def test_withdrawal_rate_with_zero_inflation():
    """With zero inflation and no other income, rate ≈ annual_expenses / start_balance."""
    monthly_expenses = 1_000.0
    initial = 200_000.0
    config = _base_config(
        initial_balance=initial,
        monthly_contribution=0.0,
        monthly_expenses=monthly_expenses,
        retirement_years=5,
        inflation_rate_mean=0.0,
        inflation_rate_volatility=0.0,
        inv1_returns_mean=0.0,
        inv1_returns_volatility=0.0,
        inv2_premium_over_inflation_mean=0.0,
        inv2_premium_over_inflation_volatility=0.0,
        inv1_use_realized_gains_tax_system=False,
        inv1_annual_tax_on_gains_rate=0.0,
        inv2_use_realized_gains_tax_system=False,
        inv2_annual_tax_on_gains_rate=0.0,
        seed=1,
        num_simulations_main=20,
    )
    sim = RetirementMonteCarloSimulator(config)
    sim.use_final_seeds()
    summary, _, _, _ = sim.run_monte_carlo_simulations(working_months=0, num_simulations=20)

    annual = monthly_expenses * MONTHS_PER_YEAR
    expected_rate = (annual / initial) * 100.0
    swr = median_first_year_withdrawal_rate(summary)
    assert abs(swr - expected_rate) < 0.5, f"SWR {swr:.3f} vs expected {expected_rate:.3f}"

    # Per-path check: gross withdrawal should equal annual expenses (no tax drag)
    for _, row in summary.iterrows():
        if row["Start Balance"] > SMALL_EPSILON:
            assert abs(row["First Year Gross Withdrawal"] - annual) < 1.0


def test_bisection_finds_true_minimum():
    """Bisection returns the true minimum against a synthetic monotone step function."""
    # Monkey-patch run_monte_carlo_simulations to a deterministic step at 37 months.
    threshold = 37
    config = _base_config(
        target_probability=90.0,
        starting_working_months_search=0,
        num_simulations_search=10,
        seed=0,
    )
    sim = RetirementMonteCarloSimulator(config)

    def fake_run(working_months: int, num_simulations: int):
        import pandas as pd

        # Success iff working_months >= threshold
        bal = 1.0 if working_months >= threshold else 0.0
        ok = working_months >= threshold
        df = pd.DataFrame(
            {
                "Start Balance": [100.0] * num_simulations,
                "Final Balance": [bal] * num_simulations,
                "Success": [ok] * num_simulations,
                "First Year Gross Withdrawal": [1.0] * num_simulations,
                "Inflation At Retirement": [1.0] * num_simulations,
            }
        )
        return df, None, None, None

    sim.run_monte_carlo_simulations = fake_run  # type: ignore[method-assign]
    months, prob = sim.find_minimum_working_months(verbose=False)
    assert months == threshold, f"expected {threshold}, got {months}"
    assert prob >= 90.0


def test_income_stream_starts_at_age():
    """Pension at start_at_age begins at max(retirement_age, start_at_age)."""
    from simulation import (
        age_at_retirement_year,
        retirement_age,
        stream_payment_start_age,
    )

    current_age = 40.0
    working_months = 240  # 20 years → retire at 60
    assert retirement_age(current_age, working_months) == pytest.approx(60.0)
    # Eligible at 65 → payments start at 65 (5 years into retirement)
    assert stream_payment_start_age(current_age, working_months, 65.0) == pytest.approx(65.0)
    assert age_at_retirement_year(current_age, working_months, 5) == pytest.approx(65.0)
    # Eligible at 55 but retire at 60 → payments start at retirement
    assert stream_payment_start_age(current_age, working_months, 55.0) == pytest.approx(60.0)

    # Path-level: zero returns/inflation, expenses covered only by pension after age 65
    config = _base_config(
        current_age=40.0,
        initial_balance=0.0,
        monthly_contribution=0.0,
        monthly_expenses=1000.0,
        retirement_years=10,
        inflation_rate_mean=0.0,
        inflation_rate_volatility=0.0,
        inv1_returns_mean=0.0,
        inv1_returns_volatility=0.0,
        inv2_premium_over_inflation_mean=0.0,
        inv2_premium_over_inflation_volatility=0.0,
        inv1_use_realized_gains_tax_system=False,
        inv1_annual_tax_on_gains_rate=0.0,
        inv2_use_realized_gains_tax_system=False,
        inv2_annual_tax_on_gains_rate=0.0,
        other_income_streams=[
            {
                "name": "Pension",
                "monthly_amount_today": 1000.0,
                "start_at_age": 65.0,
                "duration_years": None,
                "inflation_indexed": True,
                "tax_rate": 0.0,
            }
        ],
        seed=1,
        num_simulations_main=5,
    )
    # Fund enough to cover expenses for years 60–65 before pension starts
    config = config.model_copy(update={"initial_balance": 80_000.0})
    sim = RetirementMonteCarloSimulator(config)
    result = sim._run_single_simulation_path(working_months=240, path_seed=1)
    # Pension covers expenses from age 65 onward → survive with remaining principal
    assert result["Final Balance"] > 0

    # Without pension, same setup should deplete (or end much lower)
    config_no_pension = config.model_copy(update={"other_income_streams": []})
    sim2 = RetirementMonteCarloSimulator(config_no_pension)
    result2 = sim2._run_single_simulation_path(working_months=240, path_seed=1)
    assert result["Final Balance"] > result2["Final Balance"]


def test_pension_covers_after_portfolio_depleted():
    """
    Path succeeds when portfolio hits $0 before pension, then pension funds spending.
    Success is not Final Balance > 0 — living on income alone is allowed.
    """
    config = _base_config(
        current_age=60.0,
        initial_balance=12_000.0,  # exactly 1 year of $1k/mo expenses
        monthly_contribution=0.0,
        monthly_expenses=1_000.0,
        retirement_years=10,
        inflation_rate_mean=0.0,
        inflation_rate_volatility=0.0,
        inv1_returns_mean=0.0,
        inv1_returns_volatility=0.0,
        inv2_premium_over_inflation_mean=0.0,
        inv2_premium_over_inflation_volatility=0.0,
        inv1_use_realized_gains_tax_system=False,
        inv1_annual_tax_on_gains_rate=0.0,
        inv2_use_realized_gains_tax_system=False,
        inv2_annual_tax_on_gains_rate=0.0,
        other_income_streams=[
            {
                "name": "Pension",
                "monthly_amount_today": 1_000.0,
                "start_at_age": 61.0,  # after first retirement year
                "duration_years": None,
                "inflation_indexed": True,
                "tax_rate": 0.0,
            }
        ],
        seed=1,
    )
    sim = RetirementMonteCarloSimulator(config)
    # Retire immediately (age 60); deplete year 0; pension from age 61
    result = sim._run_single_simulation_path(working_months=0, path_seed=1)
    assert result["Success"] is True
    assert result["Final Balance"] == pytest.approx(0.0, abs=1e-6)

    # Without pension, same depleting portfolio fails
    config_no = config.model_copy(update={"other_income_streams": []})
    sim2 = RetirementMonteCarloSimulator(config_no)
    result2 = sim2._run_single_simulation_path(working_months=0, path_seed=1)
    assert result2["Success"] is False

    # Summary success probability uses Success, not Final Balance > 0
    sim.use_final_seeds()
    summary, _, _, _ = sim.run_monte_carlo_simulations(0, 5)
    assert sim._success_probability(summary) == pytest.approx(100.0)
    assert (summary["Final Balance"] <= SMALL_EPSILON).all()


def test_withdrawal_rate_trajectory_matches_first_year():
    """Year-0 real WR equals First Year Gross Withdrawal / Start Balance."""
    monthly_expenses = 1_000.0
    initial = 200_000.0
    config = _base_config(
        initial_balance=initial,
        monthly_contribution=0.0,
        monthly_expenses=monthly_expenses,
        retirement_years=5,
        inflation_rate_mean=0.0,
        inflation_rate_volatility=0.0,
        inv1_returns_mean=0.0,
        inv1_returns_volatility=0.0,
        inv2_premium_over_inflation_mean=0.0,
        inv2_premium_over_inflation_volatility=0.0,
        inv1_use_realized_gains_tax_system=False,
        inv1_annual_tax_on_gains_rate=0.0,
        inv2_use_realized_gains_tax_system=False,
        inv2_annual_tax_on_gains_rate=0.0,
        seed=1,
    )
    sim = RetirementMonteCarloSimulator(config)
    result = sim._run_single_simulation_path(working_months=0, path_seed=1)
    wr = result["WithdrawalRateTrajectory"]
    assert len(wr) == 5
    expected = (result["First Year Gross Withdrawal"] / result["Start Balance"]) * 100.0
    assert wr[0] == pytest.approx(expected, abs=1e-6)
    # Flat expenses, zero inflation → constant real rate each year
    assert wr[1] == pytest.approx(wr[0], abs=1e-6)

    summary, _, _, wr_pct = sim.run_monte_carlo_simulations(0, 10)
    assert wr_pct is not None and not wr_pct.empty
    assert abs(wr_pct.iloc[0][0.50] - expected) < 0.5
    swr = median_first_year_withdrawal_rate(summary)
    assert abs(swr - wr_pct.iloc[0][0.50]) < 0.5


def test_real_withdrawal_rate_flat_with_deterministic_inflation():
    """Constant real spending → real WR stays flat even when inflation compounds."""
    monthly_expenses = 1_000.0
    initial = 240_000.0  # 5% of start ≈ annual expenses
    config = _base_config(
        initial_balance=initial,
        monthly_contribution=0.0,
        monthly_expenses=monthly_expenses,
        retirement_years=8,
        inflation_rate_mean=0.06,
        inflation_rate_volatility=0.0,
        inv1_returns_mean=0.06,  # keep portfolio alive; tax off
        inv1_returns_volatility=0.0,
        inv2_premium_over_inflation_mean=0.0,
        inv2_premium_over_inflation_volatility=0.0,
        inv1_use_realized_gains_tax_system=False,
        inv1_annual_tax_on_gains_rate=0.0,
        inv2_use_realized_gains_tax_system=False,
        inv2_annual_tax_on_gains_rate=0.0,
        seed=2,
    )
    sim = RetirementMonteCarloSimulator(config)
    result = sim._run_single_simulation_path(working_months=0, path_seed=3)
    wr = result["WithdrawalRateTrajectory"]
    assert result["Success"] is True
    # Real rate should match year-0 and not drift with inflation
    for rate in wr:
        assert rate == pytest.approx(wr[0], abs=1e-4)
    assert wr[0] == pytest.approx(5.0, abs=0.05)

