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
    trajectory_time_points,
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
        summary, _, _, _, _, _, _ = sim.run_monte_carlo_simulations(months, 80)
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

    points = trajectory_time_points(working_months, config.retirement_years)
    assert points == pytest.approx([0.0, 1.0, 13 / 12, 25 / 12])
    assert len(points) == len(result["Trajectory"])


def test_partial_year_trajectory_keeps_equal_retirement_balance():
    """Equal values at distinct timestamps must not shift retirement samples."""
    config = _base_config(
        initial_balance=100_000.0,
        monthly_contribution=0.0,
        monthly_expenses=1_000.0,
        retirement_years=1,
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
    )
    result = RetirementMonteCarloSimulator(
        config
    )._run_single_simulation_path(working_months=13, path_seed=1)
    assert result["Trajectory"] == pytest.approx(
        [100_000.0, 100_000.0, 100_000.0, 88_000.0]
    )
    assert result["RealTrajectory"] == pytest.approx(result["Trajectory"])


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


def test_config_rejects_impossible_means_and_empty_search():
    """Lognormal gross means must be positive and search needs at least one path."""
    with pytest.raises(ValueError):
        _base_config(inv1_returns_mean=-1.0)
    with pytest.raises(ValueError):
        _base_config(inflation_rate_mean=-1.0)
    with pytest.raises(ValueError):
        _base_config(inv2_premium_over_inflation_mean=-1.0)
    with pytest.raises(ValueError):
        _base_config(num_simulations_search=0)
    with pytest.raises(ValueError):
        _base_config(seed=-1)

    valid = _base_config(seed=0)
    with pytest.raises(ValueError):
        RetirementMonteCarloSimulator(valid, main_seed_override=-1)


def test_perfect_equity_inflation_correlation_is_preserved():
    """Correlation endpoints ±1 must not silently fall back to zero."""
    positive = RetirementMonteCarloSimulator(
        _base_config(equity_inflation_correlation=1.0)
    )._draw_shock_path(100, path_seed=4)
    assert positive[:, 1] == pytest.approx(positive[:, 0])

    negative = RetirementMonteCarloSimulator(
        _base_config(equity_inflation_correlation=-1.0)
    )._draw_shock_path(100, path_seed=4)
    assert negative[:, 1] == pytest.approx(-negative[:, 0])


def test_allocation_weights_conserve_every_dollar():
    """Complementary allocation must not mint money through decimal rounding."""
    config = _base_config(
        initial_balance=100_000.0,
        allocation_inv1_pct=0.333333,
        monthly_expenses=0.0,
        retirement_years=1,
        inflation_rate_mean=0.0,
        inflation_rate_volatility=0.0,
        inv1_returns_mean=0.0,
        inv1_returns_volatility=0.0,
        inv2_premium_over_inflation_mean=0.0,
        inv2_premium_over_inflation_volatility=0.0,
    )
    assert config.allocation_inv1_pct + config.allocation_inv2_pct == pytest.approx(1.0)
    result = RetirementMonteCarloSimulator(
        config
    )._run_single_simulation_path(working_months=0, path_seed=1)
    assert result["Start Balance"] == pytest.approx(100_000.0)
    assert result["Trajectory"][0] == pytest.approx(100_000.0)


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
    summary, _, _, _, _, _, _ = sim.run_monte_carlo_simulations(
        working_months=0, num_simulations=20
    )

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
        return df, None, None, None, None, None, None

    sim.run_monte_carlo_simulations = fake_run  # type: ignore[method-assign]
    months, prob, curve = sim.find_minimum_working_months(verbose=False)
    assert months == threshold, f"expected {threshold}, got {months}"
    assert prob >= 90.0
    assert len(curve) >= 1
    assert all("working_months" in p and "probability" in p for p in curve)


def test_search_verification_handles_non_monotone_probabilities():
    """A locally isolated earlier pass is found despite a later probability dip."""
    import pandas as pd

    config = _base_config(
        target_probability=50.0,
        starting_working_months_search=0,
        num_simulations_search=400,
        seed=0,
    )
    sim = RetirementMonteCarloSimulator(config)

    def fake_run(working_months: int, num_simulations: int):
        if working_months == 4:
            success_count = 201  # 50.25% — first qualifying month
        elif working_months >= 24:
            success_count = 213  # 53.25%
        else:
            success_count = 199  # 49.75%, including month 12
        flags = [True] * success_count + [False] * (
            num_simulations - success_count
        )
        df = pd.DataFrame(
            {
                "Start Balance": [100.0] * num_simulations,
                "Final Balance": [1.0 if ok else 0.0 for ok in flags],
                "Success": flags,
                "First Year Gross Withdrawal": [1.0] * num_simulations,
                "Inflation At Retirement": [1.0] * num_simulations,
            }
        )
        return df, None, None, None, None, None, None

    sim.run_monte_carlo_simulations = fake_run  # type: ignore[method-assign]
    months, probability, _ = sim.find_minimum_working_months(verbose=False)
    assert months == 4
    assert probability == pytest.approx(50.25)


def test_income_stream_starts_at_age():
    """Pension at start_at_age begins at max(retirement_age, start_at_age)."""
    from simulation import (
        age_at_retirement_year,
        retirement_age,
        stream_payment_start_age,
        stream_payment_start_month_index,
    )

    current_age = 40.0
    working_months = 240  # 20 years → retire at 60
    assert retirement_age(current_age, working_months) == pytest.approx(60.0)
    # Eligible at 65 → payments start at 65 (5 years into retirement)
    assert stream_payment_start_age(current_age, working_months, 65.0) == pytest.approx(65.0)
    assert age_at_retirement_year(current_age, working_months, 5) == pytest.approx(65.0)
    assert stream_payment_start_month_index(
        current_age, working_months, 65.0
    ) == 60
    # Eligible at 55 but retire at 60 → payments start at retirement
    assert stream_payment_start_age(current_age, working_months, 55.0) == pytest.approx(60.0)
    assert stream_payment_start_month_index(
        current_age, working_months, 55.0
    ) == 0
    # Fractional age rounds up to the first monthly payment date.
    assert stream_payment_start_month_index(
        60.0, 0, 60.51
    ) == 7

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


def test_income_stream_fractional_age_starts_on_correct_month():
    """A pension at age 60.5 starts in month 7, not at the next yearly boundary."""
    config = _base_config(
        current_age=60.0,
        initial_balance=6_000.0,  # exactly funds the first six months
        monthly_contribution=0.0,
        monthly_expenses=1_000.0,
        retirement_years=2,
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
                "name": "Midyear pension",
                "monthly_amount_today": 1_000.0,
                "start_at_age": 60.5,
                "duration_years": None,
                "inflation_indexed": True,
                "tax_rate": 0.0,
            }
        ],
        seed=3,
    )
    sim = RetirementMonteCarloSimulator(config)
    result = sim._run_single_simulation_path(working_months=0, path_seed=4)
    assert result["Success"] is True
    assert result["Final Balance"] == pytest.approx(0.0, abs=1e-6)
    assert result["First Year Gross Withdrawal"] == pytest.approx(6_000.0)


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
    summary, _, _, _, _, _, _ = sim.run_monte_carlo_simulations(0, 5)
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

    summary, _, _, wr_pct, _, _, wr_counts = (
        sim.run_monte_carlo_simulations(0, 10)
    )
    assert wr_pct is not None and not wr_pct.empty
    assert wr_counts == [10] * config.retirement_years
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


def test_years_to_ruin_and_real_trajectory():
    """Failed paths report years-to-ruin; real traj ≈ nominal when inflation is 0."""
    config = _base_config(
        initial_balance=5_000.0,
        monthly_contribution=0.0,
        monthly_expenses=2_000.0,
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
        seed=9,
    )
    sim = RetirementMonteCarloSimulator(config)
    result = sim._run_single_simulation_path(working_months=0, path_seed=1)
    assert result["Success"] is False
    # $5k funds two full $2k months and half of month three.
    assert result["YearsToRuin"] == pytest.approx(3 / 12)
    assert len(result["RealTrajectory"]) == len(result["Trajectory"])
    for nom, real in zip(result["Trajectory"], result["RealTrajectory"]):
        assert real == pytest.approx(nom, abs=1e-6)

    summary, traj, _, _, real_traj, _, wr_counts = (
        sim.run_monte_carlo_simulations(0, 20)
    )
    assert (summary["Success"] == False).all()
    assert summary["YearsToRuin"].notna().all()
    assert real_traj is not None and traj is not None
    assert len(real_traj) == len(traj)
    assert wr_counts == [0] * config.retirement_years


def test_realized_tax_withdrawal_tracks_net_cash_and_loss_basis():
    """Tax capacity uses net proceeds; average basis survives unrealized losses."""
    config = _base_config(
        inv1_use_realized_gains_tax_system=True,
        inv1_realized_gains_tax_rate=0.20,
    )
    sim = RetirementMonteCarloSimulator(config)

    # All $100 is gain. Liquidating it pays $20 tax, so a $90 net target is
    # underfunded even though gross market value exceeds the requested cash.
    balance, basis, gross, net = sim._calculate_withdrawal_and_update(
        100.0, 0.0, 90.0, True, 0.20
    )
    assert balance == pytest.approx(0.0)
    assert basis == pytest.approx(0.0)
    assert gross == pytest.approx(100.0)
    assert net == pytest.approx(80.0)

    # At a loss, selling half the shares removes half the $100 basis ($50),
    # not merely the $40 proceeds.
    balance, basis, gross, net = sim._calculate_withdrawal_and_update(
        80.0, 100.0, 40.0, True, 0.20
    )
    assert balance == pytest.approx(40.0)
    assert basis == pytest.approx(50.0)
    assert gross == pytest.approx(40.0)
    assert net == pytest.approx(40.0)


def test_rebalance_is_tax_aware_and_preserves_asset_cost_basis():
    """Rebalancing pays sale tax and moves only the sold/purchased basis."""
    config = _base_config(
        allocation_inv1_pct=0.60,
        inv1_use_realized_gains_tax_system=True,
        inv1_realized_gains_tax_rate=0.10,
        inv2_use_realized_gains_tax_system=True,
        inv2_realized_gains_tax_rate=0.10,
    )
    sim = RetirementMonteCarloSimulator(config)

    bal1, cb1, bal2, cb2 = sim._rebalance_portfolio(
        bal_inv1=70.0,
        cb_inv1=50.0,
        bal_inv2=30.0,
        cb_inv2=30.0,
    )

    total = bal1 + bal2
    assert bal1 / total == pytest.approx(0.60, abs=1e-10)
    assert bal2 / total == pytest.approx(0.40, abs=1e-10)
    assert total < 100.0  # realized-gain tax was paid

    gross_sale = 70.0 - bal1
    basis_removed = 50.0 * (gross_sale / 70.0)
    taxable_gain = gross_sale - basis_removed
    tax_paid = taxable_gain * 0.10
    assert cb1 == pytest.approx(50.0 - basis_removed)
    assert cb2 == pytest.approx(30.0 + gross_sale - tax_paid)


def test_annual_tax_excludes_internal_rebalancing_transfers():
    """A zero-return asset owes no annual gains tax on transfers received."""
    common = {
        "initial_balance": 100_000.0,
        "monthly_contribution": 0.0,
        "monthly_expenses": 0.0,
        "retirement_years": 1,
        "allocation_inv1_pct": 0.50,
        "inv1_returns_mean": 0.0,
        "inv1_returns_volatility": 0.0,
        "inv1_use_realized_gains_tax_system": False,
        "inv1_realized_gains_tax_rate": 0.0,
        "inv2_premium_over_inflation_mean": 1.0,
        "inv2_premium_over_inflation_volatility": 0.0,
        "inv2_use_realized_gains_tax_system": True,
        "inv2_realized_gains_tax_rate": 0.0,
        "inflation_rate_mean": 0.0,
        "inflation_rate_volatility": 0.0,
        "seed": 11,
    }
    no_tax = _base_config(**common, inv1_annual_tax_on_gains_rate=0.0)
    full_tax = _base_config(**common, inv1_annual_tax_on_gains_rate=1.0)

    result_no_tax = RetirementMonteCarloSimulator(
        no_tax
    )._run_single_simulation_path(working_months=12, path_seed=1)
    result_full_tax = RetirementMonteCarloSimulator(
        full_tax
    )._run_single_simulation_path(working_months=12, path_seed=1)

    # Inv1 itself earned 0%. Monthly transfers into it came from Inv2 gains and
    # must not be mislabeled as Inv1 taxable gains.
    assert result_full_tax["Start Balance"] == pytest.approx(
        result_no_tax["Start Balance"], rel=1e-10
    )
    assert result_full_tax["Final Balance"] == pytest.approx(
        result_no_tax["Final Balance"], rel=1e-10
    )


def test_retirement_does_not_split_annual_tax_period():
    """A partial working year remains in the same absolute 12-month tax period."""
    config = _base_config(
        initial_balance=100.0,
        monthly_contribution=0.0,
        monthly_expenses=0.0,
        retirement_years=1,
        allocation_inv1_pct=1.0,
        inv1_returns_mean=0.12,
        inv1_returns_volatility=0.0,
        inv1_use_realized_gains_tax_system=False,
        inv1_annual_tax_on_gains_rate=0.50,
        inv2_premium_over_inflation_mean=0.0,
        inv2_premium_over_inflation_volatility=0.0,
        inv2_use_realized_gains_tax_system=False,
        inv2_annual_tax_on_gains_rate=0.0,
        inflation_rate_mean=0.0,
        inflation_rate_volatility=0.0,
        seed=12,
    )
    result = RetirementMonteCarloSimulator(
        config
    )._run_single_simulation_path(working_months=13, path_seed=1)

    monthly_gross = 1.12 ** (1 / 12)
    after_month_12_tax = 112.0 - (112.0 - 100.0) * 0.50
    expected_retirement_balance = after_month_12_tax * monthly_gross
    assert result["Start Balance"] == pytest.approx(
        expected_retirement_balance, rel=1e-10
    )


def test_api_outcomes_keep_success_flags_and_zero_balance_median():
    """Dashboard histogram cohort must match the backend successful-path cohort."""
    import pandas as pd
    from server import SimulationResponse, _build_result

    config = _base_config(
        num_simulations_main=3,
        retirement_years=1,
        other_income_streams=[],
    )
    summary = pd.DataFrame(
        {
            "Start Balance": [100.0, 100.0, 100.0],
            "Final Balance": [0.0, 50.0, 25.0],
            "Success": [True, True, False],
            "YearsToRuin": [float("nan"), float("nan"), 0.5],
            "First Year Gross Withdrawal": [0.0, 10.0, 10.0],
            "First Year Real Gross Withdrawal": [0.0, 10.0, 10.0],
            "Inflation At Retirement": [1.0, 1.0, 1.0],
        }
    )

    class FakeSimulator:
        def run_monte_carlo_simulations(self, **_kwargs):
            return summary, None, None, None, None, None, None

    result = _build_result(
        config,
        FakeSimulator(),  # type: ignore[arg-type]
        required_w_months=0,
        search_curve=[],
    )
    SimulationResponse.model_validate(result)

    assert result["summary"]["success_probability"] == pytest.approx(66.67)
    assert result["summary"]["median_final_balance_successful"] == pytest.approx(
        25.0
    )
    assert result["histogram"]["final_balances"] == [0.0, 50.0, 25.0]
    assert result["histogram"]["success_flags"] == [True, True, False]
    assert result["ruin_histogram"]["failure_count"] == 1
    assert result["ruin_histogram"]["years_to_ruin"] == [0.5]


def test_api_preserves_exact_fractional_timeline():
    """API formatting must not shift 13-month retirement markers to 1.1 years."""
    from server import SimulationResponse, _build_result

    config = _base_config(
        num_simulations_main=2,
        num_processes=1,
        retirement_years=1,
        monthly_expenses=0.0,
        seed=5,
    )
    simulator = RetirementMonteCarloSimulator(config)
    result = _build_result(
        config,
        simulator,
        required_w_months=13,
        search_curve=[
            {
                "working_months": 13,
                "working_years": 1.1,
                "probability": 100.0,
            }
        ],
    )
    SimulationResponse.model_validate(result)

    retirement_year = 13 / 12
    assert result["trajectory"]["years"] == pytest.approx(
        [0.0, 1.0, retirement_year, retirement_year + 1]
    )
    assert result["withdrawal_rate"]["years"][0] == pytest.approx(
        retirement_year
    )
    assert result["reference_lines"][0]["year"] == pytest.approx(
        retirement_year
    )
    assert result["summary"]["working_period_is_estimate"] is True


def test_streaming_endpoint_emits_schema_valid_result():
    """The dashboard SSE endpoint must validate and deliver the same response schema."""
    import json

    from fastapi.testclient import TestClient
    from server import SimulationResponse, app

    config = _base_config(
        num_simulations_main=2,
        num_processes=1,
        retirement_years=1,
        monthly_expenses=0.0,
        seed=8,
    )
    body = {
        "config": config.model_dump(by_alias=True),
        "working_months_override": 13,
    }

    with TestClient(app) as client:
        response = client.post("/api/simulate/stream", json=body)

    assert response.status_code == 200
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    result_events = [event for event in events if event.get("type") == "result"]
    assert len(result_events) == 1
    parsed = SimulationResponse.model_validate(result_events[0]["data"])
    assert parsed.summary.required_working_months == 13
    assert parsed.summary.working_period_is_estimate is False

