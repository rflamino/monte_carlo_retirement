import math
import multiprocessing
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from config import Config
from constants import MONTHS_PER_YEAR, SMALL_EPSILON
from utils import _generate_seed_from_timestamp


def arithmetic_to_log_params(mean: float, vol: float) -> Tuple[float, float]:
    """
    Convert arithmetic annual mean/vol to lognormal parameters so that
    E[annual gross return] == 1 + mean.
    """
    if vol <= 0:
        # Degenerate: deterministic growth
        return math.log(max(1.0 + mean, SMALL_EPSILON)), 0.0
    one_plus_mean = 1.0 + mean
    if one_plus_mean <= 0:
        # Pathological mean <= -100%; clamp for numerical safety
        one_plus_mean = SMALL_EPSILON
    sigma_log = math.sqrt(math.log(1.0 + (vol**2) / (one_plus_mean**2)))
    mu_log = math.log(one_plus_mean) - 0.5 * sigma_log**2
    return mu_log, sigma_log


def retirement_age(current_age: float, working_months: int) -> float:
    """Age at the start of retirement given age at T=0 and months worked."""
    return current_age + working_months / MONTHS_PER_YEAR


def stream_payment_start_age(
    current_age: float, working_months: int, start_at_age: float
) -> float:
    """
    Age when income payments actually begin during the retirement phase.
    Eligible from start_at_age, but only paid after retirement starts.
    """
    return max(retirement_age(current_age, working_months), float(start_at_age))


def age_at_retirement_year(
    current_age: float, working_months: int, year_num: int
) -> float:
    """Age at the start of retirement year ``year_num`` (0 = first retirement year)."""
    return retirement_age(current_age, working_months) + year_num


def years_from_t0_to_age(current_age: float, target_age: float) -> float:
    """Years from simulation start until reaching ``target_age`` (may be 0 if already past)."""
    return max(0.0, float(target_age) - float(current_age))


def median_first_year_withdrawal_rate(summary_df: pd.DataFrame) -> float:
    """
    Median per-path first-year gross withdrawal / start-of-retirement balance.
    Both terms are nominal and dated to the retirement date.
    """
    if summary_df.empty:
        return float("nan")
    start = summary_df["Start Balance"]
    withdraw = summary_df["First Year Gross Withdrawal"]
    valid = start > SMALL_EPSILON
    if not valid.any():
        return float("nan")
    rates = (withdraw[valid] / start[valid]) * 100.0
    return float(rates.median())


def trajectory_retirement_year_index(working_months: int) -> int:
    """
    Year-grid index where the retirement-start balance is recorded.
    Matches padding/trajectory logic: ceil(working_months / 12) for working_months > 0.
    """
    if working_months <= 0:
        return 0
    return (working_months + MONTHS_PER_YEAR - 1) // MONTHS_PER_YEAR


class RetirementMonteCarloSimulator:
    """
    A Monte Carlo simulator for retirement planning.

    Simulates portfolio performance over a working accumulation phase and a retirement
    decumulation phase, taking into account inflation, taxes, contributions, expenses,
    and other variables.
    """

    def __init__(self, params_model: Config, main_seed_override: Optional[int] = None):
        self.params_model = params_model.model_copy(deep=True)

        if main_seed_override is not None:
            self.main_seed = main_seed_override
        elif self.params_model.seed is not None:
            self.main_seed = self.params_model.seed
        else:
            self.main_seed = _generate_seed_from_timestamp()

        # Independent seed streams: search vs final run (avoids selection bias).
        seed_seq = np.random.SeedSequence(self.main_seed)
        self._search_seed_seq, self._final_seed_seq = seed_seq.spawn(2)
        self._stream_name = "final"
        self._active_seed_seq = self._final_seed_seq
        # Cache path seeds per (stream, n) so common random numbers are reused
        # across working-month candidates without advancing the SeedSequence.
        self._path_seed_cache: Dict[Tuple[str, int], List[int]] = {}

        p = self.params_model
        self._inv1_mu_log, self._inv1_sigma_log = arithmetic_to_log_params(
            p.inv1_returns_mean, p.inv1_returns_volatility
        )
        self._inf_mu_log, self._inf_sigma_log = arithmetic_to_log_params(
            p.inflation_rate_mean, p.inflation_rate_volatility
        )
        self._inv2_prem_mu_log, self._inv2_prem_sigma_log = arithmetic_to_log_params(
            p.inv2_premium_over_inflation_mean,
            p.inv2_premium_over_inflation_volatility,
        )

        # Correlation matrix over (equity, inflation, inv2 premium); Cholesky factor.
        rho = p.equity_inflation_correlation
        corr = np.array(
            [
                [1.0, rho, 0.0],
                [rho, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        # Ensure positive-definite for |rho| near 1
        try:
            self._chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            logger.warning(
                f"Correlation matrix not PD (rho={rho}); falling back to zero correlation."
            )
            self._chol = np.eye(3)

        logger.info(
            f"Simulator initialized for scenario '{self.params_model.Nickname}' "
            f"with main seed: {self.main_seed}"
        )

    def use_search_seeds(self) -> None:
        """Use the search seed stream for subsequent simulation batches."""
        self._stream_name = "search"
        self._active_seed_seq = self._search_seed_seq

    def use_final_seeds(self) -> None:
        """Use the independent final-run seed stream."""
        self._stream_name = "final"
        self._active_seed_seq = self._final_seed_seq

    def _path_seeds(self, num_simulations: int) -> List[int]:
        """
        Derive a fixed list of path seeds from the active SeedSequence.
        Same seed list is reused across working-month candidates (common random numbers).
        """
        cache_key = (self._stream_name, num_simulations)
        if cache_key not in self._path_seed_cache:
            # Spawn once and cache; do not re-spawn on subsequent calls.
            children = self._active_seed_seq.spawn(num_simulations)
            self._path_seed_cache[cache_key] = [
                int(c.generate_state(1)[0]) for c in children
            ]
        return self._path_seed_cache[cache_key]

    def _calculate_withdrawal_and_update(
        self,
        bal_inv: float,
        cb_inv: float,
        net_withdrawal_target_for_inv: float,
        use_real_tax: bool,
        real_tax_rate: float,
    ) -> Tuple[float, float, float]:
        """
        Calculates the gross withdrawal needed to meet a net target, considering taxes,
        and updates the cost basis.
        """
        final_gross_withdrawal = net_withdrawal_target_for_inv
        principal_component_of_withdrawal = net_withdrawal_target_for_inv

        if (
            use_real_tax
            and real_tax_rate > 0
            and net_withdrawal_target_for_inv > SMALL_EPSILON
            and bal_inv > SMALL_EPSILON
        ):
            total_gain_in_inv = max(0, bal_inv - cb_inv)
            if total_gain_in_inv > SMALL_EPSILON:
                gain_proportion_of_balance = total_gain_in_inv / bal_inv
                denominator = 1.0 - (gain_proportion_of_balance * real_tax_rate)
                if denominator > SMALL_EPSILON:
                    final_gross_withdrawal = net_withdrawal_target_for_inv / denominator
                else:
                    final_gross_withdrawal = min(
                        net_withdrawal_target_for_inv * 2, bal_inv
                    )
                final_gross_withdrawal = min(final_gross_withdrawal, bal_inv)
                realized_gain_from_this_withdrawal = (
                    final_gross_withdrawal * gain_proportion_of_balance
                )
                principal_component_of_withdrawal = (
                    final_gross_withdrawal - realized_gain_from_this_withdrawal
                )

        final_gross_withdrawal = min(final_gross_withdrawal, bal_inv)
        principal_component_of_withdrawal = min(
            principal_component_of_withdrawal, cb_inv
        )
        new_balance_inv = bal_inv - final_gross_withdrawal
        new_cost_basis_inv = cb_inv - principal_component_of_withdrawal
        return (
            max(0, new_balance_inv),
            max(0, new_cost_basis_inv),
            final_gross_withdrawal,
        )

    def _rebalance_portfolio(
        self,
        bal_inv1: float,
        cb_inv1: float,
        bal_inv2: float,
        cb_inv2: float,
    ) -> Tuple[float, float, float, float]:
        """
        Rebalances the two-asset portfolio to target allocations, applying
        realized gains taxes on any sales required for rebalancing.

        Returns (new_bal_inv1, new_cb_inv1, new_bal_inv2, new_cb_inv2).
        """
        p = self.params_model
        total_balance = bal_inv1 + bal_inv2

        if total_balance <= SMALL_EPSILON:
            return bal_inv1, cb_inv1, bal_inv2, cb_inv2

        target_bal1 = total_balance * p.allocation_inv1_pct
        amount_to_sell_inv1 = 0.0
        amount_to_sell_inv2 = 0.0
        tax_from_rebalancing = 0.0

        if bal_inv1 > target_bal1 + SMALL_EPSILON:
            amount_to_sell_inv1 = bal_inv1 - target_bal1
            if (
                p.inv1_use_realized_gains_tax_system
                and p.inv1_realized_gains_tax_rate > 0
                and amount_to_sell_inv1 > 0
            ):
                gain = max(0, bal_inv1 - cb_inv1)
                if gain > 0 and bal_inv1 > SMALL_EPSILON:
                    prop_sold = amount_to_sell_inv1 / bal_inv1
                    tax_from_rebalancing = (
                        gain * prop_sold * p.inv1_realized_gains_tax_rate
                    )

        elif bal_inv1 < target_bal1 - SMALL_EPSILON:
            target_bal2 = total_balance * p.allocation_inv2_pct
            amount_to_sell_inv2 = bal_inv2 - target_bal2
            if (
                p.inv2_use_realized_gains_tax_system
                and p.inv2_realized_gains_tax_rate > 0
                and amount_to_sell_inv2 > 0
            ):
                gain = max(0, bal_inv2 - cb_inv2)
                if gain > 0 and bal_inv2 > SMALL_EPSILON:
                    prop_sold = amount_to_sell_inv2 / bal_inv2
                    tax_from_rebalancing = (
                        gain * prop_sold * p.inv2_realized_gains_tax_rate
                    )

        total_after_tax = total_balance - tax_from_rebalancing
        new_bal_inv1 = total_after_tax * p.allocation_inv1_pct
        new_bal_inv2 = total_after_tax * p.allocation_inv2_pct

        total_cb = cb_inv1 + cb_inv2
        if amount_to_sell_inv1 > 0 and bal_inv1 > SMALL_EPSILON:
            prop_sold = amount_to_sell_inv1 / bal_inv1
            total_cb = total_cb - (cb_inv1 * prop_sold) + amount_to_sell_inv1
        elif amount_to_sell_inv2 > 0 and bal_inv2 > SMALL_EPSILON:
            prop_sold = amount_to_sell_inv2 / bal_inv2
            total_cb = total_cb - (cb_inv2 * prop_sold) + amount_to_sell_inv2

        new_cb_inv1 = min(
            total_cb * p.allocation_inv1_pct,
            new_bal_inv1 if new_bal_inv1 > 0 else 0,
        )
        new_cb_inv2 = min(
            total_cb * p.allocation_inv2_pct,
            new_bal_inv2 if new_bal_inv2 > 0 else 0,
        )

        return new_bal_inv1, new_cb_inv1, new_bal_inv2, new_cb_inv2

    def _draw_shock_path(self, n_months: int, path_seed: int) -> np.ndarray:
        """
        Pre-draw correlated standard-normal shocks of shape (n_months, 3) for
        (equity, inflation, inv2 premium).
        """
        rng = np.random.default_rng(path_seed)
        z = rng.standard_normal((n_months, 3))
        return z @ self._chol.T

    def _monthly_gross_from_shock(
        self, mu_log: float, sigma_log: float, z: float
    ) -> float:
        """Monthly gross return factor from annual log params and a unit shock."""
        return float(
            math.exp(mu_log / MONTHS_PER_YEAR + sigma_log / math.sqrt(MONTHS_PER_YEAR) * z)
        )

    def _run_single_simulation_path(
        self, working_months: int, path_seed: int
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Runs a single simulation path for a given number of working months and seed.

        Returns a dict with Start Balance, Final Balance, Success (funded all spending),
        First Year Gross Withdrawal, Trajectory, WithdrawalRateTrajectory (real % of
        start-of-retirement balance per retirement year), and Inflation At Retirement.
        """
        p = self.params_model
        total_months = working_months + p.retirement_years * MONTHS_PER_YEAR
        shocks = self._draw_shock_path(max(total_months, 1), path_seed)

        yearly_trajectory: List[float] = [p.initial_balance]
        # Price level (cumulative inflation from T=0) at each trajectory point.
        trajectory_price_levels: List[float] = [1.0]
        # Real withdrawal rate vs nest egg at retirement (Bengen/Trinity style):
        # (nominal gross withdraw / start_balance) * (I_ret / I_year_start) * 100.
        withdrawal_rate_trajectory: List[float] = []
        years_to_ruin: float = float("nan")  # 1-based years into retirement; NaN if success
        # Price level at the start of each simulation year (idx 0 = T0).
        yearly_master_inflation_factors: List[float] = [1.0]

        balance_inv1 = p.initial_balance * p.allocation_inv1_pct
        balance_inv2 = p.initial_balance * p.allocation_inv2_pct
        cost_basis_inv1 = balance_inv1
        cost_basis_inv2 = balance_inv2

        current_monthly_contribution = p.monthly_contribution
        bal_inv1_start_tax_year_acc = balance_inv1
        bal_inv2_start_tax_year_acc = balance_inv2
        contrib_inv1_tax_year, contrib_inv2_tax_year = 0.0, 0.0

        master_cumulative_inflation = 1.0
        shock_idx = 0

        # --- ACCUMULATION (WORKING) PHASE ---
        for m_idx in range(1, working_months + 1):
            if (m_idx - 1) % MONTHS_PER_YEAR == 0 and m_idx > 1:
                # Start of a new calendar year: record price level, grow contributions.
                yearly_master_inflation_factors.append(master_cumulative_inflation)
                if p.contribution_growth_rate_annual > 0:
                    current_monthly_contribution *= 1 + p.contribution_growth_rate_annual

            z_eq, z_inf, z_prem = shocks[shock_idx]
            shock_idx += 1

            monthly_gross_inv1 = self._monthly_gross_from_shock(
                self._inv1_mu_log, self._inv1_sigma_log, z_eq
            )
            monthly_gross_inf = self._monthly_gross_from_shock(
                self._inf_mu_log, self._inf_sigma_log, z_inf
            )
            monthly_gross_prem = self._monthly_gross_from_shock(
                self._inv2_prem_mu_log, self._inv2_prem_sigma_log, z_prem
            )
            # Inv2: inflation component * premium component (both as gross factors).
            monthly_gross_inv2 = monthly_gross_inf * monthly_gross_prem

            balance_inv1 *= monthly_gross_inv1
            balance_inv2 *= monthly_gross_inv2
            master_cumulative_inflation *= monthly_gross_inf

            contrib_m_inv1 = current_monthly_contribution * p.allocation_inv1_pct
            contrib_m_inv2 = current_monthly_contribution * p.allocation_inv2_pct
            balance_inv1 += contrib_m_inv1
            cost_basis_inv1 += contrib_m_inv1
            balance_inv2 += contrib_m_inv2
            cost_basis_inv2 += contrib_m_inv2

            contrib_inv1_tax_year += contrib_m_inv1
            contrib_inv2_tax_year += contrib_m_inv2

            balance_inv1, cost_basis_inv1, balance_inv2, cost_basis_inv2 = (
                self._rebalance_portfolio(
                    balance_inv1, cost_basis_inv1, balance_inv2, cost_basis_inv2
                )
            )

            if m_idx % MONTHS_PER_YEAR == 0 or m_idx == working_months:
                eoy_balance_inv1_before_tax, eoy_balance_inv2_before_tax = (
                    balance_inv1,
                    balance_inv2,
                )
                if (
                    not p.inv1_use_realized_gains_tax_system
                    and p.inv1_annual_tax_on_gains_rate > 0
                ):
                    gain_inv1 = (
                        eoy_balance_inv1_before_tax
                        - bal_inv1_start_tax_year_acc
                        - contrib_inv1_tax_year
                    )
                    if gain_inv1 > 0:
                        balance_inv1 -= gain_inv1 * p.inv1_annual_tax_on_gains_rate
                if (
                    not p.inv2_use_realized_gains_tax_system
                    and p.inv2_annual_tax_on_gains_rate > 0
                ):
                    gain_inv2 = (
                        eoy_balance_inv2_before_tax
                        - bal_inv2_start_tax_year_acc
                        - contrib_inv2_tax_year
                    )
                    if gain_inv2 > 0:
                        balance_inv2 -= gain_inv2 * p.inv2_annual_tax_on_gains_rate

                total_balance = balance_inv1 + balance_inv2
                if m_idx % MONTHS_PER_YEAR == 0:
                    yearly_trajectory.append(total_balance)
                    trajectory_price_levels.append(master_cumulative_inflation)

                bal_inv1_start_tax_year_acc = total_balance * p.allocation_inv1_pct
                bal_inv2_start_tax_year_acc = total_balance * p.allocation_inv2_pct
                contrib_inv1_tax_year, contrib_inv2_tax_year = 0.0, 0.0

        balance_at_retirement_start = balance_inv1 + balance_inv2
        inflation_at_retirement = master_cumulative_inflation

        # Record price level at retirement start if not already (partial final year).
        num_working_years = (
            (working_months + MONTHS_PER_YEAR - 1) // MONTHS_PER_YEAR
            if working_months > 0
            else 0
        )
        # yearly_master_inflation_factors should have an entry for retirement start.
        # After full years we have entries at year starts; append current level for
        # the retirement-start year index if needed.
        while len(yearly_master_inflation_factors) <= num_working_years:
            yearly_master_inflation_factors.append(master_cumulative_inflation)

        if working_months > 0 and working_months % MONTHS_PER_YEAR != 0:
            if (
                not yearly_trajectory
                or abs(yearly_trajectory[-1] - balance_at_retirement_start)
                > SMALL_EPSILON
            ):
                yearly_trajectory.append(balance_at_retirement_start)
                trajectory_price_levels.append(inflation_at_retirement)
        elif working_months == 0 and len(yearly_trajectory) == 0:
            yearly_trajectory.append(p.initial_balance)
            trajectory_price_levels.append(1.0)

        # Pre-calculate fixed nominal amounts for non-inflation-indexed income streams
        path_specific_other_income_streams_details = []
        for income_config in p.other_income_streams:
            stream_detail = income_config.model_copy(deep=True)
            pay_start_age = stream_payment_start_age(
                p.current_age, working_months, income_config.start_at_age
            )
            stream_detail._payment_start_age = pay_start_age
            # Map payment-start age to the nearest year-grid inflation index.
            years_to_start = years_from_t0_to_age(p.current_age, pay_start_age)
            year_income_starts_abs_sim_idx = int(math.floor(years_to_start + SMALL_EPSILON))
            if year_income_starts_abs_sim_idx < len(yearly_master_inflation_factors):
                stream_detail._master_inflation_at_start = (
                    yearly_master_inflation_factors[year_income_starts_abs_sim_idx]
                )
            if (
                not income_config.inflation_indexed
                and stream_detail._master_inflation_at_start is not None
            ):
                stream_detail._nominal_fixed_monthly_amount = (
                    income_config.monthly_amount_today
                    * stream_detail._master_inflation_at_start
                )
            path_specific_other_income_streams_details.append(stream_detail)

        first_year_gross_withdrawal = 0.0
        # True iff every retirement year funded spending (portfolio and/or other income).
        # A $0 portfolio is allowed when net other income covers expenses.
        path_succeeded = True

        # --- DECUMULATION (RETIREMENT) PHASE ---
        for year_num in range(p.retirement_years):
            # Price expenses/income at the START of the retirement year.
            price_level_at_year_start = master_cumulative_inflation
            yearly_master_inflation_factors.append(price_level_at_year_start)
            age_this_year = age_at_retirement_year(
                p.current_age, working_months, year_num
            )

            # Update any deferred non-indexed stream start factors.
            for stream_detail in path_specific_other_income_streams_details:
                if (
                    not stream_detail.inflation_indexed
                    and stream_detail._master_inflation_at_start is None
                ):
                    pay_start = stream_detail._payment_start_age
                    if pay_start is not None and age_this_year + SMALL_EPSILON >= pay_start:
                        stream_detail._master_inflation_at_start = price_level_at_year_start
                        stream_detail._nominal_fixed_monthly_amount = (
                            stream_detail.monthly_amount_today
                            * stream_detail._master_inflation_at_start
                        )

            nominal_annual_expenses = (
                p.monthly_expenses * MONTHS_PER_YEAR * price_level_at_year_start
            )

            net_other_annual_income = 0.0
            for income_stream_details in path_specific_other_income_streams_details:
                pay_start = income_stream_details._payment_start_age
                if pay_start is None:
                    pay_start = stream_payment_start_age(
                        p.current_age,
                        working_months,
                        income_stream_details.start_at_age,
                    )
                if age_this_year + SMALL_EPSILON >= pay_start:
                    years_receiving = age_this_year - pay_start
                    if (
                        income_stream_details.duration_years is None
                        or years_receiving < income_stream_details.duration_years
                    ):
                        if income_stream_details.inflation_indexed:
                            current_nominal_monthly_val = (
                                income_stream_details.monthly_amount_today
                                * price_level_at_year_start
                            )
                        else:
                            if (
                                income_stream_details._nominal_fixed_monthly_amount
                                is not None
                            ):
                                current_nominal_monthly_val = (
                                    income_stream_details._nominal_fixed_monthly_amount
                                )
                            else:
                                current_nominal_monthly_val = (
                                    income_stream_details.monthly_amount_today
                                )

                        stream_annual_pre_tax = (
                            current_nominal_monthly_val * MONTHS_PER_YEAR
                        )
                        stream_tax = (
                            stream_annual_pre_tax * income_stream_details.tax_rate
                        )
                        net_other_annual_income += stream_annual_pre_tax - stream_tax

            required_annual_withdrawal = max(
                0, nominal_annual_expenses - net_other_annual_income
            )
            monthly_withdrawal_needed = required_annual_withdrawal / MONTHS_PER_YEAR

            # Empty portfolio is fine when income covers spending; fail only if cash needed.
            if (
                balance_inv1 + balance_inv2 <= SMALL_EPSILON
                and monthly_withdrawal_needed > SMALL_EPSILON
            ):
                path_succeeded = False
                years_to_ruin = float(year_num + 1)
                break

            bal_inv1_start_tax_year_ret, bal_inv2_start_tax_year_ret = (
                balance_inv1,
                balance_inv2,
            )
            total_gross_withdraw_inv1_this_year = 0.0
            total_gross_withdraw_inv2_this_year = 0.0
            year_funding_failed = False

            for _month_in_ret_year_idx in range(MONTHS_PER_YEAR):
                total_balance_before_month = balance_inv1 + balance_inv2
                if (
                    total_balance_before_month <= SMALL_EPSILON
                    and monthly_withdrawal_needed > SMALL_EPSILON
                ):
                    year_funding_failed = True
                    break

                z_eq, z_inf, z_prem = shocks[min(shock_idx, len(shocks) - 1)]
                shock_idx += 1

                monthly_gross_inv1 = self._monthly_gross_from_shock(
                    self._inv1_mu_log, self._inv1_sigma_log, z_eq
                )
                monthly_gross_inf = self._monthly_gross_from_shock(
                    self._inf_mu_log, self._inf_sigma_log, z_inf
                )
                monthly_gross_prem = self._monthly_gross_from_shock(
                    self._inv2_prem_mu_log, self._inv2_prem_sigma_log, z_prem
                )
                monthly_gross_inv2 = monthly_gross_inf * monthly_gross_prem

                balance_inv1 *= monthly_gross_inv1
                balance_inv2 *= monthly_gross_inv2
                master_cumulative_inflation *= monthly_gross_inf
                total_after_growth = balance_inv1 + balance_inv2

                if (
                    total_after_growth <= SMALL_EPSILON
                    and monthly_withdrawal_needed > SMALL_EPSILON
                ):
                    balance_inv1 = max(0, balance_inv1)
                    balance_inv2 = max(0, balance_inv2)
                    year_funding_failed = True
                    break

                actual_monthly_withdrawal_target = min(
                    monthly_withdrawal_needed, total_after_growth
                )
                actual_monthly_withdrawal_target = max(
                    0, actual_monthly_withdrawal_target
                )
                if (
                    monthly_withdrawal_needed > SMALL_EPSILON
                    and actual_monthly_withdrawal_target
                    < monthly_withdrawal_needed - SMALL_EPSILON
                ):
                    year_funding_failed = True

                prop1 = (
                    balance_inv1 / total_after_growth
                    if total_after_growth > SMALL_EPSILON
                    else p.allocation_inv1_pct
                )
                prop2 = 1.0 - prop1

                balance_inv1, cost_basis_inv1, gw1 = (
                    self._calculate_withdrawal_and_update(
                        balance_inv1,
                        cost_basis_inv1,
                        actual_monthly_withdrawal_target * prop1,
                        p.inv1_use_realized_gains_tax_system,
                        p.inv1_realized_gains_tax_rate,
                    )
                )
                total_gross_withdraw_inv1_this_year += gw1

                balance_inv2, cost_basis_inv2, gw2 = (
                    self._calculate_withdrawal_and_update(
                        balance_inv2,
                        cost_basis_inv2,
                        actual_monthly_withdrawal_target * prop2,
                        p.inv2_use_realized_gains_tax_system,
                        p.inv2_realized_gains_tax_rate,
                    )
                )
                total_gross_withdraw_inv2_this_year += gw2

                balance_inv1 = max(0, balance_inv1)
                balance_inv2 = max(0, balance_inv2)
                cost_basis_inv1 = min(
                    cost_basis_inv1, balance_inv1 if balance_inv1 > 0 else 0
                )
                cost_basis_inv2 = min(
                    cost_basis_inv2, balance_inv2 if balance_inv2 > 0 else 0
                )

                balance_inv1, cost_basis_inv1, balance_inv2, cost_basis_inv2 = (
                    self._rebalance_portfolio(
                        balance_inv1, cost_basis_inv1, balance_inv2, cost_basis_inv2
                    )
                )

                if year_funding_failed:
                    break

            year_gross_withdrawal = (
                total_gross_withdraw_inv1_this_year + total_gross_withdraw_inv2_this_year
            )
            # Deflate to retirement-date real $ so a classic constant-real draw is flat.
            if (
                balance_at_retirement_start > SMALL_EPSILON
                and price_level_at_year_start > SMALL_EPSILON
            ):
                year_wr_pct = (
                    year_gross_withdrawal
                    / balance_at_retirement_start
                    * (inflation_at_retirement / price_level_at_year_start)
                ) * 100.0
            else:
                year_wr_pct = 0.0

            if year_funding_failed:
                path_succeeded = False
                years_to_ruin = float(year_num + 1)
                yearly_trajectory.append(max(0.0, balance_inv1 + balance_inv2))
                trajectory_price_levels.append(master_cumulative_inflation)
                # Partial year — still record observed draw vs start nest egg.
                withdrawal_rate_trajectory.append(year_wr_pct)
                if year_num == 0:
                    first_year_gross_withdrawal = year_gross_withdrawal
                break

            withdrawal_rate_trajectory.append(year_wr_pct)

            if year_num == 0:
                first_year_gross_withdrawal = year_gross_withdrawal

            if (
                not p.inv1_use_realized_gains_tax_system
                and p.inv1_annual_tax_on_gains_rate > 0
            ):
                gain_inv1_ret_year = (
                    balance_inv1 + total_gross_withdraw_inv1_this_year
                ) - bal_inv1_start_tax_year_ret
                if gain_inv1_ret_year > 0:
                    balance_inv1 -= gain_inv1_ret_year * p.inv1_annual_tax_on_gains_rate

            if (
                not p.inv2_use_realized_gains_tax_system
                and p.inv2_annual_tax_on_gains_rate > 0
            ):
                gain_inv2_ret_year = (
                    balance_inv2 + total_gross_withdraw_inv2_this_year
                ) - bal_inv2_start_tax_year_ret
                if gain_inv2_ret_year > 0:
                    balance_inv2 -= gain_inv2_ret_year * p.inv2_annual_tax_on_gains_rate

            balance_inv1 = max(0, balance_inv1)
            balance_inv2 = max(0, balance_inv2)
            cost_basis_inv1 = min(cost_basis_inv1, balance_inv1)
            cost_basis_inv2 = min(cost_basis_inv2, balance_inv2)

            total_balance_after_annual_tax = balance_inv1 + balance_inv2
            yearly_trajectory.append(total_balance_after_annual_tax)
            trajectory_price_levels.append(master_cumulative_inflation)

            # Keep going with a $0 portfolio when income covers future spending.
            if total_balance_after_annual_tax > SMALL_EPSILON:
                balance_inv1 = total_balance_after_annual_tax * p.allocation_inv1_pct
                balance_inv2 = total_balance_after_annual_tax * p.allocation_inv2_pct
                total_cb = cost_basis_inv1 + cost_basis_inv2
                cost_basis_inv1 = total_cb * p.allocation_inv1_pct
                cost_basis_inv2 = total_cb * p.allocation_inv2_pct
                cost_basis_inv1 = min(cost_basis_inv1, balance_inv1)
                cost_basis_inv2 = min(cost_basis_inv2, balance_inv2)

        final_total_balance = balance_inv1 + balance_inv2

        expected_len = 1 + num_working_years + p.retirement_years
        current_len = len(yearly_trajectory)

        if current_len < expected_len:
            padding_value = (
                0.0
                if not path_succeeded
                else (yearly_trajectory[-1] if yearly_trajectory else 0.0)
            )
            pad_n = expected_len - current_len
            yearly_trajectory.extend([padding_value] * pad_n)
            last_px = (
                trajectory_price_levels[-1] if trajectory_price_levels else 1.0
            )
            trajectory_price_levels.extend([last_px] * pad_n)
        elif current_len > expected_len:
            yearly_trajectory = yearly_trajectory[:expected_len]
            trajectory_price_levels = trajectory_price_levels[:expected_len]

        # Align price-level series length with trajectory.
        while len(trajectory_price_levels) < len(yearly_trajectory):
            trajectory_price_levels.append(
                trajectory_price_levels[-1] if trajectory_price_levels else 1.0
            )
        trajectory_price_levels = trajectory_price_levels[: len(yearly_trajectory)]

        real_trajectory = [
            (nom / px if px > SMALL_EPSILON else 0.0)
            for nom, px in zip(yearly_trajectory, trajectory_price_levels)
        ]

        # Pad remaining retirement years as NaN (no observation after failure / early stop).
        while len(withdrawal_rate_trajectory) < p.retirement_years:
            withdrawal_rate_trajectory.append(float("nan"))
        if len(withdrawal_rate_trajectory) > p.retirement_years:
            withdrawal_rate_trajectory = withdrawal_rate_trajectory[: p.retirement_years]

        return {
            "Start Balance": balance_at_retirement_start,
            "Final Balance": max(0, final_total_balance),
            "Success": bool(path_succeeded),
            "YearsToRuin": years_to_ruin,
            "First Year Gross Withdrawal": first_year_gross_withdrawal,
            "Trajectory": yearly_trajectory,
            "RealTrajectory": real_trajectory,
            "WithdrawalRateTrajectory": withdrawal_rate_trajectory,
            "Inflation At Retirement": inflation_at_retirement,
        }

    def run_monte_carlo_simulations(
        self, working_months: int, num_simulations: int
    ) -> Tuple[
        pd.DataFrame,
        Optional[pd.DataFrame],
        Optional[List[List[float]]],
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[List[List[float]]],
    ]:
        """
        Runs multiple simulation paths, either sequentially or in parallel.

        Uses common random numbers: path seeds are derived from the active seed
        stream and are identical across different working_months candidates.

        Returns summary_df, nominal trajectory percentiles, sample paths,
        withdrawal-rate percentiles, real trajectory percentiles, real sample paths.
        """
        path_seeds = self._path_seeds(num_simulations)
        num_procs_to_use = (
            self.params_model.num_processes
            if self.params_model.num_processes is not None
            else 1
        )

        all_results_list: List[Dict[str, Union[float, List[float]]]]

        if num_procs_to_use <= 1:
            logger.debug(
                f"Running {num_simulations} simulations sequentially for "
                f"{working_months} working months."
            )
            all_results_list = [
                self._run_single_simulation_path(working_months, seed)
                for seed in path_seeds
            ]
        else:
            logger.debug(
                f"Running {num_simulations} simulations in parallel using "
                f"{num_procs_to_use} processes for {working_months} working months."
            )
            args_for_starmap = [(working_months, seed) for seed in path_seeds]
            try:
                with multiprocessing.Pool(processes=num_procs_to_use) as pool:
                    all_results_list = pool.starmap(
                        self._run_single_simulation_path, args_for_starmap
                    )
            except Exception as e:
                logger.error(
                    f"Multiprocessing pool error: {e}. Falling back to sequential execution.",
                    exc_info=True,
                )
                all_results_list = [
                    self._run_single_simulation_path(working_months, seed)
                    for seed in path_seeds
                ]

        summary_results_list = [
            {
                "Start Balance": r["Start Balance"],
                "Final Balance": r["Final Balance"],
                "Success": bool(r.get("Success", r["Final Balance"] > SMALL_EPSILON)),
                "YearsToRuin": r.get("YearsToRuin", float("nan")),
                "First Year Gross Withdrawal": r["First Year Gross Withdrawal"],
                "Inflation At Retirement": r.get("Inflation At Retirement", 1.0),
            }
            for r in all_results_list
        ]
        summary_df = pd.DataFrame(summary_results_list)

        trajectories_raw = [
            r["Trajectory"]
            for r in all_results_list
            if "Trajectory" in r and r["Trajectory"]
        ]
        real_trajectories_raw = [
            r["RealTrajectory"]
            for r in all_results_list
            if r.get("RealTrajectory")
        ]

        trajectory_percentiles_df: Optional[pd.DataFrame] = None
        real_trajectory_percentiles_df: Optional[pd.DataFrame] = None
        sample_trajectories_list: Optional[List[List[float]]] = None
        sample_real_trajectories_list: Optional[List[List[float]]] = None

        percentiles_to_calc = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        if trajectories_raw:
            try:
                min_len = min(map(len, trajectories_raw))
                max_len = max(map(len, trajectories_raw))
                if min_len != max_len:
                    logger.warning(
                        f"Trajectory lengths are inconsistent: min={min_len}, max={max_len}."
                    )

                trajectory_df = pd.DataFrame(trajectories_raw).transpose()

                if not trajectory_df.empty:
                    trajectory_percentiles_df = trajectory_df.quantile(
                        percentiles_to_calc, axis=1
                    ).transpose()

                    num_sample_paths = min(len(trajectory_df.columns), 5)
                    if num_sample_paths > 0:
                        actual_num_to_sample = min(
                            num_sample_paths, trajectory_df.shape[1]
                        )
                        sampled = trajectory_df.sample(
                            n=actual_num_to_sample,
                            axis=1,
                            random_state=self.main_seed,
                        )
                        sample_trajectories_list = sampled.values.T.tolist()
                        if real_trajectories_raw:
                            real_df_full = pd.DataFrame(real_trajectories_raw).transpose()
                            sample_real_trajectories_list = real_df_full.loc[
                                :, sampled.columns
                            ].values.T.tolist()
            except ValueError as ve:
                logger.error(
                    f"Error processing trajectories, possibly due to inconsistent lengths: {ve}",
                    exc_info=True,
                )
            except Exception as e:
                logger.error(f"Error processing trajectories: {e}", exc_info=True)

        if real_trajectories_raw:
            try:
                real_df = pd.DataFrame(real_trajectories_raw).transpose()
                if not real_df.empty:
                    real_trajectory_percentiles_df = real_df.quantile(
                        percentiles_to_calc, axis=1
                    ).transpose()
            except Exception as e:
                logger.error(f"Error processing real trajectories: {e}", exc_info=True)

        wr_percentiles_df: Optional[pd.DataFrame] = None
        wr_raw = [
            r["WithdrawalRateTrajectory"]
            for r in all_results_list
            if r.get("WithdrawalRateTrajectory")
        ]
        if wr_raw:
            try:
                wr_df = pd.DataFrame(wr_raw).transpose()
                if not wr_df.empty:
                    wr_percentiles_df = wr_df.quantile(
                        [0.05, 0.25, 0.50, 0.75, 0.95], axis=1
                    ).transpose()
            except Exception as e:
                logger.error(
                    f"Error processing withdrawal-rate trajectories: {e}",
                    exc_info=True,
                )

        return (
            summary_df,
            trajectory_percentiles_df,
            sample_trajectories_list,
            wr_percentiles_df,
            real_trajectory_percentiles_df,
            sample_real_trajectories_list,
        )

    def _success_probability(self, summary_df: pd.DataFrame) -> float:
        """Share of paths that funded all retirement spending (may end at $0)."""
        if summary_df.empty:
            return 0.0
        if "Success" in summary_df.columns:
            return float(summary_df["Success"].astype(bool).mean() * 100.0)
        return float((summary_df["Final Balance"] > SMALL_EPSILON).mean() * 100.0)

    def find_minimum_working_months(
        self,
        verbose: bool = True,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> Tuple[int, float, List[Dict[str, float]]]:
        """
        Finds the minimum working months to achieve the target success probability
        via a coarse bracket scan followed by bisection.

        Uses the search seed stream with common random numbers across candidates.

        Returns (months, probability, search_curve) where search_curve is a list of
        {working_months, working_years, probability} probe points (may include
        duplicates if the same month is retested).
        """
        self.use_search_seeds()
        p = self.params_model
        starting_working_months = p.starting_working_months_search
        target_probability_pct = p.target_probability
        sim_count = p.num_simulations_search
        max_total_months = starting_working_months + 70 * MONTHS_PER_YEAR
        search_curve: List[Dict[str, float]] = []

        if verbose:
            logger.info(
                f"Searching for minimum working months to achieve "
                f"{target_probability_pct:.2f}% success for '{p.Nickname}'."
            )
            logger.info(
                f"Starting search from {starting_working_months} months. "
                f"Simulations per test: {sim_count}."
            )

        search_iteration = 0
        highest_prob_if_target_not_met = -1.0
        lo = starting_working_months
        hi: Optional[int] = None

        def _test(months: int) -> float:
            nonlocal search_iteration, highest_prob_if_target_not_met
            search_iteration += 1
            if verbose:
                logger.info(
                    f"Search iter {search_iteration}: Testing {months} m "
                    f"({months / MONTHS_PER_YEAR:.1f} yrs) with {sim_count} sims."
                )
            summary_df, _, _, _, _, _ = self.run_monte_carlo_simulations(months, sim_count)
            prob = self._success_probability(summary_df)
            if verbose:
                logger.info(
                    f"  Search iter {search_iteration}: Prob for {months} m: "
                    f"{prob:.2f}% (Target: {target_probability_pct:.2f}%)"
                )
            point = {
                "working_months": months,
                "working_years": round(months / MONTHS_PER_YEAR, 1),
                "probability": round(prob, 2),
            }
            search_curve.append(point)
            if progress_callback:
                progress_callback(
                    {
                        "type": "search_iter",
                        "iteration": search_iteration,
                        "working_months": months,
                        "working_years": round(months / MONTHS_PER_YEAR, 1),
                        "probability": round(prob, 2),
                        "target": target_probability_pct,
                        "sim_count": sim_count,
                        "lo": lo,
                        "hi": hi,
                    }
                )
            if prob > highest_prob_if_target_not_met:
                highest_prob_if_target_not_met = prob
            return prob

        # --- Phase 1: Bracket ---
        step = 12
        current = starting_working_months
        prob_at_lo = _test(current)

        if prob_at_lo >= target_probability_pct:
            if verbose:
                logger.info(f"  Target met at starting point {current} months.")
            return current, prob_at_lo, search_curve

        while current < max_total_months:
            # Grow step while far from target
            gap = target_probability_pct - prob_at_lo
            if gap > 20:
                step = max(step, 24)
            elif gap > 10:
                step = max(step, 12)
            else:
                step = max(step, 6)

            next_months = min(current + step, max_total_months)
            if next_months <= current:
                break
            prob = _test(next_months)
            if prob >= target_probability_pct:
                lo = current
                hi = next_months
                best_prob = prob
                if verbose:
                    logger.info(
                        f"  Bracketed: lo={lo} m (miss), hi={hi} m (hit). Bisecting…"
                    )
                if progress_callback:
                    progress_callback(
                        {
                            "type": "search_refining",
                            "working_months": hi,
                            "lo": lo,
                            "hi": hi,
                        }
                    )
                break
            lo = next_months
            prob_at_lo = prob
            current = next_months

        if hi is None:
            if verbose:
                logger.warning(
                    f"Search for '{p.Nickname}' reached max limit "
                    f"({max_total_months / MONTHS_PER_YEAR:.1f} yrs). Target NOT met."
                )
                logger.warning(
                    f"Highest probability achieved: {highest_prob_if_target_not_met:.2f}%."
                )
            return -1, highest_prob_if_target_not_met, search_curve

        # --- Phase 2: Bisect [lo, hi], track smallest month count that met target ---
        best = hi
        while hi - lo > 1:
            mid = (lo + hi) // 2
            prob = _test(mid)
            if prob >= target_probability_pct:
                best = mid
                best_prob = prob
                hi = mid
            else:
                lo = mid

        if verbose:
            logger.info(
                f"  Search complete: minimum {best} months "
                f"({best / MONTHS_PER_YEAR:.1f} yrs) with prob {best_prob:.2f}%."
            )
        return best, best_prob, search_curve
