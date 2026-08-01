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
    if mean <= -1.0:
        raise ValueError("Arithmetic mean must be greater than -100%.")
    if vol < 0:
        raise ValueError("Volatility cannot be negative.")
    if vol == 0:
        # Degenerate: deterministic growth
        return math.log(1.0 + mean), 0.0
    one_plus_mean = 1.0 + mean
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


def stream_payment_start_month_index(
    current_age: float, working_months: int, start_at_age: float
) -> int:
    """First retirement-month index whose payment date is at/after eligibility."""
    retirement_start = retirement_age(current_age, working_months)
    eligible_age = stream_payment_start_age(
        current_age, working_months, start_at_age
    )
    return max(
        0,
        int(
            math.ceil(
                (eligible_age - retirement_start) * MONTHS_PER_YEAR
                - SMALL_EPSILON
            )
        ),
    )


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
    Median per-path first-year real gross withdrawal / start-of-retirement balance.
    Withdrawals are deflated to retirement-date dollars (Trinity/Bengen basis).
    """
    if summary_df.empty:
        return float("nan")
    start = summary_df["Start Balance"]
    withdrawal_column = (
        "First Year Real Gross Withdrawal"
        if "First Year Real Gross Withdrawal" in summary_df.columns
        else "First Year Gross Withdrawal"
    )
    withdraw = summary_df[withdrawal_column]
    valid = start > SMALL_EPSILON
    if not valid.any():
        return float("nan")
    rates = (withdraw[valid] / start[valid]) * 100.0
    return float(rates.median())


def trajectory_time_points(
    working_months: int, retirement_years: int
) -> List[float]:
    """
    Actual year values for yearly trajectory samples.

    Full accumulation years are sampled at integer years. A partial final working
    year adds a sample at the exact retirement date, then retirement samples occur
    at one-year intervals from that date.
    """
    full_working_years, remaining_months = divmod(
        working_months, MONTHS_PER_YEAR
    )
    points: List[float] = [0.0]
    points.extend(float(year) for year in range(1, full_working_years + 1))

    retirement_time = working_months / MONTHS_PER_YEAR
    if remaining_months:
        points.append(retirement_time)

    points.extend(
        retirement_time + year
        for year in range(1, retirement_years + 1)
    )
    return points


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
            if main_seed_override < 0:
                raise ValueError("main_seed_override must be nonnegative.")
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

        # Direct correlated-normal construction supports the full [-1, 1] range,
        # including the singular perfect-correlation endpoints.
        self._equity_inflation_rho = p.equity_inflation_correlation

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
    ) -> Tuple[float, float, float, float]:
        """
        Calculates the gross withdrawal needed to meet a net target, considering taxes,
        and updates average cost basis.

        Cost basis may legitimately exceed market value after a loss.  The basis removed
        on sale is therefore proportional to shares sold, not capped at sale proceeds.

        Returns (new_balance, new_cost_basis, gross_withdrawal, net_cash_delivered).
        """
        if bal_inv <= SMALL_EPSILON or net_withdrawal_target_for_inv <= 0:
            return max(0.0, bal_inv), max(0.0, cb_inv), 0.0, 0.0

        gain_fraction = max(0.0, bal_inv - cb_inv) / bal_inv
        effective_tax_fraction = (
            gain_fraction * real_tax_rate
            if use_real_tax and real_tax_rate > 0
            else 0.0
        )
        net_fraction = max(SMALL_EPSILON, 1.0 - effective_tax_fraction)
        gross_withdrawal = min(
            net_withdrawal_target_for_inv / net_fraction,
            bal_inv,
        )

        fraction_sold = min(1.0, gross_withdrawal / bal_inv)
        basis_removed = min(cb_inv, cb_inv * fraction_sold)
        taxable_gain = max(0.0, gross_withdrawal - basis_removed)
        tax_paid = (
            taxable_gain * real_tax_rate
            if use_real_tax and real_tax_rate > 0
            else 0.0
        )
        net_cash_delivered = max(0.0, gross_withdrawal - tax_paid)

        new_balance_inv = max(0.0, bal_inv - gross_withdrawal)
        new_cost_basis_inv = max(0.0, cb_inv - basis_removed)
        if new_balance_inv <= SMALL_EPSILON:
            new_balance_inv = 0.0
            new_cost_basis_inv = 0.0

        return (
            new_balance_inv,
            new_cost_basis_inv,
            gross_withdrawal,
            net_cash_delivered,
        )

    @staticmethod
    def _net_liquidation_value(
        balance: float,
        cost_basis: float,
        use_realized_gains_tax: bool,
        realized_gains_tax_rate: float,
    ) -> float:
        """Cash available after fully liquidating an asset and paying gains tax."""
        if balance <= SMALL_EPSILON:
            return 0.0
        taxable_gain = max(0.0, balance - cost_basis)
        tax = (
            taxable_gain * realized_gains_tax_rate
            if use_realized_gains_tax and realized_gains_tax_rate > 0
            else 0.0
        )
        return max(0.0, balance - tax)

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

        target_bal1_before_tax = total_balance * p.allocation_inv1_pct
        drift1 = bal_inv1 - target_bal1_before_tax
        if abs(drift1) <= SMALL_EPSILON:
            return bal_inv1, cb_inv1, bal_inv2, cb_inv2

        if drift1 > 0:
            # Sell Inv1.  Because tax reduces the final portfolio, solve the sale
            # amount x from: bal1 - x = alloc1 * (total - tax_rate_on_sale * x).
            gain_fraction = max(0.0, bal_inv1 - cb_inv1) / bal_inv1
            tax_per_dollar = (
                gain_fraction * p.inv1_realized_gains_tax_rate
                if p.inv1_use_realized_gains_tax_system
                else 0.0
            )
            denominator = max(
                SMALL_EPSILON,
                1.0 - p.allocation_inv1_pct * tax_per_dollar,
            )
            gross_sale = min(bal_inv1, drift1 / denominator)
            fraction_sold = gross_sale / bal_inv1
            basis_removed = min(cb_inv1, cb_inv1 * fraction_sold)
            taxable_gain = max(0.0, gross_sale - basis_removed)
            tax_paid = (
                taxable_gain * p.inv1_realized_gains_tax_rate
                if p.inv1_use_realized_gains_tax_system
                else 0.0
            )
            net_purchase = gross_sale - tax_paid

            new_bal_inv1 = max(0.0, bal_inv1 - gross_sale)
            new_cb_inv1 = max(0.0, cb_inv1 - basis_removed)
            new_bal_inv2 = bal_inv2 + net_purchase
            new_cb_inv2 = cb_inv2 + net_purchase
        else:
            # Sell Inv2, with the symmetric tax-aware target equation.
            drift2 = bal_inv2 - total_balance * p.allocation_inv2_pct
            gain_fraction = max(0.0, bal_inv2 - cb_inv2) / bal_inv2
            tax_per_dollar = (
                gain_fraction * p.inv2_realized_gains_tax_rate
                if p.inv2_use_realized_gains_tax_system
                else 0.0
            )
            denominator = max(
                SMALL_EPSILON,
                1.0 - p.allocation_inv2_pct * tax_per_dollar,
            )
            gross_sale = min(bal_inv2, drift2 / denominator)
            fraction_sold = gross_sale / bal_inv2
            basis_removed = min(cb_inv2, cb_inv2 * fraction_sold)
            taxable_gain = max(0.0, gross_sale - basis_removed)
            tax_paid = (
                taxable_gain * p.inv2_realized_gains_tax_rate
                if p.inv2_use_realized_gains_tax_system
                else 0.0
            )
            net_purchase = gross_sale - tax_paid

            new_bal_inv2 = max(0.0, bal_inv2 - gross_sale)
            new_cb_inv2 = max(0.0, cb_inv2 - basis_removed)
            new_bal_inv1 = bal_inv1 + net_purchase
            new_cb_inv1 = cb_inv1 + net_purchase

        if new_bal_inv1 <= SMALL_EPSILON:
            new_bal_inv1, new_cb_inv1 = 0.0, 0.0
        if new_bal_inv2 <= SMALL_EPSILON:
            new_bal_inv2, new_cb_inv2 = 0.0, 0.0
        return new_bal_inv1, new_cb_inv1, new_bal_inv2, new_cb_inv2

    def _apply_annual_gain_taxes(
        self,
        balance_inv1: float,
        cost_basis_inv1: float,
        balance_inv2: float,
        cost_basis_inv2: float,
        gain_inv1: float,
        gain_inv2: float,
    ) -> Tuple[float, float, float, float, bool]:
        """
        Deduct annual mark-to-market taxes for one completed tax period.

        Gains are market P&L accumulated monthly, so contributions, withdrawals, and
        internal rebalancing transfers are excluded. The combined bill is paid as a
        portfolio cash outflow; selling an asset that uses realized-gain taxation can
        itself require additional gross proceeds. Returns ``tax_failed`` when net
        liquidation value cannot pay the assessed bill.
        """
        p = self.params_model
        tax_due_inv1 = (
            max(0.0, gain_inv1) * p.inv1_annual_tax_on_gains_rate
            if not p.inv1_use_realized_gains_tax_system
            else 0.0
        )
        tax_due_inv2 = (
            max(0.0, gain_inv2) * p.inv2_annual_tax_on_gains_rate
            if not p.inv2_use_realized_gains_tax_system
            else 0.0
        )
        total_tax_due = tax_due_inv1 + tax_due_inv2

        capacity_inv1 = self._net_liquidation_value(
            balance_inv1,
            cost_basis_inv1,
            p.inv1_use_realized_gains_tax_system,
            p.inv1_realized_gains_tax_rate,
        )
        capacity_inv2 = self._net_liquidation_value(
            balance_inv2,
            cost_basis_inv2,
            p.inv2_use_realized_gains_tax_system,
            p.inv2_realized_gains_tax_rate,
        )
        total_capacity = capacity_inv1 + capacity_inv2
        net_tax_payment = min(total_tax_due, total_capacity)
        tax_failed = net_tax_payment < total_tax_due - SMALL_EPSILON

        if total_capacity > SMALL_EPSILON and net_tax_payment > 0:
            share_inv1 = capacity_inv1 / total_capacity
            share_inv2 = 1.0 - share_inv1
            balance_inv1, cost_basis_inv1, _, net1 = (
                self._calculate_withdrawal_and_update(
                    balance_inv1,
                    cost_basis_inv1,
                    net_tax_payment * share_inv1,
                    p.inv1_use_realized_gains_tax_system,
                    p.inv1_realized_gains_tax_rate,
                )
            )
            balance_inv2, cost_basis_inv2, _, net2 = (
                self._calculate_withdrawal_and_update(
                    balance_inv2,
                    cost_basis_inv2,
                    net_tax_payment * share_inv2,
                    p.inv2_use_realized_gains_tax_system,
                    p.inv2_realized_gains_tax_rate,
                )
            )
            if net1 + net2 < total_tax_due - SMALL_EPSILON:
                tax_failed = True

        (
            balance_inv1,
            cost_basis_inv1,
            balance_inv2,
            cost_basis_inv2,
        ) = self._rebalance_portfolio(
            balance_inv1,
            cost_basis_inv1,
            balance_inv2,
            cost_basis_inv2,
        )

        return (
            balance_inv1,
            cost_basis_inv1,
            balance_inv2,
            cost_basis_inv2,
            tax_failed,
        )

    def _draw_shock_path(self, n_months: int, path_seed: int) -> np.ndarray:
        """
        Pre-draw correlated standard-normal shocks of shape (n_months, 3) for
        (equity, inflation, inv2 premium).
        """
        rng = np.random.default_rng(path_seed)
        independent = rng.standard_normal((n_months, 3))
        equity = independent[:, 0]
        rho = self._equity_inflation_rho
        inflation = (
            rho * equity
            + math.sqrt(max(0.0, 1.0 - rho * rho)) * independent[:, 1]
        )
        premium = independent[:, 2]
        return np.column_stack((equity, inflation, premium))

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
        # Elapsed retirement years at the first unfunded monthly payment; NaN if success.
        years_to_ruin: float = float("nan")

        balance_inv1 = p.initial_balance * p.allocation_inv1_pct
        balance_inv2 = p.initial_balance - balance_inv1
        cost_basis_inv1 = balance_inv1
        cost_basis_inv2 = balance_inv2

        current_monthly_contribution = p.monthly_contribution
        gain_inv1_tax_year_acc = 0.0
        gain_inv2_tax_year_acc = 0.0

        master_cumulative_inflation = 1.0
        shock_idx = 0
        pre_retirement_tax_failed = False

        # --- ACCUMULATION (WORKING) PHASE ---
        for m_idx in range(1, working_months + 1):
            if (m_idx - 1) % MONTHS_PER_YEAR == 0 and m_idx > 1:
                # Start of a new contribution year.
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

            gain_inv1_tax_year_acc += balance_inv1 * (monthly_gross_inv1 - 1.0)
            gain_inv2_tax_year_acc += balance_inv2 * (monthly_gross_inv2 - 1.0)
            balance_inv1 *= monthly_gross_inv1
            balance_inv2 *= monthly_gross_inv2
            master_cumulative_inflation *= monthly_gross_inf

            contrib_m_inv1 = (
                current_monthly_contribution * p.allocation_inv1_pct
            )
            contrib_m_inv2 = current_monthly_contribution - contrib_m_inv1
            balance_inv1 += contrib_m_inv1
            cost_basis_inv1 += contrib_m_inv1
            balance_inv2 += contrib_m_inv2
            cost_basis_inv2 += contrib_m_inv2

            balance_inv1, cost_basis_inv1, balance_inv2, cost_basis_inv2 = (
                self._rebalance_portfolio(
                    balance_inv1, cost_basis_inv1, balance_inv2, cost_basis_inv2
                )
            )

            # Annual taxes follow absolute 12-month tax periods. Retirement does
            # not create an artificial tax boundary for a partial working year.
            if m_idx % MONTHS_PER_YEAR == 0:
                (
                    balance_inv1,
                    cost_basis_inv1,
                    balance_inv2,
                    cost_basis_inv2,
                    tax_failed,
                ) = self._apply_annual_gain_taxes(
                    balance_inv1,
                    cost_basis_inv1,
                    balance_inv2,
                    cost_basis_inv2,
                    gain_inv1_tax_year_acc,
                    gain_inv2_tax_year_acc,
                )
                if tax_failed:
                    pre_retirement_tax_failed = True
                total_balance = balance_inv1 + balance_inv2
                yearly_trajectory.append(total_balance)
                trajectory_price_levels.append(master_cumulative_inflation)

                gain_inv1_tax_year_acc = 0.0
                gain_inv2_tax_year_acc = 0.0

        balance_at_retirement_start = balance_inv1 + balance_inv2
        inflation_at_retirement = master_cumulative_inflation

        # Record price level at retirement start if not already (partial final year).
        num_working_years = (
            (working_months + MONTHS_PER_YEAR - 1) // MONTHS_PER_YEAR
            if working_months > 0
            else 0
        )
        if working_months > 0 and working_months % MONTHS_PER_YEAR != 0:
            # This is a distinct timestamp even when its balance happens to equal
            # the preceding full-year sample.
            yearly_trajectory.append(balance_at_retirement_start)
            trajectory_price_levels.append(inflation_at_retirement)
        elif working_months == 0 and len(yearly_trajectory) == 0:
            yearly_trajectory.append(p.initial_balance)
            trajectory_price_levels.append(1.0)

        # Convert stream age dates to retirement-month indices. This honors fractional
        # retirement ages and starts each stream on the first monthly payment date at
        # or after start_at_age. Duration is then exactly N * 12 payments.
        path_specific_other_income_streams_details: List[dict] = []
        for income_config in p.other_income_streams:
            start_month = stream_payment_start_month_index(
                p.current_age,
                working_months,
                income_config.start_at_age,
            )
            duration_months = (
                None
                if income_config.duration_years is None
                else income_config.duration_years * MONTHS_PER_YEAR
            )
            path_specific_other_income_streams_details.append(
                {
                    "config": income_config,
                    "start_month": start_month,
                    "duration_months": duration_months,
                    "nominal_fixed_monthly_amount": None,
                }
            )

        first_year_gross_withdrawal = 0.0
        first_year_real_gross_withdrawal = 0.0
        # True iff every retirement year funded spending (portfolio and/or other income).
        # A $0 portfolio is allowed when net other income covers expenses.
        path_succeeded = not pre_retirement_tax_failed
        if pre_retirement_tax_failed:
            years_to_ruin = 0.0

        # --- DECUMULATION (RETIREMENT) PHASE ---
        for year_num in range(p.retirement_years):
            if pre_retirement_tax_failed:
                break
            total_gross_withdraw_inv1_this_year = 0.0
            total_gross_withdraw_inv2_this_year = 0.0
            total_real_gross_withdraw_this_year = 0.0
            year_funding_failed = False

            for month_in_ret_year_idx in range(MONTHS_PER_YEAR):
                retirement_month_index = (
                    year_num * MONTHS_PER_YEAR + month_in_ret_year_idx
                )
                price_level_at_month_start = master_cumulative_inflation
                nominal_monthly_expenses = (
                    p.monthly_expenses * price_level_at_month_start
                )

                net_other_monthly_income = 0.0
                for stream_detail in path_specific_other_income_streams_details:
                    start_month = stream_detail["start_month"]
                    duration_months = stream_detail["duration_months"]
                    is_active = retirement_month_index >= start_month and (
                        duration_months is None
                        or retirement_month_index < start_month + duration_months
                    )
                    if not is_active:
                        continue

                    income_config = stream_detail["config"]
                    if income_config.inflation_indexed:
                        nominal_monthly_income = (
                            income_config.monthly_amount_today
                            * price_level_at_month_start
                        )
                    else:
                        if stream_detail["nominal_fixed_monthly_amount"] is None:
                            stream_detail["nominal_fixed_monthly_amount"] = (
                                income_config.monthly_amount_today
                                * price_level_at_month_start
                            )
                        nominal_monthly_income = stream_detail[
                            "nominal_fixed_monthly_amount"
                        ]
                    net_other_monthly_income += nominal_monthly_income * (
                        1.0 - income_config.tax_rate
                    )

                monthly_withdrawal_needed = max(
                    0.0,
                    nominal_monthly_expenses - net_other_monthly_income,
                )

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

                gain_inv1_tax_year_acc += balance_inv1 * (
                    monthly_gross_inv1 - 1.0
                )
                gain_inv2_tax_year_acc += balance_inv2 * (
                    monthly_gross_inv2 - 1.0
                )
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

                net_capacity_inv1 = self._net_liquidation_value(
                    balance_inv1,
                    cost_basis_inv1,
                    p.inv1_use_realized_gains_tax_system,
                    p.inv1_realized_gains_tax_rate,
                )
                net_capacity_inv2 = self._net_liquidation_value(
                    balance_inv2,
                    cost_basis_inv2,
                    p.inv2_use_realized_gains_tax_system,
                    p.inv2_realized_gains_tax_rate,
                )
                total_net_capacity = net_capacity_inv1 + net_capacity_inv2
                actual_monthly_withdrawal_target = max(
                    0.0,
                    min(monthly_withdrawal_needed, total_net_capacity),
                )
                if (
                    monthly_withdrawal_needed > SMALL_EPSILON
                    and actual_monthly_withdrawal_target
                    < monthly_withdrawal_needed - SMALL_EPSILON
                ):
                    year_funding_failed = True

                net_prop1 = (
                    net_capacity_inv1 / total_net_capacity
                    if total_net_capacity > SMALL_EPSILON
                    else p.allocation_inv1_pct
                )
                net_prop2 = 1.0 - net_prop1

                balance_inv1, cost_basis_inv1, gw1, nw1 = (
                    self._calculate_withdrawal_and_update(
                        balance_inv1,
                        cost_basis_inv1,
                        actual_monthly_withdrawal_target * net_prop1,
                        p.inv1_use_realized_gains_tax_system,
                        p.inv1_realized_gains_tax_rate,
                    )
                )
                total_gross_withdraw_inv1_this_year += gw1

                balance_inv2, cost_basis_inv2, gw2, nw2 = (
                    self._calculate_withdrawal_and_update(
                        balance_inv2,
                        cost_basis_inv2,
                        actual_monthly_withdrawal_target * net_prop2,
                        p.inv2_use_realized_gains_tax_system,
                        p.inv2_realized_gains_tax_rate,
                    )
                )
                total_gross_withdraw_inv2_this_year += gw2
                total_real_gross_withdraw_this_year += (
                    (gw1 + gw2)
                    * inflation_at_retirement
                    / max(price_level_at_month_start, SMALL_EPSILON)
                )

                net_cash_delivered = nw1 + nw2
                if (
                    monthly_withdrawal_needed > SMALL_EPSILON
                    and net_cash_delivered
                    < monthly_withdrawal_needed - SMALL_EPSILON
                ):
                    year_funding_failed = True

                balance_inv1, cost_basis_inv1, balance_inv2, cost_basis_inv2 = (
                    self._rebalance_portfolio(
                        balance_inv1, cost_basis_inv1, balance_inv2, cost_basis_inv2
                    )
                )

                absolute_month_number = (
                    working_months + retirement_month_index + 1
                )
                if (
                    not year_funding_failed
                    and absolute_month_number % MONTHS_PER_YEAR == 0
                ):
                    (
                        balance_inv1,
                        cost_basis_inv1,
                        balance_inv2,
                        cost_basis_inv2,
                        tax_failed,
                    ) = self._apply_annual_gain_taxes(
                        balance_inv1,
                        cost_basis_inv1,
                        balance_inv2,
                        cost_basis_inv2,
                        gain_inv1_tax_year_acc,
                        gain_inv2_tax_year_acc,
                    )
                    gain_inv1_tax_year_acc = 0.0
                    gain_inv2_tax_year_acc = 0.0
                    if tax_failed:
                        year_funding_failed = True

                if year_funding_failed:
                    years_to_ruin = (
                        retirement_month_index + 1
                    ) / MONTHS_PER_YEAR
                    break

            year_gross_withdrawal = (
                total_gross_withdraw_inv1_this_year + total_gross_withdraw_inv2_this_year
            )
            # Each monthly gross draw has already been deflated to retirement-date $.
            if balance_at_retirement_start > SMALL_EPSILON:
                year_wr_pct = (
                    total_real_gross_withdraw_this_year
                    / balance_at_retirement_start
                ) * 100.0
            else:
                year_wr_pct = 0.0

            if year_funding_failed:
                path_succeeded = False
                if math.isnan(years_to_ruin):
                    years_to_ruin = (
                        retirement_month_index + 1
                    ) / MONTHS_PER_YEAR
                yearly_trajectory.append(max(0.0, balance_inv1 + balance_inv2))
                trajectory_price_levels.append(master_cumulative_inflation)
                # A partial failure year is not comparable with full annual rates.
                withdrawal_rate_trajectory.append(float("nan"))
                if year_num == 0:
                    first_year_gross_withdrawal = year_gross_withdrawal
                    first_year_real_gross_withdrawal = (
                        total_real_gross_withdraw_this_year
                    )
                break

            withdrawal_rate_trajectory.append(year_wr_pct)

            if year_num == 0:
                first_year_gross_withdrawal = year_gross_withdrawal
                first_year_real_gross_withdrawal = (
                    total_real_gross_withdraw_this_year
                )

            yearly_trajectory.append(balance_inv1 + balance_inv2)
            trajectory_price_levels.append(master_cumulative_inflation)

        # Close out a final partial tax period so reported terminal wealth is net of
        # accrued annual-tax liability. Regular tax dates remain absolute 12-month
        # boundaries and are not reset by retirement.
        total_simulation_months = (
            working_months + p.retirement_years * MONTHS_PER_YEAR
        )
        if (
            path_succeeded
            and total_simulation_months % MONTHS_PER_YEAR != 0
        ):
            (
                balance_inv1,
                cost_basis_inv1,
                balance_inv2,
                cost_basis_inv2,
                tax_failed,
            ) = self._apply_annual_gain_taxes(
                balance_inv1,
                cost_basis_inv1,
                balance_inv2,
                cost_basis_inv2,
                gain_inv1_tax_year_acc,
                gain_inv2_tax_year_acc,
            )
            if tax_failed:
                path_succeeded = False
                years_to_ruin = float(p.retirement_years)
            if yearly_trajectory:
                yearly_trajectory[-1] = balance_inv1 + balance_inv2

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
            "First Year Real Gross Withdrawal": first_year_real_gross_withdrawal,
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
        Optional[List[int]],
    ]:
        """
        Runs multiple simulation paths, either sequentially or in parallel.

        Uses common random numbers: path seeds are derived from the active seed
        stream and are identical across different working_months candidates.

        Returns summary_df, nominal trajectory percentiles, sample paths,
        withdrawal-rate percentiles, real trajectory percentiles, real sample paths,
        and the number of full-year withdrawal-rate observations at each year.
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
                "First Year Real Gross Withdrawal": r.get(
                    "First Year Real Gross Withdrawal",
                    r["First Year Gross Withdrawal"],
                ),
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
        wr_observation_counts: Optional[List[int]] = None
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
                    wr_observation_counts = [
                        int(v) for v in wr_df.count(axis=1).tolist()
                    ]
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
            wr_observation_counts,
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
        Estimates the minimum working months that achieve the target probability.

        Uses coarse bracketing and bisection, then verifies every month in the
        statistically plausible transition region. The result remains a Monte Carlo
        estimate; an independent larger final run may differ.

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
        probability_cache: Dict[int, float] = {}

        if verbose:
            logger.info(
                f"Estimating working months to achieve "
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
            if months in probability_cache:
                return probability_cache[months]
            search_iteration += 1
            if verbose:
                logger.info(
                    f"Search iter {search_iteration}: Testing {months} m "
                    f"({months / MONTHS_PER_YEAR:.1f} yrs) with {sim_count} sims."
                )
            summary_df, _, _, _, _, _, _ = self.run_monte_carlo_simulations(
                months, sim_count
            )
            prob = self._success_probability(summary_df)
            probability_cache[months] = prob
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

        # Monte Carlo estimates can be locally non-monotone. Verify every month
        # starting one tested point before the first statistically plausible target
        # region (a conservative three-sigma worst-case binomial margin).
        uncertainty_margin_pct = min(
            100.0,
            150.0 / math.sqrt(sim_count),
        )
        tested_before_best = sorted(
            month for month in probability_cache if month <= best
        )
        near_target_index = next(
            (
                i
                for i, month in enumerate(tested_before_best)
                if probability_cache[month]
                >= target_probability_pct - uncertainty_margin_pct
            ),
            len(tested_before_best) - 1,
        )
        verification_index = max(0, near_target_index - 1)
        verification_start = max(
            starting_working_months,
            tested_before_best[verification_index],
        )
        if verbose:
            logger.info(
                f"  Verifying each month from {verification_start} to {best} "
                "to handle locally non-monotone Monte Carlo estimates."
            )
        for month in range(verification_start, best + 1):
            _test(month)

        qualifying_months = [
            month
            for month, probability in probability_cache.items()
            if (
                starting_working_months <= month <= best
                and probability >= target_probability_pct
            )
        ]
        if qualifying_months:
            best = min(qualifying_months)
            best_prob = probability_cache[best]

        if verbose:
            logger.info(
                f"  Search complete: estimated minimum {best} months "
                f"({best / MONTHS_PER_YEAR:.1f} yrs) with prob {best_prob:.2f}%."
            )
        return best, best_prob, search_curve
