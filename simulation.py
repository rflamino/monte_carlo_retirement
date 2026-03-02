import numpy as np
import pandas as pd
import multiprocessing
from typing import Dict, Union, List, Tuple, Optional
from loguru import logger

from config import Config
from constants import (
    SMALL_EPSILON,
    MONTHS_PER_YEAR,
    MINIMUM_SIMULATIONS_FOR_SEARCH_STEP,
)
from utils import _generate_seed_from_timestamp


class RetirementMonteCarloSimulator:
    """
    A Monte Carlo simulator for retirement planning.

    Simulates portfolio performance over a working accumulation phase and a retirement decumulation phase,
    taking into account inflation, taxes, contributions, Expenses, and other variables.
    """

    def __init__(self, params_model: Config, main_seed_override: Optional[int] = None):
        self.params_model = params_model.model_copy(
            deep=True
        )  # Use a copy to modify (e.g. income stream nominal values)

        if main_seed_override is not None:
            self.main_seed = main_seed_override
        elif self.params_model.seed is not None:
            self.main_seed = self.params_model.seed
        else:
            self.main_seed = _generate_seed_from_timestamp()
        logger.info(
            f"Simulator initialized for scenario '{self.params_model.Nickname}' with main seed: {self.main_seed}"
        )

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

    def _run_single_simulation_path(
        self, working_months: int, path_seed: int
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Runs a single simulation path for a given number of working months and seed.

        Args:
            working_months: Number of months to simulate working/accumulation phase.
            path_seed: Random seed for this specific simulation path.

        Returns:
            A dictionary containing simulation results: 'Start Balance', 'Final Balance', and 'Trajectory'.
        """
        np.random.seed(path_seed)
        p = self.params_model

        yearly_trajectory: List[float] = [p.initial_balance]
        # Store master cumulative inflation factors at the start of each year (idx 0 = T0, idx 1 = start of year 1, etc.)
        yearly_master_inflation_factors: List[float] = [1.0]

        balance_inv1 = p.initial_balance * p.allocation_inv1_pct
        balance_inv2 = p.initial_balance * p.allocation_inv2_pct
        cost_basis_inv1 = balance_inv1
        cost_basis_inv2 = balance_inv2

        current_monthly_contribution = p.monthly_contribution
        bal_inv1_start_tax_year_acc = balance_inv1
        bal_inv2_start_tax_year_acc = balance_inv2
        contrib_inv1_tax_year, contrib_inv2_tax_year = 0.0, 0.0

        # Master cumulative inflation factor from T=0
        master_cumulative_inflation: float = (
            1.0  # Starts at 1.0 at T=0 (simulation start)
        )
        current_year_annual_inflation = np.random.normal(
            p.inflation_rate_mean, p.inflation_rate_volatility
        )  # Inflation for year 0

        # --- ACCUMULATION (WORKING) PHASE ---
        for m_idx in range(1, working_months + 1):
            if (m_idx - 1) % MONTHS_PER_YEAR == 0:  # Start of a new year
                if m_idx > 1:  # Not the very first month of simulation
                    master_cumulative_inflation *= (
                        1 + current_year_annual_inflation
                    )  # Apply previous year's inflation
                    yearly_master_inflation_factors.append(master_cumulative_inflation)
                    current_year_annual_inflation = np.random.normal(
                        p.inflation_rate_mean, p.inflation_rate_volatility
                    )
                    if p.contribution_growth_rate_annual > 0:
                        current_monthly_contribution *= (
                            1 + p.contribution_growth_rate_annual
                        )
                # For the very first year (m_idx=1 up to 12), current_year_annual_inflation is already set for year 0
                # master_cumulative_inflation is 1.0 for year 0. yearly_master_inflation_factors[0] is 1.0

            monthly_ret_inv1 = np.random.normal(
                p.inv1_returns_mean / MONTHS_PER_YEAR,
                p.inv1_returns_volatility / np.sqrt(MONTHS_PER_YEAR),
            )
            monthly_ret_inv2_premium = np.random.normal(
                p.inv2_premium_over_inflation_mean / MONTHS_PER_YEAR,
                p.inv2_premium_over_inflation_volatility / np.sqrt(MONTHS_PER_YEAR),
            )
            # Inv2 return uses the current year's annual inflation, divided monthly
            monthly_ret_inv2 = (
                current_year_annual_inflation / MONTHS_PER_YEAR
            ) + monthly_ret_inv2_premium

            balance_inv1 *= 1 + monthly_ret_inv1
            balance_inv2 *= 1 + monthly_ret_inv2

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

                bal_inv1_start_tax_year_acc = (
                    total_balance * p.allocation_inv1_pct
                )  # After tax, re-allocate for next year start
                bal_inv2_start_tax_year_acc = total_balance * p.allocation_inv2_pct
                contrib_inv1_tax_year, contrib_inv2_tax_year = 0.0, 0.0

        # End of accumulation phase, apply last year's inflation to master factor
        if working_months > 0:
            master_cumulative_inflation *= 1 + current_year_annual_inflation
            yearly_master_inflation_factors.append(
                master_cumulative_inflation
            )  # Factor at retirement start

        balance_at_retirement_start = balance_inv1 + balance_inv2
        if working_months > 0 and working_months % MONTHS_PER_YEAR != 0:
            if (
                not yearly_trajectory
                or abs(yearly_trajectory[-1] - balance_at_retirement_start)
                > SMALL_EPSILON
            ):
                yearly_trajectory.append(balance_at_retirement_start)
        elif (
            working_months == 0 and not yearly_trajectory
        ):  # if starting directly in retirement
            yearly_trajectory.append(
                p.initial_balance
            )  # yearly_master_inflation_factors[0] is 1.0

        # Pre-calculate fixed nominal monthly amounts for non-inflation-indexed income streams
        path_specific_other_income_streams_details = []
        num_working_years_for_indexing = (
            (working_months + MONTHS_PER_YEAR - 1) // MONTHS_PER_YEAR
            if working_months > 0
            else 0
        )

        for income_config in p.other_income_streams:
            stream_detail = income_config.model_copy(deep=True)
            # Determine the master inflation factor at the year this stream starts
            year_income_starts_abs_sim_idx = (
                num_working_years_for_indexing
                + income_config.start_after_retirement_years
            )

            if year_income_starts_abs_sim_idx < len(yearly_master_inflation_factors):
                stream_detail._master_inflation_at_start = (
                    yearly_master_inflation_factors[year_income_starts_abs_sim_idx]
                )
            else:
                # This would require projecting inflation factors further.
                pass

            if (
                not income_config.inflation_indexed
                and stream_detail._master_inflation_at_start is not None
            ):  # Will be set later if None now
                stream_detail._nominal_fixed_monthly_amount = (
                    income_config.monthly_amount_today
                    * stream_detail._master_inflation_at_start
                )
            path_specific_other_income_streams_details.append(stream_detail)

        # --- DECUMULATION (RETIREMENT) PHASE ---
        # master_cumulative_inflation is already at the value for the start of retirement.
        # yearly_master_inflation_factors also contains this.

        for year_num in range(
            p.retirement_years
        ):  # year_num is 0 for first year of retirement
            if (
                balance_inv1 + balance_inv2 <= SMALL_EPSILON
                and p.monthly_expenses > SMALL_EPSILON
            ):
                break

            # Inflation for this retirement year
            annual_inflation_this_ret_year = np.random.normal(
                p.inflation_rate_mean, p.inflation_rate_volatility
            )
            master_cumulative_inflation *= (
                1 + annual_inflation_this_ret_year
            )  # Update master inflation factor
            yearly_master_inflation_factors.append(
                master_cumulative_inflation
            )  # Store it

            # Now that yearly_master_inflation_factors is populated for this year, update any income streams
            # that didn't have their _master_inflation_at_start set yet.
            for stream_detail in path_specific_other_income_streams_details:
                if (
                    not stream_detail.inflation_indexed
                    and stream_detail._master_inflation_at_start is None
                ):
                    year_income_starts_abs_sim_idx = (
                        num_working_years_for_indexing
                        + stream_detail.start_after_retirement_years
                    )
                    if year_income_starts_abs_sim_idx < len(
                        yearly_master_inflation_factors
                    ):
                        stream_detail._master_inflation_at_start = (
                            yearly_master_inflation_factors[
                                year_income_starts_abs_sim_idx
                            ]
                        )
                        stream_detail._nominal_fixed_monthly_amount = (
                            stream_detail.monthly_amount_today
                            * stream_detail._master_inflation_at_start
                        )

            # Nominal annual expenses for this retirement year
            nominal_annual_expenses = (
                p.monthly_expenses * MONTHS_PER_YEAR * master_cumulative_inflation
            )

            net_other_annual_income = 0.0

            for income_stream_details in path_specific_other_income_streams_details:
                if year_num >= income_stream_details.start_after_retirement_years:
                    if (
                        income_stream_details.duration_years is None
                        or (
                            year_num
                            - income_stream_details.start_after_retirement_years
                        )
                        < income_stream_details.duration_years
                    ):
                        current_nominal_monthly_val: float
                        if income_stream_details.inflation_indexed:
                            current_nominal_monthly_val = (
                                income_stream_details.monthly_amount_today
                                * master_cumulative_inflation
                            )
                        else:  # Not inflation_indexed, use pre-calculated fixed nominal amount
                            if (
                                income_stream_details._nominal_fixed_monthly_amount
                                is not None
                            ):
                                current_nominal_monthly_val = (
                                    income_stream_details._nominal_fixed_monthly_amount
                                )
                            else:  # Fallback if somehow not calculated (should not happen)
                                current_nominal_monthly_val = (
                                    income_stream_details.monthly_amount_today
                                )  # Effectively T=0 nominal value

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

            bal_inv1_start_tax_year_ret, bal_inv2_start_tax_year_ret = (
                balance_inv1,
                balance_inv2,
            )
            total_gross_withdraw_inv1_this_year, total_gross_withdraw_inv2_this_year = (
                0.0,
                0.0,
            )

            for month_in_ret_year_idx in range(MONTHS_PER_YEAR):
                total_balance_before_month = balance_inv1 + balance_inv2
                if (
                    total_balance_before_month <= SMALL_EPSILON
                    and monthly_withdrawal_needed > SMALL_EPSILON
                ):
                    break

                m_ret1 = np.random.normal(
                    p.inv1_returns_mean / MONTHS_PER_YEAR,
                    p.inv1_returns_volatility / np.sqrt(MONTHS_PER_YEAR),
                )
                m_ret2_prem = np.random.normal(
                    p.inv2_premium_over_inflation_mean / MONTHS_PER_YEAR,
                    p.inv2_premium_over_inflation_volatility / np.sqrt(MONTHS_PER_YEAR),
                )
                # Monthly part of Inv2 return uses this retirement year's annual inflation
                m_ret2 = (
                    annual_inflation_this_ret_year / MONTHS_PER_YEAR
                ) + m_ret2_prem

                balance_inv1 *= 1 + m_ret1
                balance_inv2 *= 1 + m_ret2
                total_after_growth = balance_inv1 + balance_inv2

                if (
                    total_after_growth <= SMALL_EPSILON
                    and monthly_withdrawal_needed > SMALL_EPSILON
                ):
                    balance_inv1 = max(0, balance_inv1)
                    balance_inv2 = max(0, balance_inv2)
                    break

                actual_monthly_withdrawal_target = min(
                    monthly_withdrawal_needed, total_after_growth
                )
                actual_monthly_withdrawal_target = max(
                    0, actual_monthly_withdrawal_target
                )

                prop1 = (
                    balance_inv1 / total_after_growth
                    if total_after_growth > SMALL_EPSILON
                    else p.allocation_inv1_pct
                )
                prop2 = 1.0 - prop1

                balance_inv1_after_growth, cost_basis_inv1_after_growth = (
                    balance_inv1,
                    cost_basis_inv1,
                )
                balance_inv2_after_growth, cost_basis_inv2_after_growth = (
                    balance_inv2,
                    cost_basis_inv2,
                )

                balance_inv1, cost_basis_inv1, gw1 = (
                    self._calculate_withdrawal_and_update(
                        balance_inv1_after_growth,
                        cost_basis_inv1_after_growth,
                        actual_monthly_withdrawal_target * prop1,
                        p.inv1_use_realized_gains_tax_system,
                        p.inv1_realized_gains_tax_rate,
                    )
                )
                total_gross_withdraw_inv1_this_year += gw1

                balance_inv2, cost_basis_inv2, gw2 = (
                    self._calculate_withdrawal_and_update(
                        balance_inv2_after_growth,
                        cost_basis_inv2_after_growth,
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
                )  # CB cannot exceed balance
                cost_basis_inv2 = min(
                    cost_basis_inv2, balance_inv2 if balance_inv2 > 0 else 0
                )

                balance_inv1, cost_basis_inv1, balance_inv2, cost_basis_inv2 = (
                    self._rebalance_portfolio(
                        balance_inv1, cost_basis_inv1, balance_inv2, cost_basis_inv2
                    )
                )

                total_balance_after_withdrawal_and_rebalance = (
                    balance_inv1 + balance_inv2
                )
                if (
                    total_balance_after_withdrawal_and_rebalance < SMALL_EPSILON
                    and monthly_withdrawal_needed > SMALL_EPSILON
                ):
                    break
            # --- End of Monthly Loop in Retirement Year ---

            # --- End of Retirement Year: Apply annual taxes if not using realized system ---
            if (
                not p.inv1_use_realized_gains_tax_system
                and p.inv1_annual_tax_on_gains_rate > 0
            ):
                # Gain is current EOY balance + gross withdrawals made during year - balance at start of year
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

            if (
                total_balance_after_annual_tax <= SMALL_EPSILON
                and required_annual_withdrawal > SMALL_EPSILON
            ):
                break

            # Rebalance at EOY after taxes (if any) - ensures start of next year is allocated correctly
            if (
                total_balance_after_annual_tax > SMALL_EPSILON
            ):  # Only rebalance if there is money
                balance_inv1 = total_balance_after_annual_tax * p.allocation_inv1_pct
                balance_inv2 = total_balance_after_annual_tax * p.allocation_inv2_pct
                # And apportion total cost basis
                total_cb = cost_basis_inv1 + cost_basis_inv2
                cost_basis_inv1 = total_cb * p.allocation_inv1_pct
                cost_basis_inv2 = total_cb * p.allocation_inv2_pct
                cost_basis_inv1 = min(cost_basis_inv1, balance_inv1)
                cost_basis_inv2 = min(cost_basis_inv2, balance_inv2)

        final_total_balance = balance_inv1 + balance_inv2

        num_working_years = (
            (working_months + MONTHS_PER_YEAR - 1) // MONTHS_PER_YEAR
            if working_months > 0
            else 0
        )
        # Expected length: 1 (initial) + num_working_years (EOY balances) + num_retirement_years (EOY balances)
        expected_len = 1 + num_working_years + p.retirement_years
        current_len = len(yearly_trajectory)

        if current_len < expected_len:
            padding_value = yearly_trajectory[-1] if yearly_trajectory else 0.0
            yearly_trajectory.extend([padding_value] * (expected_len - current_len))
        elif current_len > expected_len:
            yearly_trajectory = yearly_trajectory[:expected_len]

        return {
            "Start Balance": balance_at_retirement_start,
            "Final Balance": max(0, final_total_balance),
            "Trajectory": yearly_trajectory,
        }

    def run_monte_carlo_simulations(
        self, working_months: int, num_simulations: int
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[List[List[float]]]]:
        """
        Runs multiple simulation paths, either sequentially or in parallel.
        """
        path_seeds = [self.main_seed + i for i in range(num_simulations)]
        num_procs_to_use = (
            self.params_model.num_processes
            if self.params_model.num_processes is not None
            else 1
        )

        all_results_list: List[Dict[str, Union[float, List[float]]]]

        if num_procs_to_use <= 1:
            logger.debug(
                f"Running {num_simulations} simulations sequentially for {working_months} working months."
            )
            all_results_list = [
                self._run_single_simulation_path(working_months, seed)
                for seed in path_seeds
            ]
        else:
            logger.debug(
                f"Running {num_simulations} simulations in parallel using {num_procs_to_use} processes for {working_months} working months."
            )
            args_for_starmap = [(working_months, seed) for seed in path_seeds]
            try:
                # Consider get_context for spawn/forkserver if issues arise, esp. on macOS or Windows.
                # context = multiprocessing.get_context('spawn')
                # with context.Pool(processes=num_procs_to_use) as pool:
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
            {"Start Balance": r["Start Balance"], "Final Balance": r["Final Balance"]}
            for r in all_results_list
        ]
        summary_df = pd.DataFrame(summary_results_list)

        trajectories_raw = [
            r["Trajectory"]
            for r in all_results_list
            if "Trajectory" in r and r["Trajectory"]
        ]

        trajectory_percentiles_df: Optional[pd.DataFrame] = None
        sample_trajectories_list: Optional[List[List[float]]] = None

        if trajectories_raw:
            try:
                # Ensure all trajectories have the same length (padding is done in _run_single_simulation_path)
                # Check for length consistency before creating DataFrame if strictness is needed
                min_len = min(map(len, trajectories_raw))
                max_len = max(map(len, trajectories_raw))
                if min_len != max_len:
                    logger.warning(
                        f"Trajectory lengths are inconsistent: min={min_len}, max={max_len}. This might affect percentile calculations. Using min_len for safety if issues occur, but padding should handle."
                    )

                trajectory_df = pd.DataFrame(
                    trajectories_raw
                ).transpose()  # Rows are years, columns are simulations

                if not trajectory_df.empty:
                    percentiles_to_calc = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
                    trajectory_percentiles_df = trajectory_df.quantile(
                        percentiles_to_calc, axis=1
                    ).transpose()

                    num_sample_paths = min(len(trajectory_df.columns), 5)
                    if num_sample_paths > 0:
                        actual_num_to_sample = min(
                            num_sample_paths, trajectory_df.shape[1]
                        )
                        sample_trajectories_list = trajectory_df.sample(
                            n=actual_num_to_sample, axis=1, random_state=self.main_seed
                        ).values.T.tolist()
            except ValueError as ve:  # Handles cases like inconsistent trajectory lengths if padding failed
                logger.error(
                    f"Error processing trajectories, possibly due to inconsistent lengths: {ve}",
                    exc_info=True,
                )
            except Exception as e:
                logger.error(f"Error processing trajectories: {e}", exc_info=True)

        return summary_df, trajectory_percentiles_df, sample_trajectories_list

    def _get_dynamic_sim_count(
        self,
        achieved_prob_prev_iter: float,
        target_prob_pct: float,
        base_sim_count: int,
    ) -> int:
        """
        Adjusts the number of simulations for the next search iteration.
        More simulations are used when closer to the target for higher precision.
        """
        if achieved_prob_prev_iter < 0:  # First iteration or no prior data
            num_s = int(base_sim_count * 0.5)
        else:
            delta_prob = abs(achieved_prob_prev_iter - target_prob_pct)
            if delta_prob <= 1.0:
                num_s = int(base_sim_count * 4.00)  # Very close
            elif delta_prob <= 1.5:
                num_s = int(base_sim_count * 2.00)  # Even closer
            elif delta_prob <= 2.0:
                num_s = int(base_sim_count * 1.00)  # Getting closer
            elif delta_prob <= 3.0:
                num_s = int(base_sim_count * 0.80)
            elif delta_prob <= 4.0:
                num_s = int(base_sim_count * 0.70)
            elif delta_prob <= 5.0:
                num_s = int(base_sim_count * 0.50)
            else:
                num_s = int(base_sim_count * 0.20)  # Far

        # Ensure the number of simulations is not excessively small or large
        final_sim_count = max(
            MINIMUM_SIMULATIONS_FOR_SEARCH_STEP, min(num_s, base_sim_count * 5)
        )  # Cap max multiplier
        delta_val_display = (
            abs(achieved_prob_prev_iter - target_prob_pct)
            if achieved_prob_prev_iter >= 0
            else -1.0
        )
        logger.debug(
            f"Dynamic sim count: prev_prob={achieved_prob_prev_iter:.2f}, target={target_prob_pct:.2f}, delta={delta_val_display:.2f}, base_count={base_sim_count}, new_count={final_sim_count}"
        )
        return final_sim_count

    def find_minimum_working_months(self, verbose: bool = True) -> Tuple[int, float]:
        """
        Iteratively searches for the minimum number of working months required
        to achieve the target success probability.
        """
        p = self.params_model
        starting_working_months = p.starting_working_months_search
        target_probability_pct = p.target_probability
        base_num_simulations_per_test = p.num_simulations_search

        max_search_increase_months = (
            70 * MONTHS_PER_YEAR
        )  # Max ~70 additional years of search
        max_total_months_to_test = starting_working_months + max_search_increase_months

        current_test_months = starting_working_months
        achieved_probability_at_prev_months = (
            -1.0
        )  # Probability from the *previous* test_months iteration
        highest_prob_if_target_not_met = -1.0

        if verbose:
            logger.info(
                f"Searching for minimum working months to achieve {target_probability_pct:.2f}% success for '{p.Nickname}'."
            )
            logger.info(
                f"Starting search from {starting_working_months} months. Base simulations per test step: {base_num_simulations_per_test}."
            )

        search_iteration = 0
        # Simple linear scan; could be optimized (e.g., binary search if prob is monotonic)
        # For now, dynamic sim count helps manage computational cost.
        months_increment_step = 12  # Search year by year initially for speed

        while current_test_months <= max_total_months_to_test:
            search_iteration += 1
            iter_sim_count = self._get_dynamic_sim_count(
                achieved_probability_at_prev_months,
                target_probability_pct,
                base_num_simulations_per_test,
            )

            if verbose:
                logger.info(
                    f"Search iter {search_iteration}: Testing {current_test_months} m ({current_test_months / MONTHS_PER_YEAR:.1f} yrs) with {iter_sim_count} sims."
                )

            summary_df, _, _ = self.run_monte_carlo_simulations(
                current_test_months, iter_sim_count
            )

            current_iter_achieved_probability_pct: float
            if summary_df.empty:
                logger.warning(
                    f"Search iter {search_iteration}: No sim results for {current_test_months} months."
                )
                current_iter_achieved_probability_pct = 0.0
            else:
                current_iter_achieved_probability_pct = (
                    summary_df["Final Balance"] > SMALL_EPSILON
                ).mean() * 100.0

            if verbose:
                logger.info(
                    f"  Search iter {search_iteration}: Prob for {current_test_months} m: {current_iter_achieved_probability_pct:.2f}% (Target: {target_probability_pct:.2f}%)"
                )

            if current_iter_achieved_probability_pct > highest_prob_if_target_not_met:
                highest_prob_if_target_not_met = current_iter_achieved_probability_pct

            if current_iter_achieved_probability_pct >= target_probability_pct:
                if verbose:
                    logger.info(f"  Target met at {current_test_months} months.")
                # If we overshot with a large step, we might want to backtrack and refine
                if months_increment_step > 1:
                    logger.info(
                        f"  Target met with step {months_increment_step}. Refining search by smaller steps..."
                    )
                    # Backtrack one big step, then search month by month
                    current_test_months = max(
                        starting_working_months,
                        current_test_months - months_increment_step + 1,
                    )
                    months_increment_step = 1  # Switch to fine-grained search
                    achieved_probability_at_prev_months = (
                        -1
                    )  # Reset for dynamic sim count when refining
                    continue  # Restart loop with new current_test_months and step
                return current_test_months, current_iter_achieved_probability_pct

            achieved_probability_at_prev_months = (
                current_iter_achieved_probability_pct  # Store for next dynamic count
            )

            # Adaptive step: if far from target, take larger steps.
            # This is a simple heuristic for adapting step size.
            if (
                target_probability_pct - current_iter_achieved_probability_pct > 20
                and months_increment_step < 24
            ):  # If very far
                months_increment_step = 24  # 2 years
            elif (
                target_probability_pct - current_iter_achieved_probability_pct > 10
                and months_increment_step < 12
            ):  # If moderately far
                months_increment_step = 12  # 1 year
            elif (
                target_probability_pct - current_iter_achieved_probability_pct > 3
                and months_increment_step < 6
            ):
                months_increment_step = 6  # 6 months
            else:  # Close or overshot but not met target (should refine)
                months_increment_step = (
                    1  # Search month by month when close or needing refinement
                )

            current_test_months += months_increment_step

        if verbose:
            logger.warning(
                f"Search for '{p.Nickname}' reached max limit ({max_total_months_to_test / MONTHS_PER_YEAR:.1f} yrs). Target NOT met."
            )
            logger.warning(
                f"Highest probability achieved: {highest_prob_if_target_not_met:.2f}% at/before last test."
            )
        return -1, highest_prob_if_target_not_met
