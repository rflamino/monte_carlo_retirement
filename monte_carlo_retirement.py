# monte_carlo_retirement - Retirement Monte Carlo Simulator
# Author: Reinaldo S. Flamino
# Description: A robust Python-based Monte Carlo simulation tool designed to project portfolio longevity in retirement.

import datetime as _dt
import hashlib
import json
import multiprocessing
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.ticker import FuncFormatter



# Pydantic for configuration validation and data management
from pydantic import BaseModel, Field, field_validator, ValidationInfo

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# --- CONSTANTS ---
MONTHS_PER_YEAR: int = 12
DEFAULT_PLOT_FILENAME: str = "retirement_projection.png"
MINIMUM_SIMULATIONS_FOR_SEARCH_STEP: int = 100 
SMALL_EPSILON: float = 1e-6 

# --- PYDANTIC MODELS FOR CONFIGURATION ---

class OtherIncomeStreamConfig(BaseModel):
    """Configuration for an additional income stream during retirement."""
    name: str = Field(..., description="Name of the income stream (e.g., 'Rental Income', 'Pension').")
    monthly_amount_today: float = Field(..., ge=0, description="Current monthly amount of this income in today's (T=0) real terms.")
    start_after_retirement_years: int = Field(..., ge=0, description="Years after retirement starts that this income begins.")
    duration_years: Optional[int] = Field(None, ge=0, description="How many years this income lasts. None means indefinitely or until end of retirement.")
    inflation_indexed: bool = Field(True, description="If True, keeps pace with inflation from T=0. If False, its nominal value is fixed based on its value at its start date.")
    tax_rate: float = Field(..., ge=0.0, le=1.0, description="Tax rate applied to this income stream.")
    # Store the pre-calculated nominal value if not inflation_indexed, or the T=0 real value if it is
    _nominal_fixed_monthly_amount: Optional[float] = None 
    _master_inflation_at_start: Optional[float] = None 


class Config(BaseModel):
    """Main configuration model for the retirement simulation."""
    Nickname: str = Field("DefaultScenario", alias="scenario", description="A nickname for this simulation scenario.")
    initial_balance: float = Field(..., ge=0)
    monthly_contribution: float = Field(..., ge=0)
    contribution_growth_rate_annual: float = Field(0.0, ge=0)
    monthly_expenses: float = Field(..., ge=0, description="Monthly expenses in today's (T=0) real terms.")
    retirement_years: int = Field(..., gt=0)

    allocation_inv1_pct: float = Field(..., ge=0.0, le=1.0)
    inv1_returns_mean: float = Field(...)
    inv1_returns_volatility: float = Field(..., ge=0.0)
    inv1_annual_tax_on_gains_rate: float = Field(..., ge=0.0, le=1.0)
    inv1_realized_gains_tax_rate: float = Field(0.0, ge=0.0, le=1.0)
    inv1_use_realized_gains_tax_system: bool = Field(False)

    inv2_premium_over_inflation_mean: float = Field(...)
    inv2_premium_over_inflation_volatility: float = Field(..., ge=0.0)
    inv2_annual_tax_on_gains_rate: float = Field(..., ge=0.0, le=1.0)
    inv2_realized_gains_tax_rate: float = Field(0.0, ge=0.0, le=1.0)
    inv2_use_realized_gains_tax_system: bool = Field(True)

    inflation_rate_mean: float = Field(...)
    inflation_rate_volatility: float = Field(..., ge=0.0)

    num_simulations_main: int = Field(..., gt=0)
    num_simulations_search: int = Field(..., gt=MINIMUM_SIMULATIONS_FOR_SEARCH_STEP // 2)
    target_probability: float = Field(..., ge=0.0, le=100.0)
    starting_working_months_search: int = Field(..., ge=0)
    seed: Optional[int] = Field(None)
    num_processes: Optional[int] = Field(1, ge=1)

    other_income_streams: List[OtherIncomeStreamConfig] = Field([])

    model_config = {
        "validate_by_name": True,
        "validate_assignment": True
    }

    @field_validator('inflation_rate_volatility')
    @classmethod
    def check_inflation_volatility(cls, v: float, info: ValidationInfo) -> float:
        if v > 0.05:
            # Safe access to nickname in case validation fails before nickname is set
            scen_name = info.data.get('Nickname', 'N/A')
            logger.warning(f"Inflation volatility ({v*100:.1f}%) is relatively high for scenario '{scen_name}'.")
        return v

    @property
    def allocation_inv2_pct(self) -> float:
        return round(1.0 - self.allocation_inv1_pct, 4)

# --- LOGGER SETUP ---
# logger is imported from loguru directly

# --- UTILITY FUNCTIONS ---
def _generate_seed_from_timestamp() -> int:
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return int.from_bytes(hashlib.sha256(ts.encode()).digest()[:8], "big") % (2**32 - 1)

def load_config_from_json(file_path: str) -> Dict[str, Any]:
    """Loads and returns the configuration dictionary from a JSON file."""
    if not os.path.exists(file_path):
        logger.error(f"Configuration file not found at: {file_path}")
        sys.exit(1)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file '{file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error reading config file: {e}")
        sys.exit(1)

# --- CORE SIMULATION LOGIC ---
class RetirementMonteCarloSimulator:
    """
    A Monte Carlo simulator for retirement planning.

    Simulates portfolio performance over a working accumulation phase and a retirement decumulation phase,
    taking into account inflation, taxes, contributions, Expenses, and other variables.
    """
    def __init__(self, params_model: Config, main_seed_override: Optional[int] = None):
        self.params_model = params_model.model_copy(deep=True) # Use a copy to modify (e.g. income stream nominal values)

        if main_seed_override is not None:
            self.main_seed = main_seed_override
        elif self.params_model.seed is not None:
            self.main_seed = self.params_model.seed
        else:
            self.main_seed = _generate_seed_from_timestamp()
        logger.info(f"Simulator initialized for scenario '{self.params_model.Nickname}' with main seed: {self.main_seed}")

    def _calculate_withdrawal_and_update(
        self,
        bal_inv: float,
        cb_inv: float,
        net_withdrawal_target_for_inv: float,
        use_real_tax: bool,
        real_tax_rate: float
    ) -> Tuple[float, float, float]:
        """
        Calculates the gross withdrawal needed to meet a net target, considering taxes,
        and updates the cost basis.
        """
        final_gross_withdrawal = net_withdrawal_target_for_inv
        principal_component_of_withdrawal = net_withdrawal_target_for_inv

        if use_real_tax and real_tax_rate > 0 and net_withdrawal_target_for_inv > SMALL_EPSILON and bal_inv > SMALL_EPSILON:
            total_gain_in_inv = max(0, bal_inv - cb_inv)
            if total_gain_in_inv > SMALL_EPSILON:
                gain_proportion_of_balance = total_gain_in_inv / bal_inv
                denominator = 1.0 - (gain_proportion_of_balance * real_tax_rate)
                if denominator > SMALL_EPSILON:
                    final_gross_withdrawal = net_withdrawal_target_for_inv / denominator
                else:
                    final_gross_withdrawal = min(net_withdrawal_target_for_inv * 2, bal_inv)
                final_gross_withdrawal = min(final_gross_withdrawal, bal_inv)
                realized_gain_from_this_withdrawal = final_gross_withdrawal * gain_proportion_of_balance
                principal_component_of_withdrawal = final_gross_withdrawal - realized_gain_from_this_withdrawal

        final_gross_withdrawal = min(final_gross_withdrawal, bal_inv)
        principal_component_of_withdrawal = min(principal_component_of_withdrawal, cb_inv)
        new_balance_inv = bal_inv - final_gross_withdrawal
        new_cost_basis_inv = cb_inv - principal_component_of_withdrawal
        return max(0, new_balance_inv), max(0, new_cost_basis_inv), final_gross_withdrawal

    def _run_single_simulation_path(self, working_months: int, path_seed: int) -> Dict[str, Union[float, List[float]]]:
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
        master_cumulative_inflation: float = 1.0 # Starts at 1.0 at T=0 (simulation start)
        current_year_annual_inflation = np.random.normal(p.inflation_rate_mean, p.inflation_rate_volatility) # Inflation for year 0

        # --- ACCUMULATION (WORKING) PHASE ---
        for m_idx in range(1, working_months + 1):


            if (m_idx - 1) % MONTHS_PER_YEAR == 0: # Start of a new year
                if m_idx > 1: # Not the very first month of simulation
                    master_cumulative_inflation *= (1 + current_year_annual_inflation) # Apply previous year's inflation
                    yearly_master_inflation_factors.append(master_cumulative_inflation)
                    current_year_annual_inflation = np.random.normal(p.inflation_rate_mean, p.inflation_rate_volatility)
                    if p.contribution_growth_rate_annual > 0:
                        current_monthly_contribution *= (1 + p.contribution_growth_rate_annual)
                # For the very first year (m_idx=1 up to 12), current_year_annual_inflation is already set for year 0
                # master_cumulative_inflation is 1.0 for year 0. yearly_master_inflation_factors[0] is 1.0

            monthly_ret_inv1 = np.random.normal(p.inv1_returns_mean / MONTHS_PER_YEAR, p.inv1_returns_volatility / np.sqrt(MONTHS_PER_YEAR))
            monthly_ret_inv2_premium = np.random.normal(p.inv2_premium_over_inflation_mean / MONTHS_PER_YEAR, p.inv2_premium_over_inflation_volatility / np.sqrt(MONTHS_PER_YEAR))
            # Inv2 return uses the current year's annual inflation, divided monthly
            monthly_ret_inv2 = (current_year_annual_inflation / MONTHS_PER_YEAR) + monthly_ret_inv2_premium

            balance_inv1 *= (1 + monthly_ret_inv1)
            balance_inv2 *= (1 + monthly_ret_inv2)

            contrib_m_inv1 = current_monthly_contribution * p.allocation_inv1_pct
            contrib_m_inv2 = current_monthly_contribution * p.allocation_inv2_pct
            balance_inv1 += contrib_m_inv1
            cost_basis_inv1 += contrib_m_inv1
            balance_inv2 += contrib_m_inv2
            cost_basis_inv2 += contrib_m_inv2

            contrib_inv1_tax_year += contrib_m_inv1
            contrib_inv2_tax_year += contrib_m_inv2

            # --- Rebalancing with potential realized gains tax ---
            bal1_pre_rebal, cb1_pre_rebal = balance_inv1, cost_basis_inv1
            bal2_pre_rebal, cb2_pre_rebal = balance_inv2, cost_basis_inv2
            total_balance_pre_rebal = bal1_pre_rebal + bal2_pre_rebal
            tax_from_rebalancing = 0.0

            if total_balance_pre_rebal > SMALL_EPSILON:
                target_bal1_ideal = total_balance_pre_rebal * p.allocation_inv1_pct
                amount_to_sell_from_inv1 = 0.0
                amount_to_sell_from_inv2 = 0.0

                if bal1_pre_rebal > target_bal1_ideal + SMALL_EPSILON: # Inv1 is overweight
                    amount_to_sell_from_inv1 = bal1_pre_rebal - target_bal1_ideal
                    if p.inv1_use_realized_gains_tax_system and p.inv1_realized_gains_tax_rate > 0 and amount_to_sell_from_inv1 > 0:
                        gain_in_inv1 = max(0, bal1_pre_rebal - cb1_pre_rebal)
                        if gain_in_inv1 > 0 and bal1_pre_rebal > SMALL_EPSILON:
                            proportion_of_inv1_value_sold = amount_to_sell_from_inv1 / bal1_pre_rebal
                            realized_gain = gain_in_inv1 * proportion_of_inv1_value_sold
                            tax_from_rebalancing = realized_gain * p.inv1_realized_gains_tax_rate
                
                elif bal1_pre_rebal < target_bal1_ideal - SMALL_EPSILON: # Inv2 is overweight
                    # Corrected logic: Calculate target first, then subtract to find sell amount
                    target_bal2_ideal = total_balance_pre_rebal * p.allocation_inv2_pct
                    amount_to_sell_from_inv2 = bal2_pre_rebal - target_bal2_ideal 

                    if p.inv2_use_realized_gains_tax_system and p.inv2_realized_gains_tax_rate > 0 and amount_to_sell_from_inv2 > 0:
                        gain_in_inv2 = max(0, bal2_pre_rebal - cb2_pre_rebal)
                        if gain_in_inv2 > 0 and bal2_pre_rebal > SMALL_EPSILON:
                            proportion_of_inv2_value_sold = amount_to_sell_from_inv2 / bal2_pre_rebal
                            realized_gain = gain_in_inv2 * proportion_of_inv2_value_sold
                            tax_from_rebalancing = realized_gain * p.inv2_realized_gains_tax_rate
                
                total_balance_after_rebal_tax = total_balance_pre_rebal - tax_from_rebalancing
                
                # Update balances to target allocations of the new total
                balance_inv1 = total_balance_after_rebal_tax * p.allocation_inv1_pct
                balance_inv2 = total_balance_after_rebal_tax * p.allocation_inv2_pct

                # Adjust cost basis:
                # The total cost basis is preserved unless tax reduces the overall value more than gains.
                # Simpler: re-apportion total cost basis. First, reduce CB of sold asset proportionally.
                current_total_cost_basis = cb1_pre_rebal + cb2_pre_rebal
                if amount_to_sell_from_inv1 > 0 and bal1_pre_rebal > SMALL_EPSILON:
                    prop_sold = amount_to_sell_from_inv1 / bal1_pre_rebal
                    cb_reduction = cb1_pre_rebal * prop_sold
                    current_total_cost_basis -= cb_reduction # Part of CB is "used up" by sale
                    current_total_cost_basis += amount_to_sell_from_inv1 # And "transferred"
                elif amount_to_sell_from_inv2 > 0 and bal2_pre_rebal > SMALL_EPSILON:
                    prop_sold = amount_to_sell_from_inv2 / bal2_pre_rebal
                    cb_reduction = cb2_pre_rebal * prop_sold
                    current_total_cost_basis -= cb_reduction
                    current_total_cost_basis += amount_to_sell_from_inv2
                
                # Apportion the (potentially adjusted by sale) total cost basis
                cost_basis_inv1 = current_total_cost_basis * p.allocation_inv1_pct
                cost_basis_inv2 = current_total_cost_basis * p.allocation_inv2_pct
            else: # No rebalancing if total balance is zero
                pass # Balances and cost basis remain as they were

            cost_basis_inv1 = min(cost_basis_inv1, balance_inv1 if balance_inv1 > 0 else 0)
            cost_basis_inv2 = min(cost_basis_inv2, balance_inv2 if balance_inv2 > 0 else 0)


            if m_idx % MONTHS_PER_YEAR == 0 or m_idx == working_months:
                eoy_balance_inv1_before_tax, eoy_balance_inv2_before_tax = balance_inv1, balance_inv2
                if not p.inv1_use_realized_gains_tax_system and p.inv1_annual_tax_on_gains_rate > 0:
                    gain_inv1 = eoy_balance_inv1_before_tax - bal_inv1_start_tax_year_acc - contrib_inv1_tax_year
                    if gain_inv1 > 0:
                        balance_inv1 -= gain_inv1 * p.inv1_annual_tax_on_gains_rate
                if not p.inv2_use_realized_gains_tax_system and p.inv2_annual_tax_on_gains_rate > 0:
                    gain_inv2 = eoy_balance_inv2_before_tax - bal_inv2_start_tax_year_acc - contrib_inv2_tax_year
                    if gain_inv2 > 0:
                        balance_inv2 -= gain_inv2 * p.inv2_annual_tax_on_gains_rate
                
                total_balance = balance_inv1 + balance_inv2
                if m_idx % MONTHS_PER_YEAR == 0:
                    yearly_trajectory.append(total_balance)

                bal_inv1_start_tax_year_acc = total_balance * p.allocation_inv1_pct # After tax, re-allocate for next year start
                bal_inv2_start_tax_year_acc = total_balance * p.allocation_inv2_pct
                contrib_inv1_tax_year, contrib_inv2_tax_year = 0.0, 0.0
        
        # End of accumulation phase, apply last year's inflation to master factor
        if working_months > 0:
             master_cumulative_inflation *= (1 + current_year_annual_inflation)
             yearly_master_inflation_factors.append(master_cumulative_inflation) # Factor at retirement start

        balance_at_retirement_start = balance_inv1 + balance_inv2
        if working_months > 0 and working_months % MONTHS_PER_YEAR != 0:
            if not yearly_trajectory or abs(yearly_trajectory[-1] - balance_at_retirement_start) > SMALL_EPSILON:
                yearly_trajectory.append(balance_at_retirement_start)
        elif working_months == 0 and not yearly_trajectory : # if starting directly in retirement
             yearly_trajectory.append(p.initial_balance) # yearly_master_inflation_factors[0] is 1.0


        # Pre-calculate fixed nominal monthly amounts for non-inflation-indexed income streams
        path_specific_other_income_streams_details = []
        num_working_years_for_indexing = (working_months + MONTHS_PER_YEAR -1) // MONTHS_PER_YEAR if working_months > 0 else 0

        for income_config in p.other_income_streams:
            stream_detail = income_config.model_copy(deep=True)
            # Determine the master inflation factor at the year this stream starts
            year_income_starts_abs_sim_idx = num_working_years_for_indexing + income_config.start_after_retirement_years
            
            if year_income_starts_abs_sim_idx < len(yearly_master_inflation_factors):
                 stream_detail._master_inflation_at_start = yearly_master_inflation_factors[year_income_starts_abs_sim_idx]
            else: 
                 # This would require projecting inflation factors further.
                 pass 

            if not income_config.inflation_indexed and stream_detail._master_inflation_at_start is not None: # Will be set later if None now
                stream_detail._nominal_fixed_monthly_amount = income_config.monthly_amount_today * stream_detail._master_inflation_at_start
            path_specific_other_income_streams_details.append(stream_detail)


        # --- DECUMULATION (RETIREMENT) PHASE ---
        # master_cumulative_inflation is already at the value for the start of retirement.
        # yearly_master_inflation_factors also contains this.

        for year_num in range(p.retirement_years): # year_num is 0 for first year of retirement
            if balance_inv1 + balance_inv2 <= SMALL_EPSILON and p.monthly_expenses > SMALL_EPSILON:
                break

            # Inflation for this retirement year
            annual_inflation_this_ret_year = np.random.normal(p.inflation_rate_mean, p.inflation_rate_volatility)
            master_cumulative_inflation *= (1 + annual_inflation_this_ret_year) # Update master inflation factor
            yearly_master_inflation_factors.append(master_cumulative_inflation) # Store it

            # Now that yearly_master_inflation_factors is populated for this year, update any income streams
            # that didn't have their _master_inflation_at_start set yet.
            for stream_detail in path_specific_other_income_streams_details:
                if not stream_detail.inflation_indexed and stream_detail._master_inflation_at_start is None:
                    year_income_starts_abs_sim_idx = num_working_years_for_indexing + stream_detail.start_after_retirement_years
                    if year_income_starts_abs_sim_idx < len(yearly_master_inflation_factors):
                         stream_detail._master_inflation_at_start = yearly_master_inflation_factors[year_income_starts_abs_sim_idx]
                         stream_detail._nominal_fixed_monthly_amount = stream_detail.monthly_amount_today * stream_detail._master_inflation_at_start


            # Nominal annual expenses for this retirement year
            nominal_annual_expenses = p.monthly_expenses * MONTHS_PER_YEAR * master_cumulative_inflation

            net_other_annual_income = 0.0


            for income_stream_details in path_specific_other_income_streams_details:
                if year_num >= income_stream_details.start_after_retirement_years:
                    if income_stream_details.duration_years is None or (year_num - income_stream_details.start_after_retirement_years) < income_stream_details.duration_years:
                        current_nominal_monthly_val: float
                        if income_stream_details.inflation_indexed:
                            current_nominal_monthly_val = income_stream_details.monthly_amount_today * master_cumulative_inflation
                        else: # Not inflation_indexed, use pre-calculated fixed nominal amount
                            if income_stream_details._nominal_fixed_monthly_amount is not None:
                                current_nominal_monthly_val = income_stream_details._nominal_fixed_monthly_amount
                            else: # Fallback if somehow not calculated (should not happen)
                                current_nominal_monthly_val = income_stream_details.monthly_amount_today # Effectively T=0 nominal value
                        
                        stream_annual_pre_tax = current_nominal_monthly_val * MONTHS_PER_YEAR
                        stream_tax = stream_annual_pre_tax * income_stream_details.tax_rate
                        net_other_annual_income += (stream_annual_pre_tax - stream_tax)
            
            required_annual_withdrawal = max(0, nominal_annual_expenses - net_other_annual_income)
            monthly_withdrawal_needed = required_annual_withdrawal / MONTHS_PER_YEAR

            bal_inv1_start_tax_year_ret, bal_inv2_start_tax_year_ret = balance_inv1, balance_inv2
            total_gross_withdraw_inv1_this_year, total_gross_withdraw_inv2_this_year = 0.0, 0.0

            for month_in_ret_year_idx in range(MONTHS_PER_YEAR):
                total_balance_before_month = balance_inv1 + balance_inv2
                if total_balance_before_month <= SMALL_EPSILON and monthly_withdrawal_needed > SMALL_EPSILON:
                    break 

                m_ret1 = np.random.normal(p.inv1_returns_mean/MONTHS_PER_YEAR, p.inv1_returns_volatility/np.sqrt(MONTHS_PER_YEAR))
                m_ret2_prem = np.random.normal(p.inv2_premium_over_inflation_mean/MONTHS_PER_YEAR, p.inv2_premium_over_inflation_volatility/np.sqrt(MONTHS_PER_YEAR))
                # Monthly part of Inv2 return uses this retirement year's annual inflation
                m_ret2 = (annual_inflation_this_ret_year / MONTHS_PER_YEAR) + m_ret2_prem

                balance_inv1 *= (1 + m_ret1)
                balance_inv2 *= (1 + m_ret2)
                total_after_growth = balance_inv1 + balance_inv2

                if total_after_growth <= SMALL_EPSILON and monthly_withdrawal_needed > SMALL_EPSILON:
                    balance_inv1 = max(0, balance_inv1)
                    balance_inv2 = max(0, balance_inv2)
                    break
                
                actual_monthly_withdrawal_target = min(monthly_withdrawal_needed, total_after_growth)
                actual_monthly_withdrawal_target = max(0, actual_monthly_withdrawal_target)

                prop1 = balance_inv1 / total_after_growth if total_after_growth > SMALL_EPSILON else p.allocation_inv1_pct
                prop2 = 1.0 - prop1

                balance_inv1_after_growth, cost_basis_inv1_after_growth = balance_inv1, cost_basis_inv1
                balance_inv2_after_growth, cost_basis_inv2_after_growth = balance_inv2, cost_basis_inv2

                balance_inv1, cost_basis_inv1, gw1 = self._calculate_withdrawal_and_update(
                    balance_inv1_after_growth, cost_basis_inv1_after_growth, actual_monthly_withdrawal_target * prop1,
                    p.inv1_use_realized_gains_tax_system, p.inv1_realized_gains_tax_rate
                )
                total_gross_withdraw_inv1_this_year += gw1

                balance_inv2, cost_basis_inv2, gw2 = self._calculate_withdrawal_and_update(
                    balance_inv2_after_growth, cost_basis_inv2_after_growth, actual_monthly_withdrawal_target * prop2,
                    p.inv2_use_realized_gains_tax_system, p.inv2_realized_gains_tax_rate
                )
                total_gross_withdraw_inv2_this_year += gw2
                
                balance_inv1 = max(0, balance_inv1)
                balance_inv2 = max(0, balance_inv2)
                cost_basis_inv1 = min(cost_basis_inv1, balance_inv1 if balance_inv1 > 0 else 0) # CB cannot exceed balance
                cost_basis_inv2 = min(cost_basis_inv2, balance_inv2 if balance_inv2 > 0 else 0)


                # --- Rebalancing after withdrawal (same logic as in accumulation) ---
                bal1_pre_rebal_ret, cb1_pre_rebal_ret = balance_inv1, cost_basis_inv1
                bal2_pre_rebal_ret, cb2_pre_rebal_ret = balance_inv2, cost_basis_inv2
                total_balance_pre_rebal_ret = bal1_pre_rebal_ret + bal2_pre_rebal_ret
                tax_from_rebalancing_ret = 0.0

                if total_balance_pre_rebal_ret > SMALL_EPSILON:
                    target_bal1_ideal_ret = total_balance_pre_rebal_ret * p.allocation_inv1_pct
                    amount_to_sell_from_inv1_ret = 0.0
                    amount_to_sell_from_inv2_ret = 0.0

                    if bal1_pre_rebal_ret > target_bal1_ideal_ret + SMALL_EPSILON: # Inv1 is overweight
                        amount_to_sell_from_inv1_ret = bal1_pre_rebal_ret - target_bal1_ideal_ret
                        if p.inv1_use_realized_gains_tax_system and p.inv1_realized_gains_tax_rate > 0 and amount_to_sell_from_inv1_ret > 0:
                            gain_in_inv1 = max(0, bal1_pre_rebal_ret - cb1_pre_rebal_ret)
                            if gain_in_inv1 > 0 and bal1_pre_rebal_ret > SMALL_EPSILON:
                                prop_sold = amount_to_sell_from_inv1_ret / bal1_pre_rebal_ret
                                realized_gain = gain_in_inv1 * prop_sold
                                tax_from_rebalancing_ret = realized_gain * p.inv1_realized_gains_tax_rate
                    elif bal1_pre_rebal_ret < target_bal1_ideal_ret - SMALL_EPSILON: # Inv2 is overweight
                        target_bal2_ideal_ret = total_balance_pre_rebal_ret * p.allocation_inv2_pct
                        amount_to_sell_from_inv2_ret = bal2_pre_rebal_ret - target_bal2_ideal_ret
                        if p.inv2_use_realized_gains_tax_system and p.inv2_realized_gains_tax_rate > 0 and amount_to_sell_from_inv2_ret > 0:
                            gain_in_inv2 = max(0, bal2_pre_rebal_ret - cb2_pre_rebal_ret)
                            if gain_in_inv2 > 0 and bal2_pre_rebal_ret > SMALL_EPSILON:
                                prop_sold = amount_to_sell_from_inv2_ret / bal2_pre_rebal_ret
                                realized_gain = gain_in_inv2 * prop_sold
                                tax_from_rebalancing_ret = realized_gain * p.inv2_realized_gains_tax_rate
                    
                    total_balance_after_rebal_tax_ret = total_balance_pre_rebal_ret - tax_from_rebalancing_ret
                    balance_inv1 = total_balance_after_rebal_tax_ret * p.allocation_inv1_pct
                    balance_inv2 = total_balance_after_rebal_tax_ret * p.allocation_inv2_pct

                    current_total_cost_basis_ret = cb1_pre_rebal_ret + cb2_pre_rebal_ret
                    if amount_to_sell_from_inv1_ret > 0 and bal1_pre_rebal_ret > SMALL_EPSILON:
                        prop_sold = amount_to_sell_from_inv1_ret / bal1_pre_rebal_ret
                        cb_reduction = cb1_pre_rebal_ret * prop_sold
                        current_total_cost_basis_ret = (current_total_cost_basis_ret - cb_reduction) + amount_to_sell_from_inv1_ret # Simplified CB transfer
                    elif amount_to_sell_from_inv2_ret > 0 and bal2_pre_rebal_ret > SMALL_EPSILON:
                        prop_sold = amount_to_sell_from_inv2_ret / bal2_pre_rebal_ret
                        cb_reduction = cb2_pre_rebal_ret * prop_sold
                        current_total_cost_basis_ret = (current_total_cost_basis_ret - cb_reduction) + amount_to_sell_from_inv2_ret

                    cost_basis_inv1 = current_total_cost_basis_ret * p.allocation_inv1_pct
                    cost_basis_inv2 = current_total_cost_basis_ret * p.allocation_inv2_pct
                
                cost_basis_inv1 = min(cost_basis_inv1, balance_inv1 if balance_inv1 > 0 else 0)
                cost_basis_inv2 = min(cost_basis_inv2, balance_inv2 if balance_inv2 > 0 else 0)

                total_balance_after_withdrawal_and_rebalance = balance_inv1 + balance_inv2
                if total_balance_after_withdrawal_and_rebalance < SMALL_EPSILON and monthly_withdrawal_needed > SMALL_EPSILON:
                     break 
            # --- End of Monthly Loop in Retirement Year ---

            # --- End of Retirement Year: Apply annual taxes if not using realized system ---
            if not p.inv1_use_realized_gains_tax_system and p.inv1_annual_tax_on_gains_rate > 0:
                # Gain is current EOY balance + gross withdrawals made during year - balance at start of year
                gain_inv1_ret_year = (balance_inv1 + total_gross_withdraw_inv1_this_year) - bal_inv1_start_tax_year_ret
                if gain_inv1_ret_year > 0:
                    balance_inv1 -= gain_inv1_ret_year * p.inv1_annual_tax_on_gains_rate
            
            if not p.inv2_use_realized_gains_tax_system and p.inv2_annual_tax_on_gains_rate > 0:
                gain_inv2_ret_year = (balance_inv2 + total_gross_withdraw_inv2_this_year) - bal_inv2_start_tax_year_ret
                if gain_inv2_ret_year > 0:
                    balance_inv2 -= gain_inv2_ret_year * p.inv2_annual_tax_on_gains_rate

            balance_inv1 = max(0, balance_inv1)
            balance_inv2 = max(0, balance_inv2)
            cost_basis_inv1 = min(cost_basis_inv1, balance_inv1)
            cost_basis_inv2 = min(cost_basis_inv2, balance_inv2)


            total_balance_after_annual_tax = balance_inv1 + balance_inv2
            yearly_trajectory.append(total_balance_after_annual_tax)

            if total_balance_after_annual_tax <= SMALL_EPSILON and required_annual_withdrawal > SMALL_EPSILON :
                break 

            # Rebalance at EOY after taxes (if any) - ensures start of next year is allocated correctly
            if total_balance_after_annual_tax > SMALL_EPSILON: # Only rebalance if there is money
                balance_inv1 = total_balance_after_annual_tax * p.allocation_inv1_pct
                balance_inv2 = total_balance_after_annual_tax * p.allocation_inv2_pct
                # And apportion total cost basis
                total_cb = cost_basis_inv1 + cost_basis_inv2
                cost_basis_inv1 = total_cb * p.allocation_inv1_pct
                cost_basis_inv2 = total_cb * p.allocation_inv2_pct
                cost_basis_inv1 = min(cost_basis_inv1, balance_inv1)
                cost_basis_inv2 = min(cost_basis_inv2, balance_inv2)


        final_total_balance = balance_inv1 + balance_inv2

        num_working_years = (working_months + MONTHS_PER_YEAR - 1) // MONTHS_PER_YEAR if working_months > 0 else 0
        # Expected length: 1 (initial) + num_working_years (EOY balances) + num_retirement_years (EOY balances)
        expected_len = 1 + num_working_years + p.retirement_years
        current_len = len(yearly_trajectory)

        if current_len < expected_len:
            padding_value = yearly_trajectory[-1] if yearly_trajectory else 0.0
            yearly_trajectory.extend([padding_value] * (expected_len - current_len))
        elif current_len > expected_len:
            yearly_trajectory = yearly_trajectory[:expected_len]

        return {
            'Start Balance': balance_at_retirement_start,
            'Final Balance': max(0, final_total_balance),
            'Trajectory': yearly_trajectory
        }
    
    def run_monte_carlo_simulations(self, working_months: int, num_simulations: int) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[List[List[float]]]]:
        """
        Runs multiple simulation paths, either sequentially or in parallel.
        """
        path_seeds = [self.main_seed + i for i in range(num_simulations)] 
        num_procs_to_use = self.params_model.num_processes if self.params_model.num_processes is not None else 1

        all_results_list: List[Dict[str, Union[float, List[float]]]]

        if num_procs_to_use <= 1:
            logger.debug(f"Running {num_simulations} simulations sequentially for {working_months} working months.")
            all_results_list = [self._run_single_simulation_path(working_months, seed) for seed in path_seeds]
        else:
            logger.debug(f"Running {num_simulations} simulations in parallel using {num_procs_to_use} processes for {working_months} working months.")
            args_for_starmap = [(working_months, seed) for seed in path_seeds]
            try:
                # Consider get_context for spawn/forkserver if issues arise, esp. on macOS or Windows.
                # context = multiprocessing.get_context('spawn')
                # with context.Pool(processes=num_procs_to_use) as pool:
                with multiprocessing.Pool(processes=num_procs_to_use) as pool:
                    all_results_list = pool.starmap(self._run_single_simulation_path, args_for_starmap)
            except Exception as e:
                logger.error(f"Multiprocessing pool error: {e}. Falling back to sequential execution.", exc_info=True)
                all_results_list = [self._run_single_simulation_path(working_months, seed) for seed in path_seeds]
        
        summary_results_list = [{'Start Balance': r['Start Balance'], 'Final Balance': r['Final Balance']} for r in all_results_list]
        summary_df = pd.DataFrame(summary_results_list)
        
        trajectories_raw = [r['Trajectory'] for r in all_results_list if 'Trajectory' in r and r['Trajectory']]
        
        trajectory_percentiles_df: Optional[pd.DataFrame] = None
        sample_trajectories_list: Optional[List[List[float]]] = None

        if trajectories_raw:
            try:
                # Ensure all trajectories have the same length (padding is done in _run_single_simulation_path)
                # Check for length consistency before creating DataFrame if strictness is needed
                min_len = min(map(len, trajectories_raw))
                max_len = max(map(len, trajectories_raw))
                if min_len != max_len:
                    logger.warning(f"Trajectory lengths are inconsistent: min={min_len}, max={max_len}. This might affect percentile calculations. Using min_len for safety if issues occur, but padding should handle.")
                
                trajectory_df = pd.DataFrame(trajectories_raw).transpose() # Rows are years, columns are simulations

                if not trajectory_df.empty:
                    percentiles_to_calc = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
                    trajectory_percentiles_df = trajectory_df.quantile(percentiles_to_calc, axis=1).transpose()
                    
                    num_sample_paths = min(len(trajectory_df.columns), 5) 
                    if num_sample_paths > 0 :
                        actual_num_to_sample = min(num_sample_paths, trajectory_df.shape[1])
                        sample_trajectories_list = trajectory_df.sample(n=actual_num_to_sample, axis=1, random_state=self.main_seed).values.T.tolist()
            except ValueError as ve: # Handles cases like inconsistent trajectory lengths if padding failed
                logger.error(f"Error processing trajectories, possibly due to inconsistent lengths: {ve}", exc_info=True)
            except Exception as e:
                logger.error(f"Error processing trajectories: {e}", exc_info=True)
        
        return summary_df, trajectory_percentiles_df, sample_trajectories_list

    def _get_dynamic_sim_count(self, achieved_prob_prev_iter: float, target_prob_pct: float, base_sim_count: int) -> int:
        """
        Adjusts the number of simulations for the next search iteration.
        More simulations are used when closer to the target for higher precision.
        """
        if achieved_prob_prev_iter < 0: # First iteration or no prior data
            num_s = int(base_sim_count * 0.5) 
        else:
            delta_prob = abs(achieved_prob_prev_iter - target_prob_pct)
            if delta_prob <= 1.0:
                num_s = int(base_sim_count * 4.00) # Very close
            elif delta_prob <= 1.5:
                num_s = int(base_sim_count * 2.00) # Even closer
            elif delta_prob <= 2.0:
                num_s = int(base_sim_count * 1.00) # Getting closer            
            elif delta_prob <= 3.0:
                num_s = int(base_sim_count * 0.80)
            elif delta_prob <= 4.0:
                num_s = int(base_sim_count * 0.70)            
            elif delta_prob <= 5.0:
                num_s = int(base_sim_count * 0.50)
            else:
                num_s = int(base_sim_count * 0.20) # Far
        
        # Ensure the number of simulations is not excessively small or large
        final_sim_count = max(MINIMUM_SIMULATIONS_FOR_SEARCH_STEP, min(num_s, base_sim_count * 5)) # Cap max multiplier
        delta_val_display = abs(achieved_prob_prev_iter - target_prob_pct) if achieved_prob_prev_iter >= 0 else -1.0
        logger.debug(f"Dynamic sim count: prev_prob={achieved_prob_prev_iter:.2f}, target={target_prob_pct:.2f}, delta={delta_val_display:.2f}, base_count={base_sim_count}, new_count={final_sim_count}")
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
        
        max_search_increase_months = 70 * MONTHS_PER_YEAR  # Max ~70 additional years of search
        max_total_months_to_test = starting_working_months + max_search_increase_months

        current_test_months = starting_working_months
        achieved_probability_at_prev_months = -1.0  # Probability from the *previous* test_months iteration
        highest_prob_if_target_not_met = -1.0 

        if verbose:
            logger.info(f"Searching for minimum working months to achieve {target_probability_pct:.2f}% success for '{p.Nickname}'.")
            logger.info(f"Starting search from {starting_working_months} months. Base simulations per test step: {base_num_simulations_per_test}.")

        search_iteration = 0
        # Simple linear scan; could be optimized (e.g., binary search if prob is monotonic)
        # For now, dynamic sim count helps manage computational cost.
        months_increment_step = 12 # Search year by year initially for speed
        
        while current_test_months <= max_total_months_to_test:
            search_iteration +=1
            iter_sim_count = self._get_dynamic_sim_count(
                achieved_probability_at_prev_months, 
                target_probability_pct,
                base_num_simulations_per_test
            )
            
            if verbose:
                logger.info(f"Search iter {search_iteration}: Testing {current_test_months} m ({current_test_months/MONTHS_PER_YEAR:.1f} yrs) with {iter_sim_count} sims.")
            
            summary_df, _, _ = self.run_monte_carlo_simulations(current_test_months, iter_sim_count)
            
            current_iter_achieved_probability_pct: float
            if summary_df.empty:
                logger.warning(f"Search iter {search_iteration}: No sim results for {current_test_months} months.")
                current_iter_achieved_probability_pct = 0.0
            else:
                current_iter_achieved_probability_pct = (summary_df['Final Balance'] > SMALL_EPSILON).mean() * 100.0

            if verbose:
                logger.info(f"  Search iter {search_iteration}: Prob for {current_test_months} m: {current_iter_achieved_probability_pct:.2f}% (Target: {target_probability_pct:.2f}%)")

            if current_iter_achieved_probability_pct > highest_prob_if_target_not_met:
                highest_prob_if_target_not_met = current_iter_achieved_probability_pct

            if current_iter_achieved_probability_pct >= target_probability_pct:
                if verbose:
                    logger.info(f"  Target met at {current_test_months} months.")
                # If we overshot with a large step, we might want to backtrack and refine
                if months_increment_step > 1:
                    logger.info(f"  Target met with step {months_increment_step}. Refining search by smaller steps...")
                    # Backtrack one big step, then search month by month
                    current_test_months = max(starting_working_months, current_test_months - months_increment_step + 1) 
                    months_increment_step = 1 # Switch to fine-grained search
                    achieved_probability_at_prev_months = -1 # Reset for dynamic sim count when refining
                    continue # Restart loop with new current_test_months and step
                return current_test_months, current_iter_achieved_probability_pct
            
            achieved_probability_at_prev_months = current_iter_achieved_probability_pct # Store for next dynamic count

            # Adaptive step: if far from target, take larger steps.
            # This is a simple heuristic for adapting step size.
            if target_probability_pct - current_iter_achieved_probability_pct > 20 and months_increment_step < 24: # If very far
                months_increment_step = 24 # 2 years
            elif target_probability_pct - current_iter_achieved_probability_pct > 10 and months_increment_step < 12: # If moderately far
                months_increment_step = 12 # 1 year
            elif target_probability_pct - current_iter_achieved_probability_pct > 3 and months_increment_step < 6:
                 months_increment_step = 6 # 6 months
            else: # Close or overshot but not met target (should refine)
                months_increment_step = 1 # Search month by month when close or needing refinement

            current_test_months += months_increment_step
        
        if verbose:
            logger.warning(f"Search for '{p.Nickname}' reached max limit ({max_total_months_to_test/MONTHS_PER_YEAR:.1f} yrs). Target NOT met.")
            logger.warning(f"Highest probability achieved: {highest_prob_if_target_not_met:.2f}% at/before last test.")
        return -1, highest_prob_if_target_not_met

# --- PLOTTING UTILITY (HISTOGRAM) ---
def plot_simulation_results(results_df: pd.DataFrame,
                            input_config: Config,
                            analysis_summary: Dict[str, Any],
                            filename: str = DEFAULT_PLOT_FILENAME):
    plt.figure(figsize=(12, 7.5)) # Adjusted for better aspect ratio if text is inside
    ax = plt.gca()

    successful_outcomes = results_df[results_df['Final Balance'] > SMALL_EPSILON]
    success_rate_display = (len(successful_outcomes) / len(results_df) * 100) if len(results_df) > 0 else 0.0
    
    balances_in_millions = successful_outcomes['Final Balance'] / 1e6 if not successful_outcomes.empty else pd.Series(dtype=float)
    
    if not balances_in_millions.empty:
        plt.hist(balances_in_millions, bins=100, edgecolor='black', alpha=0.7, label=f'Successful Outcomes ({success_rate_display:.1f}%) (Final Bal > 0)')
    else:
        logger.info(f"No successful outcomes to plot in histogram for {filename}.")
        # Optionally, plot an empty state or just text
        ax.text(0.5, 0.5, "No successful outcomes to display.", transform=ax.transAxes, ha="center", va="center")


    TEXT_INPUT_COLOR = '#1f77b4'
    TEXT_OUTPUT_COLOR = '#ff7f0e'
    p = input_config
    
    # Prepare text content (condense if needed for space)
    input_lines = [
        f"Scenario: {p.Nickname}",
        f"Sims (Main): {p.num_simulations_main:,}, Sims (Search Base): {p.num_simulations_search:,}",        
        f"Init Bal: ${p.initial_balance:,.0f}, Contr: ${p.monthly_contribution:,.0f} (Grows @ {p.contribution_growth_rate_annual*100:.1f}%)",
        f"Monthly Exp (T0): ${p.monthly_expenses:,.0f}, Ret Yrs: {p.retirement_years}",
        "--- Investments ---",
        f"Inv1 ({p.allocation_inv1_pct*100:.0f}%): {p.inv1_returns_mean*100:.1f}%R {p.inv1_returns_volatility*100:.1f}%Vol",
        f" Tax Model: {'Realz' if p.inv1_use_realized_gains_tax_system else 'Ann'}, Ann.Tax {p.inv1_annual_tax_on_gains_rate*100:.0f}%, Real.Tax {p.inv1_realized_gains_tax_rate*100:.0f}%",
        f"Inv2 ({p.allocation_inv2_pct*100:.0f}%): {p.inv2_premium_over_inflation_mean*100:.1f}%Prem {p.inv2_premium_over_inflation_volatility*100:.1f}%Vol",
        f" Tax Model: {'Realz' if p.inv2_use_realized_gains_tax_system else 'Ann'}, Ann.Tax {p.inv2_annual_tax_on_gains_rate*100:.0f}%, Real.Tax {p.inv2_realized_gains_tax_rate*100:.0f}%",
        f"Inflation: {p.inflation_rate_mean*100:.1f}% Mean, {p.inflation_rate_volatility*100:.1f}% Vol",
    ]
    if p.other_income_streams:
        input_lines.append("--- Other Income (T0 Real Values) ---")
        for i, stream in enumerate(p.other_income_streams):
            if i < 2 : # Limit to a couple for brevity on plot
                duration_str = f", {stream.duration_years}yrs" if stream.duration_years is not None else ""
                input_lines.append(f" {stream.name[:10]}: ${stream.monthly_amount_today:,.0f}/mo, from ret.yr {stream.start_after_retirement_years+1}{duration_str}, {stream.tax_rate*100:.0f}%Tax")

    output_lines = [
        '--- Results ---',
        f"Req.Work: {analysis_summary['required_working_months']}mo ({analysis_summary['required_working_months']/MONTHS_PER_YEAR:.1f}yr)",
        f"Success: {analysis_summary['final_success_probability']:.1f}% (Target: {p.target_probability:.1f}%)",
        f"Med Start Bal: ${analysis_summary['median_start_retirement_balance']:,.0f}",
        f"Med Final Bal (succ): ${analysis_summary['median_final_balance']:,.0f}",
        f"SWR (Med Start): {analysis_summary.get('SWR', float('nan')):.2f}%",
    ]
    
    # Text Block Positioning
    x_pos_text = 0.98  # Right edge of text block at 98% of axes width
    y_coord_start = 0.98 # Top edge of text block at 98% of axes height
    line_spacing_val = 0.035 # Adjusted for better readability with smaller font
    fontsize_text = 6.5    # Smaller font size to fit more text inside

    # Plot input parameters
    for i, line_text in enumerate(input_lines):
        ax.text(x_pos_text, y_coord_start - i * line_spacing_val, line_text, 
                transform=ax.transAxes, ha='right', va='top', 
                fontsize=fontsize_text, color=TEXT_INPUT_COLOR,
                bbox=dict(facecolor='white', alpha=0.80, pad=2, edgecolor='lightgrey', boxstyle='round,pad=0.3'))
    
    # Gap before output lines
    output_y_start_offset = (len(input_lines) * line_spacing_val) + (line_spacing_val * 0.75) 
    
    # Plot output/results
    for j, line_text in enumerate(output_lines):
        ax.text(x_pos_text, y_coord_start - output_y_start_offset - (j * line_spacing_val), line_text, 
                transform=ax.transAxes, ha='right', va='top', 
                fontsize=fontsize_text, color=TEXT_OUTPUT_COLOR, fontweight='bold', # Same font size for consistency
                bbox=dict(facecolor='white', alpha=0.85, pad=2, edgecolor='lightgrey', boxstyle='round,pad=0.3'))

    if not balances_in_millions.empty:
        plt.axvline(balances_in_millions.median(), color='blue', linestyle='dashed', linewidth=1.2, label=f'Median (Succ.): ${balances_in_millions.median():.2f}M')
    plt.axvline(0, color='red', linestyle='-', linewidth=1.0, label='Zero Balance')
    
    plt.title(f'Final Balance Distribution: {input_config.Nickname}', fontsize=14)
    plt.xlabel('Final Balance (Millions of $)', fontsize=10) 
    plt.ylabel('Frequency', fontsize=10) # Simplified
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    # Adjust legend position if text box is in upper right
    # If the text box is well-placed, the legend might need to move or be omitted if labels are clear
    handles, labels = ax.get_legend_handles_labels()
    if handles: # Only show legend if there are items to show
        ax.legend(handles, labels, fontsize=7, loc='upper left', bbox_to_anchor=(0.01, 0.98)) # Move legend to upper left

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout() # Call tight_layout AFTER all elements are added

    try:
        plt.savefig(filename, dpi=150)
        logger.info(f"Histogram plot saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving histogram plot '{filename}': {e}", exc_info=True)
    plt.close()


def plot_portfolio_trajectories(
    trajectory_percentiles_df: Optional[pd.DataFrame],
    sample_trajectories: Optional[List[List[float]]],
    working_months: int,
    input_config: Config,
    filename: str,
    dpi_setting: int = 300 # Allow DPI to be configurable
):
    """
    Plots portfolio balance trajectories, including percentile bands, sample paths,
    a retirement line, and lines for the start of other income streams.

    Args:
        trajectory_percentiles_df: DataFrame with percentile values for trajectories.
                                   Index should represent time (e.g., years), columns
                                   should be percentile values (e.g., 0.1, 0.5, 0.9).
        sample_trajectories: A list of lists, where each inner list is a single
                             simulated portfolio trajectory over time.
        working_months: The number of months until retirement.
        input_config: A Config object containing scenario settings, including
                      Nickname and other_income_streams.
        filename: The full path and filename to save the plot to.
        dpi_setting: The DPI (dots per inch) for the saved image.
    """
    if trajectory_percentiles_df is None or trajectory_percentiles_df.empty:
        logger.warning(f"No trajectory percentile data to plot for '{filename}'. Skipping.")
        return

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    years_x_axis = np.arange(len(trajectory_percentiles_df))
    max_years_plot = len(years_x_axis) - 1 if len(years_x_axis) > 0 else 0


    if sample_trajectories:
        for i, trajectory in enumerate(sample_trajectories):
            if len(trajectory) == len(years_x_axis):
                ax.plot(years_x_axis, np.array(trajectory) / 1e6, color='grey', alpha=0.20, linewidth=0.6, label='_nolegend_')
            else:
                logger.warning(f"Sample trajectory {i} for '{filename}' length mismatch (expected {len(years_x_axis)}, got {len(trajectory)}). Skipping.")

    if 0.5 in trajectory_percentiles_df.columns:
        ax.plot(years_x_axis, trajectory_percentiles_df[0.5] / 1e6, color='blue', linewidth=1.8, label='Median (50th Percentile)')
    else:
        logger.warning("Median (0.5 percentile) not found in trajectory_percentiles_df.")

    percentile_bands_to_plot = []
    if 0.05 in trajectory_percentiles_df.columns and 0.95 in trajectory_percentiles_df.columns:
        percentile_bands_to_plot.append({'low': 0.05, 'high': 0.95, 'color': 'salmon', 'alpha': 0.15, 'label': '5th-95th Percentile Range'})
    elif 0.1 in trajectory_percentiles_df.columns and 0.9 in trajectory_percentiles_df.columns:
        percentile_bands_to_plot.append({'low': 0.1, 'high': 0.9, 'color': 'orangered', 'alpha': 0.15, 'label': '10th-90th Percentile Range'})

    if 0.25 in trajectory_percentiles_df.columns and 0.75 in trajectory_percentiles_df.columns:
        percentile_bands_to_plot.append({'low': 0.25, 'high': 0.75, 'color': 'skyblue', 'alpha': 0.25, 'label': '25th-75th Percentile Range'})

    for band in percentile_bands_to_plot:
        if band['low'] in trajectory_percentiles_df.columns and band['high'] in trajectory_percentiles_df.columns:
            ax.fill_between(years_x_axis,
                            trajectory_percentiles_df[band['low']] / 1e6,
                            trajectory_percentiles_df[band['high']] / 1e6,
                            color=band['color'], alpha=band['alpha'], label=band['label'], interpolate=True)
        else:
            logger.warning(f"Columns for percentile band {band['label']} (low: {band['low']}, high: {band['high']}) not found. Skipping band.")

    working_years_float = working_months / MONTHS_PER_YEAR
    if len(years_x_axis) > 0 and 0 <= working_years_float <= max_years_plot:
        ax.axvline(x=working_years_float, color='black', linestyle='--', linewidth=1.2, label=f'Retirement Starts ({working_years_float:.1f} yrs)')
    elif len(years_x_axis) > 0 and working_years_float > max_years_plot:
        logger.info(f"Retirement at {working_years_float:.1f} yrs is beyond the plot's x-axis range ({max_years_plot} yrs). Retirement line not plotted.")
    elif working_years_float < 0 :
         logger.info(f"Retirement at {working_years_float:.1f} yrs is before simulation start. Retirement line not plotted.")


    if input_config.other_income_streams:
        line_styles = ['-', '--', '-.', ':']
        colors = ['green', 'purple', 'brown', 'cyan', 'magenta', 'olive']
        for i, stream in enumerate(input_config.other_income_streams):
            if not hasattr(stream, 'name') or not hasattr(stream, 'start_after_retirement_years'):
                logger.warning(f"Income stream at index {i} is missing 'name' or 'start_after_retirement_years' attribute. Skipping.")
                continue

            stream_start_sim_year = working_years_float + stream.start_after_retirement_years
            
            if len(years_x_axis) > 0 and 0 <= stream_start_sim_year <= max_years_plot:
                ax.axvline(x=stream_start_sim_year,
                            color=colors[i % len(colors)],
                            linestyle=line_styles[i % len(line_styles)],
                            linewidth=1.0,
                            label=f'{stream.name} Starts (yr {stream_start_sim_year:.1f})')
            elif len(years_x_axis) > 0 and stream_start_sim_year > max_years_plot:
                logger.warning(f"Income stream '{stream.name}' starts at simulation year {stream_start_sim_year:.1f}, which is beyond the plot's x-axis range ({max_years_plot} yrs). Skipping line.")
            elif stream_start_sim_year < 0 :
                logger.info(f"Income stream '{stream.name}' starts at simulation year {stream_start_sim_year:.1f}, which is before plot's x-axis range (0 yrs). Skipping line.")


    ax.set_xlabel("Years from Simulation Start", fontsize=9)
    ax.set_ylabel("Portfolio Balance (Millions of $)", fontsize=9)
    ax.set_title(f"Portfolio Balance Trajectories - Scenario: {input_config.Nickname}", fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid(True, linestyle=':', alpha=0.6)

    def millions_formatter(x_val, pos):
        return f'{x_val:.1f}M' if x_val != 0 else '0'
    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

    if len(years_x_axis) > 0:
        ax.set_xlim(left=0, right=max_years_plot)
    
    min_y_val_plot = 0
    if not trajectory_percentiles_df.empty:
        percentile_cols_for_ylim = [col for col in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] if col in trajectory_percentiles_df.columns]
        if percentile_cols_for_ylim:
            all_percentile_values = trajectory_percentiles_df[percentile_cols_for_ylim].values.flatten()
            # Check if sample trajectories also contribute to y-axis range
            if sample_trajectories:
                all_sample_values = np.array([val for traj in sample_trajectories for val in traj if len(traj) == len(years_x_axis)])
                if len(all_sample_values) > 0:
                    all_percentile_values = np.concatenate((all_percentile_values, all_sample_values))
            
            if len(all_percentile_values) > 0:
                min_data_val_abs = np.min(all_percentile_values) / 1e6
                max_data_val_abs = np.max(all_percentile_values) / 1e6

                if min_data_val_abs < 0:
                    min_y_val_plot = min_data_val_abs * 1.05 # Add 5% buffer for negative values
                # If all values are positive, min_y_val_plot remains 0 unless specified otherwise
                
                # Ensure the top of the plot has some space too
                ax.set_ylim(bottom=min_y_val_plot, top=max_data_val_abs * 1.05 if max_data_val_abs > 0 else 1) # Add 5% buffer at top, or default 1 if max is 0
            else: # No data to determine range, default y-lim
                 ax.set_ylim(bottom=0, top=1) # Default if no data points
        else: # No percentile columns found, default y-lim
            ax.set_ylim(bottom=0, top=1) # Default if no percentile columns
    else: # No x-axis, default y-lim
        ax.set_ylim(bottom=0, top=1)


    ax.legend(fontsize=7.5, loc='best')
    plt.tight_layout()

    # --- Save Plot ---
    try:
        file_directory = os.path.dirname(filename)
        # Only attempt to create directories if a non-empty path is specified
        if file_directory: # This ensures we don't call os.makedirs with an empty string
            os.makedirs(file_directory, exist_ok=True)
        
        plt.savefig(filename, dpi=dpi_setting)
        logger.info(f"Trajectory plot saved to {filename} (DPI: {dpi_setting})")
    except Exception as e:
        logger.error(f"Error saving trajectory plot '{filename}': {e}", exc_info=True)
    finally:
        plt.close()


def log_input_parameters(config: Config) -> None:
    """Logs the input parameters for the simulation."""
    logger.info(f"--- Input Parameters For Scenario: {config.Nickname} ---")
    config_as_dict_for_logging = config.model_dump(by_alias=False)
    for key, value in config_as_dict_for_logging.items():
        if key == "Nickname":
            continue
        if key == "other_income_streams":
            logger.info(f"{key.replace('_', ' ').title()}:")
            if config.other_income_streams:
                for stream_model in config.other_income_streams:
                    duration_str = f", lasts {stream_model.duration_years} yrs" if stream_model.duration_years is not None else ", lasts indefinitely"
                    inflation_idx_str = " (Fully Inflation Adj.)" if stream_model.inflation_indexed else " (Nominal Fixed at Stream Start)"
                    logger.info(f"  - {stream_model.name}: ${stream_model.monthly_amount_today:,.0f}/mo (T=0 real value), "
                                f"starts after {stream_model.start_after_retirement_years} ret. yrs{duration_str}{inflation_idx_str}, "
                                f"Tax: {stream_model.tax_rate*100:.0f}%")
            else:
                logger.info("  - None")
        elif key == "target_probability":
            logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}%")
        elif isinstance(value, float) and ("rate" in key or "mean" in key or "volatility" in key or "pct" in key) and \
             key not in ["initial_balance", "monthly_contribution", "monthly_expenses", "monthly_amount_today"]:
            logger.info(f"{key.replace('_', ' ').title()}: {value*100:.2f}%")
        elif isinstance(value, (float, int)) and any(curr_kw in key for curr_kw in ["balance", "contribution", "expenses", "amount"]):
             logger.info(f"{key.replace('_', ' ').title()}: ${value:,.2f}") # Assuming $
        else:
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
    logger.info(f"Allocation Inv2 Pct (Calculated): {config.allocation_inv2_pct*100:.2f}%")
    logger.info("--- End of Input Parameters ---")


def log_simulation_results(
    config: Config,
    required_w_months: int,
    final_success_prob_pct: float,
    median_start_ret_bal: float,
    median_final_bal_successful: float,
    swr: float,
    final_summary_df: pd.DataFrame
) -> None:
    """Logs the final results of the simulation."""
    logger.info(f"--- Final Simulation Results for Scenario: '{config.Nickname}' ---")
    logger.info(f"Determined Required Working Months: {required_w_months} ({required_w_months/MONTHS_PER_YEAR:.1f} years)")
    logger.info(f"Probability of Not Running Out of Money (Final Sims): {final_success_prob_pct:.2f}% (Target: {config.target_probability:.2f}%)")
    logger.info(f"Median Balance at Start of Retirement (All Sims): ${median_start_ret_bal:,.2f}")
    logger.info(f"Median Final Balance (Successful Sims Only): ${median_final_bal_successful:,.2f}")
    logger.info(f"Est. SWR (Nominal, 1st yr, on Median Start Bal, using T=0 expenses): {swr:.2f}%")

    percentiles_final_balance = final_summary_df['Final Balance'].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    logger.info("Final Balance Percentiles (All Sims, $):")
    for p_val, value in percentiles_final_balance.items():
        logger.info(f"  {p_val*100:.0f}th: {max(0, value):,.2f}")


# --- MAIN EXECUTION SCRIPT ---
def main():
    """
    Main execution entry point.
    
    Loads configuration, runs the simulation search for minimum working months,
    executes the final simulation set, logs results, and generates plots.
    """
    current_timestamp_str = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"ret_proj_log_{current_timestamp_str}.log"
    
    # Configure loguru
    logger.remove() # Remove default handler
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
    logger.add(log_filename, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO", rotation="10 MB")
    
    logger.info(f"Logging initialized. Log file: {log_filename}")

    # --- LOAD CONFIGURATION FROM JSON ---
    if len(sys.argv) > 1:
        json_filename = sys.argv[1]
    else:
        json_filename = "config.json"
        logger.info(f"No config file specified via argument. Defaulting to '{json_filename}'")

    logger.info(f"Loading configuration from: {json_filename}")
    config_dict = load_config_from_json(json_filename)

    config: Optional[Config] = None
    try:
        config = Config(**config_dict)
        logger.info(f"Configuration for scenario '{config.Nickname}' loaded and validated successfully.")
    except Exception as e: 
        logger.error(f"Configuration error: {e}", exc_info=True)
        return 

    log_input_parameters(config)

    simulator = RetirementMonteCarloSimulator(config)

    logger.info(f"--- Starting Search for Minimum Working Months for '{config.Nickname}' ---")
    required_w_months, achieved_prob_search = simulator.find_minimum_working_months(verbose=True)

    if required_w_months == -1:
        logger.error(f"Target probability of {config.target_probability:.2f}% could not be met for '{config.Nickname}'.")
        logger.error(f"Highest probability achieved: {achieved_prob_search:.2f}%. Consider adjusting parameters or target.")
        return 

    logger.info(f"--- Search Complete for '{config.Nickname}'. Required: {required_w_months} m ({required_w_months/MONTHS_PER_YEAR:.1f} yrs) with prob: {achieved_prob_search:.2f}%. ---")
    logger.info(f"--- Running Final Detailed Simulation for '{config.Nickname}' ({config.num_simulations_main} sims) using {required_w_months} working months. ---")

    final_summary_df, final_trajectory_percentiles_df, final_sample_trajectories = \
        simulator.run_monte_carlo_simulations(
            working_months=required_w_months,
            num_simulations=config.num_simulations_main
        )

    if final_summary_df.empty:
        logger.error(f"Final simulation for '{config.Nickname}' yielded no results.")
        return

    final_success_prob_pct = (final_summary_df['Final Balance'] > SMALL_EPSILON).mean() * 100.0
    successful_final_balances = final_summary_df.loc[final_summary_df['Final Balance'] > SMALL_EPSILON, 'Final Balance']
    median_final_bal_successful = successful_final_balances.median() if not successful_final_balances.empty else 0.0
    median_start_ret_bal = final_summary_df['Start Balance'].median()

    # SWR based on T=0 expenses, not inflation-adjusted to retirement start here, as it's a common way to quote SWR
    # The simulation correctly uses nominal expenses. This SWR is just an output metric.
    initial_annual_expenses_t0 = config.monthly_expenses * MONTHS_PER_YEAR 
    swr = (initial_annual_expenses_t0 * 100.0) / median_start_ret_bal if median_start_ret_bal > SMALL_EPSILON else float('nan')

    log_simulation_results(
        config,
        required_w_months,
        final_success_prob_pct,
        median_start_ret_bal,
        median_final_bal_successful,
        swr,
        final_summary_df
    )

    safe_nickname = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in config.Nickname)
    plot_file_base = f"ret_proj_{safe_nickname}_{current_timestamp_str}"

    analysis_summary_for_plot = {
        'required_working_months': required_w_months,
        'final_success_probability': final_success_prob_pct,
        'median_start_retirement_balance': median_start_ret_bal,
        'median_final_balance': median_final_bal_successful,
        'SWR': swr,
    }
    
    plot_filename_hist = f"{plot_file_base}_HIST.png"
    plot_simulation_results(final_summary_df, config, analysis_summary_for_plot, plot_filename_hist)
    
    plot_filename_traj = f"{plot_file_base}_TRAJ.png"
    if final_trajectory_percentiles_df is not None:
        plot_portfolio_trajectories(
            final_trajectory_percentiles_df,  
            final_sample_trajectories,
            required_w_months,
            config,  
            plot_filename_traj
        )
    else:
        logger.warning(f"Skipping trajectory plot for '{config.Nickname}' as trajectory data is missing.")

    logger.info(f"--- Main execution finished for scenario '{config.Nickname}'. Outputs in current directory. Log: {log_filename} ---")


if __name__ == '__main__':
    multiprocessing.freeze_support()  
    main()