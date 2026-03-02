import datetime as _dt
import hashlib
import pandas as pd
from loguru import logger
from config import Config
from constants import MONTHS_PER_YEAR


def _generate_seed_from_timestamp() -> int:
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return int.from_bytes(hashlib.sha256(ts.encode()).digest()[:8], "big") % (2**32 - 1)


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
                    duration_str = (
                        f", lasts {stream_model.duration_years} yrs"
                        if stream_model.duration_years is not None
                        else ", lasts indefinitely"
                    )
                    inflation_idx_str = (
                        " (Fully Inflation Adj.)"
                        if stream_model.inflation_indexed
                        else " (Nominal Fixed at Stream Start)"
                    )
                    logger.info(
                        f"  - {stream_model.name}: ${stream_model.monthly_amount_today:,.0f}/mo (T=0 real value), "
                        f"starts after {stream_model.start_after_retirement_years} ret. yrs{duration_str}{inflation_idx_str}, "
                        f"Tax: {stream_model.tax_rate * 100:.0f}%"
                    )
            else:
                logger.info("  - None")
        elif key == "target_probability":
            logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}%")
        elif (
            isinstance(value, float)
            and ("rate" in key or "mean" in key or "volatility" in key or "pct" in key)
            and key
            not in [
                "initial_balance",
                "monthly_contribution",
                "monthly_expenses",
                "monthly_amount_today",
            ]
        ):
            logger.info(f"{key.replace('_', ' ').title()}: {value * 100:.2f}%")
        elif isinstance(value, (float, int)) and any(
            curr_kw in key
            for curr_kw in ["balance", "contribution", "expenses", "amount"]
        ):
            logger.info(f"{key.replace('_', ' ').title()}: ${value:,.2f}")  # Assuming $
        else:
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
    logger.info(
        f"Allocation Inv2 Pct (Calculated): {config.allocation_inv2_pct * 100:.2f}%"
    )
    logger.info("--- End of Input Parameters ---")


def log_simulation_results(
    config: Config,
    required_w_months: int,
    final_success_prob_pct: float,
    median_start_ret_bal: float,
    median_final_bal_successful: float,
    swr: float,
    final_summary_df: pd.DataFrame,
) -> None:
    """Logs the final results of the simulation."""
    logger.info(f"--- Final Simulation Results for Scenario: '{config.Nickname}' ---")
    logger.info(
        f"Determined Required Working Months: {required_w_months} ({required_w_months / MONTHS_PER_YEAR:.1f} years)"
    )
    logger.info(
        f"Probability of Not Running Out of Money (Final Sims): {final_success_prob_pct:.2f}% (Target: {config.target_probability:.2f}%)"
    )
    logger.info(
        f"Median Balance at Start of Retirement (All Sims): ${median_start_ret_bal:,.2f}"
    )
    logger.info(
        f"Median Final Balance (Successful Sims Only): ${median_final_bal_successful:,.2f}"
    )
    logger.info(
        f"Est. SWR (Nominal, 1st yr, on Median Start Bal, using T=0 expenses): {swr:.2f}%"
    )

    percentiles_final_balance = final_summary_df["Final Balance"].quantile(
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    logger.info("Final Balance Percentiles (All Sims, $):")
    for p_val, value in percentiles_final_balance.items():
        logger.info(f"  {p_val * 100:.0f}th: {max(0, value):,.2f}")
