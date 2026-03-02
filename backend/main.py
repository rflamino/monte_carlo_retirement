import sys
import datetime as _dt
import multiprocessing
from loguru import logger

from config import Config, ConfigurationError, load_config_from_json
from utils import log_input_parameters, log_simulation_results
from simulation import RetirementMonteCarloSimulator
from plotting import plot_simulation_results, plot_portfolio_trajectories
from constants import SMALL_EPSILON, MONTHS_PER_YEAR


def main():
    """
    Main execution entry point.

    Loads configuration, runs the simulation search for minimum working months,
    executes the final simulation set, logs results, and generates plots.
    """
    current_timestamp_str = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"ret_proj_log_{current_timestamp_str}.log"

    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    logger.add(
        log_filename,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="10 MB",
    )

    logger.info(f"Logging initialized. Log file: {log_filename}")

    # --- LOAD CONFIGURATION FROM JSON ---
    if len(sys.argv) > 1:
        json_filename = sys.argv[1]
    else:
        json_filename = "config.json"
        logger.info(
            f"No config file specified via argument. Defaulting to '{json_filename}'"
        )

    logger.info(f"Loading configuration from: {json_filename}")
    try:
        config_dict = load_config_from_json(json_filename)
        config = Config(**config_dict)
        logger.info(
            f"Configuration for scenario '{config.Nickname}' loaded and validated successfully."
        )
    except ConfigurationError as e:
        logger.error(f"Configuration file error: {e}")
        return
    except Exception as e:
        logger.error(f"Configuration validation error: {e}", exc_info=True)
        return

    log_input_parameters(config)

    simulator = RetirementMonteCarloSimulator(config)

    logger.info(
        f"--- Starting Search for Minimum Working Months for '{config.Nickname}' ---"
    )
    required_w_months, achieved_prob_search = simulator.find_minimum_working_months(
        verbose=True
    )

    if required_w_months == -1:
        logger.error(
            f"Target probability of {config.target_probability:.2f}% could not be met for '{config.Nickname}'."
        )
        logger.error(
            f"Highest probability achieved: {achieved_prob_search:.2f}%. Consider adjusting parameters or target."
        )
        logger.error("Skipping final simulation.")
        return

    logger.info(
        f"--- Search Complete for '{config.Nickname}'. Required: {required_w_months} m ({required_w_months / MONTHS_PER_YEAR:.1f} yrs) with prob: {achieved_prob_search:.2f}%. ---"
    )
    logger.info(
        f"--- Running Final Detailed Simulation for '{config.Nickname}' ({config.num_simulations_main} sims) using {required_w_months} working months. ---"
    )

    final_summary_df, final_trajectory_percentiles_df, final_sample_trajectories = (
        simulator.run_monte_carlo_simulations(
            working_months=required_w_months,
            num_simulations=config.num_simulations_main,
        )
    )

    if final_summary_df.empty:
        logger.error(f"Final simulation for '{config.Nickname}' yielded no results.")
        return

    final_success_prob_pct = (
        final_summary_df["Final Balance"] > SMALL_EPSILON
    ).mean() * 100.0
    successful_final_balances = final_summary_df.loc[
        final_summary_df["Final Balance"] > SMALL_EPSILON, "Final Balance"
    ]
    median_final_bal_successful = (
        successful_final_balances.median()
        if not successful_final_balances.empty
        else 0.0
    )
    median_start_ret_bal = final_summary_df["Start Balance"].median()

    # SWR based on T=0 expenses, not inflation-adjusted to retirement start here, as it's a common way to quote SWR
    # The simulation correctly uses nominal expenses. This SWR is just an output metric.
    initial_annual_expenses_t0 = config.monthly_expenses * MONTHS_PER_YEAR
    swr = (
        (initial_annual_expenses_t0 * 100.0) / median_start_ret_bal
        if median_start_ret_bal > SMALL_EPSILON
        else float("nan")
    )

    log_simulation_results(
        config,
        required_w_months,
        final_success_prob_pct,
        median_start_ret_bal,
        median_final_bal_successful,
        swr,
        final_summary_df,
    )

    safe_nickname = "".join(
        c if c.isalnum() or c in ["_", "-"] else "_" for c in config.Nickname
    )
    plot_file_base = f"ret_proj_{safe_nickname}_{current_timestamp_str}"

    analysis_summary_for_plot = {
        "required_working_months": required_w_months,
        "final_success_probability": final_success_prob_pct,
        "median_start_retirement_balance": median_start_ret_bal,
        "median_final_balance": median_final_bal_successful,
        "SWR": swr,
    }

    plot_filename_hist = f"{plot_file_base}_HIST.png"
    plot_simulation_results(
        final_summary_df, config, analysis_summary_for_plot, plot_filename_hist
    )

    plot_filename_traj = f"{plot_file_base}_TRAJ.png"
    if final_trajectory_percentiles_df is not None:
        plot_portfolio_trajectories(
            final_trajectory_percentiles_df,
            final_sample_trajectories,
            required_w_months,
            config,
            plot_filename_traj,
        )
    else:
        logger.warning(
            f"Skipping trajectory plot for '{config.Nickname}' as trajectory data is missing."
        )

    logger.info(
        f"--- Main execution finished for scenario '{config.Nickname}'. Outputs in current directory. Log: {log_filename} ---"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
