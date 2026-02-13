import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.ticker import FuncFormatter
from typing import Dict, Any, List, Optional

from config import Config
from constants import (
    SMALL_EPSILON,
    TEXT_INPUT_COLOR,
    TEXT_OUTPUT_COLOR,
    MONTHS_PER_YEAR,
)


def plot_simulation_results(
    results_df: pd.DataFrame,
    input_config: Config,
    analysis_summary: Dict[str, Any],
    filename: str,
):
    plt.figure(figsize=(12, 7.5))  # Adjusted for better aspect ratio if text is inside
    ax = plt.gca()

    successful_outcomes = results_df[results_df["Final Balance"] > SMALL_EPSILON]
    success_rate_display = (
        (len(successful_outcomes) / len(results_df) * 100)
        if len(results_df) > 0
        else 0.0
    )

    balances_in_millions = (
        successful_outcomes["Final Balance"] / 1e6
        if not successful_outcomes.empty
        else pd.Series(dtype=float)
    )

    if not balances_in_millions.empty:
        plt.hist(
            balances_in_millions,
            bins=100,
            edgecolor="black",
            alpha=0.7,
            label=f"Successful Outcomes ({success_rate_display:.1f}%) (Final Bal > 0)",
        )
    else:
        logger.info(f"No successful outcomes to plot in histogram for {filename}.")
        # Optionally, plot an empty state or just text
        ax.text(
            0.5,
            0.5,
            "No successful outcomes to display.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    p = input_config

    # Prepare text content (condense if needed for space)
    input_lines = [
        f"Scenario: {p.Nickname}",
        f"Sims (Main): {p.num_simulations_main:,}, Sims (Search Base): {p.num_simulations_search:,}",
        f"Init Bal: ${p.initial_balance:,.0f}, Contr: ${p.monthly_contribution:,.0f} (Grows @ {p.contribution_growth_rate_annual * 100:.1f}%)",
        f"Monthly Exp (T0): ${p.monthly_expenses:,.0f}, Ret Yrs: {p.retirement_years}",
        "--- Investments ---",
        f"Inv1 ({p.allocation_inv1_pct * 100:.0f}%): {p.inv1_returns_mean * 100:.1f}%R {p.inv1_returns_volatility * 100:.1f}%Vol",
        f" Tax Model: {'Realz' if p.inv1_use_realized_gains_tax_system else 'Ann'}, Ann.Tax {p.inv1_annual_tax_on_gains_rate * 100:.0f}%, Real.Tax {p.inv1_realized_gains_tax_rate * 100:.0f}%",
        f"Inv2 ({p.allocation_inv2_pct * 100:.0f}%): {p.inv2_premium_over_inflation_mean * 100:.1f}%Prem {p.inv2_premium_over_inflation_volatility * 100:.1f}%Vol",
        f" Tax Model: {'Realz' if p.inv2_use_realized_gains_tax_system else 'Ann'}, Ann.Tax {p.inv2_annual_tax_on_gains_rate * 100:.0f}%, Real.Tax {p.inv2_realized_gains_tax_rate * 100:.0f}%",
        f"Inflation: {p.inflation_rate_mean * 100:.1f}% Mean, {p.inflation_rate_volatility * 100:.1f}% Vol",
    ]
    if p.other_income_streams:
        input_lines.append("--- Other Income (T0 Real Values) ---")
        for i, stream in enumerate(p.other_income_streams):
            if i < 2:  # Limit to a couple for brevity on plot
                duration_str = (
                    f", {stream.duration_years}yrs"
                    if stream.duration_years is not None
                    else ""
                )
                input_lines.append(
                    f" {stream.name[:10]}: ${stream.monthly_amount_today:,.0f}/mo, from ret.yr {stream.start_after_retirement_years + 1}{duration_str}, {stream.tax_rate * 100:.0f}%Tax"
                )

    output_lines = [
        "--- Results ---",
        f"Req.Work: {analysis_summary['required_working_months']}mo ({analysis_summary['required_working_months'] / MONTHS_PER_YEAR:.1f}yr)",
        f"Success: {analysis_summary['final_success_probability']:.1f}% (Target: {p.target_probability:.1f}%)",
        f"Med Start Bal: ${analysis_summary['median_start_retirement_balance']:,.0f}",
        f"Med Final Bal (succ): ${analysis_summary['median_final_balance']:,.0f}",
        f"SWR (Med Start): {analysis_summary.get('SWR', float('nan')):.2f}%",
    ]

    # Text Block Positioning
    x_pos_text = 0.98  # Right edge of text block at 98% of axes width
    y_coord_start = 0.98  # Top edge of text block at 98% of axes height
    line_spacing_val = 0.035  # Adjusted for better readability with smaller font
    fontsize_text = 6.5  # Smaller font size to fit more text inside

    # Plot input parameters
    for i, line_text in enumerate(input_lines):
        ax.text(
            x_pos_text,
            y_coord_start - i * line_spacing_val,
            line_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=fontsize_text,
            color=TEXT_INPUT_COLOR,
            bbox=dict(
                facecolor="white",
                alpha=0.80,
                pad=2,
                edgecolor="lightgrey",
                boxstyle="round,pad=0.3",
            ),
        )

    # Gap before output lines
    output_y_start_offset = (len(input_lines) * line_spacing_val) + (
        line_spacing_val * 0.75
    )

    # Plot output/results
    for j, line_text in enumerate(output_lines):
        ax.text(
            x_pos_text,
            y_coord_start - output_y_start_offset - (j * line_spacing_val),
            line_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=fontsize_text,
            color=TEXT_OUTPUT_COLOR,
            fontweight="bold",  # Same font size for consistency
            bbox=dict(
                facecolor="white",
                alpha=0.85,
                pad=2,
                edgecolor="lightgrey",
                boxstyle="round,pad=0.3",
            ),
        )

    if not balances_in_millions.empty:
        plt.axvline(
            balances_in_millions.median(),
            color="blue",
            linestyle="dashed",
            linewidth=1.2,
            label=f"Median (Succ.): ${balances_in_millions.median():.2f}M",
        )
    plt.axvline(0, color="red", linestyle="-", linewidth=1.0, label="Zero Balance")

    plt.title(f"Final Balance Distribution: {input_config.Nickname}", fontsize=14)
    plt.xlabel("Final Balance (Millions of $)", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)  # Simplified
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Adjust legend position if text box is in upper right
    # If the text box is well-placed, the legend might need to move or be omitted if labels are clear
    handles, labels = ax.get_legend_handles_labels()
    if handles:  # Only show legend if there are items to show
        ax.legend(
            handles, labels, fontsize=7, loc="upper left", bbox_to_anchor=(0.01, 0.98)
        )  # Move legend to upper left

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()  # Call tight_layout AFTER all elements are added

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
    dpi_setting: int = 300,  # Allow DPI to be configurable
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
        logger.warning(
            f"No trajectory percentile data to plot for '{filename}'. Skipping."
        )
        return

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    years_x_axis = np.arange(len(trajectory_percentiles_df))
    max_years_plot = len(years_x_axis) - 1 if len(years_x_axis) > 0 else 0

    if sample_trajectories:
        for i, trajectory in enumerate(sample_trajectories):
            if len(trajectory) == len(years_x_axis):
                ax.plot(
                    years_x_axis,
                    np.array(trajectory) / 1e6,
                    color="grey",
                    alpha=0.20,
                    linewidth=0.6,
                    label="_nolegend_",
                )
            else:
                logger.warning(
                    f"Sample trajectory {i} for '{filename}' length mismatch (expected {len(years_x_axis)}, got {len(trajectory)}). Skipping."
                )

    if 0.5 in trajectory_percentiles_df.columns:
        ax.plot(
            years_x_axis,
            trajectory_percentiles_df[0.5] / 1e6,
            color="blue",
            linewidth=1.8,
            label="Median (50th Percentile)",
        )
    else:
        logger.warning(
            "Median (0.5 percentile) not found in trajectory_percentiles_df."
        )

    percentile_bands_to_plot = []
    if (
        0.05 in trajectory_percentiles_df.columns
        and 0.95 in trajectory_percentiles_df.columns
    ):
        percentile_bands_to_plot.append(
            {
                "low": 0.05,
                "high": 0.95,
                "color": "salmon",
                "alpha": 0.15,
                "label": "5th-95th Percentile Range",
            }
        )
    elif (
        0.1 in trajectory_percentiles_df.columns
        and 0.9 in trajectory_percentiles_df.columns
    ):
        percentile_bands_to_plot.append(
            {
                "low": 0.1,
                "high": 0.9,
                "color": "orangered",
                "alpha": 0.15,
                "label": "10th-90th Percentile Range",
            }
        )

    if (
        0.25 in trajectory_percentiles_df.columns
        and 0.75 in trajectory_percentiles_df.columns
    ):
        percentile_bands_to_plot.append(
            {
                "low": 0.25,
                "high": 0.75,
                "color": "skyblue",
                "alpha": 0.25,
                "label": "25th-75th Percentile Range",
            }
        )

    for band in percentile_bands_to_plot:
        if (
            band["low"] in trajectory_percentiles_df.columns
            and band["high"] in trajectory_percentiles_df.columns
        ):
            ax.fill_between(
                years_x_axis,
                trajectory_percentiles_df[band["low"]] / 1e6,
                trajectory_percentiles_df[band["high"]] / 1e6,
                color=band["color"],
                alpha=band["alpha"],
                label=band["label"],
                interpolate=True,
            )
        else:
            logger.warning(
                f"Columns for percentile band {band['label']} (low: {band['low']}, high: {band['high']}) not found. Skipping band."
            )

    working_years_float = working_months / MONTHS_PER_YEAR
    if len(years_x_axis) > 0 and 0 <= working_years_float <= max_years_plot:
        ax.axvline(
            x=working_years_float,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Retirement Starts ({working_years_float:.1f} yrs)",
        )
    elif len(years_x_axis) > 0 and working_years_float > max_years_plot:
        logger.info(
            f"Retirement at {working_years_float:.1f} yrs is beyond the plot's x-axis range ({max_years_plot} yrs). Retirement line not plotted."
        )
    elif working_years_float < 0:
        logger.info(
            f"Retirement at {working_years_float:.1f} yrs is before simulation start. Retirement line not plotted."
        )

    if input_config.other_income_streams:
        line_styles = ["-", "--", "-.", ":"]
        colors = ["green", "purple", "brown", "cyan", "magenta", "olive"]
        for i, stream in enumerate(input_config.other_income_streams):
            if not hasattr(stream, "name") or not hasattr(
                stream, "start_after_retirement_years"
            ):
                logger.warning(
                    f"Income stream at index {i} is missing 'name' or 'start_after_retirement_years' attribute. Skipping."
                )
                continue

            stream_start_sim_year = (
                working_years_float + stream.start_after_retirement_years
            )

            if len(years_x_axis) > 0 and 0 <= stream_start_sim_year <= max_years_plot:
                ax.axvline(
                    x=stream_start_sim_year,
                    color=colors[i % len(colors)],
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=1.0,
                    label=f"{stream.name} Starts (yr {stream_start_sim_year:.1f})",
                )
            elif len(years_x_axis) > 0 and stream_start_sim_year > max_years_plot:
                logger.warning(
                    f"Income stream '{stream.name}' starts at simulation year {stream_start_sim_year:.1f}, which is beyond the plot's x-axis range ({max_years_plot} yrs). Skipping line."
                )
            elif stream_start_sim_year < 0:
                logger.info(
                    f"Income stream '{stream.name}' starts at simulation year {stream_start_sim_year:.1f}, which is before plot's x-axis range (0 yrs). Skipping line."
                )

    ax.set_xlabel("Years from Simulation Start", fontsize=9)
    ax.set_ylabel("Portfolio Balance (Millions of $)", fontsize=9)
    ax.set_title(
        f"Portfolio Balance Trajectories - Scenario: {input_config.Nickname}",
        fontsize=11,
    )
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.grid(True, linestyle=":", alpha=0.6)

    def millions_formatter(x_val, pos):
        return f"{x_val:.1f}M" if x_val != 0 else "0"

    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

    if len(years_x_axis) > 0:
        ax.set_xlim(left=0, right=max_years_plot)

    min_y_val_plot = 0
    if not trajectory_percentiles_df.empty:
        percentile_cols_for_ylim = [
            col
            for col in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
            if col in trajectory_percentiles_df.columns
        ]
        if percentile_cols_for_ylim:
            all_percentile_values = trajectory_percentiles_df[
                percentile_cols_for_ylim
            ].values.flatten()
            # Check if sample trajectories also contribute to y-axis range
            if sample_trajectories:
                all_sample_values = np.array(
                    [
                        val
                        for traj in sample_trajectories
                        for val in traj
                        if len(traj) == len(years_x_axis)
                    ]
                )
                if len(all_sample_values) > 0:
                    all_percentile_values = np.concatenate(
                        (all_percentile_values, all_sample_values)
                    )

            if len(all_percentile_values) > 0:
                min_data_val_abs = np.min(all_percentile_values) / 1e6
                max_data_val_abs = np.max(all_percentile_values) / 1e6

                if min_data_val_abs < 0:
                    min_y_val_plot = (
                        min_data_val_abs * 1.05
                    )  # Add 5% buffer for negative values
                # If all values are positive, min_y_val_plot remains 0 unless specified otherwise

                # Ensure the top of the plot has some space too
                ax.set_ylim(
                    bottom=min_y_val_plot,
                    top=max_data_val_abs * 1.05 if max_data_val_abs > 0 else 1,
                )  # Add 5% buffer at top, or default 1 if max is 0
            else:  # No data to determine range, default y-lim
                ax.set_ylim(bottom=0, top=1)  # Default if no data points
        else:  # No percentile columns found, default y-lim
            ax.set_ylim(bottom=0, top=1)  # Default if no percentile columns
    else:  # No x-axis, default y-lim
        ax.set_ylim(bottom=0, top=1)

    ax.legend(fontsize=7.5, loc="best")
    plt.tight_layout()

    # --- Save Plot ---
    try:
        file_directory = os.path.dirname(filename)
        # Only attempt to create directories if a non-empty path is specified
        if (
            file_directory
        ):  # This ensures we don't call os.makedirs with an empty string
            os.makedirs(file_directory, exist_ok=True)

        plt.savefig(filename, dpi=dpi_setting)
        logger.info(f"Trajectory plot saved to {filename} (DPI: {dpi_setting})")
    except Exception as e:
        logger.error(f"Error saving trajectory plot '{filename}': {e}", exc_info=True)
    finally:
        plt.close()
