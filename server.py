import asyncio
import json
import math
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from config import Config, ConfigurationError
from constants import MONTHS_PER_YEAR, SMALL_EPSILON
from simulation import RetirementMonteCarloSimulator


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SimulationSummary(BaseModel):
    required_working_months: int
    required_working_years: float
    success_probability: float
    target_probability: float
    median_start_balance: float
    median_final_balance_successful: float
    swr: Optional[float] = None
    final_balance_percentiles: Dict[str, float]


class TrajectoryData(BaseModel):
    years: List[int]
    percentiles: Dict[str, List[float]]
    sample_paths: List[List[float]]


class HistogramData(BaseModel):
    final_balances: List[float]
    start_balances: List[float]


class SimulationResponse(BaseModel):
    scenario: str
    summary: SimulationSummary
    trajectory: Optional[TrajectoryData] = None
    histogram: HistogramData


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    config: Dict[str, Any] = Field(
        ...,
        description="Simulation configuration (same schema as config.json).",
    )
    working_months_override: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "If provided, skip the search phase and run the final simulation "
            "directly with this many working months."
        ),
    )


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )
    logger.add(
        "server.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="10 MB",
    )


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _configure_logging()
    logger.info("Monte Carlo Retirement API starting up")
    yield
    logger.info("Monte Carlo Retirement API shutting down")


app = FastAPI(
    title="Monte Carlo Retirement Simulator API",
    description="Backend API for running Monte Carlo retirement simulations and returning data for frontend visualisation.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: float) -> Optional[float]:
    """Convert NaN / Inf to None so JSON serialisation stays valid."""
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, 2)


def _run_simulation(
    config: Config,
    working_months_override: Optional[int] = None,
) -> dict:
    """Heavy, synchronous work -- called via ``asyncio.to_thread``."""
    simulator = RetirementMonteCarloSimulator(config)

    if working_months_override is not None:
        required_w_months = working_months_override
        logger.info(
            f"Using working-months override: {required_w_months} "
            f"({required_w_months / MONTHS_PER_YEAR:.1f} yrs)"
        )
    else:
        logger.info(f"Searching for minimum working months for '{config.Nickname}'")
        required_w_months, achieved_prob = simulator.find_minimum_working_months(
            verbose=True,
        )
        if required_w_months == -1:
            raise ValueError(
                f"Target probability of {config.target_probability:.2f}% could not be "
                f"met. Highest achieved: {achieved_prob:.2f}%"
            )

    logger.info(
        f"Running final simulation for '{config.Nickname}' "
        f"({config.num_simulations_main} sims, {required_w_months} working months)"
    )

    summary_df, traj_pct_df, sample_trajectories = (
        simulator.run_monte_carlo_simulations(
            working_months=required_w_months,
            num_simulations=config.num_simulations_main,
        )
    )

    if summary_df.empty:
        raise ValueError(f"Simulation for '{config.Nickname}' yielded no results.")

    # ---- summary statistics ----
    success_prob = (summary_df["Final Balance"] > SMALL_EPSILON).mean() * 100.0
    successful = summary_df.loc[
        summary_df["Final Balance"] > SMALL_EPSILON, "Final Balance"
    ]
    median_final = float(successful.median()) if not successful.empty else 0.0
    median_start = float(summary_df["Start Balance"].median())

    annual_expenses_t0 = config.monthly_expenses * MONTHS_PER_YEAR
    swr = (
        (annual_expenses_t0 * 100.0) / median_start
        if median_start > SMALL_EPSILON
        else float("nan")
    )

    pct_raw = summary_df["Final Balance"].quantile(
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    balance_percentiles = {
        f"p{int(k * 100)}": round(max(0.0, float(v)), 2)
        for k, v in pct_raw.items()
    }

    # ---- trajectory data ----
    trajectory_data = None
    if traj_pct_df is not None and not traj_pct_df.empty:
        years = list(range(len(traj_pct_df)))
        pct_dict: Dict[str, List[float]] = {}
        for col in traj_pct_df.columns:
            pct_dict[f"p{int(col * 100)}"] = [
                round(float(v), 2) for v in traj_pct_df[col]
            ]
        trajectory_data = {
            "years": years,
            "percentiles": pct_dict,
            "sample_paths": (
                [[round(float(v), 2) for v in path] for path in sample_trajectories]
                if sample_trajectories
                else []
            ),
        }

    return {
        "scenario": config.Nickname,
        "summary": {
            "required_working_months": required_w_months,
            "required_working_years": round(required_w_months / MONTHS_PER_YEAR, 1),
            "success_probability": round(float(success_prob), 2),
            "target_probability": config.target_probability,
            "median_start_balance": round(median_start, 2),
            "median_final_balance_successful": round(median_final, 2),
            "swr": _safe_float(swr),
            "final_balance_percentiles": balance_percentiles,
        },
        "trajectory": trajectory_data,
        "histogram": {
            "final_balances": [
                round(float(v), 2) for v in summary_df["Final Balance"]
            ],
            "start_balances": [
                round(float(v), 2) for v in summary_df["Start Balance"]
            ],
        },
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.get("/api/config/default")
async def get_default_config():
    """Return the bundled ``config.json`` as a ready-to-use template."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Default config.json not found.")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/validate")
async def validate_config(body: SimulationRequest):
    """Validate a configuration without running any simulation."""
    try:
        config = Config(**body.config)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid configuration: {e}")
    return {"valid": True, "scenario": config.Nickname}


@app.post("/api/simulate", response_model=SimulationResponse)
async def simulate(body: SimulationRequest):
    """Run the Monte Carlo simulation and return all data needed for plots."""
    try:
        config = Config(**body.config)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid configuration: {e}")

    logger.info(f"Received simulation request for scenario '{config.Nickname}'")

    try:
        result = await asyncio.to_thread(
            _run_simulation, config, body.working_months_override,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation error: {e}")

    logger.info(f"Simulation complete for '{config.Nickname}'")
    return result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _configure_logging()
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
