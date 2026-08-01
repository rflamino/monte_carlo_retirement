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
from starlette.responses import StreamingResponse

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from config import Config, ConfigurationError
from constants import MONTHS_PER_YEAR, SMALL_EPSILON
from simulation import (
    RetirementMonteCarloSimulator,
    median_first_year_withdrawal_rate,
    retirement_age,
    stream_payment_start_month_index,
    trajectory_time_points,
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SimulationSummary(BaseModel):
    required_working_months: int
    required_working_years: float
    working_period_is_estimate: bool = True
    retirement_age: Optional[float] = None
    success_probability: float
    target_probability: float
    median_start_balance: float
    median_final_balance_successful: float
    swr: Optional[float] = Field(
        None,
        description=(
            "Median first-year real gross withdrawal divided by "
            "start-of-retirement balance, as a percentage."
        ),
    )
    final_balance_percentiles: Dict[str, float]


class TrajectoryData(BaseModel):
    years: List[float]
    percentiles: Dict[str, List[float]]
    sample_paths: List[List[float]]


class WithdrawalRateData(BaseModel):
    """
    Real withdrawal rate by year from T=0: inflation-adjusted portfolio withdrawal
    as a percentage of the balance at retirement start (Trinity/Bengen basis).
    """

    years: List[float]
    percentiles: Dict[str, List[Optional[float]]]
    observation_counts: List[int]
    total_paths: int


class SearchCurvePoint(BaseModel):
    working_months: int
    working_years: float
    probability: float


class SearchCurveData(BaseModel):
    points: List[SearchCurvePoint]
    target_probability: float
    selected_working_months: int


class RuinHistogramData(BaseModel):
    """Elapsed retirement years at the first unfunded month (failed paths only)."""

    years_to_ruin: List[float]
    failure_count: int
    total_paths: int


class HistogramData(BaseModel):
    final_balances: List[float]
    start_balances: List[float]
    success_flags: List[bool]


class ReferenceLineData(BaseModel):
    name: str
    year: float


class SimulationResponse(BaseModel):
    scenario: str
    summary: SimulationSummary
    trajectory: Optional[TrajectoryData] = None
    trajectory_real: Optional[TrajectoryData] = None
    withdrawal_rate: Optional[WithdrawalRateData] = None
    search_curve: Optional[SearchCurveData] = None
    ruin_histogram: Optional[RuinHistogramData] = None
    histogram: HistogramData
    reference_lines: List[ReferenceLineData] = []


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


def _dedupe_search_curve(points: List[dict]) -> List[dict]:
    """Keep latest probability for each working_months, sorted ascending."""
    by_months: Dict[int, dict] = {}
    for p in points:
        by_months[int(p["working_months"])] = p
    return [by_months[m] for m in sorted(by_months)]


def _traj_payload(
    traj_pct_df, sample_paths, years: List[float]
) -> Optional[dict]:
    if traj_pct_df is None or traj_pct_df.empty:
        return None
    if len(years) != len(traj_pct_df):
        raise ValueError(
            "Trajectory time-point count does not match trajectory data "
            f"({len(years)} != {len(traj_pct_df)})."
        )
    pct_dict: Dict[str, List[float]] = {}
    for col in traj_pct_df.columns:
        pct_dict[f"p{int(col * 100)}"] = [
            round(float(v), 2) for v in traj_pct_df[col]
        ]
    return {
        "years": years,
        "percentiles": pct_dict,
        "sample_paths": (
            [[round(float(v), 2) for v in path] for path in sample_paths]
            if sample_paths
            else []
        ),
    }


def _run_simulation(
    config: Config,
    working_months_override: Optional[int] = None,
) -> dict:
    """Heavy, synchronous work -- called via ``asyncio.to_thread``."""
    simulator = RetirementMonteCarloSimulator(config)
    search_curve: List[dict] = []

    if working_months_override is not None:
        required_w_months = working_months_override
        logger.info(
            f"Using working-months override: {required_w_months} "
            f"({required_w_months / MONTHS_PER_YEAR:.1f} yrs)"
        )
    else:
        logger.info(
            f"Estimating required working months for '{config.Nickname}'"
        )
        required_w_months, achieved_prob, search_curve = (
            simulator.find_minimum_working_months(verbose=True)
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

    simulator.use_final_seeds()
    return _build_result(
        config, simulator, required_w_months, search_curve=search_curve
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.get("/api/config/default")
async def get_default_config():
    """Return the bundled ``config.json`` as a ready-to-use template."""
    config_path = os.path.join(_PROJECT_ROOT, "config.json")
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


@app.post("/api/simulate/stream")
async def simulate_stream(body: SimulationRequest):
    """Run the simulation and stream progress events via SSE."""
    try:
        config = Config(**body.config)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid configuration: {e}")

    logger.info(f"Received streaming simulation request for '{config.Nickname}'")

    async def event_generator():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _emit(event: dict):
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def _run():
            try:
                simulator = RetirementMonteCarloSimulator(config)

                if body.working_months_override is not None:
                    required_w_months = body.working_months_override
                    search_curve: List[dict] = []
                    _emit({
                        "type": "phase",
                        "phase": "final_sim",
                        "message": f"Using override: {required_w_months} months",
                    })
                else:
                    _emit({
                        "type": "phase",
                        "phase": "search",
                        "message": "Estimating required working months\u2026",
                    })
                    required_w_months, achieved_prob, search_curve = (
                        simulator.find_minimum_working_months(
                            verbose=True,
                            progress_callback=_emit,
                        )
                    )
                    if required_w_months == -1:
                        _emit({
                            "type": "error",
                            "message": (
                                f"Target {config.target_probability:.1f}% not met. "
                                f"Highest: {achieved_prob:.1f}%"
                            ),
                        })
                        return
                    _emit({
                        "type": "search_complete",
                        "working_months": required_w_months,
                        "working_years": round(required_w_months / MONTHS_PER_YEAR, 1),
                        "probability": round(achieved_prob, 2),
                    })

                _emit({
                    "type": "phase",
                    "phase": "final_sim",
                    "message": (
                        f"Running {config.num_simulations_main} final simulations "
                        f"with {required_w_months} working months\u2026"
                    ),
                })

                simulator.use_final_seeds()
                result = _build_result(
                    config,
                    simulator,
                    required_w_months,
                    search_curve=search_curve,
                )
                validated_result = SimulationResponse.model_validate(
                    result
                ).model_dump(mode="json")
                _emit({"type": "result", "data": validated_result})

            except Exception as exc:
                _emit({"type": "error", "message": str(exc)})
            finally:
                _emit(None)

        asyncio.get_event_loop().run_in_executor(None, _run)

        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, allow_nan=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _build_result(
    config: Config,
    simulator: RetirementMonteCarloSimulator,
    required_w_months: int,
    search_curve: Optional[List[dict]] = None,
) -> dict:
    """Run final simulation and assemble the response dict."""
    (
        summary_df,
        traj_pct_df,
        sample_trajectories,
        wr_pct_df,
        real_traj_pct_df,
        sample_real_trajectories,
        wr_observation_counts,
    ) = simulator.run_monte_carlo_simulations(
        working_months=required_w_months,
        num_simulations=config.num_simulations_main,
    )

    if summary_df.empty:
        raise ValueError(f"Simulation for '{config.Nickname}' yielded no results.")

    success_prob = (
        summary_df["Success"].astype(bool).mean() * 100.0
        if "Success" in summary_df.columns
        else (summary_df["Final Balance"] > SMALL_EPSILON).mean() * 100.0
    )
    successful_mask = (
        summary_df["Success"].astype(bool)
        if "Success" in summary_df.columns
        else summary_df["Final Balance"] > SMALL_EPSILON
    )
    successful = summary_df.loc[successful_mask, "Final Balance"]
    median_final = float(successful.median()) if not successful.empty else 0.0
    median_start = float(summary_df["Start Balance"].median())

    swr = median_first_year_withdrawal_rate(summary_df)

    pct_raw = summary_df["Final Balance"].quantile(
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    balance_percentiles = {
        f"p{int(k * 100)}": round(max(0.0, float(v)), 2)
        for k, v in pct_raw.items()
    }

    trajectory_years = trajectory_time_points(
        required_w_months, config.retirement_years
    )
    trajectory_data = _traj_payload(
        traj_pct_df, sample_trajectories, trajectory_years
    )
    trajectory_real_data = _traj_payload(
        real_traj_pct_df, sample_real_trajectories, trajectory_years
    )

    retirement_year_index = required_w_months / MONTHS_PER_YEAR
    required_working_years = round(required_w_months / MONTHS_PER_YEAR, 1)
    ret_age = retirement_age(config.current_age, required_w_months)
    reference_lines = []
    reference_lines.append({"name": "Retirement Starts", "year": retirement_year_index})
    for stream in (config.other_income_streams or []):
        if (
            stream.monthly_amount_today <= SMALL_EPSILON
            or stream.duration_years == 0
        ):
            continue
        pay_start_month = stream_payment_start_month_index(
            config.current_age, required_w_months, stream.start_at_age
        )
        start_yr = round(
            retirement_year_index + pay_start_month / MONTHS_PER_YEAR,
            3,
        )
        reference_lines.append({
            "name": stream.name,
            "year": start_yr,
        })

    withdrawal_rate_data = None
    if wr_pct_df is not None and not wr_pct_df.empty:
        wr_years = [
            retirement_year_index + i for i in range(len(wr_pct_df))
        ]
        wr_pct_dict: Dict[str, List[Optional[float]]] = {}
        for col in wr_pct_df.columns:
            series = []
            for v in wr_pct_df[col]:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    series.append(None)
                else:
                    series.append(round(float(v), 3))
            wr_pct_dict[f"p{int(col * 100)}"] = series
        withdrawal_rate_data = {
            "years": wr_years,
            "percentiles": wr_pct_dict,
            "observation_counts": wr_observation_counts or [],
            "total_paths": int(len(summary_df)),
        }

    search_curve_data = None
    if search_curve:
        search_curve_data = {
            "points": _dedupe_search_curve(search_curve),
            "target_probability": config.target_probability,
            "selected_working_months": required_w_months,
        }

    ruin_histogram = None
    if "YearsToRuin" in summary_df.columns:
        failed = summary_df.loc[~successful_mask, "YearsToRuin"].dropna()
        ruin_histogram = {
            "years_to_ruin": [round(float(v), 1) for v in failed],
            "failure_count": int(len(failed)),
            "total_paths": int(len(summary_df)),
        }

    return {
        "scenario": config.Nickname,
        "summary": {
            "required_working_months": required_w_months,
            "required_working_years": required_working_years,
            "working_period_is_estimate": bool(search_curve),
            "retirement_age": round(ret_age, 1),
            "success_probability": round(float(success_prob), 2),
            "target_probability": config.target_probability,
            "median_start_balance": round(median_start, 2),
            "median_final_balance_successful": round(median_final, 2),
            "swr": _safe_float(swr),
            "final_balance_percentiles": balance_percentiles,
        },
        "trajectory": trajectory_data,
        "trajectory_real": trajectory_real_data,
        "withdrawal_rate": withdrawal_rate_data,
        "search_curve": search_curve_data,
        "ruin_histogram": ruin_histogram,
        "histogram": {
            "final_balances": [
                round(float(v), 2) for v in summary_df["Final Balance"]
            ],
            "start_balances": [
                round(float(v), 2) for v in summary_df["Start Balance"]
            ],
            "success_flags": [
                bool(v) for v in summary_df["Success"]
            ],
        },
        "reference_lines": reference_lines,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _configure_logging()
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_dirs=[_BACKEND_DIR],
    )
