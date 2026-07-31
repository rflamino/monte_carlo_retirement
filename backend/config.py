import os
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, PrivateAttr, field_validator, ValidationInfo
from loguru import logger


class ConfigurationError(Exception):
    """Raised when the configuration file cannot be loaded or parsed."""


class OtherIncomeStreamConfig(BaseModel):
    """Configuration for an additional income stream during retirement."""

    name: str = Field(
        ..., description="Name of the income stream (e.g., 'Rental Income', 'Pension')."
    )
    monthly_amount_today: float = Field(
        ...,
        ge=0,
        description="Current monthly amount of this income in today's (T=0) real terms.",
    )
    start_at_age: float = Field(
        ...,
        ge=0,
        le=120,
        description=(
            "Age when this income becomes eligible. Payments begin at "
            "max(retirement_age, start_at_age), where retirement_age = "
            "current_age + working_months/12."
        ),
    )
    duration_years: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "How many years payments last after they begin "
            "(from max(retirement_age, start_at_age)). None = indefinitely."
        ),
    )
    inflation_indexed: bool = Field(
        True,
        description="If True, keeps pace with inflation from T=0. If False, its nominal value is fixed based on its value at its start date.",
    )
    tax_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Tax rate applied to this income stream."
    )
    _nominal_fixed_monthly_amount: Optional[float] = PrivateAttr(default=None)
    _master_inflation_at_start: Optional[float] = PrivateAttr(default=None)
    _payment_start_age: Optional[float] = PrivateAttr(default=None)


class Config(BaseModel):
    """Main configuration model for the retirement simulation."""

    Nickname: str = Field(
        "DefaultScenario",
        alias="scenario",
        description="A nickname for this simulation scenario.",
    )
    initial_balance: float = Field(..., ge=0)
    monthly_contribution: float = Field(..., ge=0)
    contribution_growth_rate_annual: float = Field(0.0, ge=0)
    monthly_expenses: float = Field(
        ..., ge=0, description="Monthly expenses in today's (T=0) real terms."
    )
    current_age: float = Field(
        ...,
        ge=0,
        le=120,
        description="Age at simulation start (T=0). Used with start_at_age on income streams.",
    )
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
    equity_inflation_correlation: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Correlation between equity log-returns and inflation log-rates.",
    )

    num_simulations_main: int = Field(..., gt=0)
    num_simulations_search: int = Field(...)
    target_probability: float = Field(..., ge=0.0, le=100.0)
    starting_working_months_search: int = Field(..., ge=0)
    seed: Optional[int] = Field(None)
    num_processes: Optional[int] = Field(1, ge=1)

    other_income_streams: List[OtherIncomeStreamConfig] = Field([])

    model_config = {"validate_by_name": True, "validate_assignment": True}

    @field_validator("inflation_rate_volatility")
    @classmethod
    def check_inflation_volatility(cls, v: float, info: ValidationInfo) -> float:
        if v > 0.05:
            scen_name = info.data.get("Nickname", "N/A")
            logger.warning(
                f"Inflation volatility ({v * 100:.1f}%) is relatively high for scenario '{scen_name}'."
            )
        return v

    @field_validator("inv1_returns_volatility")
    @classmethod
    def check_equity_volatility(cls, v: float, info: ValidationInfo) -> float:
        if v < 0.05:
            scen_name = info.data.get("Nickname", "N/A")
            logger.warning(
                f"Equity (Inv1) volatility ({v * 100:.1f}%) is unusually low for scenario "
                f"'{scen_name}'. Typical equity vol is ~15%. Results will understate sequence-of-returns risk."
            )
        return v

    @property
    def allocation_inv2_pct(self) -> float:
        return round(1.0 - self.allocation_inv1_pct, 4)


def load_config_from_json(file_path: str) -> Dict[str, Any]:
    """Loads and returns the configuration dictionary from a JSON file."""
    if not os.path.exists(file_path):
        raise ConfigurationError(f"Configuration file not found at: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"Error parsing JSON file '{file_path}': {e}"
        ) from e
    except Exception as e:
        raise ConfigurationError(
            f"Unexpected error reading config file '{file_path}': {e}"
        ) from e
