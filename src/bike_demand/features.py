from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import TARGET_COLUMN, validate_dataset

NUMERIC_FEATURES = [
    "temp (°C)",
    "humidity (%)",
    "windspeed (m/s)",
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "is_peak_hour",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

CATEGORICAL_FEATURES = ["season", "weather"]
# These columns directly compose the target in common bike-sharing schemas.
# They must not be used as predictors when the target is total ride count.
LEAKAGE_COLUMNS = ["casual_riders", "registered_riders"]


@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str | None


def _title_case(value: object) -> object:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned.title() if cleaned else np.nan
    return value


def _holiday_flags(dates: pd.Series, country: str | None) -> pd.Series:
    """Return holiday flags; if the optional package is absent, use zero flags explicitly."""
    if country is None:
        return pd.Series(0, index=dates.index, dtype="int8")

    try:
        import holidays
    except ImportError:
        warnings.warn(
            "The 'holidays' package is not installed; is_holiday is set to 0. "
            "Install project requirements for calendar-holiday features.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.Series(0, index=dates.index, dtype="int8")

    calendar = holidays.country_holidays(country)
    return dates.dt.date.map(lambda value: int(value in calendar)).astype("int8")


def prepare_dataset(
    df: pd.DataFrame,
    *,
    training: bool = True,
    holiday_country: str | None = "IN",
) -> PreparedDataset:
    """Create model features without dropping rows for missing predictor values.

    Missing predictor values are intentionally retained for the sklearn imputers.
    Rows with invalid timestamps are removed; training rows also require a valid target.
    """
    validate_dataset(df, require_target=training)
    prepared = df.copy()

    prepared["datetime"] = pd.to_datetime(prepared["datetime"], errors="coerce")
    prepared = prepared.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    for column in ["season", "weather"]:
        prepared[column] = prepared[column].map(_title_case).astype("object")

    for column in ["temp (°C)", "humidity (%)", "windspeed (m/s)"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared[column] = prepared[column].replace([np.inf, -np.inf], np.nan)

    dt = prepared["datetime"]
    prepared["hour"] = dt.dt.hour.astype("int16")
    prepared["day"] = dt.dt.day.astype("int16")
    prepared["month"] = dt.dt.month.astype("int16")
    prepared["day_of_week"] = dt.dt.dayofweek.astype("int8")
    prepared["is_weekend"] = (prepared["day_of_week"] >= 5).astype("int8")
    prepared["is_peak_hour"] = prepared["hour"].isin([8, 9, 17, 18, 19]).astype("int8")
    prepared["is_holiday"] = _holiday_flags(dt, holiday_country)

    prepared["hour_sin"] = np.sin(2 * np.pi * prepared["hour"] / 24.0)
    prepared["hour_cos"] = np.cos(2 * np.pi * prepared["hour"] / 24.0)
    prepared["month_sin"] = np.sin(2 * np.pi * prepared["month"] / 12.0)
    prepared["month_cos"] = np.cos(2 * np.pi * prepared["month"] / 12.0)

    prepared = prepared.drop(columns=LEAKAGE_COLUMNS, errors="ignore")

    modeling_columns = ["datetime", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    target_column = TARGET_COLUMN if training else None
    if training:
        prepared[TARGET_COLUMN] = pd.to_numeric(prepared[TARGET_COLUMN], errors="coerce")
        prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].replace([np.inf, -np.inf], np.nan)
        modeling_columns.append(TARGET_COLUMN)

    prepared = prepared[modeling_columns]
    if training:
        prepared = prepared.dropna(subset=[TARGET_COLUMN])
        if prepared.empty:
            raise ValueError("No rows with both a valid datetime and numeric target remain.")
        if (prepared[TARGET_COLUMN] < 0).any():
            raise ValueError("The ride-count target 'count' must not contain negative values.")
    return PreparedDataset(
        frame=prepared.reset_index(drop=True),
        feature_columns=[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES],
        target_column=target_column,
    )
