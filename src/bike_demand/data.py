from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_COLUMN_ALIASES = {
    "temp (Â°C)": "temp (°C)",
    "temp (�C)": "temp (°C)",
}

REQUIRED_COLUMNS = {
    "datetime",
    "season",
    "weather",
    "temp (°C)",
    "humidity (%)",
    "windspeed (m/s)",
}

TARGET_COLUMN = "count"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim column names and normalize known temperature-header encodings."""
    standardized = df.copy()
    standardized.columns = [str(column).strip() for column in standardized.columns]
    standardized = standardized.rename(columns=RAW_COLUMN_ALIASES)

    duplicated = standardized.columns[standardized.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Dataset contains duplicate column names after normalization: {duplicated}")
    return standardized


def validate_dataset(df: pd.DataFrame, *, require_target: bool = True) -> None:
    """Validate the minimum schema required by the feature pipeline."""
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must include target column '{TARGET_COLUMN}'.")


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and validate its required columns."""
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place an authorized CSV at this path; "
            "see data/README.md for the expected schema."
        )

    df = pd.read_csv(path)
    df = standardize_columns(df)
    validate_dataset(df, require_target=TARGET_COLUMN in df.columns)
    return df
