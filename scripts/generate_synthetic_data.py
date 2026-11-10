from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WEATHER_TYPES = np.array(["Clear", "Mist", "Light Rain", "Heavy Rain", "Storm"])
WEATHER_PROBABILITIES = np.array([0.52, 0.22, 0.15, 0.08, 0.03])
WEATHER_MULTIPLIERS = {
    "Clear": 1.00,
    "Mist": 0.94,
    "Light Rain": 0.78,
    "Heavy Rain": 0.55,
    "Storm": 0.35,
}


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4):
        return "Spring"
    if month in (5, 6):
        return "Summer"
    if month in (7, 8, 9):
        return "Monsoon"
    return "Autumn"


def _make_observations(timestamps: pd.DatetimeIndex, rng: np.random.Generator, include_target: bool) -> pd.DataFrame:
    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    month = timestamps.month.to_numpy()

    # Synthetic weather values are generated from broad seasonal patterns,
    # with noise so the dataset is useful for demonstrating a real ML workflow.
    annual_temperature = 27 + 5 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    daily_temperature = 2.5 * np.sin(2 * np.pi * (hour - 8) / 24)
    temperature = annual_temperature + daily_temperature + rng.normal(0, 2.2, len(timestamps))
    humidity = np.clip(67 - 0.65 * (temperature - 25) + rng.normal(0, 9, len(timestamps)), 25, 100)
    windspeed = np.clip(rng.gamma(shape=2.2, scale=3.0, size=len(timestamps)), 0, 35)

    month_weather_weights = np.tile(WEATHER_PROBABILITIES, (len(timestamps), 1))
    monsoon_mask = np.isin(month, [6, 7, 8, 9])
    month_weather_weights[monsoon_mask] = np.array([0.30, 0.20, 0.25, 0.17, 0.08])
    weather_indices = np.array([
        rng.choice(len(WEATHER_TYPES), p=weights / weights.sum())
        for weights in month_weather_weights
    ])
    weather = WEATHER_TYPES[weather_indices]
    season = np.array([_season(int(value)) for value in month])

    frame = pd.DataFrame(
        {
            "datetime": timestamps,
            "season": season,
            "weather": weather,
            "temp (°C)": np.round(temperature, 2),
            "humidity (%)": np.round(humidity, 2),
            "windspeed (m/s)": np.round(windspeed, 2),
        }
    )

    if include_target:
        morning_peak = 42 * np.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
        evening_peak = 58 * np.exp(-0.5 * ((hour - 18) / 2.4) ** 2)
        midday_activity = 16 * np.exp(-0.5 * ((hour - 13) / 3.2) ** 2)
        night_drop = np.where((hour <= 5) | (hour >= 23), -18, 0)
        weekday_factor = np.where(day_of_week < 5, 1.12, 0.88)
        weather_factor = np.array([WEATHER_MULTIPLIERS[item] for item in weather])
        comfort_factor = np.clip(1.0 - np.abs(temperature - 27) * 0.012, 0.78, 1.05)
        gradual_trend = np.linspace(0, 12, len(timestamps))
        expected = (
            48 + morning_peak + evening_peak + midday_activity + night_drop + gradual_trend
        ) * weekday_factor * weather_factor * comfort_factor
        noise = rng.normal(0, 7.0, len(timestamps))
        frame["count"] = np.maximum(0, np.rint(expected + noise)).astype(int)

    return frame


def generate_datasets(
    rows: int = 8760,
    start: str = "2025-01-01 00:00:00",
    training_output: str | Path = "data/sample/synthetic_bike_demand.csv",
    future_output: str | Path = "data/sample/future_observations.csv",
    seed: int = 42,
    future_hours: int = 24,
) -> tuple[Path, Path]:
    if rows < 100:
        raise ValueError("At least 100 hourly rows are required for chronological train/validation/test splits.")
    if future_hours < 1:
        raise ValueError("future_hours must be at least 1.")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=rows, freq="h")
    training = _make_observations(timestamps, rng, include_target=True)
    future_timestamps = pd.date_range(start=timestamps[-1] + pd.Timedelta(hours=1), periods=future_hours, freq="h")
    future = _make_observations(future_timestamps, rng, include_target=False)

    training_path = Path(training_output)
    future_path = Path(future_output)
    training_path.parent.mkdir(parents=True, exist_ok=True)
    future_path.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(training_path, index=False)
    future.to_csv(future_path, index=False)
    return training_path, future_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible synthetic bike-demand demo data")
    parser.add_argument("--rows", type=int, default=8760, help="Number of hourly training observations")
    parser.add_argument("--start", default="2025-01-01 00:00:00", help="First training timestamp")
    parser.add_argument("--training-output", default="data/sample/synthetic_bike_demand.csv")
    parser.add_argument("--future-output", default="data/sample/future_observations.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--future-hours", type=int, default=24)
    args = parser.parse_args()

    training_path, future_path = generate_datasets(
        rows=args.rows,
        start=args.start,
        training_output=args.training_output,
        future_output=args.future_output,
        seed=args.seed,
        future_hours=args.future_hours,
    )
    training_df = pd.read_csv(training_path)
    future_df = pd.read_csv(future_path)
    print(f"Synthetic training data: {training_path} ({len(training_df):,} rows)")
    print(f"Future prediction inputs: {future_path} ({len(future_df):,} rows)")
    print("Reminder: these are simulated observations, not real customer or operational records.")


if __name__ == "__main__":
    main()
