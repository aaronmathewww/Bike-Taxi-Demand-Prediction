from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data import TARGET_COLUMN, load_dataset
from .features import prepare_dataset

sns.set_theme(style="whitegrid", context="talk")


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_visual_report(
    data_path: str | Path,
    predictions_path: str | Path,
    leaderboard_path: str | Path,
    feature_importance_path: str | Path,
    output_dir: str | Path = "reports/figures",
) -> list[Path]:
    output_dir = Path(output_dir)
    raw_df = load_dataset(data_path)
    prepared = prepare_dataset(raw_df, training=True).frame
    predictions_df = pd.read_csv(predictions_path, parse_dates=["datetime"])
    leaderboard_df = pd.read_csv(leaderboard_path)
    feature_importance_df = pd.read_csv(feature_importance_path).head(12)

    required_prediction_columns = {"datetime", TARGET_COLUMN, "predicted_count", "residual"}
    missing_prediction_columns = required_prediction_columns.difference(predictions_df.columns)
    if missing_prediction_columns:
        raise ValueError(
            f"Prediction file is missing required columns: {sorted(missing_prediction_columns)}"
        )
    if predictions_df.empty or leaderboard_df.empty:
        raise ValueError("Prediction and leaderboard files must contain at least one row.")

    generated_paths: list[Path] = []

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=prepared, x="datetime", y=TARGET_COLUMN, linewidth=2.2, color="#1f77b4")
    plt.title("Bike Demand Over Time")
    plt.xlabel("Datetime")
    plt.ylabel("Ride Count")
    path = output_dir / "demand_over_time.png"
    _save_plot(path)
    generated_paths.append(path)

    plt.figure(figsize=(10, 5))
    hourly = prepared.groupby("hour", as_index=False)[TARGET_COLUMN].mean()
    sns.barplot(data=hourly, x="hour", y=TARGET_COLUMN, color="#2a9d8f")
    plt.title("Average Demand by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Ride Count")
    path = output_dir / "average_demand_by_hour.png"
    _save_plot(path)
    generated_paths.append(path)

    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=prepared,
        x="weather",
        y=TARGET_COLUMN,
        hue="weather",
        palette="Blues",
        legend=False,
    )
    plt.title("Demand Distribution Across Weather Conditions")
    plt.xlabel("Weather")
    plt.ylabel("Ride Count")
    plt.xticks(rotation=25, ha="right")
    path = output_dir / "demand_by_weather.png"
    _save_plot(path)
    generated_paths.append(path)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=leaderboard_df,
        x="validation_mae",
        y="model",
        hue="model",
        palette="viridis",
        legend=False,
    )
    plt.title("Model Comparison on Validation MAE")
    plt.xlabel("Validation MAE")
    plt.ylabel("Model")
    path = output_dir / "model_leaderboard.png"
    _save_plot(path)
    generated_paths.append(path)

    plt.figure(figsize=(7, 7))
    sns.scatterplot(
        data=predictions_df,
        x=TARGET_COLUMN,
        y="predicted_count",
        s=70,
        alpha=0.8,
        color="#e76f51",
    )
    diagonal_limit = max(predictions_df[TARGET_COLUMN].max(), predictions_df["predicted_count"].max())
    plt.plot([0, diagonal_limit], [0, diagonal_limit], linestyle="--", color="black", linewidth=1)
    plt.title("Actual vs Predicted Demand")
    plt.xlabel("Actual Count")
    plt.ylabel("Predicted Count")
    path = output_dir / "actual_vs_predicted.png"
    _save_plot(path)
    generated_paths.append(path)

    plt.figure(figsize=(10, 5))
    sns.histplot(predictions_df["residual"], bins=20, kde=True, color="#f4a261")
    plt.title("Residual Distribution")
    plt.xlabel("Actual - Predicted")
    plt.ylabel("Frequency")
    path = output_dir / "residual_distribution.png"
    _save_plot(path)
    generated_paths.append(path)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance_df,
        x="importance",
        y="feature",
        hue="feature",
        palette="mako",
        legend=False,
    )
    plt.title("Top Feature Drivers")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    path = output_dir / "feature_importance.png"
    _save_plot(path)
    generated_paths.append(path)

    return generated_paths


def write_markdown_summary(
    best_model_name: str,
    test_metrics: dict[str, float],
    generated_paths: list[Path],
    output_path: str | Path = "reports/model_report.md",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Bike Demand Modeling Report",
        "",
        f"## Best Model",
        f"- {best_model_name}",
        "",
        "## Holdout Test Metrics",
        f"- MAE: {test_metrics['mae']:.3f}",
        f"- RMSE: {test_metrics['rmse']:.3f}",
        f"- R2: {test_metrics['r2']:.3f}",
        f"- WAPE: {test_metrics['wape']:.2f}%",
        "",
        "## Generated Visuals",
    ]
    lines.extend([f"- {path.as_posix()}" for path in generated_paths])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
