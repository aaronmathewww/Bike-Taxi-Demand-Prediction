from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import TARGET_COLUMN, load_dataset
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare_dataset


@dataclass
class TrainingResult:
    best_model_name: str
    leaderboard: pd.DataFrame
    test_metrics: dict[str, float]
    artifact_path: Path
    leaderboard_path: Path
    predictions_path: Path
    feature_importance_path: Path


def _split_dataset(
    df: pd.DataFrame,
    random_state: int = 22,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically so future observations never enter earlier partitions.

    random_state is retained for API compatibility; chronological splitting is
    deterministic and intentionally does not shuffle temporal observations.
    """
    del random_state
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Train and validation fractions must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1.")

    ordered = df.sort_values("datetime").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)

    if train_end < 1 or validation_end <= train_end or validation_end >= n_rows:
        raise ValueError(
            "Dataset is too small for non-empty chronological train, validation, and test splits."
        )

    return (
        ordered.iloc[:train_end].copy().reset_index(drop=True),
        ordered.iloc[train_end:validation_end].copy().reset_index(drop=True),
        ordered.iloc[validation_end:].copy().reset_index(drop=True),
    )


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def _candidate_models(random_state: int = 22) -> dict[str, Any]:
    models: dict[str, Any] = {
        "BaselineMean": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=random_state),
        "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.2, random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
        )
    except ImportError:
        # XGBoost is optional; the sklearn baselines remain available without it.
        pass
    return models


def _make_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            ("model", model),
        ]
    )


def _regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    absolute_actuals = np.abs(y_true.to_numpy())
    wape_denominator = float(np.sum(absolute_actuals))
    wape = (
        float(np.sum(np.abs(y_true.to_numpy() - predictions)) / wape_denominator * 100)
        if wape_denominator > 0
        else float("nan")
    )
    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "wape": float(wape),
    }


def _extract_feature_importance(trained_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    model = trained_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(np.ravel(model.coef_))
    else:
        importance = np.zeros(shape=len(feature_names), dtype=float)

    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_and_select_model(
    data_path: str | Path,
    artifact_path: str | Path = "artifacts/model_bundle.joblib",
    leaderboard_path: str | Path = "artifacts/model_leaderboard.csv",
    predictions_path: str | Path = "artifacts/test_predictions.csv",
    feature_importance_path: str | Path = "artifacts/feature_importance.csv",
    random_state: int = 22,
) -> TrainingResult:
    df = load_dataset(data_path)
    prepared = prepare_dataset(df, training=True)
    dataset = prepared.frame

    train_df, validation_df, test_df = _split_dataset(dataset, random_state=random_state)

    x_train = train_df[prepared.feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(float)
    x_validation = validation_df[prepared.feature_columns]
    y_validation = validation_df[TARGET_COLUMN].astype(float)
    x_test = test_df[prepared.feature_columns]
    y_test = test_df[TARGET_COLUMN].astype(float)

    leaderboard_rows: list[dict[str, float | str]] = []
    best_model_name = ""
    best_pipeline: Pipeline | None = None
    best_validation_mae = float("inf")

    for model_name, model in _candidate_models(random_state=random_state).items():
        pipeline = _make_pipeline(model)
        pipeline.fit(x_train, y_train)

        validation_predictions = pipeline.predict(x_validation)
        train_predictions = pipeline.predict(x_train)

        train_metrics = _regression_metrics(y_train, train_predictions)
        validation_metrics = _regression_metrics(y_validation, validation_predictions)

        leaderboard_rows.append(
            {
                "model": model_name,
                "train_mae": train_metrics["mae"],
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_r2": validation_metrics["r2"],
                "validation_wape": validation_metrics["wape"],
            }
        )

        if validation_metrics["mae"] < best_validation_mae:
            best_validation_mae = validation_metrics["mae"]
            best_model_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model was trained successfully.")

    # After model selection, refit the selected pipeline on train + validation.
    # The chronological test partition remains untouched until final evaluation.
    x_train_validation = pd.concat([x_train, x_validation], ignore_index=True)
    y_train_validation = pd.concat([y_train, y_validation], ignore_index=True)
    best_pipeline.fit(x_train_validation, y_train_validation)

    test_predictions = best_pipeline.predict(x_test)
    test_metrics = _regression_metrics(y_test, test_predictions)
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("validation_mae").reset_index(drop=True)

    test_predictions_df = test_df[["datetime", TARGET_COLUMN]].copy()
    test_predictions_df["predicted_count"] = test_predictions
    test_predictions_df["residual"] = test_predictions_df[TARGET_COLUMN] - test_predictions_df["predicted_count"]

    feature_importance_df = _extract_feature_importance(best_pipeline)

    artifact = {
        "pipeline": best_pipeline,
        "feature_columns": prepared.feature_columns,
        "target_column": TARGET_COLUMN,
        "best_model_name": best_model_name,
        "leaderboard": leaderboard,
        "test_metrics": test_metrics,
        "split_strategy": "chronological",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "test_start": str(test_df["datetime"].min()),
        "test_end": str(test_df["datetime"].max()),
    }

    artifact_path = Path(artifact_path)
    leaderboard_path = Path(leaderboard_path)
    predictions_path = Path(predictions_path)
    feature_importance_path = Path(feature_importance_path)
    for path in [artifact_path, leaderboard_path, predictions_path, feature_importance_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, artifact_path)
    leaderboard.to_csv(leaderboard_path, index=False)
    test_predictions_df.to_csv(predictions_path, index=False)
    feature_importance_df.to_csv(feature_importance_path, index=False)

    return TrainingResult(
        best_model_name=best_model_name,
        leaderboard=leaderboard,
        test_metrics=test_metrics,
        artifact_path=artifact_path,
        leaderboard_path=leaderboard_path,
        predictions_path=predictions_path,
        feature_importance_path=feature_importance_path,
    )
