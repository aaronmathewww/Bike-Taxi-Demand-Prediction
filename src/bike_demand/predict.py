from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .data import load_dataset
from .features import prepare_dataset


def predict_from_csv(
    input_csv: str | Path,
    artifact_path: str | Path = "artifacts/model_bundle.joblib",
    output_csv: str | Path = "artifacts/predictions.csv",
) -> pd.DataFrame:
    input_df = load_dataset(input_csv)

    bundle = joblib.load(artifact_path)
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    prepared = prepare_dataset(input_df, training=False)
    processed = prepared.frame

    missing_cols = [col for col in feature_columns if col not in processed.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    x = processed[feature_columns].copy()
    predictions = pipeline.predict(x)

    output = processed[["datetime"]].copy()
    output["predicted_count"] = predictions

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    return output
