from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bike_demand.data import load_dataset
from bike_demand.features import prepare_dataset
from bike_demand.predict import predict_from_csv
from bike_demand.train import _split_dataset, train_and_select_model
from scripts.generate_synthetic_data import generate_datasets


def make_dataset(rows: int = 120) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h")
    hour = timestamps.hour.to_numpy()
    count = 25 + 8 * np.sin(2 * np.pi * hour / 24) + np.arange(rows) * 0.03
    return pd.DataFrame(
        {
            "datetime": timestamps.astype(str),
            "season": np.where(timestamps.month <= 6, "Spring", "Summer"),
            "weather": np.where(hour % 3 == 0, "Clear", "Mist"),
            "temp (°C)": 20 + 5 * np.sin(2 * np.pi * hour / 24),
            "humidity (%)": 55 + (hour % 10),
            "windspeed (m/s)": 8 + (hour % 5),
            "count": count,
            "casual_riders": np.full(rows, 5),
            "registered_riders": np.full(rows, 20),
        }
    )


class PipelineTests(unittest.TestCase):
    def test_synthetic_generator_creates_training_and_future_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            training_path, future_path = generate_datasets(
                rows=120,
                training_output=temp / "training.csv",
                future_output=temp / "future.csv",
                seed=7,
                future_hours=6,
            )
            training = pd.read_csv(training_path)
            future = pd.read_csv(future_path)

            self.assertEqual(len(training), 120)
            self.assertEqual(len(future), 6)
            self.assertIn("count", training.columns)
            self.assertNotIn("count", future.columns)
            self.assertTrue((training["count"] >= 0).all())
            self.assertEqual(
                pd.to_datetime(future["datetime"]).min(),
                pd.to_datetime(training["datetime"]).max() + pd.Timedelta(hours=1),
            )

    def test_load_dataset_reports_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_dataset("data/raw/does-not-exist.csv")

    def test_prepare_dataset_removes_leakage_and_keeps_missing_features(self) -> None:
        df = make_dataset()
        df.loc[4, "temp (°C)"] = np.nan
        prepared = prepare_dataset(df, training=True, holiday_country=None)

        self.assertNotIn("casual_riders", prepared.frame.columns)
        self.assertNotIn("registered_riders", prepared.frame.columns)
        self.assertIn("count", prepared.frame.columns)
        self.assertTrue(prepared.frame["datetime"].is_monotonic_increasing)
        self.assertTrue(pd.isna(prepared.frame.loc[4, "temp (°C)"]))

    def test_split_is_chronological_and_non_overlapping(self) -> None:
        prepared = prepare_dataset(make_dataset(), training=True, holiday_country=None).frame
        train, validation, test = _split_dataset(prepared)

        self.assertLess(train["datetime"].max(), validation["datetime"].min())
        self.assertLess(validation["datetime"].max(), test["datetime"].min())
        self.assertEqual(len(train) + len(validation) + len(test), len(prepared))

    def test_training_persistence_and_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            train_path = temp / "train.csv"
            inference_path = temp / "inference.csv"
            training_data = make_dataset(120)
            training_data.loc[10, "temp (°C)"] = np.nan
            training_data.to_csv(train_path, index=False)
            make_dataset(8).drop(columns=["count", "casual_riders", "registered_riders"]).to_csv(
                inference_path, index=False
            )

            result = train_and_select_model(
                data_path=train_path,
                artifact_path=temp / "model_bundle.joblib",
                leaderboard_path=temp / "leaderboard.csv",
                predictions_path=temp / "test_predictions.csv",
                feature_importance_path=temp / "feature_importance.csv",
            )

            self.assertTrue(result.artifact_path.is_file())
            self.assertTrue(result.leaderboard_path.is_file())
            self.assertTrue(result.predictions_path.is_file())
            self.assertTrue(result.feature_importance_path.is_file())
            self.assertGreaterEqual(result.test_metrics["mae"], 0)

            output = predict_from_csv(
                inference_path,
                artifact_path=result.artifact_path,
                output_csv=temp / "predictions.csv",
            )
            self.assertEqual(len(output), 8)
            self.assertIn("predicted_count", output.columns)


if __name__ == "__main__":
    unittest.main()
