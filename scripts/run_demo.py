from __future__ import annotations

import argparse
from pathlib import Path

from bike_demand.predict import predict_from_csv
from bike_demand.reporting import generate_visual_report, write_markdown_summary
from bike_demand.train import train_and_select_model
from generate_synthetic_data import generate_datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete reproducible synthetic bike-demand ML demonstration"
    )
    parser.add_argument("--data", default="data/sample/synthetic_bike_demand.csv")
    parser.add_argument("--future", default="data/sample/future_observations.csv")
    parser.add_argument("--regenerate-data", action="store_true", help="Regenerate the synthetic sample files")
    args = parser.parse_args()

    training_path = Path(args.data)
    future_path = Path(args.future)
    if args.regenerate_data or not training_path.is_file() or not future_path.is_file():
        generate_datasets(training_output=training_path, future_output=future_path)

    print("[1/4] Training and comparing candidate models...")
    result = train_and_select_model(data_path=training_path)

    print("[2/4] Predicting demand for future synthetic observations...")
    future_predictions = predict_from_csv(
        input_csv=future_path,
        artifact_path=result.artifact_path,
        output_csv="artifacts/future_predictions.csv",
    )

    print("[3/4] Generating evaluation charts...")
    figures = generate_visual_report(
        data_path=training_path,
        predictions_path=result.predictions_path,
        leaderboard_path=result.leaderboard_path,
        feature_importance_path=result.feature_importance_path,
    )

    print("[4/4] Writing the model report...")
    report_path = write_markdown_summary(
        best_model_name=result.best_model_name,
        test_metrics=result.test_metrics,
        generated_paths=figures,
        output_path="reports/model_report.md",
    )
    with report_path.open("a", encoding="utf-8") as report:
        report.write(
            "\n\n## Dataset Note\n"
            "This run uses a reproducible synthetic dataset generated for pipeline demonstration. "
            "The metrics measure performance on simulated data and must not be interpreted as "
            "real-world bike-taxi demand accuracy.\n"
        )

    print("\nDemo completed successfully.")
    print(f"Selected model: {result.best_model_name}")
    print("Chronological holdout metrics:")
    for name, value in result.test_metrics.items():
        print(f"  {name.upper()}: {value:.4f}")
    print(f"Future predictions: {len(future_predictions)} rows -> artifacts/future_predictions.csv")
    print(f"Model bundle: {result.artifact_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
