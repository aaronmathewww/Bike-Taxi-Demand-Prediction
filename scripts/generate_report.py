from __future__ import annotations

import argparse

from bike_demand.reporting import generate_visual_report, write_markdown_summary
from bike_demand.train import train_and_select_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the project and generate visual report assets")
    parser.add_argument("--data", default="data/sample/synthetic_bike_demand.csv", help="Path to the source dataset")
    parser.add_argument("--artifact", default="artifacts/model_bundle.joblib", help="Model bundle output path")
    parser.add_argument(
        "--leaderboard",
        default="artifacts/model_leaderboard.csv",
        help="Leaderboard CSV output path",
    )
    parser.add_argument(
        "--predictions",
        default="artifacts/test_predictions.csv",
        help="Test predictions CSV output path",
    )
    parser.add_argument(
        "--importance",
        default="artifacts/feature_importance.csv",
        help="Feature importance CSV output path",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures",
        help="Directory where PNG visualizations will be written",
    )
    parser.add_argument(
        "--report",
        default="reports/model_report.md",
        help="Path to the markdown summary report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_select_model(
        data_path=args.data,
        artifact_path=args.artifact,
        leaderboard_path=args.leaderboard,
        predictions_path=args.predictions,
        feature_importance_path=args.importance,
    )
    figure_paths = generate_visual_report(
        data_path=args.data,
        predictions_path=args.predictions,
        leaderboard_path=args.leaderboard,
        feature_importance_path=args.importance,
        output_dir=args.figures_dir,
    )
    report_path = write_markdown_summary(
        best_model_name=result.best_model_name,
        test_metrics=result.test_metrics,
        generated_paths=figure_paths,
        output_path=args.report,
    )

    print(f"Best model: {result.best_model_name}")
    print(f"Report written to: {report_path}")
    print("Generated figures:")
    for path in figure_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
