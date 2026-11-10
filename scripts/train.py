from __future__ import annotations

import argparse

from bike_demand.train import train_and_select_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bike demand prediction models")
    parser.add_argument(
        "--data",
        default="data/sample/synthetic_bike_demand.csv",
        help="Path to training CSV (defaults to the included synthetic demo dataset)",
    )
    parser.add_argument(
        "--artifact",
        default="artifacts/model_bundle.joblib",
        help="Path to save trained model bundle",
    )
    parser.add_argument(
        "--leaderboard",
        default="artifacts/model_leaderboard.csv",
        help="Path to save the model comparison table",
    )
    parser.add_argument(
        "--predictions",
        default="artifacts/test_predictions.csv",
        help="Path to save holdout predictions",
    )
    parser.add_argument(
        "--importance",
        default="artifacts/feature_importance.csv",
        help="Path to save feature importance values",
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

    print("Training completed")
    print(f"Best model: {result.best_model_name}")
    print("\nLeaderboard:")
    print(result.leaderboard.to_string(index=False))
    print("\nHoldout test metrics:")
    for metric_name, metric_value in result.test_metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.4f}")
    print(f"\nSaved artifact: {result.artifact_path}")
    print(f"Saved leaderboard: {result.leaderboard_path}")
    print(f"Saved predictions: {result.predictions_path}")
    print(f"Saved feature importance: {result.feature_importance_path}")


if __name__ == "__main__":
    main()
