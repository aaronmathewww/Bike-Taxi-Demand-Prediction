from __future__ import annotations

import argparse

from bike_demand.predict import predict_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict bike demand from input csv")
    parser.add_argument(
        "--input",
        default="data/sample/future_observations.csv",
        help="Path to prediction input CSV (defaults to the included future synthetic observations)",
    )
    parser.add_argument(
        "--artifact",
        default="artifacts/model_bundle.joblib",
        help="Path to trained model bundle",
    )
    parser.add_argument(
        "--output",
        default="artifacts/predictions.csv",
        help="Path to write prediction csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = predict_from_csv(args.input, args.artifact, args.output)
    print(f"Predictions saved to: {args.output}")
    print(output.head().to_string(index=False))


if __name__ == "__main__":
    main()
