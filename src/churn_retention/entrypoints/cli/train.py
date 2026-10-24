#!/usr/bin/env python3
"""CLI entrypoint for training churn prediction models."""

import argparse
import sys
from pathlib import Path

from churn_retention.workflows.train_model import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train a churn prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["logistic", "catboost"],
        help="Type of model to train",
    )

    parser.add_argument(
        "--train-path",
        type=Path,
        required=True,
        help="Path to training CSV file",
    )

    parser.add_argument(
        "--val-path",
        type=Path,
        required=True,
        help="Path to validation CSV file",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config if provided)",
    )

    parser.add_argument(
        "--class-weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use balanced class weights",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/bundles"),
        help="Output directory for model bundles",
    )

    args = parser.parse_args()

    try:
        result = train_model(
            config_path=args.config,
            model_type=args.model_type,
            class_weight=args.class_weight,
            seed=args.seed,
            train_path=args.train_path,
            val_path=args.val_path,
            out_dir=args.out_dir,
        )

        print("\n=== Training Complete ===")
        print(f"Model version: {result['model_version']}")
        print(f"Bundle: {result['bundle_dir']}")
        print(f"MLflow run: {result['run_id']}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
