"""Script to prepare train and validation splits from raw Telco Churn data."""

import argparse
import sys
from pathlib import Path

import pandas as pd


def make_splits(
    raw_churn_path: Path,
    splits_path: Path,
    train_out: Path,
    val_out: Path,
) -> dict[str, int]:
    """Read raw churn and split assignments, handle TotalCharges, and save train/val sets."""
    if not raw_churn_path.exists():
        raise FileNotFoundError(f"Raw churn dataset not found: {raw_churn_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Split assignments file not found: {splits_path}")

    raw_df = pd.read_csv(raw_churn_path)
    splits_df = pd.read_csv(splits_path)

    merged = raw_df.merge(splits_df, on="customerID", how="inner")
    if len(merged) != len(raw_df):
        raise ValueError(
            f"Merge row count mismatch: raw={len(raw_df)}, merged={len(merged)}"
        )

    # Detect empty or blank TotalCharges
    is_empty = merged["TotalCharges"].isna() | (
        merged["TotalCharges"].astype(str).str.strip() == ""
    )

    # Validate: If empty and tenure > 0, raise ValueError mentioning customerID
    invalid = merged[is_empty & (merged["tenure"] > 0)]
    if not invalid.empty:
        invalid_ids = invalid["customerID"].tolist()
        raise ValueError(
            f"Empty TotalCharges with tenure > 0 for customerID: {invalid_ids}"
        )

    # Count empty rows to be filled in train and val
    train_filled_count = int((is_empty & (merged["split"] == "train")).sum())
    val_filled_count = int((is_empty & (merged["split"] == "val")).sum())

    # Fill 0.0 when tenure == 0
    merged.loc[is_empty, "TotalCharges"] = "0.0"
    merged["TotalCharges"] = pd.to_numeric(merged["TotalCharges"], errors="raise")

    # Split train and val, dropping split column
    train_df = merged[merged["split"] == "train"].drop(columns=["split"])
    val_df = merged[merged["split"] == "val"].drop(columns=["split"])

    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    print(f"Train split saved to: {train_out} ({len(train_df)} rows)")
    print(f"Validation split saved to: {val_out} ({len(val_df)} rows)")
    print(
        f"TotalCharges filled with 0.0 -> Train: {train_filled_count}, Val: {val_filled_count}"
    )

    return {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_filled": train_filled_count,
        "val_filled": val_filled_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare train and validation splits from raw Telco Churn data"
    )
    parser.add_argument(
        "--raw-churn-path",
        type=Path,
        default=Path("data/raw/telco_churn.csv"),
        help="Path to raw telco churn CSV",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=Path("data/raw/dataset_splits.csv"),
        help="Path to dataset split assignments CSV",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Output path for train CSV",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=Path("data/processed/val.csv"),
        help="Output path for val CSV",
    )

    args = parser.parse_args()
    make_splits(
        raw_churn_path=args.raw_churn_path,
        splits_path=args.splits_path,
        train_out=args.train_out,
        val_out=args.val_out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
