from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "telco_churn.csv"
MANIFEST = ROOT / "artifacts" / "splits" / "dataset_splits.csv"
OUTPUT = ROOT / "data" / "processed"


def main() -> None:
    data = pd.read_csv(SOURCE)
    splits = pd.read_csv(MANIFEST)

    if data["customerID"].isna().any() or data["customerID"].duplicated().any():
        raise ValueError("Dữ liệu gốc có customerID thiếu hoặc trùng.")

    if splits["customerID"].isna().any() or splits["customerID"].duplicated().any():
        raise ValueError("Manifest có customerID thiếu hoặc trùng.")

    if set(data["customerID"]) != set(splits["customerID"]):
        raise ValueError("ID trong manifest không khớp dữ liệu gốc.")

    if not set(splits["split"]).issubset({"train", "val", "test"}):
        raise ValueError("Manifest chứa tên split không hợp lệ.")

    merged = data.merge(
        splits,
        on="customerID",
        how="inner",
        validate="one_to_one",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        frame = merged.loc[merged["split"] == split].drop(columns="split")
        frame.to_csv(OUTPUT / f"{split}.csv", index=False)
        print(f"{split}: {len(frame)} rows")


if __name__ == "__main__":
    main()
