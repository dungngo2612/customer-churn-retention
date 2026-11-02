"""Small RAG-only fixtures: no classifier, training helper or v1 builder."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from churn_retention.adapters.retrieval._reference_v2_format import ReferenceManifestV2
from churn_retention.workflows.build_reference_v2 import build_reference_v2

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES: dict[str, list[str]] = json.loads(
    (ROOT / "tests/fixtures/llm_rag_v1_golden.json").read_text()
)["preprocessing"]["categorical_vocabulary"]


@dataclass
class Source:
    train: Path
    splits: Path
    output: Path
    frame: pd.DataFrame

    def build(self) -> ReferenceManifestV2:
        return build_reference_v2(self.train, self.splits, self.output)


def make_source(directory: Path, n: int = 5) -> Source:
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "customerID": f"unit-{n - i:03d}",
                "Churn": "Yes" if i % 2 else "No",
                "SeniorCitizen": i % 2,
                "tenure": i,
                "MonthlyCharges": float(20 + i),
                "TotalCharges": float(i * (20 + i)),
                **{
                    name: values[i % len(values)] for name, values in CATEGORIES.items()
                },
            }
            for i in range(n)
        ]
    )
    train, splits = directory / "train.csv", directory / "splits.csv"
    frame.to_csv(train, index=False)
    pd.DataFrame(
        {
            "customerID": frame.customerID.tolist() + ["held-out-val", "held-out-test"],
            "split": ["train"] * n + ["val", "test"],
        }
    ).to_csv(splits, index=False)
    return Source(train, splits, directory / "reference-v2.zip", frame)


def query(source: Source) -> dict[str, object]:
    return {str(name): value for name, value in source.frame.iloc[0].items()}
