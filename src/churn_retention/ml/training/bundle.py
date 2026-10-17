import importlib.metadata
import json
import pickle
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline


def get_git_info() -> dict[str, Any]:
    """Get git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()

        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()

        dirty = len(status) > 0

        if dirty:
            warnings.warn("Git repository has uncommitted changes (dirty=True)")

        return {"git_commit": commit, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "git_dirty": None}


def get_library_versions() -> dict[str, str]:
    """Get versions of key dependencies."""
    libraries = ["scikit-learn", "catboost", "pandas", "numpy"]
    versions = {"python": importlib.metadata.version("customer-churn-retention").split("+")[0]}

    for lib in libraries:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "unknown"

    # Get python version
    import sys
    versions["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    return versions


def create_feature_schema(
    df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> list[dict]:
    """Create feature schema for bundle."""
    schema = []

    for col in numeric_cols:
        schema.append({
            "name": col,
            "kind": "numeric",
            "dtype": df[col].dtype.name,
        })

    for col in categorical_cols:
        schema.append({
            "name": col,
            "kind": "categorical",
            "dtype": df[col].dtype.name,
        })

    return schema


def save_bundle(
    pipeline: Pipeline,
    bundle_dir: Path,
    feature_schema: list[dict],
    metadata: dict,
) -> None:
    """Save model bundle to directory."""
    try:
        bundle_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"Bundle directory already exists: {bundle_dir}"
        ) from None

    # Save pipeline
    with open(bundle_dir / "model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    # Save feature schema
    with open(bundle_dir / "feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=2)

    # Save metadata
    with open(bundle_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


@dataclass
class LoadedBundle:
    _pipeline: Pipeline
    metadata: dict[str, Any]
    feature_schema: list[dict[str, str]]

    @property
    def model_version(self) -> str:
        return self.metadata["model_version"]

    def predict_proba(self, df: pd.DataFrame) -> NDArray[np.float64]:
        """Predict encoded class 1, preserving row order and validating features."""
        feature_names = [column["name"] for column in self.feature_schema]
        missing = [name for name in feature_names if name not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        extra = [name for name in df.columns if name not in feature_names]
        if extra:
            warnings.warn(f"Ignoring extra columns: {extra}", stacklevel=2)

        for column in self.feature_schema:
            if column["kind"] != "numeric":
                continue
            name = column["name"]
            values = df[name]
            invalid = values.notna() & pd.to_numeric(values, errors="coerce").isna()
            if invalid.any():
                samples = values[invalid].head(5).tolist()
                raise ValueError(
                    f"Numeric column '{name}' contains non-numeric values. "
                    f"Examples: {samples}"
                )

        features = df.loc[:, feature_names]
        positive_index = list(self._pipeline.classes_).index(1)
        return np.asarray(
            self._pipeline.predict_proba(features)[:, positive_index], dtype=np.float64
        )


def load_bundle(path: str | Path) -> LoadedBundle:
    """Load a trusted local bundle containing a pickled, fitted pipeline."""
    bundle_dir = Path(path)
    with open(bundle_dir / "model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open(bundle_dir / "feature_schema.json", encoding="utf-8") as f:
        feature_schema = json.load(f)
    with open(bundle_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return LoadedBundle(pipeline, metadata, feature_schema)
