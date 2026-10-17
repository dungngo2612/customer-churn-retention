import importlib.metadata
import json
import subprocess
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
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
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Save pipeline
    import pickle
    with open(bundle_dir / "model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    # Save feature schema
    with open(bundle_dir / "feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=2)

    # Save metadata
    with open(bundle_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
