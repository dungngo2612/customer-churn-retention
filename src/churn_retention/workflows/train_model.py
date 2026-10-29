import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from churn_retention.adapters.model_registry.mlflow_adapter import (
    flatten_params,
    log_training_run,
)
from churn_retention.ml.training.baseline import train_logistic
from churn_retention.ml.training.bundle import (
    create_feature_schema,
    get_git_info,
    get_library_versions,
    save_bundle,
)
from churn_retention.ml.training.catboost_model import train_catboost
from churn_retention.ml.training.metrics import compute_metrics

# Without IDs, different customers may have identical rows. Reject near copies
# of train only, allowing occasional matches such as 3 of 1056 validation rows.
_MIN_VAL_ROW_OVERLAP_RATIO = 0.9


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_dataframe(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    positive_label: Any,
    drop_cols: list[str],
    *,
    allowed_labels: Sequence[Any] | None = None,
) -> None:
    """
    Validate DataFrame schema and data quality.

    Raises:
        ValueError: If critical validation fails.
    """
    groups = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "drop_cols": drop_cols,
        "target_col": [target_col],
    }
    memberships: dict[str, list[str]] = {}
    for group, columns in groups.items():
        for column in set(columns):
            memberships.setdefault(column, []).append(group)
    overlaps = [
        f"{column!r}: {', '.join(names)}"
        for column, names in sorted(memberships.items())
        if len(names) > 1
    ]
    if overlaps:
        raise ValueError(f"Overlapping column groups: {'; '.join(overlaps)}")

    # Check required columns exist
    required_cols = set(numeric_cols + categorical_cols + [target_col])
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Warn about drop_cols that don't exist
    missing_drop = set(drop_cols) - set(df.columns)
    if missing_drop:
        warnings.warn(f"drop_cols contains non-existent columns: {missing_drop}")

    # Check numeric columns have valid types and values
    for col in numeric_cols:
        # Check for non-numeric values
        non_null_vals = df[col].dropna()
        converted = pd.to_numeric(non_null_vals, errors="coerce")
        invalid_mask = converted.isna() & non_null_vals.notna()

        if invalid_mask.any():
            invalid_samples = non_null_vals[invalid_mask].head(5).tolist()
            raise ValueError(
                f"Numeric column '{col}' contains non-numeric values. "
                f"Examples: {invalid_samples}"
            )

        # Warn about NaN counts
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            warnings.warn(f"Numeric column '{col}' has {nan_count} NaN values")

    # Check target column
    if df[target_col].isna().any():
        raise ValueError(f"Target column '{target_col}' contains NaN values")

    target_values = df[target_col].unique()
    if allowed_labels is not None:
        unknown = (
            df.loc[~df[target_col].isin(allowed_labels), target_col].unique().tolist()
        )
        if unknown:
            raise ValueError(f"Unknown target labels in '{target_col}': {unknown}")
    elif len(target_values) > 2:
        raise ValueError(
            f"Target column '{target_col}' must have at most 2 unique values, "
            f"got {len(target_values)}: {target_values}"
        )

    if allowed_labels is None and positive_label not in target_values:
        raise ValueError(
            f"positive_label '{positive_label}' not found in target column. "
            f"Values: {target_values}"
        )

    # Warn about unused columns
    used_cols = set(numeric_cols + categorical_cols + [target_col] + drop_cols)
    unused_cols = set(df.columns) - used_cols
    if unused_cols:
        warnings.warn(f"Columns not in any config group: {unused_cols}")


def prepare_features_and_target(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    positive_label: str,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features (X) and target (y) from DataFrame.

    Returns:
        X: Feature DataFrame with columns in order: numeric_cols + categorical_cols
        y: Binary target Series (1 = positive_label, 0 = other)
    """
    # Encode target
    y = (df[target_col] == positive_label).astype(int)

    # Select and order features
    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()

    return X, y


def _validate_split_separation(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Reject shared customers, or near-copied validation data without customer IDs."""
    id_columns = ("customerID", "customer_id")
    train_id_columns = [col for col in id_columns if col in train_df.columns]
    val_id_columns = [col for col in id_columns if col in val_df.columns]
    if train_id_columns and val_id_columns:
        train_ids = {
            str(value).strip()
            for col in train_id_columns
            for value in train_df[col].dropna()
        }
        val_ids = {
            str(value).strip()
            for col in val_id_columns
            for value in val_df[col].dropna()
        }
        train_ids.discard("")
        val_ids.discard("")
        if shared_ids := train_ids & val_ids:
            raise ValueError(
                f"Training and validation data share {len(shared_ids)} customer ID(s); "
                "splits must be disjoint (customerID/customer_id)."
            )
        if train_ids and val_ids:
            return

    # Align columns and dtypes, ignore row indices and any remaining ID columns.
    rows = pd.concat(
        [
            frame.drop(columns=list(id_columns), errors="ignore")
            for frame in (train_df, val_df)
        ],
        ignore_index=True,
        sort=True,
    )
    hashes = pd.util.hash_pandas_object(rows, index=False)
    matching_rows = int(
        hashes.iloc[len(train_df) :].isin(hashes.iloc[: len(train_df)]).sum()
    )
    val_rows = len(val_df)
    if val_rows and matching_rows / val_rows >= _MIN_VAL_ROW_OVERLAP_RATIO:
        raise ValueError(
            "Training and validation data contain identical rows "
            f"for {matching_rows}/{val_rows} validation rows "
            f"({matching_rows / val_rows:.1%}); "
            "customer IDs are unavailable in one or both splits."
        )


def train_model(
    config_path: Path,
    model_type: str,
    class_weight: bool,
    seed: int,
    train_path: Path,
    val_path: Path,
    out_dir: Path,
) -> dict:
    """
    Train a churn prediction model.

    Args:
        config_path: Path to YAML config file.
        model_type: Model type ("logistic" or "catboost").
        class_weight: Whether to use balanced class weights.
        seed: Random seed.
        train_path: Path to training CSV.
        val_path: Path to validation CSV.
        out_dir: Output directory for bundles.

    Returns:
        Dictionary with run info (bundle_dir, run_id, metrics).

    Raises:
        ValueError: If paths resolve to the same file or the splits overlap.
    """
    if train_path.resolve() == val_path.resolve():
        raise ValueError(
            "train_path and val_path resolve to the same file; "
            "use separate training and validation files."
        )

    # Load config
    cfg = load_config(config_path)

    # Override seed if provided
    if seed is not None:
        cfg["seed"] = seed
    else:
        seed = cfg.get("seed", 42)

    # Extract config sections
    features_cfg = cfg["features"]
    numeric_cols = features_cfg["numeric_cols"]
    categorical_cols = features_cfg["categorical_cols"]
    drop_cols = features_cfg.get("drop_cols", [])
    target_col = features_cfg["target_col"]
    positive_label = features_cfg["positive_label"]

    mlflow_cfg = cfg.get("mlflow", {})
    experiment_name = mlflow_cfg.get("experiment_name", "churn-retention")

    # Build training config for model
    model_cfg = {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
    }

    if model_type == "logistic":
        model_cfg["logistic"] = cfg["models"]["logistic"]
    elif model_type == "catboost":
        model_cfg["catboost"] = cfg["models"]["catboost"]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load and validate data
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    validate_dataframe(
        train_df, numeric_cols, categorical_cols, target_col, positive_label, drop_cols
    )
    X_train, y_train = prepare_features_and_target(
        train_df, numeric_cols, categorical_cols, target_col, positive_label, drop_cols
    )

    print(f"Loading validation data from {val_path}...")
    val_df = pd.read_csv(val_path)
    _validate_split_separation(train_df, val_df)
    validate_dataframe(
        val_df,
        numeric_cols,
        categorical_cols,
        target_col,
        positive_label,
        drop_cols,
        allowed_labels=train_df[target_col].unique().tolist(),
    )
    X_val, y_val = prepare_features_and_target(
        val_df, numeric_cols, categorical_cols, target_col, positive_label, drop_cols
    )

    # Train model
    print(f"Training {model_type} model (class_weight={class_weight}, seed={seed})...")
    import time

    start_time = time.time()

    if model_type == "logistic":
        pipeline = train_logistic(X_train, y_train, model_cfg, seed, class_weight)
    elif model_type == "catboost":
        pipeline = train_catboost(X_train, y_train, model_cfg, seed, class_weight)

    duration_seconds = time.time() - start_time
    print(f"Training completed in {duration_seconds:.2f}s")

    # Compute validation metrics
    print("Computing validation metrics...")
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    # prepare_features_and_target encodes CSV labels with astype(int).
    y_val_values = cast(np.ndarray, y_val.values)
    metrics = compute_metrics(y_val_values, y_val_prob)

    print("Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Create model version and bundle directory
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    model_version = f"{model_type}-{timestamp}-seed{seed}"
    bundle_dir = out_dir / model_version

    print(f"Saving bundle to {bundle_dir}...")

    # Create feature schema
    feature_schema = create_feature_schema(X_train, numeric_cols, categorical_cols)

    # Create metadata
    git_info = get_git_info()
    library_versions = get_library_versions()

    created_at = datetime.now(UTC).isoformat()

    metadata = {
        "model_version": model_version,
        "model_type": model_type,
        "class_weight": class_weight,
        "seed": seed,
        "config": cfg,
        "label": {
            "target_col": target_col,
            "positive_label": positive_label,
        },
        "created_at": created_at,
        "training": {
            "duration_seconds": duration_seconds,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "created_at": created_at,
        },
        "metric_threshold": 0.5,
        "metric_threshold_note": "tạm, chưa chốt với nhóm",
        "git": git_info,
        "libraries": library_versions,
        "metrics": {f"val_{k}": v for k, v in metrics.items()},
    }

    # Save bundle
    save_bundle(pipeline, bundle_dir, feature_schema, metadata)

    # Log to MLflow
    print("Logging to MLflow...")

    # Flatten params for MLflow
    params = {
        "model_type": model_type,
        "seed": seed,
        "class_weight": class_weight,
        "metric_threshold": 0.5,
    }

    # Add model-specific params
    if model_type == "logistic":
        params.update(flatten_params({"logistic": cfg["models"]["logistic"]}))
    elif model_type == "catboost":
        params.update(flatten_params({"catboost": cfg["models"]["catboost"]}))

    tags = {"model_version": model_version}

    # Save effective config (after CLI overrides) as YAML for MLflow artifact
    effective_config_path = bundle_dir / "effective_config.yaml"
    with open(effective_config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    run_id = log_training_run(
        experiment_name=experiment_name,
        params=params,
        metrics=metrics,
        bundle_dir=bundle_dir,
        config_path=config_path,
        tags=tags,
        effective_config_path=effective_config_path,
    )

    print(f"MLflow run_id: {run_id}")
    print(f"Bundle saved: {bundle_dir}")

    return {
        "bundle_dir": bundle_dir,
        "run_id": run_id,
        "model_version": model_version,
        "metrics": metrics,
    }
