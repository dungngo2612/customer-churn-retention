import warnings
from datetime import datetime, timezone
from pathlib import Path

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


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_dataframe(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    positive_label: str,
    drop_cols: list[str],
) -> None:
    """
    Validate DataFrame schema and data quality.

    Raises:
        ValueError: If critical validation fails.
    """
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
    if len(target_values) != 2:
        raise ValueError(
            f"Target column '{target_col}' must have exactly 2 unique values, "
            f"got {len(target_values)}: {target_values}"
        )

    if positive_label not in target_values:
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
    negative_label = [v for v in df[target_col].unique() if v != positive_label][0]
    y = (df[target_col] == positive_label).astype(int)

    # Select and order features
    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()

    return X, y


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
    """
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
    validate_dataframe(
        val_df, numeric_cols, categorical_cols, target_col, positive_label, drop_cols
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
    metrics = compute_metrics(y_val.values, y_val_prob)

    print("Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Create model version and bundle directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    model_version = f"{model_type}-{timestamp}-seed{seed}"
    bundle_dir = out_dir / model_version

    print(f"Saving bundle to {bundle_dir}...")

    # Create feature schema
    feature_schema = create_feature_schema(X_train, numeric_cols, categorical_cols)

    # Create metadata
    git_info = get_git_info()
    library_versions = get_library_versions()

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
        "training": {
            "duration_seconds": duration_seconds,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
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
    }

    # Add model-specific params
    if model_type == "logistic":
        params.update(flatten_params({"logistic": cfg["models"]["logistic"]}))
    elif model_type == "catboost":
        params.update(flatten_params({"catboost": cfg["models"]["catboost"]}))

    tags = {"model_version": model_version}

    run_id = log_training_run(
        experiment_name=experiment_name,
        params=params,
        metrics=metrics,
        bundle_dir=bundle_dir,
        config_path=config_path,
        tags=tags,
    )

    print(f"MLflow run_id: {run_id}")
    print(f"Bundle saved: {bundle_dir}")

    return {
        "bundle_dir": bundle_dir,
        "run_id": run_id,
        "model_version": model_version,
        "metrics": metrics,
    }
