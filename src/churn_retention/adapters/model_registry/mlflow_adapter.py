import os
from pathlib import Path
from typing import Any

import mlflow


def get_tracking_uri() -> str:
    """Get MLflow tracking URI from environment or use default."""
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def log_training_run(
    experiment_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    bundle_dir: Path,
    config_path: Path,
    tags: dict[str, str] | None = None,
    effective_config_path: Path | None = None,
) -> str:
    """
    Log a training run to MLflow.

    Args:
        experiment_name: Name of MLflow experiment.
        params: Flattened parameters to log.
        metrics: Metrics to log (will be prefixed with val_).
        bundle_dir: Path to model bundle directory to log as artifact.
        config_path: Path to config YAML to log as artifact.
        tags: Optional tags for the run.
        effective_config_path: Path to effective config YAML (after CLI overrides).

    Returns:
        MLflow run ID.
    """
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # Log params
        mlflow.log_params(params)

        # Log metrics with val_ prefix
        val_metrics = {f"val_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(val_metrics)

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log config as artifact
        mlflow.log_artifact(str(config_path), artifact_path="config")

        # Log effective config as artifact
        if effective_config_path is not None:
            mlflow.log_artifact(str(effective_config_path), artifact_path="effective_config")

        # Log bundle directory
        mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")

        return run.info.run_id


def flatten_params(params: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """
    Flatten nested dict for MLflow params.

    Example:
        {"models": {"logistic": {"C": 1.0}}} -> {"models.logistic.C": 1.0}
    """
    items: list[tuple[str, Any]] = []
    for k, v in params.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_params(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)
