import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def log_training_run(run_info: dict, output_dir: str | Path) -> Path:
    """
    Log a training run to a JSON file.

    Args:
        run_info: Dictionary containing:
            - model_name: str
            - config: dict (copy of training config)
            - metrics: dict
            - dataset_id: str
            - split_id: str
            - evaluation_split: str (e.g., "validation", "test")
            - duration_seconds: float
        output_dir: Directory to write the log file.

    Returns:
        Path to the created JSON file.

    Raises:
        FileExistsError: If a file with the generated run_id already exists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique run_id
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    # Build complete log entry
    log_entry = {
        "run_id": run_id,
        "created_at": created_at,
        **run_info,
    }

    # Write to file
    output_path = output_dir / f"{run_id}.json"
    if output_path.exists():
        raise FileExistsError(f"Run log file already exists: {output_path}")

    with open(output_path, "w") as f:
        json.dump(log_entry, f, indent=2)

    return output_path
