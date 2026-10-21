import json
from pathlib import Path

import pandas as pd


def compare_runs(run_paths: list[str | Path]) -> pd.DataFrame:
    """
    Compare multiple training runs by loading their log files.

    Args:
        run_paths: List of paths to run log JSON files.

    Returns:
        DataFrame with run information and metrics, sorted by average_precision
        descending.
    """
    runs = []
    for path in run_paths:
        with open(path) as f:
            run_data = json.load(f)
            runs.append(run_data)

    df = pd.json_normalize(runs)

    # Sort by average_precision descending
    if "metrics.average_precision" in df.columns:
        df = df.sort_values("metrics.average_precision", ascending=False)

    return df
