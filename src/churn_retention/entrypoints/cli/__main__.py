"""Experiment CLI: python -m churn_retention.entrypoints.cli tune --config ..."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from churn_retention.entrypoints.cli.batch_scoring import register_commands, run_command
from churn_retention.ml.training.tune import (
    load_tune_config,
    run_experiments,
    write_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Churn model experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    tune = commands.add_parser("tune", help="Compare models with train-only CV")
    tune.add_argument("--config", required=True, type=Path)
    tune.add_argument(
        "--smoke",
        action="store_true",
        help="Use 2 folds, first 500 train rows, <=2 trials",
    )
    register_commands(commands)
    args = parser.parse_args(argv)
    if args.command in {"score-batch", "job-status"}:
        return run_command(args)
    try:
        config = load_tune_config(args.config)
        report = run_experiments(config, smoke=args.smoke)
        destination = config.source.parent.parent / "docs" / "experiments.md"
        write_report(report, destination)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Experiment failed: {exc}\n")
    completed = sum(bool(trial.cv) for trial in report.trials)
    evaluated = sum(bool(trial.val) for trial in report.trials)
    print(
        f"CV completed: {completed}; val evaluated: {evaluated}; "
        f"stop: {report.stop_reason}; report: {destination}"
    )
    return 0 if completed and report.stop_reason == "configured_trials_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
