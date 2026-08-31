"""Command-line entry point for one isolated comparative experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.config import ExperimentConfig
from experiments.model_registry import MODEL_REGISTRY
from experiments.trainer import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--initial-epochs", type=int, default=30)
    parser.add_argument("--fine-tune-epochs", type=int, default=25)
    parser.add_argument("--results-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExperimentConfig(
        model_key=args.model,
        batch_size=args.batch_size,
        initial_epochs=args.initial_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
    )
    if args.results_root is not None:
        config.results_root = args.results_root
    try:
        output = run_experiment(config)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Experiment completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
