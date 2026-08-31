"""Generate comparison reports from completed real experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.config import DEFAULT_RESULTS_ROOT
from experiments.reporting import create_comparison_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments", required=True, nargs="+", type=Path
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = create_comparison_report(args.experiments, args.results_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Comparison report created: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
