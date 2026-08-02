#!/usr/bin/env python3
"""Run the local 500 ms -> 100 ms CellSight-style 6DoF LR experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fovsim.dof_prediction import run_dof_lr_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=Path("trace_csvs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/dof_lr_500ms_to_100ms")
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--history-ms", type=int, default=500)
    parser.add_argument("--horizon-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    result = run_dof_lr_experiment(
        trace_dir=args.trace_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        history_ms=args.history_ms,
        horizon_ms=args.horizon_ms,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

