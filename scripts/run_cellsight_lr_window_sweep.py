#!/usr/bin/env python3
"""Evaluate exact CellSight LR30/LR90 temporal baselines locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fovsim.dof_prediction import run_cellsight_lr_window_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=Path("trace_csvs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/cellsight_lr30_lr90")
    )
    parser.add_argument("--history-samples", type=int, nargs="+", default=(30, 90))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--horizon-ms", type=int, default=100)
    args = parser.parse_args()
    result = run_cellsight_lr_window_sweep(
        trace_dir=args.trace_dir,
        output_dir=args.output_dir,
        history_samples=tuple(args.history_samples),
        fps=args.fps,
        horizon_ms=args.horizon_ms,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
