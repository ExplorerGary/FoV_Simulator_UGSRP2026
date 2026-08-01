#!/usr/bin/env python3
"""Write a compact, auditable summary of one 500 ms LR experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-ms", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    horizon = summary["horizons"][f"{args.horizon_ms}ms"]
    model = horizon["visibility"]
    persistence = horizon["persistence"]
    model_classification = model["classification"]
    persistence_classification = persistence["classification"]
    report = {
        "status": summary["status"],
        "mechanism": summary["mechanism"],
        "training_target_mode": summary["config"]["training_target_mode"],
        "history_ms": summary["config"]["history_ms"],
        "requested_horizon_ms": args.horizon_ms,
        "mean_actual_horizon_ms": horizon["mean_actual_horizon_ms"],
        "training_traces": summary["split"]["training_traces"],
        "test_traces": summary["split"]["test_traces"],
        "decision_threshold": horizon["threshold_calibration"][
            "selected_decision_threshold"
        ],
        "safe_threshold_calibration": horizon.get(
            "safe_threshold_calibration"
        ),
        "model": {
            "mse_against_fraction_target": model["mse"],
            **model_classification,
        },
        "persistence": {
            "mse_against_fraction_target": persistence["mse"],
            **persistence_classification,
        },
        "tradeoff_vs_persistence": {
            "missed_cells_per_frame_delta": (
                model_classification["missed_visible_cells_per_frame"]
                - persistence_classification["missed_visible_cells_per_frame"]
            ),
            "extra_cells_per_frame_delta": (
                model_classification["extra_cells_per_frame"]
                - persistence_classification["extra_cells_per_frame"]
            ),
            "f2_delta": (
                model_classification["f2"] - persistence_classification["f2"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
