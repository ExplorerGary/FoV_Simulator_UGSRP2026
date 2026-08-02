#!/usr/bin/env python3
"""Convert one visibility CSV score definition into Base/E3 decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visibility", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score",
        choices=("contributing_fraction", "rasterized_fraction", "image_share"),
        required=True,
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--threshold", type=float)
    selection.add_argument("--cumulative-coverage", type=float)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--predicted-pose-visibility", action="store_true",
        help="Mark scores as geometry evaluated at a predicted future pose.",
    )
    return parser.parse_args()


def score(row: dict[str, str], definition: str) -> float:
    if definition == "contributing_fraction":
        return float(row["contributing_gaussian_fraction"])
    if definition == "rasterized_fraction":
        active = int(row["active_gaussian_count"])
        return int(row["rasterized_gaussian_count"]) / active if active else 0.0
    return float(row["image_share"])


def main() -> None:
    args = parse_args()
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be within [0, 1]")
    if (
        args.cumulative_coverage is not None
        and not 0.0 < args.cumulative_coverage <= 1.0
    ):
        raise ValueError("--cumulative-coverage must be within (0, 1]")
    with args.visibility.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    if not source_rows:
        raise ValueError("Visibility CSV is empty")
    frames: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        frames[int(row["output_frame"])].append(row)

    decisions: set[tuple[int, str]] = set()
    if args.threshold is not None:
        for frame, rows in frames.items():
            for row in rows:
                if score(row, args.score) >= args.threshold:
                    decisions.add((frame, row["cell_id"]))
    else:
        assert args.cumulative_coverage is not None
        for frame, rows in frames.items():
            ordered = sorted(rows, key=lambda row: score(row, args.score), reverse=True)
            cumulative = 0.0
            for row in ordered:
                value = score(row, args.score)
                if value <= 0.0:
                    break
                decisions.add((frame, row["cell_id"]))
                cumulative += value
                if cumulative >= args.cumulative_coverage:
                    break

    output_rows: list[dict[str, object]] = []
    selected_by_frame: dict[int, int] = defaultdict(int)
    for row in source_rows:
        frame = int(row["output_frame"])
        selected = (frame, row["cell_id"]) in decisions
        selected_by_frame[frame] += int(selected)
        output_rows.append(
            {
                **row,
                "predicted_visibility_score": score(row, args.score),
                "score_definition": args.score,
                "selection_rule": (
                    f"score >= {args.threshold}"
                    if args.threshold is not None
                    else f"descending cumulative score >= {args.cumulative_coverage}"
                ),
                "policy_threshold": args.threshold,
                "enhancement_required": int(selected),
                "target_level": 3 if selected else 0,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = args.output_dir / "cell_decisions.csv"
    with decisions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    selected_counts = list(selected_by_frame.values())
    summary = {
        "status": "PASS",
        "variant": args.variant,
        "input_contract": (
            "visibility score computed from predicted future pose"
            if args.predicted_pose_visibility
            else "precomputed per-cell visibility score"
        ),
        "future_visibility_values_used": not args.predicted_pose_visibility,
        "predicted_pose_visibility_used": args.predicted_pose_visibility,
        "oracle_score_definition_study": not args.predicted_pose_visibility,
        "score_definition": args.score,
        "selection_rule": output_rows[0]["selection_rule"],
        "mean_selected_cells": sum(selected_counts) / len(selected_counts),
        "frame_count": len(selected_counts),
        "decision_rows": len(output_rows),
        "inputs": {"visibility": str(args.visibility.resolve())},
        "outputs": {"cell_decisions": str(decisions_path.resolve())},
    }
    (args.output_dir / "prediction_policy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
