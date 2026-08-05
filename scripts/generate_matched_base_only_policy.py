#!/usr/bin/env python3
"""Derive an all-Base policy on exactly the frames of an existing policy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def generate(source: Path, output_dir: Path, variant: str) -> dict[str, object]:
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if not rows:
        raise ValueError("Source decisions CSV is empty")

    required = {"output_frame", "cell_id", "enhancement_required", "target_level"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"Source decisions are missing columns: {missing}")

    replacements = {
        "predicted_visibility_score": "0.0",
        "score_definition": "base_only",
        "selection_rule": "select no E3 cells",
        "policy_threshold": "",
        "decision_threshold": "",
        "enhancement_required": "0",
        "target_level": "0",
    }
    for name in replacements:
        if name not in fieldnames:
            fieldnames.append(name)
    for row in rows:
        row.update(replacements)

    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = output_dir / "cell_decisions.csv"
    with decisions.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    frames = {int(row["output_frame"]) for row in rows}
    summary = {
        "status": "PASS",
        "variant": variant,
        "input_contract": "all-Base decisions copied from an existing matched policy frame set",
        "future_visibility_values_used": False,
        "predicted_pose_visibility_used": False,
        "oracle_score_definition_study": False,
        "score_definition": "base_only",
        "selection_rule": "select no E3 cells",
        "mean_selected_cells": 0.0,
        "frame_count": len(frames),
        "decision_rows": len(rows),
        "inputs": {"matched_policy_decisions": str(source.resolve())},
        "outputs": {"cell_decisions": str(decisions.resolve())},
    }
    (output_dir / "prediction_policy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="matched_base_only")
    args = parser.parse_args()
    print(json.dumps(generate(args.source_decisions, args.output_dir, args.variant), indent=2))


if __name__ == "__main__":
    main()
