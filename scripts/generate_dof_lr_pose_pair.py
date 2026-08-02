#!/usr/bin/env python3
"""Apply the trained 6DoF OLS model and emit aligned predicted/actual traces."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

import numpy as np

from fovsim.dof_prediction import (
    CIRCULAR_NAMES,
    _circular_pose,
    _decode_pose,
    _metrics,
)
from fovsim.prediction import _load_sampled_trace
from fovsim.trace import TraceRow, load_trace


FIELDS = (
    "FileName", "LocationX", "LocationY", "LocationZ",
    "RotationRoll", "RotationPitch", "RotationYaw", "Frame", "Timestamp",
    "GazeHitX", "GazeHitY", "GazeHitZ", "GazeConfidence",
)


def nearest_rows(rows: list[TraceRow], times: np.ndarray) -> list[TraceRow]:
    timestamps = [row.timestamp_s for row in rows]
    selected = []
    for timestamp in times:
        right = bisect.bisect_left(timestamps, float(timestamp))
        if right <= 0:
            selected.append(rows[0])
        elif right >= len(rows):
            selected.append(rows[-1])
        else:
            before, after = rows[right - 1], rows[right]
            selected.append(
                before
                if timestamp - before.timestamp_s <= after.timestamp_s - timestamp
                else after
            )
    return selected


def predict(model_path: Path, histories: np.ndarray) -> np.ndarray:
    encoded = np.empty((len(histories), len(CIRCULAR_NAMES)), dtype=np.float64)
    with np.load(model_path, allow_pickle=False) as model:
        for index, name in enumerate(CIRCULAR_NAMES):
            coefficient = np.asarray(model[f"{name}_coefficient"], dtype=np.float64)
            intercept = float(np.asarray(model[f"{name}_intercept"]).reshape(-1)[0])
            if coefficient.shape != (histories.shape[1],):
                raise ValueError(
                    f"Model/history mismatch for {name}: {coefficient.shape} vs "
                    f"{histories.shape[1]} samples"
                )
            encoded[:, index] = histories[:, :, index] @ coefficient + intercept
    for sine, cosine in ((3, 4), (5, 6), (7, 8)):
        norm = np.hypot(encoded[:, sine], encoded[:, cosine])
        encoded[:, sine] /= np.maximum(norm, 1e-12)
        encoded[:, cosine] /= np.maximum(norm, 1e-12)
    return encoded


def trace_row(source: TraceRow, timestamp: float, pose: np.ndarray) -> dict[str, object]:
    gaze = source.gaze_direction or (0.0, 0.0, 0.0)
    return {
        "FileName": source.file_name,
        "LocationX": pose[0], "LocationY": pose[1], "LocationZ": pose[2],
        "RotationRoll": pose[3], "RotationPitch": pose[4], "RotationYaw": pose[5],
        "Frame": source.gsv_frame, "Timestamp": timestamp,
        "GazeHitX": gaze[0], "GazeHitY": gaze[1], "GazeHitZ": gaze[2],
        "GazeConfidence": source.gaze_confidence or 0.0,
    }


def write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predicted-trace", type=Path, required=True)
    parser.add_argument("--actual-trace", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sequence", default="BiancaGolden_CircleTurns")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--history-ms", type=int, default=500)
    parser.add_argument("--horizon-ms", type=int, default=100)
    args = parser.parse_args()

    history_steps = int(round(args.history_ms * args.fps / 1000.0))
    horizon_steps = int(round(args.horizon_ms * args.fps / 1000.0))
    sampled = _load_sampled_trace(args.trace, args.sequence, args.fps)
    if sampled is None:
        raise ValueError(f"Trace does not contain {args.sequence}: {args.trace}")
    encoded = _circular_pose(sampled.dof)
    current_indices = np.arange(history_steps, len(encoded) - horizon_steps)
    if len(current_indices) < 2:
        raise ValueError("Trace is too short for requested history and horizon")
    target_indices = current_indices + horizon_steps
    histories = np.stack(
        [encoded[index - history_steps : index + 1] for index in current_indices]
    )
    target_encoded = encoded[target_indices]
    target_pose = sampled.dof[target_indices]
    predicted_encoded = predict(args.model, histories)
    predicted_pose = _decode_pose(predicted_encoded)
    target_times = sampled.times_s[target_indices]

    raw_rows, _ = load_trace(args.trace)
    sequence_rows = [row for row in raw_rows if row.file_name == args.sequence]
    sources = nearest_rows(sequence_rows, target_times)
    actual_rows = [
        trace_row(source, float(timestamp), pose)
        for source, timestamp, pose in zip(sources, target_times, target_pose)
    ]
    predicted_rows = [
        trace_row(source, float(timestamp), pose)
        for source, timestamp, pose in zip(sources, target_times, predicted_pose)
    ]
    write_trace(args.actual_trace, actual_rows)
    write_trace(args.predicted_trace, predicted_rows)
    metrics = _metrics(predicted_pose, target_pose, predicted_encoded, target_encoded)
    output = {
        "status": "PASS",
        "mechanism": "trained independent-coordinate OLS: 500ms 6DoF to 100ms 6DoF",
        "trace": args.trace.stem,
        "history_ms": args.history_ms,
        "history_samples_including_current": history_steps + 1,
        "horizon_ms": args.horizon_ms,
        "horizon_steps": horizon_steps,
        "frames": len(actual_rows),
        "metrics": metrics,
        "outputs": {
            "predicted_trace": str(args.predicted_trace.resolve()),
            "actual_trace": str(args.actual_trace.resolve()),
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
