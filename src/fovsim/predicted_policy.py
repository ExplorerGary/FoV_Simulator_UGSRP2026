"""Convert a saved DoF-only linear predictor into QoE cell decisions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .io import write_csv_atomic, write_json_atomic
from .prediction import (
    StandardizedRidge,
    VisibilityFrames,
    _examples,
    _load_sampled_trace,
    load_visibility,
)


@dataclass(frozen=True, slots=True)
class FrameProvenance:
    output_frame: int
    output_time_s: float
    trace_source_row: int
    trace_timestamp_s: float
    gsv_frame: int


def _load_provenance(path: Path) -> list[FrameProvenance]:
    required = {
        "output_frame", "output_time_s", "trace_source_row",
        "trace_timestamp_s", "gsv_frame",
    }
    frames: dict[int, FrameProvenance] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Visibility CSV is missing columns: {sorted(missing)}")
        for line, raw in enumerate(reader, start=2):
            try:
                item = FrameProvenance(
                    output_frame=int(raw["output_frame"]),
                    output_time_s=float(raw["output_time_s"]),
                    trace_source_row=int(raw["trace_source_row"]),
                    trace_timestamp_s=float(raw["trace_timestamp_s"]),
                    gsv_frame=int(raw["gsv_frame"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid provenance at {path}:{line}") from exc
            previous = frames.setdefault(item.output_frame, item)
            if previous != item:
                raise ValueError(f"Inconsistent frame metadata at {path}:{line}")
    ordered = [frames[key] for key in sorted(frames)]
    if not ordered:
        raise ValueError(f"No frame provenance in {path}")
    return ordered


def _parse_cell_id(cell_id: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(value) for value in cell_id.split(":"))
    except ValueError as exc:
        raise ValueError(f"Invalid model cell ID: {cell_id!r}") from exc
    if len(values) != 3:
        raise ValueError(f"Invalid model cell ID: {cell_id!r}")
    return values


def _asset_paths(model_root: Path, gt_root: Path, asset_id: int) -> tuple[Path, ...]:
    frame = f"{asset_id:07d}"
    return (
        model_root / "SINGLE" / f"{frame}_aggressive_base_random" / "ply" / "point_cloud_8999.ply",
        model_root / "EVOGS_V1" / frame / "enhancement_03_enhanced" / "ply" / f"{frame}_enhancement.ply",
        gt_root / f"{frame}.ply",
    )


def generate_predicted_policy(
    *, trace_path: Path, visibility_path: Path, model_path: Path,
    output_dir: Path, sequence: str = "BiancaGolden_CircleTurns",
    fps: float = 30.0, history_ms: int = 500, horizon_ms: int = 500,
    model_root: Path | None = None, gt_root: Path | None = None,
    asset_frame_offset: int = 1,
    policy_mode: str = "linear",
    decision_threshold: float | None = None,
    guard_band_steps: int = 0,
    require_valid_gaze_history: bool = False,
) -> dict[str, object]:
    """Write QoE decisions without consulting future visibility fractions."""
    import numpy as np

    if (model_root is None) != (gt_root is None):
        raise ValueError("model_root and gt_root must be supplied together")
    if policy_mode not in {"linear", "persistence", "base_only"}:
        raise ValueError("Invalid policy_mode")
    if decision_threshold is not None and not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be within [0, 1]")
    if guard_band_steps < 0:
        raise ValueError("guard_band_steps must be non-negative")
    trace = _load_sampled_trace(trace_path, sequence, fps)
    if trace is None:
        raise ValueError(f"Trace does not contain FileName={sequence!r}")
    # load_visibility supplies timestamps for alignment. Its fraction values are
    # never used as features or decisions; the empty index makes that invariant
    # explicit in the generated Examples target matrices.
    visibility = load_visibility(visibility_path)
    provenance = _load_provenance(visibility_path)
    if len(provenance) != len(visibility.times_s):
        raise ValueError("Visibility/provenance frame count mismatch")
    with np.load(model_path, allow_pickle=False) as saved:
        model = StandardizedRidge(
            feature_mean=saved["feature_mean"], feature_scale=saved["feature_scale"],
            target_mean=saved["target_mean"], target_scale=saved["target_scale"],
            coefficients=saved["coefficients"], alpha=float(saved["alpha"][0]),
        )
        cell_ids = [str(value) for value in saved["cell_ids"]]
        saved_threshold = float(saved["decision_threshold"][0])
        target_threshold = float(saved["target_threshold"][0])
        feature_mode = (
            str(saved["feature_mode"][0])
            if "feature_mode" in saved.files
            else "raw_history"
        )
    threshold = saved_threshold if decision_threshold is None else decision_threshold
    history_steps = max(1, int(round(history_ms * fps / 1000.0)))
    cell_index = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    if policy_mode == "persistence":
        example_visibility = visibility
        example_cells = cell_index
    else:
        example_visibility = VisibilityFrames(
            output_frames=visibility.output_frames,
            times_s=visibility.times_s,
            values_by_frame=[{} for _ in visibility.values_by_frame],
        )
        example_cells = {}
    examples = _examples(
        trace, example_visibility, example_cells, fps=fps,
        history_steps=history_steps, horizon_s=horizon_ms / 1000.0,
        feature_mode=feature_mode,
        require_valid_gaze_history=require_valid_gaze_history,
    )
    if policy_mode == "linear":
        scores = np.clip(model.predict(examples.features), 0.0, 1.0)
    elif policy_mode == "persistence":
        scores = examples.current_visibility
        threshold = target_threshold
    else:
        scores = np.zeros((len(examples.features), len(cell_ids)))
        threshold = 1.0
    if scores.shape[1] != len(cell_ids):
        raise ValueError("Model output dimension does not match its cell IDs")
    selected_matrix = scores >= threshold
    if guard_band_steps:
        coordinates = [_parse_cell_id(cell_id) for cell_id in cell_ids]
        index_by_coordinate = {
            coordinate: index for index, coordinate in enumerate(coordinates)
        }
        for _ in range(guard_band_steps):
            expanded = selected_matrix.copy()
            for cell_index_value, (x, y, z) in enumerate(coordinates):
                neighbors = (
                    (x - 1, y, z), (x + 1, y, z),
                    (x, y - 1, z), (x, y + 1, z),
                    (x, y, z - 1), (x, y, z + 1),
                )
                for neighbor in neighbors:
                    neighbor_index = index_by_coordinate.get(neighbor)
                    if neighbor_index is not None:
                        expanded[:, neighbor_index] |= selected_matrix[:, cell_index_value]
            selected_matrix = expanded

    # Timestamp gaps can map more than one history to one future frame. Keep the
    # candidate closest to the requested horizon, yielding one policy per frame.
    selected: dict[int, int] = {}
    requested_s = horizon_ms / 1000.0
    for example_index, target_index in enumerate(examples.target_indices):
        target = int(target_index)
        incumbent = selected.get(target)
        error = abs(float(examples.actual_horizons_s[example_index]) - requested_s)
        if incumbent is None or error < abs(
            float(examples.actual_horizons_s[incumbent]) - requested_s
        ):
            selected[target] = example_index

    rows: list[dict[str, object]] = []
    skipped_assets = 0
    for target_index, example_index in sorted(selected.items()):
        target = provenance[target_index]
        asset_id = target.gsv_frame + asset_frame_offset
        if model_root is not None and not all(
            path.is_file() for path in _asset_paths(model_root, gt_root, asset_id)  # type: ignore[arg-type]
        ):
            skipped_assets += 1
            continue
        current = provenance[int(examples.current_indices[example_index])]
        actual_ms = float(examples.actual_horizons_s[example_index] * 1000.0)
        output_frame = len(rows) // len(cell_ids)
        for cell_index, cell_id in enumerate(cell_ids):
            x, y, z = _parse_cell_id(cell_id)
            score = float(scores[example_index, cell_index])
            enhancement_required = bool(selected_matrix[example_index, cell_index])
            rows.append({
                "output_frame": output_frame,
                "source_output_frame": target.output_frame,
                "output_time_s": target.output_time_s,
                "trace_source_row": target.trace_source_row,
                "trace_timestamp_s": target.trace_timestamp_s,
                "gsv_frame": target.gsv_frame,
                "asset_frame_id": asset_id,
                "cell_id": cell_id, "cell_x": x, "cell_y": y, "cell_z": z,
                "predicted_visibility_score": score,
                "decision_threshold": threshold,
                "enhancement_required": int(enhancement_required),
                "target_level": 3 if enhancement_required else 0,
                "prediction_current_output_frame": current.output_frame,
                "prediction_current_trace_timestamp_s": current.trace_timestamp_s,
                "requested_horizon_ms": horizon_ms,
                "actual_horizon_ms": actual_ms,
            })
    if not rows:
        raise ValueError("No policy frames remain after alignment/asset filtering")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(output_dir / "cell_decisions.csv", list(rows[0]), rows)
    frame_count = len(rows) // len(cell_ids)
    summary = {
        "status": "PASS",
        "policy_mode": policy_mode,
        "feature_mode": feature_mode,
        "input_contract": (
            (
                "6dof_and_gaze_history"
                if feature_mode in {"motion_gaze", "raw_gaze"}
                else "6dof_history_only"
            ) if policy_mode == "linear"
            else "current_visibility" if policy_mode == "persistence"
            else "none"
        ),
        "future_visibility_values_used": False,
        "trace": trace.name, "history_ms": history_ms, "horizon_ms": horizon_ms,
        "decision_threshold": threshold,
        "saved_decision_threshold": saved_threshold,
        "guard_band_steps": guard_band_steps,
        "require_valid_gaze_history": require_valid_gaze_history,
        "cell_count": len(cell_ids),
        "frame_count": frame_count, "skipped_missing_asset_frames": skipped_assets,
        "mean_selected_cells": sum(int(row["enhancement_required"]) for row in rows) / frame_count,
        "output": str((output_dir / "cell_decisions.csv").resolve()),
    }
    write_json_atomic(output_dir / "prediction_policy_summary.json", summary)
    return summary
