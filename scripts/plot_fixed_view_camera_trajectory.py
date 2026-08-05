#!/usr/bin/env python3
"""Render GT and predicted 6DoF position arcs from two fixed viewpoints."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SPAWN_LOCATION_CM = np.asarray((-200.0, 0.0, 30.0), dtype=np.float64)
EVOGS_TO_GSV = np.asarray(
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    dtype=np.float64,
)
GT_BLUE = "#0877e8"
PREDICTED_YELLOW = "#f5c400"
GT_TITLE = "Ground Truth DoF"
COMBINED_TITLE = "Ground Truth DoF vs Predicted DoF"


@dataclass(frozen=True)
class FixedView:
    name: str
    elevation_degrees: float
    azimuth_degrees: float


DEFAULT_VIEWS = (
    FixedView("camera_a", 24.0, -58.0),
    FixedView("camera_b", 18.0, 122.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-trace", type=Path, required=True)
    parser.add_argument("--predicted-trace", type=Path, required=True)
    parser.add_argument(
        "--gt-ply-root", type=Path, required=True,
        help="Directory containing seven-digit DanceNet3D GT PLY files",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-frame-offset", type=int, default=1)
    parser.add_argument(
        "--point-cloud-points", type=int, default=0,
        help="Maximum GT points to draw; 0 keeps the complete dancer",
    )
    parser.add_argument("--particle-count", type=int, default=240)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--camera-a-elevation", type=float, default=24.0)
    parser.add_argument("--camera-a-azimuth", type=float, default=-58.0)
    parser.add_argument("--camera-b-elevation", type=float, default=18.0)
    parser.add_argument("--camera-b-azimuth", type=float, default=122.0)
    parser.add_argument(
        "--start-mode",
        choices=("beginning", "tracked", "tracked-gaze"),
        default="tracked-gaze",
        help="Trim untracked spawn rows from the GT trace",
    )
    parser.add_argument(
        "--start-offset-seconds", type=float, default=0.0,
        help="Additional seconds to discard after the selected tracked start",
    )
    return parser.parse_args()


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "LocationX", "LocationY", "LocationZ", "RotationRoll",
            "RotationPitch", "RotationYaw", "Frame", "Timestamp",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Trace is missing columns: {sorted(missing)}")
        has_gaze = "GazeConfidence" in set(reader.fieldnames or ())
        for source_row, raw in enumerate(reader, start=2):
            rows.append(
                {
                    "source_row": source_row,
                    "timestamp_s": float(raw["Timestamp"]),
                    "gsv_frame": int(raw["Frame"]),
                    "location_cm": np.asarray(
                        [float(raw[key]) for key in ("LocationX", "LocationY", "LocationZ")],
                        dtype=np.float64,
                    ),
                    "rotation_rpy": np.asarray(
                        [float(raw[key]) for key in ("RotationRoll", "RotationPitch", "RotationYaw")],
                        dtype=np.float64,
                    ),
                    "gaze_confidence": float(raw["GazeConfidence"]) if has_gaze else None,
                }
            )
    if not rows:
        raise ValueError(f"Trace contains no rows: {path}")
    if any(
        current["timestamp_s"] <= previous["timestamp_s"]
        for previous, current in zip(rows, rows[1:])
    ):
        raise ValueError(f"Trace timestamps must be strictly increasing: {path}")
    return rows


def is_tracked_pose(row: dict[str, Any]) -> bool:
    translation = np.linalg.norm(row["location_cm"] - SPAWN_LOCATION_CM)
    rotation = np.linalg.norm(row["rotation_rpy"])
    return bool(translation > 1.0 or rotation > 1.0)


def trim_trace(rows: list[dict[str, Any]], start_mode: str) -> list[dict[str, Any]]:
    if start_mode == "beginning":
        return rows
    for index, row in enumerate(rows):
        gaze_ok = start_mode == "tracked" or (
            row["gaze_confidence"] is not None
            and float(row["gaze_confidence"]) > 0.0
        )
        if is_tracked_pose(row) and gaze_ok:
            return rows[index:]
    raise ValueError(f"No row satisfies start mode {start_mode!r}")


def interpolate_positions(rows: list[dict[str, Any]], times: np.ndarray) -> np.ndarray:
    source_times = np.asarray([row["timestamp_s"] for row in rows], dtype=np.float64)
    if times[0] < source_times[0] - 1e-8 or times[-1] > source_times[-1] + 1e-8:
        raise ValueError("Interpolation times fall outside the trace interval")
    source_xyz = np.stack([row["location_cm"] for row in rows])
    return np.column_stack(
        [np.interp(times, source_times, source_xyz[:, axis]) for axis in range(3)]
    )


def resample_interval(
    rows: list[dict[str, Any]], count: int,
    start_time: float | None = None, end_time: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("particle-count must be at least 2")
    start = float(rows[0]["timestamp_s"] if start_time is None else start_time)
    end = float(rows[-1]["timestamp_s"] if end_time is None else end_time)
    if end <= start:
        raise ValueError(f"Invalid trajectory interval: {start} to {end}")
    times = np.linspace(start, end, count)
    return times, interpolate_positions(rows, times)


def representative_gt_row(
    rows: list[dict[str, Any]], start_time: float, end_time: float
) -> dict[str, Any]:
    midpoint = (start_time + end_time) / 2.0
    return min(rows, key=lambda row: abs(float(row["timestamp_s"]) - midpoint))


def load_gt_point_cloud(
    path: Path, max_points: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData

    if max_points < 0:
        raise ValueError("point-cloud-points must be non-negative")
    vertices = PlyData.read(str(path))["vertex"].data
    names = set(vertices.dtype.names or ())
    points_m = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float64, copy=False
    )
    finite = np.isfinite(points_m).all(axis=1)
    points_m = points_m[finite]
    if {"red", "green", "blue"} <= names:
        colors = np.column_stack(
            (vertices["red"], vertices["green"], vertices["blue"])
        ).astype(np.float64)[finite] / 255.0
    elif {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        colors = 0.5 + 0.28209479177387814 * np.column_stack(
            (vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"])
        ).astype(np.float64)[finite]
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = np.full((len(points_m), 3), 0.58, dtype=np.float64)
    if max_points > 0 and len(points_m) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points_m), max_points, replace=False)
        points_m = points_m[indices]
        colors = colors[indices]
    points_cm = (points_m @ EVOGS_TO_GSV.T) * 100.0
    return points_cm, colors


def equal_axis_limits(
    scene_points: np.ndarray,
    trajectories: list[np.ndarray],
    padding_fraction: float = 0.08,
) -> tuple[tuple[float, float], ...]:
    if scene_points.ndim != 2 or scene_points.shape[1] != 3 or len(scene_points) == 0:
        raise ValueError("Expected a non-empty N x 3 GT point cloud")
    trajectory_points = np.vstack(trajectories)
    low = np.minimum(
        np.nanpercentile(scene_points, 0.5, axis=0),
        np.nanmin(trajectory_points, axis=0),
    )
    high = np.maximum(
        np.nanpercentile(scene_points, 99.5, axis=0),
        np.nanmax(trajectory_points, axis=0),
    )
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 1.0)
    radius *= 1.0 + padding_fraction
    return tuple((float(value - radius), float(value + radius)) for value in center)


def plot_view_pair(
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    gt_xyz: np.ndarray,
    predicted_xyz: np.ndarray,
    view: FixedView,
    limits: tuple[tuple[float, float], ...],
    gt_output: Path,
    combined_output: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(10.5, 8.5), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")

    axis.scatter(
        scene_points[:, 0], scene_points[:, 1], scene_points[:, 2],
        s=0.55, c=scene_colors, alpha=0.72, linewidths=0,
        depthshade=False, rasterized=True, label="DanceNet3D GT",
        zorder=1,
    )

    axis.plot(
        gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
        color=GT_BLUE, linewidth=2.4, alpha=0.90, zorder=7,
    )
    axis.scatter(
        gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
        s=22, c=GT_BLUE, edgecolors="white", linewidths=0.28,
        alpha=0.96, depthshade=False, label="GT DoF", zorder=8,
    )
    xlim, ylim, zlim = limits
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_zlim(*zlim)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=view.elevation_degrees, azim=view.azimuth_degrees)
    axis.set_xlabel("X (cm)", labelpad=9)
    axis.set_ylabel("Y (cm)", labelpad=9)
    axis.set_zlabel("Z / height (cm)", labelpad=9)
    axis.grid(True, alpha=0.28)
    axis.legend(loc="upper left", framealpha=0.95)
    axis.set_title(GT_TITLE, pad=17)
    gt_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(gt_output, dpi=dpi, bbox_inches="tight")

    axis.plot(
        predicted_xyz[:, 0], predicted_xyz[:, 1], predicted_xyz[:, 2],
        color=PREDICTED_YELLOW, linewidth=2.2, alpha=0.92, zorder=9,
    )
    axis.scatter(
        predicted_xyz[:, 0], predicted_xyz[:, 1], predicted_xyz[:, 2],
        s=20, c=PREDICTED_YELLOW, edgecolors="#6b5700", linewidths=0.25,
        alpha=0.94, depthshade=False, label="Predicted DoF", zorder=10,
    )
    axis.set_title(COMBINED_TITLE, pad=17)
    axis.legend(loc="upper left", framealpha=0.95)
    figure.savefig(combined_output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def position_error_summary(gt_xyz: np.ndarray, predicted_xyz: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(predicted_xyz - gt_xyz, axis=1)
    return {
        "mean_cm": float(np.mean(errors)),
        "rmse_cm": float(np.sqrt(np.mean(errors * errors))),
        "p95_cm": float(np.percentile(errors, 95.0)),
        "max_cm": float(np.max(errors)),
    }


def main() -> int:
    args = parse_args()
    for path in (args.gt_trace, args.predicted_trace):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required existing trace is missing (this script never regenerates it): {path}"
            )
    if not args.gt_ply_root.is_dir():
        raise FileNotFoundError(
            f"DanceNet3D GT PLY directory is missing: {args.gt_ply_root}"
        )

    gt_rows = trim_trace(load_trace(args.gt_trace), args.start_mode)
    predicted_rows = load_trace(args.predicted_trace)
    if args.start_offset_seconds < 0.0:
        raise ValueError("start-offset-seconds must be non-negative")
    tracked_start = float(gt_rows[0]["timestamp_s"])
    visualization_start = tracked_start + args.start_offset_seconds
    visualization_end = float(gt_rows[-1]["timestamp_s"])
    if visualization_start >= visualization_end:
        raise ValueError(
            "start-offset-seconds removes the complete GT trace interval"
        )
    overlap_start = max(visualization_start, predicted_rows[0]["timestamp_s"])
    overlap_end = min(gt_rows[-1]["timestamp_s"], predicted_rows[-1]["timestamp_s"])
    if overlap_end <= overlap_start:
        raise ValueError("GT and predicted traces have no overlapping timestamps")

    gt_times, gt_full_xyz = resample_interval(
        gt_rows, args.particle_count, visualization_start, visualization_end
    )
    comparison_times, gt_aligned_xyz = resample_interval(
        gt_rows, args.particle_count, overlap_start, overlap_end
    )
    predicted_xyz = interpolate_positions(predicted_rows, comparison_times)
    snapshot_row = representative_gt_row(
        gt_rows, visualization_start, visualization_end
    )
    snapshot_asset_id = int(snapshot_row["gsv_frame"]) + args.asset_frame_offset
    snapshot_path = args.gt_ply_root / f"{snapshot_asset_id:07d}.ply"
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Missing DanceNet3D GT point cloud: {snapshot_path}")
    scene_points, scene_colors = load_gt_point_cloud(
        snapshot_path, args.point_cloud_points, snapshot_asset_id
    )
    limits = equal_axis_limits(scene_points, [gt_full_xyz, predicted_xyz])
    views = (
        FixedView("camera_a", args.camera_a_elevation, args.camera_a_azimuth),
        FixedView("camera_b", args.camera_b_elevation, args.camera_b_azimuth),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for view in views:
        gt_output = args.output_dir / f"gt_only_{view.name}.png"
        combined_output = args.output_dir / f"gt_predicted_{view.name}.png"
        plot_view_pair(
            scene_points, scene_colors, gt_full_xyz, predicted_xyz, view,
            limits, gt_output, combined_output, args.dpi,
        )
        outputs[f"gt_only_{view.name}"] = str(gt_output.resolve())
        outputs[f"gt_predicted_{view.name}"] = str(combined_output.resolve())

    metadata = {
        "schema_version": 2,
        "status": "PASS",
        "gt_trace": str(args.gt_trace.resolve()),
        "predicted_trace": str(args.predicted_trace.resolve()),
        "prediction_reused": True,
        "prediction_was_generated": False,
        "coordinate_system": "trace LocationX/LocationY/LocationZ in GSV/Unreal centimetres",
        "dancenet_gt_point_cloud": str(snapshot_path.resolve()),
        "dancenet_gt_asset_id": snapshot_asset_id,
        "point_cloud_points_rendered": int(len(scene_points)),
        "tracked_start_s": tracked_start,
        "start_offset_seconds": args.start_offset_seconds,
        "gt_full_time_range_s": [float(gt_times[0]), float(gt_times[-1])],
        "comparison_time_range_s": [float(comparison_times[0]), float(comparison_times[-1])],
        "particles_per_trajectory": args.particle_count,
        "position_error": position_error_summary(gt_aligned_xyz, predicted_xyz),
        "views": [
            {"name": view.name, "elevation_degrees": view.elevation_degrees, "azimuth_degrees": view.azimuth_degrees}
            for view in views
        ],
        "outputs": outputs,
    }
    metadata_path = args.output_dir / "trajectory_views.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
