#!/usr/bin/env python3
"""Generate CellSight-style cell visibility from DanceNet3D GT PLYs.

The source of the policy signal is the ground-truth scene, never Base or E3.
For each sampled trace pose, gsplat enumerates the front-to-back
Gaussian/pixel intersections. A Gaussian contributes when at least one pixel
has compositing weight ``T * alpha >= visibility_weight_threshold``.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_standard_ply_qoe import (
    EVOGS_TO_GSV,
    camera_tensors,
    normalized_quat_to_rotmat,
)


FIELDS = (
    "schema_version",
    "output_frame",
    "source_output_frame",
    "output_time_s",
    "trace_source_row",
    "trace_timestamp_s",
    "gsv_frame",
    "asset_frame_id",
    "cell_id",
    "cell_x",
    "cell_y",
    "cell_z",
    "active_gaussian_count",
    "rasterized_gaussian_count",
    "contributing_gaussian_count",
    "contributing_gaussian_fraction",
    "raw_alpha_mass_pixels",
    "contribution_mass_pixels",
    "image_share",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument(
        "--require-model-assets-root",
        type=Path,
        help=(
            "Optional LUT output root. Frames missing Base or E3 are skipped; "
            "the files are checked for existence but never read for visibility."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--gsplat-library-path", type=Path, required=True)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--hfov", type=float, default=77.0)
    parser.add_argument("--near-cm", type=float, default=1.0)
    parser.add_argument("--cell-size-m", type=float, default=0.2)
    parser.add_argument(
        "--visibility-weight-threshold",
        type=float,
        default=1.0 / 255.0,
    )
    parser.add_argument("--asset-frame-offset", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-missing-assets",
        action="store_true",
        help="Skip trace samples whose GT PLY is absent and renumber output frames",
    )
    return parser.parse_args()


def load_trace(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "LocationX",
            "LocationY",
            "LocationZ",
            "RotationRoll",
            "RotationPitch",
            "RotationYaw",
            "Frame",
            "Timestamp",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Trace is missing columns: {sorted(missing)}")
        for source_row, raw in enumerate(reader, start=2):
            result.append(
                {
                    "source_row": source_row,
                    "timestamp_s": float(raw["Timestamp"]),
                    "gsv_frame": int(raw["Frame"]),
                    "location_cm": tuple(
                        float(raw[key])
                        for key in ("LocationX", "LocationY", "LocationZ")
                    ),
                    "rotation_rpy": tuple(
                        float(raw[key])
                        for key in (
                            "RotationRoll",
                            "RotationPitch",
                            "RotationYaw",
                        )
                    ),
                }
            )
    if not result:
        raise ValueError("Trace has no rows")
    return result


def sampled_trace_rows(
    rows: list[dict[str, Any]],
    start_time: float,
    end_time: float,
    fps: float,
) -> list[tuple[int, float, dict[str, Any]]]:
    timestamps = [float(row["timestamp_s"]) for row in rows]
    count = max(1, math.floor((end_time - start_time) * fps))
    samples: list[tuple[int, float, dict[str, Any]]] = []
    for index in range(count):
        output_time = start_time + index / fps
        right = bisect.bisect_left(timestamps, output_time)
        if right <= 0:
            row = rows[0]
        elif right >= len(rows):
            row = rows[-1]
        else:
            before, after = rows[right - 1], rows[right]
            row = (
                before
                if output_time - float(before["timestamp_s"])
                <= float(after["timestamp_s"]) - output_time
                else after
            )
        samples.append((index, output_time, row))
    return samples


def visibility_inputs(
    splats: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    source_means = splats["means"]
    means = (source_means @ EVOGS_TO_GSV.T) * 100.0
    rotations = normalized_quat_to_rotmat(splats["quats"])
    scales = torch.exp(splats["scales"])
    covars = (
        rotations
        @ torch.diag_embed(scales * scales)
        @ rotations.transpose(1, 2)
    )
    covars = (EVOGS_TO_GSV @ covars @ EVOGS_TO_GSV.T) * 10000.0
    opacities = torch.sigmoid(splats["opacities"].reshape(-1))
    count = int(means.shape[0])
    quats = torch.zeros((count, 4), dtype=torch.float32)
    quats[:, 0] = 1.0
    scales_dummy = torch.ones((count, 3), dtype=torch.float32)
    return (
        means.to(device, non_blocking=True),
        quats.to(device, non_blocking=True),
        scales_dummy.to(device, non_blocking=True),
        opacities.to(device, non_blocking=True),
        covars.to(device, non_blocking=True),
    )


def gaussian_attribution(
    meta: dict[str, torch.Tensor],
    width: int,
    height: int,
    gaussian_count: int,
    threshold: float,
    rasterize_to_indices_in_range: Any,
) -> dict[str, np.ndarray]:
    device = meta["means2d"].device
    transmittance = torch.ones(
        meta["means2d"].shape[:-2] + (height, width),
        dtype=torch.float32,
        device=device,
    )
    gaussian_ids, pixel_ids, image_ids = rasterize_to_indices_in_range(
        0,
        1 << 30,
        transmittance,
        meta["means2d"],
        meta["conics"],
        meta["opacities"],
        width,
        height,
        int(meta["tile_size"]),
        meta["isect_offsets"],
        meta["flatten_ids"],
    )
    if image_ids.numel() and torch.any(image_ids != 0):
        raise AssertionError("Visibility generator expects one camera")
    if pixel_ids.numel() and torch.any(pixel_ids[1:] < pixel_ids[:-1]):
        raise AssertionError("gsplat pixel intersections are not grouped")

    means2d = meta["means2d"][0, gaussian_ids]
    conics = meta["conics"][0, gaussian_ids]
    opacity = meta["opacities"][0, gaussian_ids]
    px = torch.stack(
        (
            pixel_ids % width,
            torch.div(pixel_ids, width, rounding_mode="floor"),
        ),
        dim=-1,
    ).to(torch.float32)
    delta = px + 0.5 - means2d
    sigma = (
        0.5
        * (
            conics[:, 0] * delta[:, 0].square()
            + conics[:, 2] * delta[:, 1].square()
        )
        + conics[:, 1] * delta[:, 0] * delta[:, 1]
    )
    alpha = torch.clamp_max(opacity * torch.exp(-sigma), 0.99)

    if alpha.numel():
        log_survival = torch.log1p(-alpha)
        exclusive_global = torch.cumsum(log_survival, 0) - log_survival
        _, counts = torch.unique_consecutive(pixel_ids, return_counts=True)
        starts = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.cumsum(counts, 0)[:-1],
            )
        )
        segment_bases = exclusive_global[starts]
        exclusive_segment = exclusive_global - torch.repeat_interleave(
            segment_bases, counts
        )
        weights = torch.exp(exclusive_segment) * alpha
    else:
        weights = alpha

    raw_mass = torch.bincount(
        gaussian_ids,
        weights=alpha,
        minlength=gaussian_count,
    )
    contribution_mass = torch.bincount(
        gaussian_ids,
        weights=weights,
        minlength=gaussian_count,
    )
    visible_pixels = torch.bincount(
        gaussian_ids[weights >= threshold],
        minlength=gaussian_count,
    )
    radii = meta["radii"][0]
    rasterized = (
        torch.any(radii > 0, dim=-1) if radii.ndim == 2 else radii > 0
    )
    return {
        "rasterized": rasterized.cpu().numpy(),
        "raw_mass": raw_mass.cpu().numpy(),
        "contribution_mass": contribution_mass.cpu().numpy(),
        "visible_pixels": visible_pixels.cpu().numpy(),
    }


def aggregate_cells(
    splats: dict[str, torch.Tensor],
    attribution: dict[str, np.ndarray],
    cell_size_m: float,
    active_threshold: float,
) -> list[dict[str, Any]]:
    means_gsv_m = (splats["means"] @ EVOGS_TO_GSV.T).numpy()
    coordinates = np.floor(means_gsv_m / cell_size_m).astype(np.int64)
    unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    cell_count = len(unique)
    opacities = torch.sigmoid(splats["opacities"].reshape(-1)).numpy()

    def counts(values: np.ndarray) -> np.ndarray:
        return np.bincount(
            inverse,
            weights=values.astype(np.int64, copy=False),
            minlength=cell_count,
        ).astype(np.int64)

    def sums(values: np.ndarray) -> np.ndarray:
        return np.bincount(
            inverse,
            weights=values.astype(np.float64, copy=False),
            minlength=cell_count,
        )

    active = opacities >= active_threshold
    active_count = counts(active)
    rasterized_count = counts(attribution["rasterized"])
    contributing_count = counts(attribution["visible_pixels"] > 0)
    raw_mass = sums(attribution["raw_mass"])
    contribution_mass = sums(attribution["contribution_mass"])
    total_contribution = float(contribution_mass.sum())
    result: list[dict[str, Any]] = []
    for index, xyz in enumerate(unique):
        fraction = (
            float(contributing_count[index] / active_count[index])
            if active_count[index] > 0
            else 0.0
        )
        result.append(
            {
                "cell_id": f"{xyz[0]}:{xyz[1]}:{xyz[2]}",
                "cell_x": int(xyz[0]),
                "cell_y": int(xyz[1]),
                "cell_z": int(xyz[2]),
                "active_gaussian_count": int(active_count[index]),
                "rasterized_gaussian_count": int(
                    rasterized_count[index]
                ),
                "contributing_gaussian_count": int(
                    contributing_count[index]
                ),
                "contributing_gaussian_fraction": fraction,
                "raw_alpha_mass_pixels": float(raw_mass[index]),
                "contribution_mass_pixels": float(
                    contribution_mass[index]
                ),
                "image_share": (
                    float(contribution_mass[index] / total_contribution)
                    if total_contribution > 0.0
                    else 0.0
                ),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if args.fps <= 0.0 or args.width <= 0 or args.height <= 0:
        raise ValueError("FPS and image dimensions must be positive")
    if args.cell_size_m <= 0.0:
        raise ValueError("--cell-size-m must be positive")
    if not 0.0 <= args.visibility_weight_threshold <= 1.0:
        raise ValueError("Visibility threshold must be within [0, 1]")

    sys.path.insert(0, str(args.gsplat_library_path.resolve()))
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range
    from gsplat.exporter import load_ply_to_splats
    from gsplat.rendering import rasterization

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    rows = load_trace(args.trace)
    start_time = (
        float(args.start_time)
        if args.start_time is not None
        else float(rows[0]["timestamp_s"])
    )
    end_time = (
        min(start_time + float(args.duration), float(rows[-1]["timestamp_s"]))
        if args.duration is not None
        else float(rows[-1]["timestamp_s"])
    )
    if end_time <= start_time:
        raise ValueError("Selected trace interval is empty")
    samples = sampled_trace_rows(rows, start_time, end_time, args.fps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata or args.output.with_suffix(".json")
    skipped: list[dict[str, Any]] = []
    emitted = 0
    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for source_output, output_time, trace in samples:
            asset_id = int(trace["gsv_frame"]) + args.asset_frame_offset
            gt_path = args.gt_root / f"{asset_id:07d}.ply"
            frame = f"{asset_id:07d}"
            required_paths = {"gt": gt_path}
            if args.require_model_assets_root is not None:
                required_paths.update(
                    {
                        "base": (
                            args.require_model_assets_root
                            / "SINGLE"
                            / f"{frame}_aggressive_base_random"
                            / "ply"
                            / "point_cloud_8999.ply"
                        ),
                        "e3": (
                            args.require_model_assets_root
                            / "EVOGS_V1"
                            / frame
                            / "enhancement_03_enhanced"
                            / "ply"
                            / f"{frame}_enhancement.ply"
                        ),
                    }
                )
            missing_paths = {
                role: str(path)
                for role, path in required_paths.items()
                if not path.is_file()
            }
            if missing_paths:
                record = {
                    "source_output_frame": source_output,
                    "gsv_frame": trace["gsv_frame"],
                    "asset_frame_id": asset_id,
                    "missing": missing_paths,
                }
                if args.skip_missing_assets:
                    skipped.append(record)
                    continue
                raise FileNotFoundError(
                    f"Missing frame assets: {missing_paths}"
                )

            splats = {
                key: value.float()
                for key, value in load_ply_to_splats(str(gt_path)).items()
            }
            inputs = visibility_inputs(splats, device)
            viewmat, intrinsics = camera_tensors(
                trace, args.width, args.height, args.hfov, device
            )
            means, quats, scales, opacities, covars = inputs
            with torch.inference_mode():
                _, _, meta = rasterization(
                    means,
                    quats,
                    scales,
                    opacities,
                    None,
                    viewmat,
                    intrinsics,
                    args.width,
                    args.height,
                    near_plane=args.near_cm,
                    eps2d=0.3,
                    packed=False,
                    tile_size=16,
                    render_mode="D",
                    covars=covars,
                )
                attribution = gaussian_attribution(
                    meta,
                    args.width,
                    args.height,
                    int(means.shape[0]),
                    args.visibility_weight_threshold,
                    rasterize_to_indices_in_range,
                )
            cell_rows = aggregate_cells(
                splats,
                attribution,
                args.cell_size_m,
                1.0 / 255.0,
            )
            common = {
                "schema_version": 2,
                "output_frame": emitted,
                "source_output_frame": source_output,
                "output_time_s": output_time - start_time,
                "trace_source_row": trace["source_row"],
                "trace_timestamp_s": trace["timestamp_s"],
                "gsv_frame": trace["gsv_frame"],
                "asset_frame_id": asset_id,
            }
            for cell in cell_rows:
                writer.writerow({**common, **cell})
            emitted += 1
            if emitted == 1 or emitted % max(1, round(args.fps)) == 0:
                print(
                    f"Visibility {emitted}/{len(samples)} | "
                    f"GSV {trace['gsv_frame']} | cells {len(cell_rows)}",
                    flush=True,
                )

    runtime = time.perf_counter() - started
    missing_role_counts: dict[str, int] = {}
    for skipped_record in skipped:
        for role in skipped_record["missing"]:
            missing_role_counts[role] = missing_role_counts.get(role, 0) + 1
    metadata = {
        "schema_version": 2,
        "pipeline_status": (
            "PASS" if emitted > 0 else "FAIL_NO_ELIGIBLE_FRAMES"
        ),
        "visibility_source": "DanceNet3D ground-truth standard PLY",
        "policy_independence": (
            "Neither Base nor E3 content is read while computing visibility; "
            "optional existence checks only prevent evaluating untrained frames"
        ),
        "definition": {
            "gaussian_contributing": (
                "at least one pixel has front-to-back compositing "
                "weight T*alpha >= threshold"
            ),
            "contributing_gaussian_fraction": (
                "contributing active Gaussians / active Gaussians in cell"
            ),
            "visibility_weight_threshold": (
                args.visibility_weight_threshold
            ),
            "active_opacity_threshold": 1.0 / 255.0,
            "cell_size_m": args.cell_size_m,
            "cell_origin_gsv_local_m": [0.0, 0.0, 0.0],
        },
        "inputs": {
            "trace": str(args.trace.resolve()),
            "gt_root": str(args.gt_root.resolve()),
            "required_model_assets_root": (
                str(args.require_model_assets_root.resolve())
                if args.require_model_assets_root is not None
                else None
            ),
        },
        "sampling": {
            "start_time_s": start_time,
            "end_time_s": end_time,
            "fps": args.fps,
            "requested_samples": len(samples),
            "emitted_frames": emitted,
            "skipped_samples": len(skipped),
            "missing_role_counts": missing_role_counts,
            "skipped": skipped,
        },
        "projection": {
            "width": args.width,
            "height": args.height,
            "hfov_degrees": args.hfov,
            "near_cm": args.near_cm,
        },
        "runtime": {
            "seconds": runtime,
            "frames_per_second": emitted / runtime if runtime else None,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "output": str(args.output.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if emitted == 0:
        first_missing = skipped[0]["missing"] if skipped else {}
        raise RuntimeError(
            "No eligible frames were emitted. Missing asset counts by role: "
            f"{missing_role_counts}. First missing paths: {first_missing}. "
            f"See {metadata_path}"
        )
    print(f"Wrote visibility: {args.output}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
