#!/usr/bin/env python3
"""Trace-driven Base/E3 QoE evaluation using only standard 3DGS PLY assets.

For every trace frame this evaluator renders three aligned images:

* GT: DanceNet3D ground-truth 3DGS PLY;
* E3: the complete EvoGS-v1 Enhancement-3 PLY;
* Policy: Base Gaussians outside selected cells plus E3 Gaussians inside them.

It reports three image pairs (E3-vs-GT, Policy-vs-GT, Policy-vs-E3), each for
the full image and a reference-alpha-defined foreground region.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from fovsim.qoe import (
    LPIPS_ALEX_MIN_CROP_SIZE,
    alex_lpips_crop_is_supported,
)


EVOGS_TO_GSV = torch.tensor(
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    dtype=torch.float32,
)
BACKGROUND_RGB = (8.0 / 255.0, 8.0 / 255.0, 10.0 / 255.0)
PAIR_SPECS = {
    "e3_vs_gt": {
        "test": "e3",
        "reference": "gt",
        "meaning": "complete E3 model quality relative to DanceNet3D GT",
    },
    "policy_vs_gt": {
        "test": "policy",
        "reference": "gt",
        "meaning": "delivered FoV-policy quality relative to DanceNet3D GT",
    },
    "policy_vs_e3": {
        "test": "policy",
        "reference": "e3",
        "meaning": "incremental quality loss caused by withholding E3 cells",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Root containing SINGLE/ and EVOGS_V1/",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        required=True,
        help="Directory containing seven-digit DanceNet3D GT PLY files",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gsplat-library-path", type=Path, required=True)
    parser.add_argument(
        "--dataset-variant",
        choices=("lut", "legacy_non_lut"),
        default="lut",
        help="Recorded in metadata and used to prevent LUT/non-LUT path mixing",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--near-cm", type=float, default=1.0)
    parser.add_argument("--cell-size-m", type=float, default=0.2)
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        choices=(0, 1, 2, 3),
        help="Camera-aware SH degree used for GT, E3, Base, and Policy",
    )
    parser.add_argument(
        "--foreground-alpha-threshold",
        type=float,
        default=1.0 / 255.0,
        help="Reference alpha threshold defining foreground pixels",
    )
    parser.add_argument(
        "--foreground-pad-px",
        type=int,
        default=24,
        help="Context padding around foreground bounds for SSIM/LPIPS",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-count", type=int)
    parser.add_argument(
        "--asset-frame-offset",
        type=int,
        default=1,
        help="PLY frame id = GSV frame + this value",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--prefetch-workers", type=int, default=2)
    parser.add_argument("--metric-batch-size", type=int, default=4)
    return parser.parse_args()


def load_trace(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
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
            raise ValueError(f"Trace CSV is missing columns: {sorted(missing)}")
        for source_row, raw in enumerate(reader, start=2):
            rows[source_row] = {
                "location_cm": tuple(
                    float(raw[key])
                    for key in ("LocationX", "LocationY", "LocationZ")
                ),
                "rotation_rpy": tuple(
                    float(raw[key])
                    for key in ("RotationRoll", "RotationPitch", "RotationYaw")
                ),
                "gsv_frame": int(raw["Frame"]),
                "timestamp_s": float(raw["Timestamp"]),
            }
    return rows


def load_decisions(path: Path) -> list[dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "output_frame",
            "output_time_s",
            "trace_source_row",
            "trace_timestamp_s",
            "gsv_frame",
            "cell_x",
            "cell_y",
            "cell_z",
            "target_level",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Decision CSV is missing columns: {sorted(missing)}"
            )
        for raw in reader:
            output_frame = int(raw["output_frame"])
            frame = frames.setdefault(
                output_frame,
                {
                    "output_frame": output_frame,
                    "output_time_s": float(raw["output_time_s"]),
                    "trace_source_row": int(raw["trace_source_row"]),
                    "trace_timestamp_s": float(raw["trace_timestamp_s"]),
                    "gsv_frame": int(raw["gsv_frame"]),
                    "selected_cells": set(),
                    "occupied_cells": 0,
                },
            )
            identity = (
                float(raw["output_time_s"]),
                int(raw["trace_source_row"]),
                float(raw["trace_timestamp_s"]),
                int(raw["gsv_frame"]),
            )
            expected = (
                frame["output_time_s"],
                frame["trace_source_row"],
                frame["trace_timestamp_s"],
                frame["gsv_frame"],
            )
            if identity != expected:
                raise ValueError(
                    f"Inconsistent metadata for output frame {output_frame}"
                )
            frame["occupied_cells"] += 1
            if int(raw["target_level"]) == 3:
                frame["selected_cells"].add(
                    (
                        int(raw["cell_x"]),
                        int(raw["cell_y"]),
                        int(raw["cell_z"]),
                    )
                )
    ordered = [frames[index] for index in sorted(frames)]
    if not ordered:
        raise ValueError("Decision CSV has no frames")
    expected_frames = list(
        range(ordered[0]["output_frame"], ordered[-1]["output_frame"] + 1)
    )
    if [frame["output_frame"] for frame in ordered] != expected_frames:
        raise ValueError("Decision frames must be contiguous")
    return ordered


def validate_trace_reference(
    trace: dict[str, Any], decision: dict[str, Any]
) -> None:
    if trace["gsv_frame"] != decision["gsv_frame"]:
        raise ValueError(
            f"Trace/GSV mismatch at output frame {decision['output_frame']}"
        )
    if abs(trace["timestamp_s"] - decision["trace_timestamp_s"]) > 1.0e-5:
        raise ValueError(
            f"Trace/timestamp mismatch at output frame "
            f"{decision['output_frame']}"
        )


def asset_paths(
    model_root: Path, gt_root: Path, asset_id: int
) -> dict[str, Path]:
    frame = f"{asset_id:07d}"
    return {
        "base": (
            model_root
            / "SINGLE"
            / f"{frame}_aggressive_base_random"
            / "ply"
            / "point_cloud_8999.ply"
        ),
        "e3": (
            model_root
            / "EVOGS_V1"
            / frame
            / "enhancement_03_enhanced"
            / "ply"
            / f"{frame}_enhancement.ply"
        ),
        "gt": gt_root / f"{frame}.ply",
    }


def load_frame_assets(
    paths: dict[str, Path],
    load_ply_to_splats: Callable[[str], dict[str, torch.Tensor]],
) -> dict[str, Any]:
    started = time.perf_counter()
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role} PLY: {path}")
    result = {
        role: {
            key: value.float()
            for key, value in load_ply_to_splats(str(path)).items()
        }
        for role, path in paths.items()
    }
    result["load_seconds"] = time.perf_counter() - started
    result["loader_thread"] = threading.current_thread().name
    return result


def cell_selection_mask(
    splats: dict[str, torch.Tensor],
    selected_cells: set[tuple[int, int, int]],
    cell_size_m: float,
) -> torch.Tensor:
    means_gsv_m = splats["means"] @ EVOGS_TO_GSV.T
    indices = torch.floor(means_gsv_m / cell_size_m).to(torch.int64)
    return torch.tensor(
        [
            tuple(int(value) for value in cell) in selected_cells
            for cell in indices.tolist()
        ],
        dtype=torch.bool,
    )


def build_ply_policy(
    base: dict[str, torch.Tensor],
    e3: dict[str, torch.Tensor],
    selected_cells: set[tuple[int, int, int]],
    cell_size_m: float,
) -> tuple[dict[str, torch.Tensor], dict[str, int | float]]:
    """Spatial cell replacement using standard PLYs only.

    Base Gaussians remain in non-selected cells. Complete E3 frontier
    Gaussians are used in selected cells. This is the PLY-only equivalent of
    transmitting E3 cell chunks and requires no training checkpoint.
    """
    if set(base) != set(e3):
        raise ValueError(
            "Base and E3 PLYs expose different Gaussian attribute fields"
        )
    base_selected = cell_selection_mask(base, selected_cells, cell_size_m)
    e3_selected = cell_selection_mask(e3, selected_cells, cell_size_m)
    policy = {
        key: torch.cat(
            (base[key][~base_selected], e3[key][e3_selected]), dim=0
        )
        for key in base
    }
    stats: dict[str, int | float] = {
        "base_gaussian_count": int(base["means"].shape[0]),
        "full_e3_gaussian_count": int(e3["means"].shape[0]),
        "base_gaussians_replaced": int(base_selected.sum()),
        "e3_gaussians_inserted": int(e3_selected.sum()),
        "policy_gaussian_count": int(policy["means"].shape[0]),
        "base_selected_fraction": float(base_selected.float().mean()),
        "e3_selected_fraction": float(e3_selected.float().mean()),
    }
    return policy, stats


def gaussian_payload_bytes(splats: dict[str, torch.Tensor]) -> int:
    """Return serialized attribute bytes represented by one standard-Ply row."""
    count = int(splats["means"].shape[0])
    if count <= 0:
        raise ValueError("Cannot infer Gaussian payload size from an empty PLY")
    total = 0
    for value in splats.values():
        if int(value.shape[0]) != count:
            raise ValueError("PLY tensors do not share one Gaussian dimension")
        total += value[0].numel() * value.element_size()
    return int(total)


def normalized_quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = torch.unbind(quat, dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def camera_source_position_m(trace: dict[str, Any]) -> torch.Tensor:
    camera_gsv_m = (
        torch.tensor(trace["location_cm"], dtype=torch.float32) / 100.0
    )
    return camera_gsv_m @ EVOGS_TO_GSV


def render_inputs(
    splats: dict[str, torch.Tensor],
    trace: dict[str, Any],
    device: torch.device,
    sh_degree: int,
    spherical_harmonics: Callable[..., torch.Tensor],
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
    transform = EVOGS_TO_GSV
    covars = (transform @ covars @ transform.T) * 10000.0
    opacities = torch.sigmoid(splats["opacities"].reshape(-1))

    coefficients = torch.cat((splats["sh0"], splats["shN"]), dim=1)
    required_coefficients = (sh_degree + 1) ** 2
    if coefficients.shape[1] < required_coefficients:
        raise ValueError(
            f"PLY has {coefficients.shape[1]} SH coefficients but degree "
            f"{sh_degree} requires {required_coefficients}"
        )
    camera_source = camera_source_position_m(trace)
    directions = F.normalize(source_means - camera_source, dim=-1)
    coefficients_device = coefficients.to(device, non_blocking=True)
    colors = spherical_harmonics(
        sh_degree,
        directions.to(device, non_blocking=True),
        coefficients_device,
    )
    colors = torch.clamp_min(colors + 0.5, 0.0)

    count = means.shape[0]
    dummy_quats = torch.zeros((count, 4), dtype=torch.float32)
    dummy_quats[:, 0] = 1.0
    dummy_scales = torch.ones((count, 3), dtype=torch.float32)
    return (
        means.to(device, non_blocking=True),
        dummy_quats.to(device, non_blocking=True),
        dummy_scales.to(device, non_blocking=True),
        opacities.to(device, non_blocking=True),
        colors,
        covars.to(device, non_blocking=True),
    )


def unreal_axes(
    roll_degrees: float, pitch_degrees: float, yaw_degrees: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roll, pitch, yaw = map(
        math.radians, (roll_degrees, pitch_degrees, yaw_degrees)
    )
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    forward = np.array((cp * cy, cp * sy, sp), dtype=np.float32)
    right = np.array(
        (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp),
        dtype=np.float32,
    )
    up = np.array(
        (
            -(cr * sp * cy + sr * sy),
            cy * sr - cr * sp * sy,
            cr * cp,
        ),
        dtype=np.float32,
    )
    return forward, right, up


def camera_tensors(
    trace: dict[str, Any],
    width: int,
    height: int,
    hfov: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward, right, up = unreal_axes(*trace["rotation_rpy"])
    camera_from_world = np.vstack((right, -up, forward)).astype(np.float32)
    location = np.asarray(trace["location_cm"], dtype=np.float32)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = camera_from_world
    viewmat[:3, 3] = -(camera_from_world @ location)
    focal = width / (2.0 * math.tan(math.radians(hfov) * 0.5))
    intrinsics = np.array(
        (
            (focal, 0.0, width * 0.5),
            (0.0, focal, height * 0.5),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    return (
        torch.from_numpy(viewmat).unsqueeze(0).to(device),
        torch.from_numpy(intrinsics).unsqueeze(0).to(device),
    )


def render(
    rasterization: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
    viewmat: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    near_cm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    means, quats, scales, opacities, colors, covars = inputs
    background = torch.tensor(
        (BACKGROUND_RGB,), dtype=torch.float32, device=means.device
    )
    image, alpha, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmat,
        intrinsics,
        width,
        height,
        near_plane=near_cm,
        eps2d=0.3,
        packed=False,
        tile_size=16,
        backgrounds=background,
        render_mode="RGB",
        covars=covars,
    )
    return (
        image[0].permute(2, 0, 1).clamp(0.0, 1.0),
        alpha[0, ..., 0].clamp(0.0, 1.0),
    )


def metric_values_batch(
    tests: list[torch.Tensor],
    references: list[torch.Tensor],
    lpips: LearnedPerceptualImagePatchSimilarity,
) -> list[dict[str, float]]:
    if not tests or len(tests) != len(references):
        raise ValueError("Metric batches must be non-empty and aligned")
    test_batch = torch.stack(tests)
    reference_batch = torch.stack(references)
    mse_values = (
        F.mse_loss(test_batch, reference_batch, reduction="none")
        .flatten(1)
        .mean(dim=1)
    )
    ssim_values = structural_similarity_index_measure(
        test_batch,
        reference_batch,
        data_range=1.0,
        reduction="none",
    ).reshape(-1)
    lpips_values = lpips(test_batch, reference_batch).reshape(-1)
    result: list[dict[str, float]] = []
    for mse_tensor, ssim_tensor, lpips_tensor in zip(
        mse_values, ssim_values, lpips_values, strict=True
    ):
        mse = float(mse_tensor.item())
        result.append(
            {
                "mse": mse,
                "psnr_db": (
                    math.inf if mse == 0.0 else -10.0 * math.log10(mse)
                ),
                "ssim": float(ssim_tensor.item()),
                "lpips_alex": float(lpips_tensor.item()),
            }
        )
    lpips.reset()
    return result


def foreground_definition(
    alpha: torch.Tensor, threshold: float, pad: int
) -> dict[str, Any] | None:
    mask = alpha >= threshold
    coordinates = torch.nonzero(mask, as_tuple=False)
    if coordinates.numel() == 0:
        return None
    y0 = max(0, int(coordinates[:, 0].min()) - pad)
    y1 = min(alpha.shape[0], int(coordinates[:, 0].max()) + pad + 1)
    x0 = max(0, int(coordinates[:, 1].min()) - pad)
    x1 = min(alpha.shape[1], int(coordinates[:, 1].max()) + pad + 1)
    return {
        "mask": mask,
        "y": slice(y0, y1),
        "x": slice(x0, x1),
        "pixel_count": int(mask.sum()),
        "bbox": (x0, y0, x1, y1),
    }


def foreground_metric_values(
    test: torch.Tensor,
    reference: torch.Tensor,
    definition: dict[str, Any] | None,
    lpips: LearnedPerceptualImagePatchSimilarity,
) -> dict[str, float] | None:
    if definition is None:
        return None
    y_slice = definition["y"]
    x_slice = definition["x"]
    crop_height = int(y_slice.stop) - int(y_slice.start)
    crop_width = int(x_slice.stop) - int(x_slice.start)
    if not alex_lpips_crop_is_supported(crop_height, crop_width):
        return None
    mask = definition["mask"][y_slice, x_slice]
    test_crop = test[:, y_slice, x_slice]
    reference_crop = reference[:, y_slice, x_slice]
    background = torch.tensor(
        BACKGROUND_RGB, dtype=test.dtype, device=test.device
    ).reshape(3, 1, 1)
    expanded_mask = mask.unsqueeze(0)
    masked_test = torch.where(expanded_mask, test_crop, background)
    masked_reference = torch.where(
        expanded_mask, reference_crop, background
    )
    values = metric_values_batch(
        [masked_test], [masked_reference], lpips
    )[0]
    squared_error = (test_crop - reference_crop).pow(2)
    masked_mse = float(
        squared_error[:, mask].mean().item()
    )
    values["mse"] = masked_mse
    values["psnr_db"] = (
        math.inf
        if masked_mse == 0.0
        else -10.0 * math.log10(masked_mse)
    )
    return values


def flatten_metrics(
    row: dict[str, Any],
    pair: str,
    scope: str,
    values: dict[str, float] | None,
) -> None:
    for metric in ("mse", "psnr_db", "ssim", "lpips_alex"):
        row[f"{pair}_{scope}_{metric}"] = (
            None if values is None else values[metric]
        )


def tensor_to_image(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach()
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def masked_reference_image(
    reference: torch.Tensor, definition: dict[str, Any] | None
) -> torch.Tensor:
    background = torch.tensor(
        BACKGROUND_RGB, dtype=reference.dtype, device=reference.device
    ).reshape(3, 1, 1)
    if definition is None:
        return background.expand_as(reference)
    return torch.where(
        definition["mask"].unsqueeze(0), reference, background
    )


def save_sample(
    path: Path,
    images: dict[str, torch.Tensor],
    definitions: dict[str, dict[str, Any] | None],
    frame_number: int,
) -> None:
    rows = [
        ("DanceNet3D GT full reference", images["gt"]),
        ("Full E3", images["e3"]),
        ("Base + FoV-selected E3 policy", images["policy"]),
        (
            "GT foreground reference (alpha mask)",
            masked_reference_image(images["gt"], definitions["gt"]),
        ),
        (
            "E3 foreground reference (alpha mask)",
            masked_reference_image(images["e3"], definitions["e3"]),
        ),
        (
            "Absolute difference E3 vs GT x8",
            (images["e3"] - images["gt"]).abs().mul(8.0).clamp(0.0, 1.0),
        ),
        (
            "Absolute difference Policy vs GT x8",
            (images["policy"] - images["gt"])
            .abs()
            .mul(8.0)
            .clamp(0.0, 1.0),
        ),
        (
            "Absolute difference Policy vs E3 x8",
            (images["policy"] - images["e3"])
            .abs()
            .mul(8.0)
            .clamp(0.0, 1.0),
        ),
    ]
    rendered = [(label, tensor_to_image(image)) for label, image in rows]
    width = rendered[0][1].width
    height = rendered[0][1].height
    canvas = Image.new("RGB", (width, height * len(rendered)))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(rendered):
        y = index * height
        canvas.paste(image, (0, y))
        text = f"{label} - frame {frame_number}"
        draw.rectangle((8, y + 8, 570, y + 38), fill=(0, 0, 0))
        draw.text((14, y + 14), text, fill=(255, 255, 255))
    canvas.save(path)


def aggregate_pair(
    rows: list[dict[str, Any]], pair: str, foreground_reference: str
) -> dict[str, Any]:
    full_mse = [float(row[f"{pair}_full_mse"]) for row in rows]
    finite_full_psnr = [
        float(row[f"{pair}_full_psnr_db"])
        for row in rows
        if math.isfinite(float(row[f"{pair}_full_psnr_db"]))
    ]
    foreground_rows = [
        row for row in rows if row[f"{pair}_fore_mse"] is not None
    ]
    mean_full_mse = float(np.mean(full_mse))
    result: dict[str, Any] = {
        "meaning": PAIR_SPECS[pair]["meaning"],
        "full": {
            "frame_count": len(rows),
            "sequence_psnr_db": (
                math.inf
                if mean_full_mse == 0.0
                else -10.0 * math.log10(mean_full_mse)
            ),
            "mean_frame_psnr_db": (
                float(np.mean(finite_full_psnr))
                if finite_full_psnr
                else math.inf
            ),
            "mean_ssim": float(
                np.mean([row[f"{pair}_full_ssim"] for row in rows])
            ),
            "mean_lpips_alex": float(
                np.mean([row[f"{pair}_full_lpips_alex"] for row in rows])
            ),
            "mean_mse": mean_full_mse,
        },
        "foreground": {
            "reference": foreground_reference,
            "definition": (
                "reference alpha >= threshold; masked-pixel PSNR; "
                "masked tight crop plus context for SSIM/LPIPS; all "
                "foreground metrics are N/A when the crop is smaller than "
                f"{LPIPS_ALEX_MIN_CROP_SIZE} pixels on either axis"
            ),
            "frame_count": len(foreground_rows),
            "n_a_frame_count": len(rows) - len(foreground_rows),
        },
    }
    if foreground_rows:
        pixel_field = f"{foreground_reference}_foreground_pixel_count"
        total_pixels = sum(int(row[pixel_field]) for row in foreground_rows)
        weighted_mse = sum(
            float(row[f"{pair}_fore_mse"]) * int(row[pixel_field])
            for row in foreground_rows
        ) / total_pixels
        finite_fore_psnr = [
            float(row[f"{pair}_fore_psnr_db"])
            for row in foreground_rows
            if math.isfinite(float(row[f"{pair}_fore_psnr_db"]))
        ]
        result["foreground"].update(
            {
                "foreground_pixel_count": total_pixels,
                "sequence_psnr_db": (
                    math.inf
                    if weighted_mse == 0.0
                    else -10.0 * math.log10(weighted_mse)
                ),
                "mean_frame_psnr_db": (
                    float(np.mean(finite_fore_psnr))
                    if finite_fore_psnr
                    else math.inf
                ),
                "mean_ssim": float(
                    np.mean(
                        [row[f"{pair}_fore_ssim"] for row in foreground_rows]
                    )
                ),
                "mean_lpips_alex": float(
                    np.mean(
                        [
                            row[f"{pair}_fore_lpips_alex"]
                            for row in foreground_rows
                        ]
                    )
                ),
                "pixel_weighted_mse": weighted_mse,
            }
        )
    return result


def json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    if not args.trace.is_file() or not args.decisions.is_file():
        raise FileNotFoundError("Trace or decision CSV does not exist")
    if not args.model_root.is_dir() or not args.gt_root.is_dir():
        raise FileNotFoundError("Model root or GT root does not exist")
    path_name = args.model_root.name.upper()
    if args.dataset_variant == "lut" and not path_name.endswith("_LUT"):
        raise ValueError(
            "--dataset-variant lut requires a model root ending in _LUT"
        )
    if args.dataset_variant == "legacy_non_lut" and path_name.endswith("_LUT"):
        raise ValueError("Legacy smoke mode cannot use a _LUT model root")
    if args.prefetch_workers < 0:
        raise ValueError("--prefetch-workers must be non-negative")
    if args.metric_batch_size <= 0:
        raise ValueError("--metric-batch-size must be positive")
    if args.frame_count is not None and args.frame_count <= 0:
        raise ValueError("--frame-count must be positive")
    if not 0.0 <= args.foreground_alpha_threshold <= 1.0:
        raise ValueError("--foreground-alpha-threshold must be within [0,1]")
    if args.foreground_pad_px < 0:
        raise ValueError("--foreground-pad-px must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.gsplat_library_path.resolve()))
    from gsplat.cuda._wrapper import spherical_harmonics
    from gsplat.exporter import load_ply_to_splats
    from gsplat.rendering import rasterization

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    trace_rows = load_trace(args.trace)
    decisions = load_decisions(args.decisions)
    if args.frame_count is not None:
        decisions = decisions[: args.frame_count]

    specs: list[dict[str, Any]] = []
    for decision in decisions:
        source_row = decision["trace_source_row"]
        if source_row not in trace_rows:
            raise ValueError(f"Unknown trace source row {source_row}")
        validate_trace_reference(trace_rows[source_row], decision)
        asset_id = decision["gsv_frame"] + args.asset_frame_offset
        paths = asset_paths(args.model_root, args.gt_root, asset_id)
        for role, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {role} PLY: {path}")
        specs.append({"asset_id": asset_id, "paths": paths})

    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="none"
    ).to(device).eval()
    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    loader_threads: set[str] = set()
    metric_batch_sizes: list[int] = []
    timing = {
        "asset_load_cpu_work_seconds": 0.0,
        "asset_wait_wall_seconds": 0.0,
        "policy_build_cpu_seconds": 0.0,
        "render_input_seconds": 0.0,
        "gt_render_seconds": 0.0,
        "e3_render_seconds": 0.0,
        "policy_render_seconds": 0.0,
        "full_metric_seconds": 0.0,
        "foreground_metric_seconds": 0.0,
        "sample_write_seconds": 0.0,
        "output_write_seconds": 0.0,
    }

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def flush_metrics() -> None:
        if not pending:
            return
        metric_batch_sizes.append(len(pending))
        full_by_pair: dict[str, list[dict[str, float]]] = {}
        synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for pair, spec in PAIR_SPECS.items():
                full_by_pair[pair] = metric_values_batch(
                    [item["images"][spec["test"]] for item in pending],
                    [item["images"][spec["reference"]] for item in pending],
                    lpips,
                )
        synchronize()
        timing["full_metric_seconds"] += time.perf_counter() - started

        for pending_index, item in enumerate(pending):
            row = dict(item["metadata"])
            foreground_started = time.perf_counter()
            for pair, spec in PAIR_SPECS.items():
                flatten_metrics(
                    row,
                    pair,
                    "full",
                    full_by_pair[pair][pending_index],
                )
                reference_role = spec["reference"]
                with torch.inference_mode():
                    foreground_values = foreground_metric_values(
                        item["images"][spec["test"]],
                        item["images"][reference_role],
                        item["definitions"][reference_role],
                        lpips,
                    )
                flatten_metrics(row, pair, "fore", foreground_values)
            synchronize()
            timing["foreground_metric_seconds"] += (
                time.perf_counter() - foreground_started
            )
            rows.append(row)
            print(
                f"[{row['display_frame']:02d}/{len(decisions)}] "
                f"GSV {row['gsv_frame']:02d} | "
                f"E3-GT {row['e3_vs_gt_full_psnr_db']:.3f} dB | "
                f"Policy-GT {row['policy_vs_gt_full_psnr_db']:.3f} dB | "
                f"Policy-E3 {row['policy_vs_e3_full_psnr_db']:.3f} dB",
                flush=True,
            )
        pending.clear()

    executor: ThreadPoolExecutor | None = None
    futures: dict[int, Future[dict[str, Any]]] = {}
    prefetch_window = max(1, args.prefetch_workers + 1)
    submitted_tasks = 0
    started_total = time.perf_counter()
    if args.prefetch_workers > 0:
        executor = ThreadPoolExecutor(
            max_workers=args.prefetch_workers,
            thread_name_prefix="qoe-ply",
        )
        for index in range(min(prefetch_window, len(specs))):
            futures[index] = executor.submit(
                load_frame_assets,
                specs[index]["paths"],
                load_ply_to_splats,
            )
            submitted_tasks += 1

    try:
        for index, (decision, spec) in enumerate(
            zip(decisions, specs, strict=True)
        ):
            position = index + 1
            trace = trace_rows[decision["trace_source_row"]]
            wait_started = time.perf_counter()
            if executor is None:
                assets = load_frame_assets(
                    spec["paths"], load_ply_to_splats
                )
            else:
                assets = futures.pop(index).result()
                next_index = index + prefetch_window
                if next_index < len(specs):
                    futures[next_index] = executor.submit(
                        load_frame_assets,
                        specs[next_index]["paths"],
                        load_ply_to_splats,
                    )
                    submitted_tasks += 1
            timing["asset_wait_wall_seconds"] += (
                time.perf_counter() - wait_started
            )
            timing["asset_load_cpu_work_seconds"] += assets["load_seconds"]
            loader_threads.add(assets["loader_thread"])

            policy_started = time.perf_counter()
            policy, policy_stats = build_ply_policy(
                assets["base"],
                assets["e3"],
                decision["selected_cells"],
                args.cell_size_m,
            )
            timing["policy_build_cpu_seconds"] += (
                time.perf_counter() - policy_started
            )

            viewmat, intrinsics = camera_tensors(
                trace, args.width, args.height, args.hfov, device
            )
            input_started = time.perf_counter()
            render_data = {
                "gt": render_inputs(
                    assets["gt"],
                    trace,
                    device,
                    args.sh_degree,
                    spherical_harmonics,
                ),
                "e3": render_inputs(
                    assets["e3"],
                    trace,
                    device,
                    args.sh_degree,
                    spherical_harmonics,
                ),
                "policy": render_inputs(
                    policy,
                    trace,
                    device,
                    args.sh_degree,
                    spherical_harmonics,
                ),
            }
            synchronize()
            timing["render_input_seconds"] += (
                time.perf_counter() - input_started
            )

            images: dict[str, torch.Tensor] = {}
            alphas: dict[str, torch.Tensor] = {}
            with torch.inference_mode():
                for role in ("gt", "e3", "policy"):
                    synchronize()
                    render_started = time.perf_counter()
                    images[role], alphas[role] = render(
                        rasterization,
                        render_data[role],
                        viewmat,
                        intrinsics,
                        args.width,
                        args.height,
                        args.near_cm,
                    )
                    synchronize()
                    timing[f"{role}_render_seconds"] += (
                        time.perf_counter() - render_started
                    )

            definitions = {
                "gt": foreground_definition(
                    alphas["gt"],
                    args.foreground_alpha_threshold,
                    args.foreground_pad_px,
                ),
                "e3": foreground_definition(
                    alphas["e3"],
                    args.foreground_alpha_threshold,
                    args.foreground_pad_px,
                ),
            }
            metadata: dict[str, Any] = {
                "display_frame": position,
                "output_frame": decision["output_frame"],
                "output_time_s": decision["output_time_s"],
                "trace_source_row": decision["trace_source_row"],
                "trace_timestamp_s": trace["timestamp_s"],
                "gsv_frame": decision["gsv_frame"],
                "asset_frame_id": spec["asset_id"],
                "selected_cell_count": len(decision["selected_cells"]),
                "occupied_cell_count": decision["occupied_cells"],
                "gt_gaussian_count": int(assets["gt"]["means"].shape[0]),
                **policy_stats,
            }
            base_record_bytes = gaussian_payload_bytes(assets["base"])
            e3_record_bytes = gaussian_payload_bytes(assets["e3"])
            base_file_bytes = int(spec["paths"]["base"].stat().st_size)
            e3_file_bytes = int(spec["paths"]["e3"].stat().st_size)
            selected_e3_payload_bytes = (
                int(policy_stats["e3_gaussians_inserted"])
                * e3_record_bytes
            )
            policy_transmission_bytes = (
                base_file_bytes + selected_e3_payload_bytes
            )
            full_progressive_bytes = base_file_bytes + e3_file_bytes
            metadata.update(
                {
                    "base_ply_file_bytes": base_file_bytes,
                    "e3_ply_file_bytes": e3_file_bytes,
                    "base_gaussian_record_bytes": base_record_bytes,
                    "e3_gaussian_record_bytes": e3_record_bytes,
                    "selected_e3_payload_bytes": selected_e3_payload_bytes,
                    "base_only_transmission_bytes": base_file_bytes,
                    "policy_transmission_bytes": policy_transmission_bytes,
                    "full_progressive_transmission_bytes": (
                        full_progressive_bytes
                    ),
                    "policy_savings_vs_full_bytes": (
                        full_progressive_bytes - policy_transmission_bytes
                    ),
                    "policy_savings_vs_full_fraction": (
                        (full_progressive_bytes - policy_transmission_bytes)
                        / full_progressive_bytes
                    ),
                }
            )
            for role in ("gt", "e3"):
                definition = definitions[role]
                if definition is None:
                    metadata.update(
                        {
                            f"{role}_foreground_pixel_count": 0,
                            f"{role}_foreground_fraction": 0.0,
                            f"{role}_foreground_bbox_x0": None,
                            f"{role}_foreground_bbox_y0": None,
                            f"{role}_foreground_bbox_x1": None,
                            f"{role}_foreground_bbox_y1": None,
                        }
                    )
                else:
                    x0, y0, x1, y1 = definition["bbox"]
                    metadata.update(
                        {
                            f"{role}_foreground_pixel_count": definition[
                                "pixel_count"
                            ],
                            f"{role}_foreground_fraction": (
                                definition["pixel_count"]
                                / (args.width * args.height)
                            ),
                            f"{role}_foreground_bbox_x0": x0,
                            f"{role}_foreground_bbox_y0": y0,
                            f"{role}_foreground_bbox_x1": x1,
                            f"{role}_foreground_bbox_y1": y1,
                        }
                    )

            if args.save_every > 0 and (
                position == 1
                or position == len(decisions)
                or (position - 1) % args.save_every == 0
            ):
                sample_started = time.perf_counter()
                save_sample(
                    args.output_dir / f"sample_{position:03d}.png",
                    images,
                    definitions,
                    position,
                )
                timing["sample_write_seconds"] += (
                    time.perf_counter() - sample_started
                )

            pending.append(
                {
                    "images": images,
                    "definitions": definitions,
                    "metadata": metadata,
                }
            )
            if len(pending) >= args.metric_batch_size:
                flush_metrics()
        flush_metrics()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    output_started = time.perf_counter()
    metrics_path = args.output_dir / "per_frame_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_seconds = time.perf_counter() - started_total
    timing["output_write_seconds"] += time.perf_counter() - output_started
    output_time_deltas = np.diff(
        np.asarray([row["output_time_s"] for row in rows], dtype=np.float64)
    )
    positive_output_deltas = output_time_deltas[output_time_deltas > 0.0]
    inferred_output_fps = (
        float(1.0 / np.median(positive_output_deltas))
        if positive_output_deltas.size
        else None
    )
    summary = {
        "schema_version": 2,
        "pipeline_status": "PASS",
        "frame_count": len(rows),
        "dataset_variant": args.dataset_variant,
        "asset_format": "standard_3dgs_ply",
        "comparisons": {
            "e3_vs_gt": aggregate_pair(rows, "e3_vs_gt", "gt"),
            "policy_vs_gt": aggregate_pair(rows, "policy_vs_gt", "gt"),
            "policy_vs_e3": aggregate_pair(rows, "policy_vs_e3", "e3"),
        },
        "foreground": {
            "mask_source": (
                "the alpha image of each pair's reference render "
                "(GT for E3-vs-GT and Policy-vs-GT; E3 for Policy-vs-E3)"
            ),
            "alpha_threshold": args.foreground_alpha_threshold,
            "binary_rule": "reference_alpha >= alpha_threshold",
            "psnr": "MSE normalized only over foreground-mask RGB pixels",
            "ssim_lpips": (
                "both images are masked to the common background, then "
                "evaluated on the tight reference-mask bounding box plus pad"
            ),
            "context_pad_pixels": args.foreground_pad_px,
            "n_a_rule": (
                "All per-frame foreground metric fields are blank (N/A) "
                "when the reference mask is empty or its padded crop is "
                f"smaller than {LPIPS_ALEX_MIN_CROP_SIZE} pixels on either "
                "axis. Full-frame metrics remain valid."
            ),
            "lpips_alex_min_crop_size_pixels": LPIPS_ALEX_MIN_CROP_SIZE,
        },
        "policy": {
            "cell_size_m": args.cell_size_m,
            "composition": (
                "Base PLY Gaussians in non-selected cells plus full E3 PLY "
                "Gaussians in selected cells"
            ),
            "mean_selected_cells": float(
                np.mean([row["selected_cell_count"] for row in rows])
            ),
            "mean_base_gaussians": float(
                np.mean([row["base_gaussian_count"] for row in rows])
            ),
            "mean_full_e3_gaussians": float(
                np.mean([row["full_e3_gaussian_count"] for row in rows])
            ),
            "mean_policy_gaussians": float(
                np.mean([row["policy_gaussian_count"] for row in rows])
            ),
        },
        "bandwidth": {
            "model": (
                "Base PLY file once per dynamic frame plus fixed-width E3 "
                "Gaussian records belonging to selected 0.2 m cells"
            ),
            "container_overhead": (
                "Base PLY header is included; per-cell packet/index headers "
                "are excluded because the final transport container is not "
                "yet specified"
            ),
            "total_base_only_bytes": int(
                sum(row["base_only_transmission_bytes"] for row in rows)
            ),
            "total_policy_bytes": int(
                sum(row["policy_transmission_bytes"] for row in rows)
            ),
            "total_full_progressive_bytes": int(
                sum(
                    row["full_progressive_transmission_bytes"]
                    for row in rows
                )
            ),
            "policy_savings_vs_full_fraction": float(
                1.0
                - sum(row["policy_transmission_bytes"] for row in rows)
                / sum(
                    row["full_progressive_transmission_bytes"]
                    for row in rows
                )
            ),
            "mean_policy_bytes_per_frame": float(
                np.mean([row["policy_transmission_bytes"] for row in rows])
            ),
            "inferred_output_fps": inferred_output_fps,
            "mean_policy_megabits_per_second": (
                float(
                    np.mean(
                        [row["policy_transmission_bytes"] for row in rows]
                    )
                    * 8.0
                    * inferred_output_fps
                    / 1_000_000.0
                )
                if inferred_output_fps is not None
                else None
            ),
        },
        "parallelism": {
            "prefetch_enabled": args.prefetch_workers > 0,
            "prefetch_workers_requested": args.prefetch_workers,
            "prefetch_tasks_submitted": submitted_tasks,
            "asset_loader_threads": sorted(loader_threads),
            "asset_loader_thread_count": len(loader_threads),
            "prefetch_parallel_verified": (
                args.prefetch_workers > 0 and len(loader_threads) > 1
            ),
            "metric_batch_size_requested": args.metric_batch_size,
            "metric_batch_sizes_observed": metric_batch_sizes,
            "metric_batch_count": len(metric_batch_sizes),
            "metric_batching_verified": any(
                size > 1 for size in metric_batch_sizes
            ),
        },
        "performance": {
            "runtime_seconds": total_seconds,
            "throughput_frames_per_second": len(rows) / total_seconds,
            "timing": timing,
        },
        "settings": {
            "trace": str(args.trace.resolve()),
            "decisions": str(args.decisions.resolve()),
            "model_root": str(args.model_root.resolve()),
            "gt_root": str(args.gt_root.resolve()),
            "width": args.width,
            "height": args.height,
            "hfov_degrees": args.hfov,
            "near_cm": args.near_cm,
            "background_rgb": [8, 8, 10],
            "sh_degree": args.sh_degree,
            "sh_evaluation": (
                "camera-aware in original DanceNet coordinates before "
                "geometry is transformed to GSV/Unreal coordinates"
            ),
            "asset_frame_offset": args.asset_frame_offset,
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(json_ready(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(json_ready(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
