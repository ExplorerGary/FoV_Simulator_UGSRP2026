#!/usr/bin/env python3
"""Render and evaluate a CellSight-style Base/E3 policy for EVOGS-v1.

The reference is either a supplied DanceNet3D 3DGS GT PLY or the full E3
frontier. The policy image keeps a root Gaussian at Base unless that root's
0.2 m cell has target_level=3, in which case the root is replaced by all of
its active E3 descendants.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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


SH_C0 = 0.28209479177387814
EVOGS_TO_GSV = torch.tensor(
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    dtype=torch.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--evogs-root", type=Path, required=True)
    parser.add_argument(
        "--gt-root",
        type=Path,
        help=(
            "Directory containing standard 3DGS PLY reference frames. When "
            "omitted, the active EVOGS E3 frontier remains the reference."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gsplat-library-path", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--near-cm", type=float, default=1.0)
    parser.add_argument("--cell-size-m", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--frame-count",
        type=int,
        help="Evaluate only the first N decision frames (useful for smoke tests)",
    )
    parser.add_argument(
        "--asset-frame-offset",
        type=int,
        default=1,
        help="EVOGS directory id = GSV frame + this value (default: 1)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save a visual sample every N frames; 0 disables samples",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=2,
        help=(
            "CPU workers used to parse upcoming EVOGS states and GT PLYs "
            "while the GPU renders (default: 2; 0 disables prefetch)"
        ),
    )
    parser.add_argument(
        "--metric-batch-size",
        type=int,
        default=4,
        help="Number of equal-size full frames evaluated per GPU metric batch",
    )
    return parser.parse_args()


def load_trace(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for source_row, raw in enumerate(reader, start=2):
            result[source_row] = {
                "location": tuple(float(raw[key]) for key in ("LocationX", "LocationY", "LocationZ")),
                "rotation": tuple(
                    float(raw[key])
                    for key in ("RotationRoll", "RotationPitch", "RotationYaw")
                ),
                "gsv_frame": int(raw["Frame"]),
                "timestamp": float(raw["Timestamp"]),
            }
    return result


def load_decisions(path: Path) -> list[dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "output_frame",
            "trace_source_row",
            "gsv_frame",
            "cell_x",
            "cell_y",
            "cell_z",
            "target_level",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Decision CSV is missing columns: {sorted(missing)}")
        for raw in reader:
            output_frame = int(raw["output_frame"])
            item = frames.setdefault(
                output_frame,
                {
                    "output_frame": output_frame,
                    "trace_source_row": int(raw["trace_source_row"]),
                    "gsv_frame": int(raw["gsv_frame"]),
                    "selected_cells": set(),
                    "occupied_cells": 0,
                },
            )
            item["occupied_cells"] += 1
            if int(raw["target_level"]) == 3:
                item["selected_cells"].add(
                    (int(raw["cell_x"]), int(raw["cell_y"]), int(raw["cell_z"]))
                )
    ordered = [frames[key] for key in sorted(frames)]
    if [item["output_frame"] for item in ordered] != list(
        range(ordered[0]["output_frame"], ordered[-1]["output_frame"] + 1)
    ):
        raise ValueError("Decision frames must be contiguous")
    return ordered


def compose_evogs_state(
    state: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Return root splats, active E3 frontier splats, and frontier root ids."""
    root = {key: value.float() for key, value in state["root_splat"].items()}
    values = {key: value.clone() for key, value in root.items()}
    n_roots = int(root["means"].shape[0])
    root_ids = torch.empty(int(state["num_nodes"]), dtype=torch.long)
    root_ids[:n_roots] = torch.arange(n_roots)
    group = {
        "means": 0,
        "quats": 1,
        "scales": 2,
        "opacities": 3,
        "sh0": 4,
        "shN": 4,
    }
    for bank in state["banks"]:
        parent_ids = bank["parent_ids"].long()
        child_ids = bank["child_ids"].long()
        alpha = bank["alpha"].float()
        delta = bank.get("delta")
        children: dict[str, torch.Tensor] = {}
        for key, all_values in values.items():
            parent = all_values.index_select(0, parent_ids)
            psi = bank["psi"][key].float()
            common = (
                delta[key].float()
                if isinstance(delta, dict) and key in delta
                else torch.zeros_like(psi)
            )
            coefficient = alpha[:, group[key]]
            coefficient = coefficient.reshape(
                (coefficient.shape[0],) + (1,) * (parent.ndim - 1)
            )
            child1 = parent + common + psi
            child2 = parent + common - coefficient * psi
            children[key] = torch.stack((child1, child2), dim=1).reshape(
                -1, *parent.shape[1:]
            )
            values[key] = torch.cat((all_values, children[key]), dim=0)
        parent_roots = root_ids.index_select(0, parent_ids)
        root_ids[child_ids.reshape(-1)] = parent_roots.repeat_interleave(2)

    frontier_ids = state["frontier_ids"].long()
    frontier = {
        key: value.index_select(0, frontier_ids) for key, value in values.items()
    }
    frontier_roots = root_ids.index_select(0, frontier_ids)
    expected = int(state["num_render_gaussians"])
    if frontier["means"].shape[0] != expected:
        raise ValueError("Composed frontier count does not match EVOGS metadata")
    return root, frontier, frontier_roots


def select_policy_frontier(
    root: dict[str, torch.Tensor],
    full: dict[str, torch.Tensor],
    full_root_ids: torch.Tensor,
    selected_cells: set[tuple[int, int, int]],
    cell_size_m: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    root_gsv = root["means"] @ EVOGS_TO_GSV.T
    cell_xyz = torch.floor(root_gsv / cell_size_m).to(torch.int64)
    selected_root = torch.tensor(
        [tuple(map(int, cell)) in selected_cells for cell in cell_xyz.tolist()],
        dtype=torch.bool,
    )
    selected_leaf = selected_root.index_select(0, full_root_ids)
    policy: dict[str, torch.Tensor] = {}
    for key in root:
        policy[key] = torch.cat(
            (root[key][~selected_root], full[key][selected_leaf]), dim=0
        )
    return policy, selected_root


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


def render_inputs(
    splats: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, ...]:
    means = (splats["means"] @ EVOGS_TO_GSV.T) * 100.0
    rotations = normalized_quat_to_rotmat(splats["quats"])
    scales = torch.exp(splats["scales"])
    covars = rotations @ torch.diag_embed(scales * scales) @ rotations.transpose(1, 2)
    transform = EVOGS_TO_GSV
    covars = transform @ covars @ transform.T
    covars = covars * 10000.0
    opacities = torch.sigmoid(splats["opacities"].reshape(-1))
    colors = 0.5 + SH_C0 * splats["sh0"].reshape(-1, 3)
    count = means.shape[0]
    dummy_quats = torch.zeros((count, 4), dtype=torch.float32)
    dummy_quats[:, 0] = 1.0
    dummy_scales = torch.ones((count, 3), dtype=torch.float32)
    return tuple(
        value.to(device, non_blocking=True)
        for value in (
            means,
            dummy_quats,
            dummy_scales,
            opacities,
            colors,
            covars,
        )
    )


def unreal_axes(
    roll_degrees: float, pitch_degrees: float, yaw_degrees: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roll, pitch, yaw = map(math.radians, (roll_degrees, pitch_degrees, yaw_degrees))
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    forward = np.array((cp * cy, cp * sy, sp), dtype=np.float32)
    right = np.array(
        (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp),
        dtype=np.float32,
    )
    up = np.array(
        (-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp),
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
    forward, right, up = unreal_axes(*trace["rotation"])
    camera_from_world = np.vstack((right, -up, forward)).astype(np.float32)
    location = np.asarray(trace["location"], dtype=np.float32)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = camera_from_world
    viewmat[:3, 3] = -(camera_from_world @ location)
    focal = width / (2.0 * math.tan(math.radians(hfov) * 0.5))
    intrinsics = np.array(
        ((focal, 0.0, width * 0.5), (0.0, focal, height * 0.5), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    return (
        torch.from_numpy(viewmat).unsqueeze(0).to(device),
        torch.from_numpy(intrinsics).unsqueeze(0).to(device),
    )


def render(
    rasterization: Any,
    inputs: tuple[torch.Tensor, ...],
    viewmat: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    near_cm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    means, quats, scales, opacities, colors, covars = inputs
    background = torch.tensor(
        ((8.0 / 255.0, 8.0 / 255.0, 10.0 / 255.0),),
        dtype=torch.float32,
        device=means.device,
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
    return image[0].permute(2, 0, 1).clamp(0.0, 1.0), alpha[0, ..., 0]


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


def metric_values(
    test: torch.Tensor,
    reference: torch.Tensor,
    lpips: LearnedPerceptualImagePatchSimilarity,
) -> dict[str, float]:
    return metric_values_batch([test], [reference], lpips)[0]


def metric_values_batch(
    tests: list[torch.Tensor],
    references: list[torch.Tensor],
    lpips: LearnedPerceptualImagePatchSimilarity,
) -> list[dict[str, float]]:
    """Compute independent image metrics in one GPU batch."""
    if not tests or len(tests) != len(references):
        raise ValueError("Metric batches must be non-empty and have equal lengths")
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
                "psnr_db": math.inf if mse == 0.0 else -10.0 * math.log10(mse),
                "ssim": float(ssim_tensor.item()),
                "lpips_alex": float(lpips_tensor.item()),
            }
        )
    lpips.reset()
    return result


def crop_from_alpha(alpha: torch.Tensor, pad: int = 24) -> tuple[slice, slice]:
    nonzero = torch.nonzero(alpha > (1.0 / 255.0), as_tuple=False)
    if nonzero.numel() == 0:
        return slice(0, alpha.shape[0]), slice(0, alpha.shape[1])
    y0 = max(0, int(nonzero[:, 0].min()) - pad)
    y1 = min(alpha.shape[0], int(nonzero[:, 0].max()) + pad + 1)
    x0 = max(0, int(nonzero[:, 1].min()) - pad)
    x1 = min(alpha.shape[1], int(nonzero[:, 1].max()) + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def save_sample(
    path: Path,
    reference: torch.Tensor,
    policy: torch.Tensor,
    frame_number: int,
    reference_label: str,
) -> None:
    reference_image = tensor_to_image(reference)
    policy_image = tensor_to_image(policy)
    difference = (reference - policy).abs().mul(8.0).clamp(0.0, 1.0)
    difference_image = tensor_to_image(difference)
    canvas = Image.new("RGB", (reference_image.width, reference_image.height * 3))
    canvas.paste(reference_image, (0, 0))
    canvas.paste(policy_image, (0, reference_image.height))
    canvas.paste(difference_image, (0, reference_image.height * 2))
    draw = ImageDraw.Draw(canvas)
    labels = (
        f"{reference_label} - frame {frame_number}",
        "Base + FoV-selected E3 policy",
        "Absolute RGB difference x8",
    )
    for row, label in enumerate(labels):
        y = row * reference_image.height + 12
        draw.rectangle((8, y - 4, 430, y + 24), fill=(0, 0, 0))
        draw.text((14, y), label, fill=(255, 255, 255))
    canvas.save(path)


def summarize(
    rows: list[dict[str, Any]],
    comparison: str,
    interpretation: str,
) -> dict[str, Any]:
    finite_psnr = [float(row["psnr_db"]) for row in rows if math.isfinite(row["psnr_db"])]
    mean_mse = float(np.mean([row["mse"] for row in rows]))
    crop_mean_mse = float(np.mean([row["crop_mse"] for row in rows]))
    affected = [row for row in rows if float(row["mse"]) > 0.0]
    summary = {
        "frame_count": len(rows),
        "affected_frame_count": len(affected),
        "comparison": comparison,
        "interpretation": interpretation,
        "full_frame": {
            "sequence_psnr_db": math.inf if mean_mse == 0.0 else -10.0 * math.log10(mean_mse),
            "mean_frame_psnr_db": float(np.mean(finite_psnr)) if finite_psnr else math.inf,
            "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
            "mean_lpips_alex": float(np.mean([row["lpips_alex"] for row in rows])),
            "mean_mse": mean_mse,
        },
        "reference_foreground_crop": {
            "sequence_psnr_db": (
                math.inf if crop_mean_mse == 0.0 else -10.0 * math.log10(crop_mean_mse)
            ),
            "mean_ssim": float(np.mean([row["crop_ssim"] for row in rows])),
            "mean_lpips_alex": float(np.mean([row["crop_lpips_alex"] for row in rows])),
            "mean_mse": crop_mean_mse,
        },
        "gaussians": {
            "mean_base_roots": float(np.mean([row["base_root_count"] for row in rows])),
            "mean_full_e3": float(np.mean([row["full_e3_gaussian_count"] for row in rows])),
            "mean_policy": float(np.mean([row["policy_gaussian_count"] for row in rows])),
            "mean_selected_root_fraction": float(
                np.mean([row["selected_root_fraction"] for row in rows])
            ),
        },
    }
    if affected:
        affected_mse = float(np.mean([row["mse"] for row in affected]))
        affected_crop_mse = float(np.mean([row["crop_mse"] for row in affected]))
        summary["affected_frames_only"] = {
            "sequence_psnr_db": -10.0 * math.log10(affected_mse),
            "mean_ssim": float(np.mean([row["ssim"] for row in affected])),
            "mean_lpips_alex": float(
                np.mean([row["lpips_alex"] for row in affected])
            ),
            "foreground_crop_sequence_psnr_db": -10.0
            * math.log10(affected_crop_mse),
            "foreground_crop_mean_ssim": float(
                np.mean([row["crop_ssim"] for row in affected])
            ),
            "foreground_crop_mean_lpips_alex": float(
                np.mean([row["crop_lpips_alex"] for row in affected])
            ),
        }
    return summary


def load_frame_assets(
    state_path: Path,
    gt_path: Path | None,
    load_ply_to_splats: Callable[[str], dict[str, torch.Tensor]] | None,
) -> dict[str, Any]:
    """Load one frame on CPU. Safe to execute in a bounded thread pool."""
    started = time.perf_counter()
    if not state_path.is_file():
        raise FileNotFoundError(f"No EVOGS E3 state: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    gt = None
    if gt_path is not None:
        if not gt_path.is_file():
            raise FileNotFoundError(f"No GT PLY: {gt_path}")
        if load_ply_to_splats is None:
            raise RuntimeError("PLY loader is unavailable")
        gt = load_ply_to_splats(str(gt_path))
    return {
        "state": state,
        "gt": gt,
        "load_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    if not args.decisions.is_file() or not args.trace.is_file():
        raise FileNotFoundError("Trace or decision CSV does not exist")
    if args.gt_root is not None and not args.gt_root.is_dir():
        raise FileNotFoundError(f"GT root does not exist: {args.gt_root}")
    if args.prefetch_workers < 0:
        raise ValueError("--prefetch-workers must be non-negative")
    if args.metric_batch_size <= 0:
        raise ValueError("--metric-batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.gsplat_library_path.resolve()))
    from gsplat.exporter import load_ply_to_splats
    from gsplat.rendering import rasterization

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    trace_rows = load_trace(args.trace)
    decisions = load_decisions(args.decisions)
    if args.frame_count is not None:
        if args.frame_count <= 0:
            raise ValueError("--frame-count must be positive")
        decisions = decisions[: args.frame_count]
    if not decisions:
        raise ValueError("No decision frames were selected")

    asset_specs: list[dict[str, Any]] = []
    for decision in decisions:
        asset_id = decision["gsv_frame"] + args.asset_frame_offset
        state_path = (
            args.evogs_root
            / f"{asset_id:07d}"
            / "enhancement_03_enhanced"
            / "ply"
            / "evo_refinement.pt"
        )
        gt_path = (
            args.gt_root / f"{asset_id:07d}.ply"
            if args.gt_root is not None
            else None
        )
        if not state_path.is_file():
            raise FileNotFoundError(
                f"No EVOGS E3 state for GSV {decision['gsv_frame']}: {state_path}"
            )
        if gt_path is not None and not gt_path.is_file():
            raise FileNotFoundError(
                f"No GT PLY for GSV {decision['gsv_frame']}: {gt_path}"
            )
        asset_specs.append(
            {
                "asset_id": asset_id,
                "state_path": state_path,
                "gt_path": gt_path,
            }
        )

    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="none"
    ).to(device).eval()
    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    timing = {
        "asset_load_cpu_work_seconds": 0.0,
        "asset_wait_wall_seconds": 0.0,
        "compose_policy_cpu_seconds": 0.0,
        "render_input_cpu_seconds": 0.0,
        "reference_render_seconds": 0.0,
        "policy_render_seconds": 0.0,
        "metric_seconds": 0.0,
        "sample_write_seconds": 0.0,
        "output_write_seconds": 0.0,
    }
    use_gt = args.gt_root is not None
    reference_label = "DanceNet3D GT" if use_gt else "Full E3 reference"
    comparison = (
        "test=Base+FoV-selected E3; reference=DanceNet3D 3DGS GT"
        if use_gt
        else "test=Base+FoV-selected E3; reference=Full E3"
    )
    interpretation = (
        "absolute reconstruction quality against the supplied DanceNet3D GT 3DGS"
        if use_gt
        else "quality penalty caused only by withheld E3 descendants"
    )

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def flush_metrics() -> None:
        if not pending:
            return
        synchronize()
        metric_started = time.perf_counter()
        with torch.inference_mode():
            full_metrics = metric_values_batch(
                [item["test"] for item in pending],
                [item["reference"] for item in pending],
                lpips,
            )
            crop_metrics = [
                metric_values(
                    item["test"][:, item["crop_y"], item["crop_x"]],
                    item["reference"][:, item["crop_y"], item["crop_x"]],
                    lpips,
                )
                for item in pending
            ]
        synchronize()
        timing["metric_seconds"] += time.perf_counter() - metric_started
        for item, metrics, crop in zip(
            pending, full_metrics, crop_metrics, strict=True
        ):
            row = {
                **item["metadata"],
                **metrics,
                "crop_mse": crop["mse"],
                "crop_psnr_db": crop["psnr_db"],
                "crop_ssim": crop["ssim"],
                "crop_lpips_alex": crop["lpips_alex"],
            }
            rows.append(row)
            print(
                f"[{row['display_frame']:02d}/{len(decisions)}] "
                f"GSV {row['gsv_frame']:02d} "
                f"PSNR {metrics['psnr_db']:.3f} dB | "
                f"SSIM {metrics['ssim']:.5f} | "
                f"LPIPS {metrics['lpips_alex']:.5f}",
                flush=True,
            )
        pending.clear()

    started = time.perf_counter()
    executor: ThreadPoolExecutor | None = None
    futures: dict[int, Future[dict[str, Any]]] = {}
    prefetch_window = max(1, args.prefetch_workers + 1)
    if args.prefetch_workers > 0:
        executor = ThreadPoolExecutor(
            max_workers=args.prefetch_workers,
            thread_name_prefix="qoe-asset",
        )
        for index in range(min(prefetch_window, len(decisions))):
            spec = asset_specs[index]
            futures[index] = executor.submit(
                load_frame_assets,
                spec["state_path"],
                spec["gt_path"],
                load_ply_to_splats if use_gt else None,
            )

    try:
        for index, (decision, spec) in enumerate(
            zip(decisions, asset_specs, strict=True)
        ):
            position = index + 1
            source_row = decision["trace_source_row"]
            trace = trace_rows[source_row]
            if trace["gsv_frame"] != decision["gsv_frame"]:
                raise ValueError(
                    f"Trace/GSV mismatch at output frame {decision['output_frame']}"
                )

            wait_started = time.perf_counter()
            if executor is None:
                assets = load_frame_assets(
                    spec["state_path"],
                    spec["gt_path"],
                    load_ply_to_splats if use_gt else None,
                )
            else:
                assets = futures.pop(index).result()
                next_index = index + prefetch_window
                if next_index < len(decisions):
                    next_spec = asset_specs[next_index]
                    futures[next_index] = executor.submit(
                        load_frame_assets,
                        next_spec["state_path"],
                        next_spec["gt_path"],
                        load_ply_to_splats if use_gt else None,
                    )
            timing["asset_wait_wall_seconds"] += time.perf_counter() - wait_started
            timing["asset_load_cpu_work_seconds"] += assets["load_seconds"]

            compose_started = time.perf_counter()
            root, full, full_root_ids = compose_evogs_state(assets["state"])
            policy, selected_root = select_policy_frontier(
                root,
                full,
                full_root_ids,
                decision["selected_cells"],
                args.cell_size_m,
            )
            reference_splats = assets["gt"] if use_gt else full
            if reference_splats is None:
                raise RuntimeError("Reference splats were not loaded")
            viewmat, intrinsics = camera_tensors(
                trace, args.width, args.height, args.hfov, device
            )
            timing["compose_policy_cpu_seconds"] += (
                time.perf_counter() - compose_started
            )

            input_started = time.perf_counter()
            reference_inputs = render_inputs(reference_splats, device)
            policy_inputs = render_inputs(policy, device)
            timing["render_input_cpu_seconds"] += (
                time.perf_counter() - input_started
            )

            with torch.inference_mode():
                synchronize()
                render_started = time.perf_counter()
                reference, alpha = render(
                    rasterization,
                    reference_inputs,
                    viewmat,
                    intrinsics,
                    args.width,
                    args.height,
                    args.near_cm,
                )
                synchronize()
                timing["reference_render_seconds"] += (
                    time.perf_counter() - render_started
                )

                render_started = time.perf_counter()
                test, _ = render(
                    rasterization,
                    policy_inputs,
                    viewmat,
                    intrinsics,
                    args.width,
                    args.height,
                    args.near_cm,
                )
                synchronize()
                timing["policy_render_seconds"] += (
                    time.perf_counter() - render_started
                )
                crop_y, crop_x = crop_from_alpha(alpha)

            if args.save_every > 0 and (
                position == 1
                or position == len(decisions)
                or (position - 1) % args.save_every == 0
            ):
                sample_started = time.perf_counter()
                save_sample(
                    args.output_dir / f"sample_{position:03d}.png",
                    reference,
                    test,
                    position,
                    reference_label,
                )
                timing["sample_write_seconds"] += (
                    time.perf_counter() - sample_started
                )

            pending.append(
                {
                    "reference": reference,
                    "test": test,
                    "crop_y": crop_y,
                    "crop_x": crop_x,
                    "metadata": {
                        "display_frame": position,
                        "output_frame": decision["output_frame"],
                        "trace_source_row": source_row,
                        "trace_timestamp_s": trace["timestamp"],
                        "gsv_frame": decision["gsv_frame"],
                        "evogs_asset_id": spec["asset_id"],
                        "reference_type": "dancenet3d_gt" if use_gt else "full_e3",
                        "selected_cells": len(decision["selected_cells"]),
                        "occupied_cells": decision["occupied_cells"],
                        "base_root_count": int(root["means"].shape[0]),
                        "selected_root_count": int(selected_root.sum()),
                        "selected_root_fraction": float(
                            selected_root.float().mean()
                        ),
                        "full_e3_gaussian_count": int(full["means"].shape[0]),
                        "policy_gaussian_count": int(policy["means"].shape[0]),
                        "reference_gaussian_count": int(
                            reference_splats["means"].shape[0]
                        ),
                    },
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
    total_seconds = time.perf_counter() - started
    timing["output_write_seconds"] += time.perf_counter() - output_started
    summary = summarize(rows, comparison, interpretation)
    summary["runtime_seconds"] = total_seconds
    summary["throughput_frames_per_second"] = len(rows) / total_seconds
    summary["timing"] = timing
    summary["settings"] = {
        "width": args.width,
        "height": args.height,
        "hfov_degrees": args.hfov,
        "cell_size_m": args.cell_size_m,
        "background_rgb": [8, 8, 10],
        "colors": "SH0 only, matching the GSV replay renderer",
        "lpips_backbone": "alex",
        "reference_type": "dancenet3d_gt" if use_gt else "full_e3",
        "gt_root": str(args.gt_root.resolve()) if use_gt else None,
        "evogs_root": str(args.evogs_root.resolve()),
        "asset_frame_offset": args.asset_frame_offset,
        "prefetch_workers": args.prefetch_workers,
        "metric_batch_size": args.metric_batch_size,
        "lut_enabled": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
