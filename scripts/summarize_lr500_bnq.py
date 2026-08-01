#!/usr/bin/env python3
"""Aggregate the complete 500 ms LR bandwidth-and-QoE sweep."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

TRACES = ("26_7_29_12_37_21", "26_7_31_15_1_21")
VARIANTS = (
    "base_only", "persistence", "lr_v1_t020", "lr_v2_t010",
    "lr_v2_t015", "lr_v2_t020", "lr_v2_t025", "lr_v2_t020_guard6",
)


def gt_bytes(decisions: Path, gt_root: Path) -> int:
    assets: dict[int, int] = {}
    with decisions.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            assets.setdefault(int(row["output_frame"]), int(row["asset_frame_id"]))
    return sum((gt_root / f"{asset:07d}.ply").stat().st_size for asset in assets.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument(
        "--experiment",
        default="500ms motion-aware quadratic Ridge LR BNQ sweep",
    )
    args = parser.parse_args()
    rows = []
    details: dict[str, object] = {}
    for variant in args.variants:
        frame_count = policy_bytes = full_bytes = raw_gt_bytes = 0
        selected_weighted = full_mse_weighted = ssim_weighted = lpips_weighted = 0.0
        e3_mse_weighted = 0.0
        fore_pixels = 0
        fore_error_sum = fore_ssim_weighted = fore_lpips_weighted = 0.0
        trace_details = {}
        for trace in TRACES:
            trace_root = args.root / variant / trace
            qoe = json.loads((trace_root / "02_bandwidth_qoe" / "summary.json").read_text())
            policy = json.loads((trace_root / "01_policy" / "prediction_policy_summary.json").read_text())
            frames = int(qoe["frame_count"])
            comparison = qoe["comparisons"]["policy_vs_gt"]
            e3 = qoe["comparisons"]["e3_vs_gt"]
            bandwidth = qoe["bandwidth"]
            frame_count += frames
            policy_bytes += int(bandwidth["total_policy_bytes"])
            full_bytes += int(bandwidth["total_full_progressive_bytes"])
            raw_gt_bytes += gt_bytes(
                trace_root / "01_policy" / "cell_decisions.csv", args.gt_root
            )
            selected_weighted += float(policy["mean_selected_cells"]) * frames
            full_mse_weighted += float(comparison["full"]["mean_mse"]) * frames
            e3_mse_weighted += float(e3["full"]["mean_mse"]) * frames
            ssim_weighted += float(comparison["full"]["mean_ssim"]) * frames
            lpips_weighted += float(comparison["full"]["mean_lpips_alex"]) * frames
            foreground = comparison["foreground"]
            pixels = int(foreground.get("foreground_pixel_count", 0))
            if pixels:
                fore_pixels += pixels
                fore_error_sum += float(foreground["pixel_weighted_mse"]) * pixels
                fore_ssim_weighted += float(foreground["mean_ssim"]) * pixels
                fore_lpips_weighted += float(foreground["mean_lpips_alex"]) * pixels
            trace_details[trace] = {"policy": policy, "qoe": qoe}
        mse = full_mse_weighted / frame_count
        e3_mse = e3_mse_weighted / frame_count
        row = {
            "variant": variant,
            "frames": frame_count,
            "mean_selected_cells": selected_weighted / frame_count,
            "policy_mbps": policy_bytes / frame_count * 30 * 8 / 1_000_000,
            "savings_vs_full_e3": 1.0 - policy_bytes / full_bytes,
            "savings_vs_dancenet3d_gt": 1.0 - policy_bytes / raw_gt_bytes,
            "policy_vs_gt_psnr_db": -10.0 * math.log10(mse),
            "full_e3_vs_gt_psnr_db": -10.0 * math.log10(e3_mse),
            "psnr_delta_vs_full_e3_db": -10.0 * math.log10(mse) + 10.0 * math.log10(e3_mse),
            "policy_vs_gt_ssim": ssim_weighted / frame_count,
            "policy_vs_gt_lpips_alex": lpips_weighted / frame_count,
            "foreground_psnr_db": -10.0 * math.log10(fore_error_sum / fore_pixels),
            "foreground_ssim": fore_ssim_weighted / fore_pixels,
            "foreground_lpips_alex": fore_lpips_weighted / fore_pixels,
        }
        rows.append(row)
        details[variant] = trace_details
    output = {
        "status": "PASS",
        "experiment": args.experiment,
        "aggregate": rows,
        "per_trace": details,
    }
    (args.root / "bnq_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (args.root / "bnq_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
