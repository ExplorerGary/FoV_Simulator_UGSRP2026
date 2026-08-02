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


def gt_bytes(
    decisions: Path, gt_roots: list[Path]
) -> tuple[int, set[int]]:
    assets: dict[int, int] = {}
    with decisions.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            assets.setdefault(int(row["output_frame"]), int(row["asset_frame_id"]))
    total = 0
    missing: set[int] = set()
    for asset in assets.values():
        candidates = [root / f"{asset:07d}.ply" for root in gt_roots]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            missing.add(asset)
        else:
            total += path.stat().st_size
    return total, missing


def policy_cell_metrics(decisions: Path, visibility: Path) -> dict[str, float | int]:
    fractions: dict[tuple[int, str], float] = {}
    with visibility.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["source_output_frame"]), row["cell_id"].strip())
            fractions[key] = float(row["contributing_gaussian_fraction"])
    tp = fp = fn = tn = samples = 0
    squared_error = 0.0
    target_sum = 0.0
    target_squared_sum = 0.0
    with decisions.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["source_output_frame"]), row["cell_id"].strip())
            if key not in fractions:
                raise ValueError(
                    "Decision has no source-frame-aligned visibility label: "
                    f"source_output_frame={key[0]}, cell_id={key[1]!r}"
                )
            fraction = fractions[key]
            expected = fraction >= 0.5
            predicted = row["target_level"].strip().lower() in {"3", "e3"}
            score = float(row["predicted_visibility_score"])
            squared_error += (score - fraction) ** 2
            target_sum += fraction
            target_squared_sum += fraction**2
            samples += 1
            if predicted and expected:
                tp += 1
            elif predicted:
                fp += 1
            elif expected:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if 4 * precision + recall else 0.0
    )
    return {
        "samples": samples,
        "squared_error": squared_error,
        "target_sum": target_sum,
        "target_squared_sum": target_squared_sum,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path)
    parser.add_argument("--visibility-root", type=Path)
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
        policy_points = full_points = gt_points = 0
        cell_samples = cell_tp = cell_fp = cell_fn = cell_tn = 0
        cell_squared_error = 0.0
        cell_target_sum = cell_target_squared_sum = 0.0
        selected_weighted = full_mse_weighted = ssim_weighted = lpips_weighted = 0.0
        e3_mse_weighted = 0.0
        fore_pixels = 0
        fore_error_sum = fore_ssim_weighted = fore_lpips_weighted = 0.0
        trace_details = {}
        missing_gt_assets: set[int] = set()
        for trace in TRACES:
            trace_root = args.root / variant / trace
            qoe = json.loads((trace_root / "02_bandwidth_qoe" / "summary.json").read_text())
            policy = json.loads((trace_root / "01_policy" / "prediction_policy_summary.json").read_text())
            frames = int(qoe["frame_count"])
            comparison = qoe["comparisons"]["policy_vs_gt"]
            e3 = qoe["comparisons"]["e3_vs_gt"]
            bandwidth = qoe["bandwidth"]
            metrics_path = trace_root / "02_bandwidth_qoe" / "per_frame_metrics.csv"
            with metrics_path.open("r", encoding="utf-8-sig", newline="") as stream:
                for metric_row in csv.DictReader(stream):
                    policy_points += int(float(metric_row["policy_gaussian_count"]))
                    full_points += int(float(metric_row["full_e3_gaussian_count"]))
                    gt_points += int(float(metric_row["gt_gaussian_count"]))
            frame_count += frames
            policy_bytes += int(bandwidth["total_policy_bytes"])
            full_bytes += int(bandwidth["total_full_progressive_bytes"])
            recorded_gt_root = Path(qoe["settings"]["gt_root"])
            gt_roots = [recorded_gt_root]
            if args.gt_root is not None:
                gt_roots.extend(
                    [args.gt_root, args.gt_root / "BiancaGolden_CircleTurns"]
                )
            trace_gt_bytes, trace_missing = gt_bytes(
                trace_root / "01_policy" / "cell_decisions.csv", gt_roots
            )
            raw_gt_bytes += trace_gt_bytes
            missing_gt_assets.update(trace_missing)
            if args.visibility_root is not None:
                cell = policy_cell_metrics(
                    trace_root / "01_policy" / "cell_decisions.csv",
                    args.visibility_root / f"{trace}.csv",
                )
                cell_samples += int(cell["samples"])
                cell_squared_error += float(cell["squared_error"])
                cell_target_sum += float(cell["target_sum"])
                cell_target_squared_sum += float(cell["target_squared_sum"])
                cell_tp += int(cell["true_positive"])
                cell_fp += int(cell["false_positive"])
                cell_fn += int(cell["false_negative"])
                cell_tn += int(cell["true_negative"])
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
        cell_precision = cell_tp / (cell_tp + cell_fp) if cell_tp + cell_fp else 0.0
        cell_recall = cell_tp / (cell_tp + cell_fn) if cell_tp + cell_fn else 0.0
        cell_f1 = (
            2 * cell_precision * cell_recall / (cell_precision + cell_recall)
            if cell_precision + cell_recall else 0.0
        )
        cell_f2 = (
            5 * cell_precision * cell_recall / (4 * cell_precision + cell_recall)
            if 4 * cell_precision + cell_recall else 0.0
        )
        cell_target_sst = (
            cell_target_squared_sum - cell_target_sum**2 / cell_samples
            if cell_samples else 0.0
        )
        row = {
            "variant": variant,
            "frames": frame_count,
            "mean_selected_cells": selected_weighted / frame_count,
            "policy_mbps": policy_bytes / frame_count * 30 * 8 / 1_000_000,
            "savings_vs_full_e3": 1.0 - policy_bytes / full_bytes,
            "savings_vs_dancenet3d_gt": (
                None
                if missing_gt_assets or raw_gt_bytes == 0
                else 1.0 - policy_bytes / raw_gt_bytes
            ),
            "mean_policy_gaussians": policy_points / frame_count,
            "mean_full_e3_gaussians": full_points / frame_count,
            "mean_gt_gaussians": gt_points / frame_count,
            "point_savings_vs_full_e3": 1.0 - policy_points / full_points,
            "point_savings_vs_dancenet3d_gt": 1.0 - policy_points / gt_points,
            "cell_mse": (
                cell_squared_error / cell_samples if cell_samples else None
            ),
            "cell_r2": (
                1.0 - cell_squared_error / cell_target_sst
                if cell_target_sst > 1.0e-12 else None
            ),
            "cell_precision": cell_precision if cell_samples else None,
            "cell_recall": cell_recall if cell_samples else None,
            "cell_f1": cell_f1 if cell_samples else None,
            "cell_f2": cell_f2 if cell_samples else None,
            "gt_size_status": "incomplete" if missing_gt_assets else "complete",
            "missing_gt_asset_ids": " ".join(
                f"{asset:07d}" for asset in sorted(missing_gt_assets)
            ),
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
