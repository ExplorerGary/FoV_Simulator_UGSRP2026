#!/usr/bin/env python3
"""Aggregate the two held-out LR QoE summaries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    traces = ("26_7_29_12_37_21", "26_7_31_15_1_21")
    summaries = []
    for trace in traces:
        path = args.root / trace / "02_bandwidth_qoe" / "summary.json"
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    fields = {
        "policy_vs_gt_psnr_db": [s["comparisons"]["policy_vs_gt"]["full"]["sequence_psnr_db"] for s in summaries],
        "policy_vs_gt_ssim": [s["comparisons"]["policy_vs_gt"]["full"]["mean_ssim"] for s in summaries],
        "policy_vs_gt_lpips_alex": [s["comparisons"]["policy_vs_gt"]["full"]["mean_lpips_alex"] for s in summaries],
        "policy_megabits_per_second": [s["bandwidth"]["mean_policy_megabits_per_second"] for s in summaries],
        "bandwidth_savings_vs_full_fraction": [s["bandwidth"]["policy_savings_vs_full_fraction"] for s in summaries],
    }
    result = {
        "status": "PASS", "mechanism": "500ms_dof_only_linear_regression",
        "traces": list(traces),
        "mean": {key: statistics.fmean(values) for key, values in fields.items()},
        "per_trace": {trace: summary for trace, summary in zip(traces, summaries)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
