#!/usr/bin/env python3
"""Plot trace-driven bandwidth and QoE metrics from one evaluator CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PAIR_LABELS = {
    "e3_vs_gt": "Full E3 vs GT",
    "policy_vs_gt": "FoV policy vs GT",
    "policy_vs_e3": "FoV policy vs Full E3",
}
PAIR_COLORS = {
    "e3_vs_gt": "#6B7280",
    "policy_vs_gt": "#2878B5",
    "policy_vs_e3": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="FoV-driven progressive Gaussian coding",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Sampling rate for converting bytes/frame to Mbit/s",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def finite_float(value: str | None) -> float:
    """Parse one optional metric cell for plotting.

    Foreground metrics are intentionally blank when the reference alpha mask
    is empty. Infinite PSNR is also valid for identical images. Neither value
    should abort the plot; matplotlib represents both as a line gap.
    """
    if value is None or not value.strip():
        return math.nan
    result = float(value)
    return result if math.isfinite(result) else math.nan


def main() -> None:
    args = parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("This plot requires matplotlib") from exc

    with args.metrics.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Metrics CSV has no rows")

    required = {
        "trace_timestamp_s",
        "output_time_s",
        "base_only_transmission_bytes",
        "policy_transmission_bytes",
        "full_progressive_transmission_bytes",
    }
    for pair in PAIR_LABELS:
        for region in ("full", "fore"):
            for metric in ("psnr_db", "ssim", "lpips_alex"):
                required.add(f"{pair}_{region}_{metric}")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(
            "Metrics CSV predates bandwidth/QoE plotting fields: "
            + ", ".join(missing)
        )

    absolute_time = [float(row["trace_timestamp_s"]) for row in rows]
    t0 = absolute_time[0]
    time_s = [value - t0 for value in absolute_time]
    if args.fps is not None:
        sample_fps = args.fps
    else:
        output_times = [float(row["output_time_s"]) for row in rows]
        deltas = [
            after - before
            for before, after in zip(output_times, output_times[1:])
            if after > before
        ]
        sample_fps = (
            1.0 / sorted(deltas)[len(deltas) // 2] if deltas else None
        )
    scale = (
        8.0 * sample_fps / 1_000_000.0
        if sample_fps is not None
        else 1.0 / (1024.0 * 1024.0)
    )
    bandwidth_series = (
        (
            "Base only",
            [
                float(row["base_only_transmission_bytes"]) * scale
                for row in rows
            ],
            "#B8C2CC",
        ),
        (
            "FoV policy",
            [
                float(row["policy_transmission_bytes"]) * scale
                for row in rows
            ],
            "#2878B5",
        ),
        (
            "Base + all E3",
            [
                float(row["full_progressive_transmission_bytes"]) * scale
                for row in rows
            ],
            "#F28E2B",
        ),
    )

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(15.0, 13.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0]},
    )
    for label, values, color in bandwidth_series:
        axes[0].plot(time_s, values, label=label, color=color, linewidth=1.25)
    axes[0].set_ylabel(
        (
            f"Application payload (Mbit/s @ {sample_fps:.3f} fps)"
            if sample_fps is not None
            else "Application payload (MiB/frame)"
        )
    )
    axes[0].set_title(args.title)
    axes[0].legend(frameon=False, ncols=3)

    metric_specs = (
        ("psnr_db", "PSNR (dB)", axes[1]),
        ("ssim", "SSIM (higher is better)", axes[2]),
        ("lpips_alex", "LPIPS-Alex (lower is better)", axes[3]),
    )
    for metric, ylabel, axis in metric_specs:
        for pair, label in PAIR_LABELS.items():
            color = PAIR_COLORS[pair]
            for region, style, suffix in (
                ("full", "-", "full"),
                ("fore", "--", "foreground"),
            ):
                values = [
                    finite_float(row[f"{pair}_{region}_{metric}"])
                    for row in rows
                ]
                axis.plot(
                    time_s,
                    values,
                    linestyle=style,
                    color=color,
                    linewidth=1.1,
                    alpha=0.95,
                    label=f"{label} ({suffix})",
                )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.2)

    axes[1].legend(
        frameon=False,
        ncols=2,
        fontsize=8,
        loc="best",
    )
    axes[3].set_xlabel("Trace time since evaluated interval start (s)")
    for axis in axes:
        axis.margins(x=0.0)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote plot: {args.output}")


if __name__ == "__main__":
    main()
