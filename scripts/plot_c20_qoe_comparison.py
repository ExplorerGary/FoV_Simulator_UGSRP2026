#!/usr/bin/env python3
"""Plot matched Full Base, Full E3, and Policy QoE for the fixed C20 trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRACE = "26_7_29_12_37_21"
REPRESENTATIONS = (
    ("Full Base", "#7F7F7F"),
    ("Full E3", "#E3B341"),
    ("Policy", "#08519C"),
)
PANELS = (
    ("PSNR", "dB", "psnr", 25.0, 31.0, (25.0, 27.0, 29.0, 31.0), 3),
    ("SSIM", "", "ssim", 0.96, 0.985, (0.96, 0.97, 0.98), 4),
    ("LPIPS-Alex", "lower is better", "lpips", 0.0, 0.055, (0.0, 0.02, 0.04), 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/hpc_dof_lr_c20")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hpc_dof_lr_c20/qoe_comparison.svg"),
    )
    return parser.parse_args()


def load_metrics(root: Path) -> tuple[dict[str, dict[str, float]], int]:
    base_path = (
        root / "matched_base_only" / TRACE / "02_bandwidth_qoe" / "summary.json"
    )
    policy_path = (
        root / "dof_lr_c20" / TRACE / "02_bandwidth_qoe" / "summary.json"
    )
    base = json.loads(base_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    comparisons = {
        "Full Base": base["comparisons"]["policy_vs_gt"]["full"],
        "Full E3": policy["comparisons"]["e3_vs_gt"]["full"],
        "Policy": policy["comparisons"]["policy_vs_gt"]["full"],
    }
    frame_counts = {int(value["frame_count"]) for value in comparisons.values()}
    if len(frame_counts) != 1:
        raise ValueError(f"QoE frame counts are not matched: {sorted(frame_counts)}")
    metrics = {
        name: {
            "psnr": float(value["sequence_psnr_db"]),
            "ssim": float(value["mean_ssim"]),
            "lpips": float(value["mean_lpips_alex"]),
        }
        for name, value in comparisons.items()
    }
    return metrics, frame_counts.pop()


def main() -> None:
    args = parse_args()
    metrics, frame_count = load_metrics(args.root)
    width, height = 1800, 650
    outer_left, outer_right = 70, 45
    panel_gap = 70
    panel_width = (width - outer_left - outer_right - 2 * panel_gap) / 3
    plot_top, plot_bottom = 170, 545
    plot_height = plot_bottom - plot_top

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<title>QoE Comparison</title>',
        f'<desc>Matched QoE comparison for trace {TRACE} across Full Base, Full E3, and Policy.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:30px;font-weight:600}.subtitle{font-size:16px}.panel{font-size:21px;font-weight:600}.axis{font-size:14px}.value{font-size:15px;font-weight:600}.legend{font-size:15px}.grid{stroke:#d9dde3;stroke-width:1}.frame{stroke:#60656d;stroke-width:1.2;fill:none}</style>',
        f'<text class="title" x="{width/2}" y="42" text-anchor="middle">QoE Comparison</text>',
        f'<text class="subtitle" x="{width/2}" y="70" text-anchor="middle">Trace {TRACE} · {frame_count:,} matched frames · reference: DanceNet3D GT</text>',
    ]

    legend_width = 220
    legend_start = width / 2 - legend_width * len(REPRESENTATIONS) / 2
    for index, (name, color) in enumerate(REPRESENTATIONS):
        x = legend_start + index * legend_width
        svg.append(f'<rect x="{x}" y="92" width="22" height="14" fill="{color}"/>')
        svg.append(f'<text class="legend" x="{x+31}" y="104">{name}</text>')

    for panel_index, (title, note, key, y_min, y_max, ticks, decimals) in enumerate(PANELS):
        panel_left = outer_left + panel_index * (panel_width + panel_gap)
        panel_right = panel_left + panel_width
        axis_left = panel_left + 68
        axis_right = panel_right - 20
        axis_width = axis_right - axis_left
        y_map = lambda value: plot_bottom - (value - y_min) / (y_max - y_min) * plot_height

        svg.append(f'<text class="panel" x="{(panel_left+panel_right)/2}" y="142" text-anchor="middle">{title}</text>')
        if note:
            svg.append(f'<text class="axis" x="{(panel_left+panel_right)/2}" y="161" text-anchor="middle">{note}</text>')
        for tick in ticks:
            y = y_map(tick)
            label = f"{tick:.{decimals}f}" if decimals else str(tick)
            svg.append(f'<line class="grid" x1="{axis_left}" y1="{y:.2f}" x2="{axis_right}" y2="{y:.2f}"/>')
            svg.append(f'<text class="axis" x="{axis_left-10}" y="{y+5:.2f}" text-anchor="end">{label}</text>')
        svg.append(f'<rect class="frame" x="{axis_left}" y="{plot_top}" width="{axis_width}" height="{plot_height}"/>')

        slot_width = axis_width / len(REPRESENTATIONS)
        bar_width = slot_width * 0.55
        for index, (name, color) in enumerate(REPRESENTATIONS):
            value = metrics[name][key]
            x = axis_left + index * slot_width + (slot_width - bar_width) / 2
            y = y_map(value)
            height_value = plot_bottom - y
            svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{height_value:.2f}" fill="{color}"/>')
            svg.append(f'<text class="value" x="{x+bar_width/2:.2f}" y="{y-10:.2f}" text-anchor="middle">{value:.{decimals}f}</text>')
            svg.append(f'<text class="axis" x="{x+bar_width/2:.2f}" y="{plot_bottom+25}" text-anchor="middle">{name}</text>')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join([*svg, "</svg>"]), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
