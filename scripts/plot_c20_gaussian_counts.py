#!/usr/bin/env python3
"""Plot per-frame Gaussian counts for the fixed C20 reporting trace."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


TRACE = "26_7_29_12_37_21"
SERIES = (
    ("gt_gaussian_count", "DanceNet3D GT", "#D62728", ""),
    ("base_gaussian_count", "Full Base", "#7F7F7F", ""),
    ("full_e3_gaussian_count", "Full E3", "#E3B341", "8 5"),
    ("policy_gaussian_count", "Policy", "#08519C", ""),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/hpc_dof_lr_c20/dof_lr_c20"),
        help="Directory containing the fixed C20 trace",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hpc_dof_lr_c20/gaussian_count_over_time.svg"),
    )
    return parser.parse_args()


def load_trace(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Metrics CSV is empty: {path}")
    required = {"output_time_s", *(name for name, _, _, _ in SERIES)}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Metrics CSV is missing columns {missing}: {path}")
    return rows


def main() -> None:
    args = parse_args()
    metrics = args.root / TRACE / "02_bandwidth_qoe" / "per_frame_metrics.csv"
    rows = load_trace(metrics)
    times = [float(row["output_time_s"]) for row in rows]
    times = [value - times[0] for value in times]
    x_max = max(times) or 1.0

    global_max = max(
        int(float(row[column]))
        for row in rows
        for column, _, _, _ in SERIES
    )
    tick_step = 10000
    y_max = max(tick_step, math.ceil(global_max / tick_step) * tick_step)

    width, height = 2000, 650
    left, right, top, bottom = 115, 55, 112, 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    plot_bottom = top + plot_height
    x_map = lambda value: left + value / x_max * plot_width
    y_map = lambda value: plot_bottom - value / y_max * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<title>Gaussian Count Over Time</title>',
        f'<desc>Per-frame Gaussian counts for trace {TRACE}, comparing DanceNet3D GT, Full Base, Full E3, and Policy.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:28px;font-weight:600}.subtitle{font-size:16px}.axis{font-size:15px}.legend{font-size:15px}.grid{stroke:#d9dde3;stroke-width:1}.frame{stroke:#60656d;stroke-width:1.2;fill:none}</style>',
        f'<text class="title" x="{width/2}" y="40" text-anchor="middle">Gaussian Count Over Time</text>',
        f'<text class="subtitle" x="{width/2}" y="68" text-anchor="middle">Trace {TRACE} · {len(rows):,} frames</text>',
    ]

    for value in range(0, y_max + 1, tick_step):
        y = y_map(value)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        svg.append(f'<text class="axis" x="{left-14}" y="{y+5:.2f}" text-anchor="end">{value/1000:g}k</text>')
    for fraction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        value = fraction * x_max
        x = x_map(value)
        svg.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{plot_bottom}"/>')
        svg.append(f'<text class="axis" x="{x:.2f}" y="{plot_bottom+27}" text-anchor="middle">{value:.0f}</text>')
    svg.append(f'<rect class="frame" x="{left}" y="{top}" width="{plot_width}" height="{plot_height}"/>')

    legend_width = 270
    legend_start = width / 2 - legend_width * len(SERIES) / 2
    for index, (column, label, color, dash) in enumerate(SERIES):
        values = [int(float(row[column])) for row in rows]
        mean = sum(values) / len(values)
        points = " ".join(
            f"{x_map(time):.2f},{y_map(value):.2f}"
            for time, value in zip(times, values, strict=True)
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash_attr} stroke-linejoin="round"/>'
        )
        legend_x = legend_start + index * legend_width
        legend_y = 92
        svg.append(
            f'<line x1="{legend_x}" y1="{legend_y-5}" x2="{legend_x+32}" y2="{legend_y-5}" stroke="{color}" stroke-width="3"{dash_attr}/>'
        )
        svg.append(
            f'<text class="legend" x="{legend_x+41}" y="{legend_y}">{label} (mean {mean:,.0f})</text>'
        )

    svg.append(
        f'<text class="axis" x="{left+plot_width/2}" y="{height-22}" text-anchor="middle">Elapsed time (s)</text>'
    )
    center_y = top + plot_height / 2
    svg.append(
        f'<text class="axis" x="27" y="{center_y}" text-anchor="middle" transform="rotate(-90 27 {center_y})">Gaussian count</text>'
    )
    svg.append('</svg>')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
