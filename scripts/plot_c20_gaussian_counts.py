#!/usr/bin/env python3
"""Plot per-frame Full Base, Full E3, and C20 policy Gaussian counts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


TRACES = ("26_7_29_12_37_21", "26_7_31_15_1_21")
SERIES = (
    ("base_gaussian_count", "Full Base", "#4C78A8"),
    ("full_e3_gaussian_count", "Full E3", "#E45756"),
    ("policy_gaussian_count", "Policy", "#59A14F"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/hpc_dof_lr_c20/dof_lr_c20"),
        help="Directory containing one subdirectory per C20 trace",
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
    required = {"output_time_s", *(name for name, _, _ in SERIES)}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Metrics CSV is missing columns {missing}: {path}")
    return rows


def main() -> None:
    args = parse_args()
    datasets: list[tuple[str, list[dict[str, str]]]] = []
    global_max = 0
    for trace in TRACES:
        metrics = args.root / trace / "02_bandwidth_qoe" / "per_frame_metrics.csv"
        rows = load_trace(metrics)
        datasets.append((trace, rows))
        global_max = max(
            global_max,
            *(int(float(row[column])) for row in rows for column, _, _ in SERIES),
        )

    width, height = 1600, 920
    left, right, top, bottom = 105, 45, 112, 62
    panel_gap = 82
    panel_height = (height - top - bottom - panel_gap) / 2
    plot_width = width - left - right
    tick_step = 5000
    y_max = max(tick_step, math.ceil(global_max / tick_step) * tick_step)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:26px;font-weight:600}.panel{font-size:19px;font-weight:600}.axis{font-size:14px}.legend{font-size:14px}.grid{stroke:#d9dde3;stroke-width:1}.frame{stroke:#60656d;stroke-width:1.2;fill:none}</style>',
        f'<text class="title" x="{width / 2}" y="38" text-anchor="middle">C20 Gaussian Count Over Time</text>',
    ]

    for panel_index, (trace, rows) in enumerate(datasets):
        panel_top = top + panel_index * (panel_height + panel_gap)
        panel_bottom = panel_top + panel_height
        times = [float(row["output_time_s"]) for row in rows]
        times = [value - times[0] for value in times]
        x_max = max(times) or 1.0
        x_map = lambda value: left + value / x_max * plot_width
        y_map = lambda value: panel_bottom - value / y_max * panel_height

        svg.append(
            f'<text class="panel" x="{left}" y="{panel_top - 25}">Trace {trace} — {len(rows):,} frames</text>'
        )
        for value in range(0, y_max + 1, tick_step):
            y = y_map(value)
            svg.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
            svg.append(f'<text class="axis" x="{left-12}" y="{y+5:.2f}" text-anchor="end">{value/1000:g}k</text>')
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            value = fraction * x_max
            x = x_map(value)
            svg.append(f'<line class="grid" x1="{x:.2f}" y1="{panel_top}" x2="{x:.2f}" y2="{panel_bottom}"/>')
            svg.append(f'<text class="axis" x="{x:.2f}" y="{panel_bottom+25}" text-anchor="middle">{value:.0f}</text>')
        svg.append(f'<rect class="frame" x="{left}" y="{panel_top}" width="{plot_width}" height="{panel_height}"/>')

        for column, label, color in SERIES:
            values = [int(float(row[column])) for row in rows]
            mean = sum(values) / len(values)
            points = " ".join(
                f"{x_map(time):.2f},{y_map(value):.2f}"
                for time, value in zip(times, values, strict=True)
            )
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.4" stroke-linejoin="round"/>'
            )
            legend_index = next(i for i, item in enumerate(SERIES) if item[0] == column)
            legend_x = width - right - 600 + legend_index * 205
            legend_y = panel_top - 25
            svg.append(f'<line x1="{legend_x}" y1="{legend_y-5}" x2="{legend_x+25}" y2="{legend_y-5}" stroke="{color}" stroke-width="3"/>')
            svg.append(f'<text class="legend" x="{legend_x+33}" y="{legend_y}">{label} ({mean:,.0f})</text>')

        svg.append(
            f'<text class="axis" x="{left + plot_width/2}" y="{panel_bottom+48}" text-anchor="middle">Elapsed time (s)</text>'
        )
        center_y = panel_top + panel_height / 2
        svg.append(
            f'<text class="axis" x="25" y="{center_y}" text-anchor="middle" transform="rotate(-90 25 {center_y})">Gaussian count</text>'
        )

    svg.append('</svg>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
