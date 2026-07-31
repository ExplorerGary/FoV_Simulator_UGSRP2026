"""Streaming policy application with trace/visibility provenance checks."""

from __future__ import annotations

import csv
import math
import platform
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .io import atomic_text_writer, write_csv_atomic, write_json_atomic
from .policy import PolicyConfig, classify_fraction
from .trace import TraceRow, load_trace


REQUIRED_VISIBILITY_COLUMNS = frozenset(
    {
        "output_frame",
        "output_time_s",
        "trace_source_row",
        "trace_timestamp_s",
        "gsv_frame",
        "cell_id",
        "active_gaussian_count",
        "contributing_gaussian_count",
        "contributing_gaussian_fraction",
        "image_share",
    }
)


@dataclass(slots=True)
class FrameAccumulator:
    output_frame: int
    output_time_s: float
    trace_timestamp_s: float
    gsv_frame: int
    occupied_cells: int = 0
    enhancement3_cells: int = 0
    active_gaussians: int = 0
    enhancement3_active_gaussians: int = 0
    enhancement3_image_share: float = 0.0

    def add(
        self,
        *,
        use_enhancement: bool,
        active_gaussian_count: int,
        image_share: float,
    ) -> None:
        self.occupied_cells += 1
        self.active_gaussians += active_gaussian_count
        if use_enhancement:
            self.enhancement3_cells += 1
            self.enhancement3_active_gaussians += active_gaussian_count
            self.enhancement3_image_share += image_share

    def as_summary_row(self, threshold: float) -> dict[str, object]:
        cell_fraction = (
            self.enhancement3_cells / self.occupied_cells
            if self.occupied_cells
            else 0.0
        )
        gaussian_fraction = (
            self.enhancement3_active_gaussians / self.active_gaussians
            if self.active_gaussians
            else 0.0
        )
        return {
            "display_frame": self.output_frame + 1,
            "output_frame": self.output_frame,
            "output_time_s": self.output_time_s,
            "trace_timestamp_s": self.trace_timestamp_s,
            "gsv_frame": self.gsv_frame,
            "policy_threshold": threshold,
            "occupied_cells": self.occupied_cells,
            "enhancement3_cells": self.enhancement3_cells,
            "base_only_cells": self.occupied_cells - self.enhancement3_cells,
            "enhancement3_cell_fraction": cell_fraction,
            "active_gaussians": self.active_gaussians,
            "enhancement3_active_gaussians": (
                self.enhancement3_active_gaussians
            ),
            "enhancement3_gaussian_fraction": gaussian_fraction,
            "enhancement3_image_share": self.enhancement3_image_share,
            "base_only_image_share": max(
                0.0,
                1.0 - self.enhancement3_image_share,
            ),
        }


@dataclass(frozen=True, slots=True)
class SimulationResult:
    output_dir: Path
    decision_rows: int
    frame_count: int
    mean_enhancement3_cells: float
    mean_enhancement3_cell_fraction: float
    mean_enhancement3_gaussian_fraction: float
    mean_enhancement3_image_share: float
    runtime_seconds: float


def _finite_float(raw: dict[str, str], field: str, row_number: int) -> float:
    try:
        value = float(raw[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid visibility field {field!r} at CSV row {row_number}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite visibility field {field!r} at CSV row {row_number}"
        )
    return value


def _nonnegative_int(
    raw: dict[str, str],
    field: str,
    row_number: int,
) -> int:
    value = _finite_float(raw, field, row_number)
    integer = int(value)
    if value != integer or integer < 0:
        raise ValueError(
            f"Visibility field {field!r} must be a non-negative integer "
            f"at CSV row {row_number}"
        )
    return integer


def _validate_trace_reference(
    raw: dict[str, str],
    row_number: int,
    trace_by_source_row: dict[int, TraceRow],
    timestamp_tolerance_s: float,
) -> TraceRow:
    source_row = _nonnegative_int(raw, "trace_source_row", row_number)
    try:
        trace_row = trace_by_source_row[source_row]
    except KeyError as exc:
        raise ValueError(
            f"Visibility row {row_number} references absent trace source "
            f"row {source_row}"
        ) from exc
    visibility_timestamp = _finite_float(
        raw,
        "trace_timestamp_s",
        row_number,
    )
    if (
        abs(visibility_timestamp - trace_row.timestamp_s)
        > timestamp_tolerance_s
    ):
        raise ValueError(
            f"Timestamp mismatch at visibility row {row_number}: "
            f"{visibility_timestamp} vs trace {trace_row.timestamp_s}"
        )
    visibility_gsv_frame = _nonnegative_int(raw, "gsv_frame", row_number)
    if visibility_gsv_frame != trace_row.gsv_frame:
        raise ValueError(
            f"GSV frame mismatch at visibility row {row_number}: "
            f"{visibility_gsv_frame} vs trace {trace_row.gsv_frame}"
        )
    return trace_row


def _mean(rows: list[dict[str, object]], field: str) -> float:
    return (
        sum(float(row[field]) for row in rows) / len(rows)
        if rows
        else 0.0
    )


def run_simulation(
    *,
    trace_path: str | Path,
    visibility_path: str | Path,
    output_dir: str | Path,
    threshold: float = 0.5,
    first_frame: int = 1,
    frame_count: int | None = None,
    timestamp_tolerance_s: float = 1e-5,
    make_plot: bool = False,
) -> SimulationResult:
    started = time.perf_counter()
    if first_frame <= 0:
        raise ValueError("first_frame must be positive and one-based")
    if frame_count is not None and frame_count <= 0:
        raise ValueError("frame_count must be positive when provided")
    if timestamp_tolerance_s < 0.0 or not math.isfinite(
        timestamp_tolerance_s
    ):
        raise ValueError("timestamp_tolerance_s must be finite and non-negative")

    config = PolicyConfig(threshold=threshold)
    trace_rows, trace_summary = load_trace(trace_path)
    trace_by_source_row = {row.source_row: row for row in trace_rows}
    if len(trace_by_source_row) != len(trace_rows):
        raise AssertionError("Trace source-row identifiers are not unique")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    decisions_path = output_path / "cell_decisions.csv"
    summary_path = output_path / "frame_summary.csv"
    metadata_path = output_path / "run_metadata.json"

    first_output_frame = first_frame - 1
    end_output_frame = (
        first_output_frame + frame_count if frame_count is not None else None
    )
    frames: dict[int, FrameAccumulator] = {}
    selected_row_count = 0

    with Path(visibility_path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_stream:
        reader = csv.DictReader(input_stream)
        source_fields = list(reader.fieldnames or ())
        missing = sorted(REQUIRED_VISIBILITY_COLUMNS - set(source_fields))
        if missing:
            raise ValueError(
                f"Visibility CSV is missing required columns: {missing}"
            )
        decision_fields = source_fields + [
            "policy_threshold",
            "enhancement_required",
            "target_level",
        ]

        with atomic_text_writer(decisions_path, newline="") as output_stream:
            writer = csv.DictWriter(
                output_stream,
                fieldnames=decision_fields,
            )
            writer.writeheader()
            for row_number, raw in enumerate(reader, start=2):
                output_frame = _nonnegative_int(
                    raw,
                    "output_frame",
                    row_number,
                )
                if output_frame < first_output_frame:
                    continue
                if (
                    end_output_frame is not None
                    and output_frame >= end_output_frame
                ):
                    continue

                trace_row = _validate_trace_reference(
                    raw,
                    row_number,
                    trace_by_source_row,
                    timestamp_tolerance_s,
                )
                fraction = _finite_float(
                    raw,
                    "contributing_gaussian_fraction",
                    row_number,
                )
                use_enhancement, target_level = classify_fraction(
                    fraction,
                    config,
                )
                active_count = _nonnegative_int(
                    raw,
                    "active_gaussian_count",
                    row_number,
                )
                image_share = _finite_float(raw, "image_share", row_number)
                if image_share < 0.0:
                    raise ValueError(
                        f"Negative image_share at visibility row {row_number}"
                    )
                output_time_s = _finite_float(
                    raw,
                    "output_time_s",
                    row_number,
                )
                accumulator = frames.get(output_frame)
                if accumulator is None:
                    accumulator = FrameAccumulator(
                        output_frame=output_frame,
                        output_time_s=output_time_s,
                        trace_timestamp_s=trace_row.timestamp_s,
                        gsv_frame=trace_row.gsv_frame,
                    )
                    frames[output_frame] = accumulator
                elif (
                    accumulator.output_time_s != output_time_s
                    or accumulator.trace_timestamp_s != trace_row.timestamp_s
                    or accumulator.gsv_frame != trace_row.gsv_frame
                ):
                    raise ValueError(
                        f"Inconsistent frame metadata for output frame "
                        f"{output_frame}"
                    )

                accumulator.add(
                    use_enhancement=use_enhancement,
                    active_gaussian_count=active_count,
                    image_share=image_share,
                )
                writer.writerow(
                    {
                        **raw,
                        "policy_threshold": config.threshold,
                        "enhancement_required": int(use_enhancement),
                        "target_level": target_level,
                    }
                )
                selected_row_count += 1

    if selected_row_count == 0:
        decisions_path.unlink(missing_ok=True)
        raise ValueError("No visibility rows fall within the selected frames")

    ordered_frames = sorted(frames)
    if frame_count is not None and len(ordered_frames) != frame_count:
        decisions_path.unlink(missing_ok=True)
        raise ValueError(
            f"Requested {frame_count} frames but found {len(ordered_frames)}"
        )
    expected_frames = list(
        range(ordered_frames[0], ordered_frames[-1] + 1)
    )
    if ordered_frames != expected_frames:
        decisions_path.unlink(missing_ok=True)
        raise ValueError("Selected visibility frames are not contiguous")

    summary_rows = [
        frames[frame].as_summary_row(config.threshold)
        for frame in ordered_frames
    ]
    write_csv_atomic(summary_path, list(summary_rows[0]), summary_rows)

    if make_plot:
        from .plotting import plot_frame_summary

        plot_frame_summary(summary_rows, output_path / "policy.png")

    runtime = time.perf_counter() - started
    result = SimulationResult(
        output_dir=output_path.resolve(),
        decision_rows=selected_row_count,
        frame_count=len(summary_rows),
        mean_enhancement3_cells=_mean(
            summary_rows,
            "enhancement3_cells",
        ),
        mean_enhancement3_cell_fraction=_mean(
            summary_rows,
            "enhancement3_cell_fraction",
        ),
        mean_enhancement3_gaussian_fraction=_mean(
            summary_rows,
            "enhancement3_gaussian_fraction",
        ),
        mean_enhancement3_image_share=_mean(
            summary_rows,
            "enhancement3_image_share",
        ),
        runtime_seconds=runtime,
    )
    metadata = {
        "schema_version": 1,
        "policy": {
            "name": "contributing_fraction_base_or_e3",
            "threshold": config.threshold,
            "comparison": ">=",
            "base_level": config.base_level,
            "enhancement_level": config.enhancement_level,
        },
        "inputs": {
            "trace": str(Path(trace_path).resolve()),
            "visibility": str(Path(visibility_path).resolve()),
        },
        "selection": {
            "first_display_frame": first_frame,
            "requested_frame_count": frame_count,
            "timestamp_tolerance_s": timestamp_tolerance_s,
        },
        "trace_summary": asdict(trace_summary),
        "result": {
            **asdict(result),
            "output_dir": str(result.output_dir),
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "outputs": {
            "cell_decisions": str(decisions_path.resolve()),
            "frame_summary": str(summary_path.resolve()),
            "plot": (
                str((output_path / "policy.png").resolve())
                if make_plot
                else None
            ),
        },
    }
    write_json_atomic(metadata_path, metadata)
    return result
