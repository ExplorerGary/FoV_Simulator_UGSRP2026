"""Strict parser for the project's 6DoF trace CSV."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


REQUIRED_TRACE_COLUMNS = frozenset(
    {
        "FileName",
        "LocationX",
        "LocationY",
        "LocationZ",
        "RotationRoll",
        "RotationPitch",
        "RotationYaw",
        "Frame",
        "Timestamp",
    }
)
GAZE_TRACE_COLUMNS = frozenset(
    {"GazeHitX", "GazeHitY", "GazeHitZ", "GazeConfidence"}
)


@dataclass(frozen=True, slots=True)
class TraceRow:
    source_row: int
    file_name: str
    location_cm: tuple[float, float, float]
    rotation_rpy_degrees: tuple[float, float, float]
    gsv_frame: int
    timestamp_s: float
    gaze_direction: tuple[float, float, float] | None = None
    gaze_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class TraceSummary:
    row_count: int
    first_timestamp_s: float
    last_timestamp_s: float
    duration_s: float
    effective_fps: float
    first_gsv_frame: int
    last_gsv_frame: int
    unique_gsv_frames: int


def _finite_float(raw: dict[str, str], name: str, source_row: int) -> float:
    try:
        value = float(raw[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {name!r} at trace source row {source_row}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite {name!r} at trace source row {source_row}"
        )
    return value


def _nonnegative_int(
    raw: dict[str, str],
    name: str,
    source_row: int,
) -> int:
    value = _finite_float(raw, name, source_row)
    integer = int(value)
    if value != integer or integer < 0:
        raise ValueError(
            f"{name!r} must be a non-negative integer at trace "
            f"source row {source_row}"
        )
    return integer


def load_trace(path: str | Path) -> tuple[list[TraceRow], TraceSummary]:
    trace_path = Path(path)
    rows: list[TraceRow] = []
    with trace_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_TRACE_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"Trace is missing required columns: {missing}")
        present_gaze = GAZE_TRACE_COLUMNS & fieldnames
        if present_gaze and present_gaze != GAZE_TRACE_COLUMNS:
            raise ValueError(
                "Trace has an incomplete gaze schema; expected columns: "
                f"{sorted(GAZE_TRACE_COLUMNS)}"
            )
        has_gaze = present_gaze == GAZE_TRACE_COLUMNS

        for source_row, raw in enumerate(reader, start=2):
            if raw.get("FileName") == "FileName":
                raise ValueError(
                    f"Repeated CSV header at trace source row {source_row}"
                )
            row = TraceRow(
                source_row=source_row,
                file_name=raw["FileName"],
                location_cm=(
                    _finite_float(raw, "LocationX", source_row),
                    _finite_float(raw, "LocationY", source_row),
                    _finite_float(raw, "LocationZ", source_row),
                ),
                rotation_rpy_degrees=(
                    _finite_float(raw, "RotationRoll", source_row),
                    _finite_float(raw, "RotationPitch", source_row),
                    _finite_float(raw, "RotationYaw", source_row),
                ),
                gsv_frame=_nonnegative_int(raw, "Frame", source_row),
                timestamp_s=_finite_float(raw, "Timestamp", source_row),
                gaze_direction=(
                    (
                        _finite_float(raw, "GazeHitX", source_row),
                        _finite_float(raw, "GazeHitY", source_row),
                        _finite_float(raw, "GazeHitZ", source_row),
                    )
                    if has_gaze
                    else None
                ),
                gaze_confidence=(
                    _finite_float(raw, "GazeConfidence", source_row)
                    if has_gaze
                    else None
                ),
            )
            if rows and row.timestamp_s <= rows[-1].timestamp_s:
                raise ValueError(
                    "Trace timestamps must be strictly increasing: "
                    f"source rows {rows[-1].source_row} and {source_row}"
                )
            rows.append(row)

    if not rows:
        raise ValueError("Trace contains no data rows")

    duration = rows[-1].timestamp_s - rows[0].timestamp_s
    effective_fps = (
        (len(rows) - 1) / duration if len(rows) > 1 and duration > 0.0 else 0.0
    )
    summary = TraceSummary(
        row_count=len(rows),
        first_timestamp_s=rows[0].timestamp_s,
        last_timestamp_s=rows[-1].timestamp_s,
        duration_s=duration,
        effective_fps=effective_fps,
        first_gsv_frame=rows[0].gsv_frame,
        last_gsv_frame=rows[-1].gsv_frame,
        unique_gsv_frames=len({row.gsv_frame for row in rows}),
    )
    return rows, summary
