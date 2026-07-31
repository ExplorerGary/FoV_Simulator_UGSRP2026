"""Job-level multiprocessing for independent trace simulations."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .io import write_csv_atomic, write_json_atomic
from .pipeline import run_simulation


@dataclass(frozen=True, slots=True)
class JobSpec:
    name: str
    trace: str
    visibility: str
    output_dir: str
    threshold: float = 0.5
    first_frame: int = 1
    frame_count: int | None = None
    timestamp_tolerance_s: float = 1e-5
    plot: bool = False


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def load_manifest(path: str | Path) -> list[JobSpec]:
    manifest_path = Path(path).resolve()
    base = manifest_path.parent
    jobs: list[JobSpec] = []
    names: set[str] = set()
    output_dirs: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at manifest line {line_number}"
                ) from exc
            required = {"name", "trace", "visibility", "output_dir"}
            missing = sorted(required - set(raw))
            if missing:
                raise ValueError(
                    f"Manifest line {line_number} is missing: {missing}"
                )
            name = str(raw["name"])
            if not name or name in names:
                raise ValueError(
                    f"Duplicate or empty job name at manifest line {line_number}"
                )
            output_dir = _resolve_path(str(raw["output_dir"]), base)
            if output_dir in output_dirs:
                raise ValueError(
                    f"Duplicate output directory at manifest line {line_number}"
                )
            names.add(name)
            output_dirs.add(output_dir)
            jobs.append(
                JobSpec(
                    name=name,
                    trace=_resolve_path(str(raw["trace"]), base),
                    visibility=_resolve_path(
                        str(raw["visibility"]),
                        base,
                    ),
                    output_dir=output_dir,
                    threshold=float(raw.get("threshold", 0.5)),
                    first_frame=int(raw.get("first_frame", 1)),
                    frame_count=(
                        int(raw["frame_count"])
                        if raw.get("frame_count") is not None
                        else None
                    ),
                    timestamp_tolerance_s=float(
                        raw.get("timestamp_tolerance_s", 1e-5)
                    ),
                    plot=bool(raw.get("plot", False)),
                )
            )
    if not jobs:
        raise ValueError("Manifest contains no jobs")
    return jobs


def execute_job(job: JobSpec) -> dict[str, object]:
    started = time.perf_counter()
    result = run_simulation(
        trace_path=job.trace,
        visibility_path=job.visibility,
        output_dir=job.output_dir,
        threshold=job.threshold,
        first_frame=job.first_frame,
        frame_count=job.frame_count,
        timestamp_tolerance_s=job.timestamp_tolerance_s,
        make_plot=job.plot,
    )
    return {
        "name": job.name,
        "status": "PASS",
        "output_dir": str(result.output_dir),
        "decision_rows": result.decision_rows,
        "frame_count": result.frame_count,
        "mean_enhancement3_cells": result.mean_enhancement3_cells,
        "mean_enhancement3_cell_fraction": (
            result.mean_enhancement3_cell_fraction
        ),
        "mean_enhancement3_gaussian_fraction": (
            result.mean_enhancement3_gaussian_fraction
        ),
        "mean_enhancement3_image_share": (
            result.mean_enhancement3_image_share
        ),
        "runtime_seconds": time.perf_counter() - started,
        "error": "",
    }


def _failed_job(job: JobSpec, exc: Exception) -> dict[str, object]:
    return {
        "name": job.name,
        "status": "FAIL",
        "output_dir": job.output_dir,
        "decision_rows": 0,
        "frame_count": 0,
        "mean_enhancement3_cells": 0.0,
        "mean_enhancement3_cell_fraction": 0.0,
        "mean_enhancement3_gaussian_fraction": 0.0,
        "mean_enhancement3_image_share": 0.0,
        "runtime_seconds": 0.0,
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_batch(
    *,
    manifest_path: str | Path,
    workers: int | None = None,
    summary_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    jobs = load_manifest(manifest_path)
    if workers is None:
        workers = min(len(jobs), os.cpu_count() or 1)
    if workers <= 0:
        raise ValueError("workers must be positive")
    workers = min(workers, len(jobs))

    results: list[dict[str, object]] = []
    if workers == 1:
        for job in jobs:
            try:
                results.append(execute_job(job))
            except Exception as exc:
                results.append(_failed_job(job, exc))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(execute_job, job): job for job in jobs
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(_failed_job(job, exc))

    order = {job.name: index for index, job in enumerate(jobs)}
    results.sort(key=lambda row: order[str(row["name"])])

    destination = (
        Path(summary_dir)
        if summary_dir is not None
        else Path(manifest_path).resolve().parent
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(
        destination / "batch_summary.csv",
        list(results[0]),
        results,
    )
    write_json_atomic(
        destination / "batch_summary.json",
        {
            "schema_version": 1,
            "manifest": str(Path(manifest_path).resolve()),
            "workers": workers,
            "jobs": [asdict(job) for job in jobs],
            "results": results,
        },
    )
    failures = [row for row in results if row["status"] != "PASS"]
    if failures:
        names = ", ".join(str(row["name"]) for row in failures)
        raise RuntimeError(f"{len(failures)} batch job(s) failed: {names}")
    return results


def run_manifest_line(
    *,
    manifest_path: str | Path,
    one_based_line: int,
) -> dict[str, object]:
    jobs = load_manifest(manifest_path)
    if one_based_line <= 0 or one_based_line > len(jobs):
        raise ValueError(
            f"job index must be within [1, {len(jobs)}]"
        )
    return execute_job(jobs[one_based_line - 1])
