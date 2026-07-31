"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .batch import run_batch, run_manifest_line
from .pipeline import run_simulation
from .trace import load_trace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fovsim",
        description=(
            "Apply a deterministic Base/E3 cell policy to 6DoF trace "
            "visibility labels."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-trace",
        help="Validate and summarize a raw 6DoF trace CSV.",
    )
    validate.add_argument("--trace", type=Path, required=True)

    run = subparsers.add_parser(
        "run",
        help="Run one trace/visibility policy simulation.",
    )
    run.add_argument("--trace", type=Path, required=True)
    run.add_argument("--visibility", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--first-frame", type=int, default=1)
    run.add_argument("--frame-count", type=int)
    run.add_argument("--timestamp-tolerance-s", type=float, default=1e-5)
    run.add_argument("--plot", action="store_true")

    batch = subparsers.add_parser(
        "batch",
        help="Run independent manifest jobs in parallel.",
    )
    batch.add_argument("--manifest", type=Path, required=True)
    batch.add_argument("--workers", type=int)
    batch.add_argument("--summary-dir", type=Path)

    manifest_job = subparsers.add_parser(
        "manifest-job",
        help="Run one manifest row, intended for Slurm arrays.",
    )
    manifest_job.add_argument("--manifest", type=Path, required=True)
    manifest_job.add_argument(
        "--job-index",
        type=int,
        required=True,
        help="One-based index among non-empty manifest rows.",
    )

    predict = subparsers.add_parser(
        "predict-linear",
        help="Train/test a direct linear 6DoF and cell-visibility predictor.",
    )
    predict.add_argument("--trace-dir", type=Path, required=True)
    predict.add_argument("--visibility-dir", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument(
        "--sequence", default="BiancaGolden_CircleTurns"
    )
    predict.add_argument("--fps", type=float, default=30.0)
    predict.add_argument("--history-ms", type=int, default=500)
    predict.add_argument(
        "--horizons-ms", type=int, nargs="+", default=(100, 200, 500)
    )
    predict.add_argument("--visibility-threshold", type=float, default=0.5)
    predict.add_argument("--test-fraction", type=float, default=0.2)
    predict.add_argument("--seed", type=int, default=20260731)
    predict.add_argument("--ridge-alpha", type=float, default=1.0)
    predict.add_argument("--expected-traces", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "validate-trace":
        _, summary = load_trace(args.trace)
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    elif args.command == "run":
        result = run_simulation(
            trace_path=args.trace,
            visibility_path=args.visibility,
            output_dir=args.output_dir,
            threshold=args.threshold,
            first_frame=args.first_frame,
            frame_count=args.frame_count,
            timestamp_tolerance_s=args.timestamp_tolerance_s,
            make_plot=args.plot,
        )
        payload = asdict(result)
        payload["output_dir"] = str(result.output_dir)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "batch":
        results = run_batch(
            manifest_path=args.manifest,
            workers=args.workers,
            summary_dir=args.summary_dir,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "jobs": len(results),
                },
                indent=2,
            )
        )
    elif args.command == "manifest-job":
        result = run_manifest_line(
            manifest_path=args.manifest,
            one_based_line=args.job_index,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "predict-linear":
        from .prediction import run_linear_prediction

        result = run_linear_prediction(
            trace_dir=args.trace_dir,
            visibility_dir=args.visibility_dir,
            output_dir=args.output_dir,
            sequence=args.sequence,
            fps=args.fps,
            history_ms=args.history_ms,
            horizons_ms=args.horizons_ms,
            visibility_threshold=args.visibility_threshold,
            test_fraction=args.test_fraction,
            seed=args.seed,
            ridge_alpha=args.ridge_alpha,
            expected_traces=args.expected_traces,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
