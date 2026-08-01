#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/scratch/$USER/fov_visibility_lr}"
VISIBILITY_ROOT="$EXPERIMENT_ROOT/visibility"
OUTPUT_ROOT="${MOTION_MODEL_ROOT:-$EXPERIMENT_ROOT/linear_prediction_lr500_motion_quadratic}"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"
mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m fovsim predict-linear \
  --trace-dir "$REPO/trace_csvs" --visibility-dir "$VISIBILITY_ROOT" \
  --output-dir "$OUTPUT_ROOT" --history-ms 500 --horizons-ms 500 \
  --visibility-threshold 0.5 --decision-threshold-min 0.05 \
  --decision-threshold-max 0.35 --decision-threshold-steps 31 \
  --target-mode binary --feature-mode motion_quadratic \
  --test-fraction 0.2 --seed 20260731 --ridge-alpha 3.0 \
  --expected-traces 10
"$PYTHON" "$REPO/scripts/summarize_lr500_result.py" \
  --summary "$OUTPUT_ROOT/linear_visibility_summary.json" \
  --output "$OUTPUT_ROOT/lr500_evaluation.json"
echo "PASS: $OUTPUT_ROOT"
