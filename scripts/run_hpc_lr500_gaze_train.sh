#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/scratch/$USER/fov_visibility_lr}"
VISIBILITY_ROOT="$EXPERIMENT_ROOT/visibility"
OUTPUT_ROOT="${GAZE_MODEL_ROOT:-$EXPERIMENT_ROOT/linear_prediction_lr500_raw_gaze}"
BASELINE_ROOT="${GAZE_BASELINE_ROOT:-$EXPERIMENT_ROOT/linear_prediction_lr500_raw_gaze_window_baseline}"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"
mkdir -p "$OUTPUT_ROOT" "$BASELINE_ROOT"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
COMMON_ARGS=(
  --trace-dir "$REPO/trace_csvs" --visibility-dir "$VISIBILITY_ROOT"
  --history-ms 500 --horizons-ms 500 --visibility-threshold 0.5
  --decision-threshold-min 0.05 --decision-threshold-max 0.35
  --decision-threshold-steps 31 --target-mode binary
  --require-valid-gaze-history --test-fraction 0.2 --seed 20260731
  --ridge-alpha 1.0 --expected-traces 10
)
"$PYTHON" -m fovsim predict-linear \
  "${COMMON_ARGS[@]}" --feature-mode raw_history --output-dir "$BASELINE_ROOT"
"$PYTHON" "$REPO/scripts/summarize_lr500_result.py" \
  --summary "$BASELINE_ROOT/linear_visibility_summary.json" \
  --output "$BASELINE_ROOT/lr500_evaluation.json"
"$PYTHON" -m fovsim predict-linear \
  "${COMMON_ARGS[@]}" --feature-mode raw_gaze --output-dir "$OUTPUT_ROOT"
"$PYTHON" "$REPO/scripts/summarize_lr500_result.py" \
  --summary "$OUTPUT_ROOT/linear_visibility_summary.json" \
  --output "$OUTPUT_ROOT/lr500_evaluation.json"
echo "PASS baseline: $BASELINE_ROOT"
echo "PASS gaze:     $OUTPUT_ROOT"
