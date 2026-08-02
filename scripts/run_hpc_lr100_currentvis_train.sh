#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/scratch/$USER/fov_visibility_lr}"
VISIBILITY_ROOT="$EXPERIMENT_ROOT/visibility"
BASELINE_ROOT="${LR100_BASELINE_MODEL_ROOT:-$EXPERIMENT_ROOT/linear_prediction_lr100_currentvis_matched_gaze}"
CURRENT_ROOT="${LR100_CURRENT_MODEL_ROOT:-$EXPERIMENT_ROOT/linear_prediction_lr100_gaze_current_visibility}"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"
mkdir -p "$BASELINE_ROOT" "$CURRENT_ROOT"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
COMMON_ARGS=(
  --trace-dir "$REPO/trace_csvs" --visibility-dir "$VISIBILITY_ROOT"
  --history-ms 500 --horizons-ms 100 --visibility-threshold 0.5
  --decision-threshold-min 0.01 --decision-threshold-max 0.50
  --decision-threshold-steps 50 --safe-recall-target 0.85
  --target-mode binary --require-valid-gaze-history
  --test-fraction 0.2 --seed 20260731 --ridge-alpha 1.0
  --expected-traces 10
)
"$PYTHON" -m fovsim predict-linear \
  "${COMMON_ARGS[@]}" --feature-mode raw_gaze --output-dir "$BASELINE_ROOT"
"$PYTHON" "$REPO/scripts/summarize_lr500_result.py" \
  --summary "$BASELINE_ROOT/linear_visibility_summary.json" \
  --output "$BASELINE_ROOT/lr100_evaluation.json" --horizon-ms 100
"$PYTHON" -m fovsim predict-linear \
  "${COMMON_ARGS[@]}" --feature-mode raw_gaze_current_visibility \
  --output-dir "$CURRENT_ROOT"
"$PYTHON" "$REPO/scripts/summarize_lr500_result.py" \
  --summary "$CURRENT_ROOT/linear_visibility_summary.json" \
  --output "$CURRENT_ROOT/lr100_evaluation.json" --horizon-ms 100
echo "PASS matched gaze:      $BASELINE_ROOT"
echo "PASS current visibility: $CURRENT_ROOT"
