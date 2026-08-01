#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/scratch/$USER/fov_visibility_lr}"
VISIBILITY_ROOT="$EXPERIMENT_ROOT/visibility"
GAZE_ROOT="${LR100_GAZE_MODEL_ROOT:-$EXPERIMENT_ROOT/linear_prediction_lr100_raw_gaze}"
HEAD_ROOT="${LR100_HEAD_MODEL_ROOT:-$EXPERIMENT_ROOT/linear_prediction_lr100_raw_head}"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"
mkdir -p "$GAZE_ROOT" "$HEAD_ROOT"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
COMMON_ARGS=(
  --trace-dir "$REPO/trace_csvs" --visibility-dir "$VISIBILITY_ROOT"
  --history-ms 500 --horizons-ms 100 --visibility-threshold 0.5
  --decision-threshold-min 0.10 --decision-threshold-max 0.50
  --decision-threshold-steps 41 --safe-recall-target 0.85
  --target-mode binary --require-valid-gaze-history
  --test-fraction 0.2 --seed 20260731 --ridge-alpha 1.0
  --expected-traces 10
)
"$PYTHON" -m fovsim predict-linear \
  "${COMMON_ARGS[@]}" --feature-mode raw_history --output-dir "$HEAD_ROOT"
"$PYTHON" "$REPO/scripts/summarize_lr500_result.py" \
  --summary "$HEAD_ROOT/linear_visibility_summary.json" \
  --output "$HEAD_ROOT/lr100_evaluation.json" --horizon-ms 100
"$PYTHON" -m fovsim predict-linear \
  "${COMMON_ARGS[@]}" --feature-mode raw_gaze --output-dir "$GAZE_ROOT"
"$PYTHON" "$REPO/scripts/summarize_lr500_result.py" \
  --summary "$GAZE_ROOT/linear_visibility_summary.json" \
  --output "$GAZE_ROOT/lr100_evaluation.json" --horizon-ms 100
echo "PASS head: $HEAD_ROOT"
echo "PASS gaze: $GAZE_ROOT"
