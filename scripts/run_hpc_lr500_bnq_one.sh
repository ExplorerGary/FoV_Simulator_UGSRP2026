#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TRACE="${TRACE:?TRACE is required}"
VARIANT="${VARIANT:?VARIANT is required}"
POLICY_MODE="${POLICY_MODE:-linear}"
MODEL="${MODEL:?MODEL is required}"
DECISION_THRESHOLD="${DECISION_THRESHOLD:-}"
GUARD_BAND_STEPS="${GUARD_BAND_STEPS:-0}"
REQUIRE_VALID_GAZE_HISTORY="${REQUIRE_VALID_GAZE_HISTORY:-0}"
TRACE_STEM="$(basename "${TRACE%.csv}")"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/scratch/$USER/fov_visibility_lr}"
VISIBILITY="$EXPERIMENT_ROOT/visibility/$TRACE_STEM.csv"
MODEL_ROOT="${MODEL_ROOT:-/scratch/$USER/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_CircleTurns_LUT}"
GT_SEQUENCE_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_lr500_bnq_v2}"
RESULT_ROOT="$BNQ_ROOT/$VARIANT/$TRACE_STEM"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"
for path in "$REPO" "$TRACE" "$VISIBILITY" "$MODEL" "$MODEL_ROOT" "$GT_SEQUENCE_ROOT"; do
  [[ -e "$path" ]] || { echo "Required path is missing: $path" >&2; exit 2; }
done
GSPLAT_LIBRARY_PATH="${GSPLAT_LIBRARY_PATH:-$(
  "$PYTHON" -c 'import pathlib, gsplat; print(pathlib.Path(gsplat.__file__).resolve().parent.parent)'
)}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
POLICY_DIR="$RESULT_ROOT/01_policy"
QUALITY_DIR="$RESULT_ROOT/02_bandwidth_qoe"
mkdir -p "$POLICY_DIR" "$QUALITY_DIR"
THRESHOLD_ARGS=()
[[ -z "$DECISION_THRESHOLD" ]] || THRESHOLD_ARGS=(--decision-threshold "$DECISION_THRESHOLD")
GAZE_ARGS=()
[[ "$REQUIRE_VALID_GAZE_HISTORY" != "1" ]] || GAZE_ARGS=(--require-valid-gaze-history)
"$PYTHON" -m fovsim predict-linear-policy \
  --trace "$TRACE" --visibility "$VISIBILITY" --model "$MODEL" \
  --output-dir "$POLICY_DIR" --history-ms 500 --horizon-ms 500 \
  --policy-mode "$POLICY_MODE" --guard-band-steps "$GUARD_BAND_STEPS" \
  --model-root "$MODEL_ROOT" --gt-root "$GT_SEQUENCE_ROOT" \
  "${THRESHOLD_ARGS[@]}" "${GAZE_ARGS[@]}"
"$PYTHON" "$REPO/scripts/evaluate_standard_ply_qoe.py" \
  --trace "$TRACE" --decisions "$POLICY_DIR/cell_decisions.csv" \
  --model-root "$MODEL_ROOT" --gt-root "$GT_SEQUENCE_ROOT" \
  --gsplat-library-path "$GSPLAT_LIBRARY_PATH" --output-dir "$QUALITY_DIR" \
  --dataset-variant lut --width 1920 --height 1080 --hfov 77 \
  --cell-size-m 0.2 --sh-degree 3 --prefetch-workers 4 \
  --metric-batch-size 4 --save-every 250 --device cuda
"$PYTHON" "$REPO/scripts/plot_qoe_bandwidth.py" \
  --metrics "$QUALITY_DIR/per_frame_metrics.csv" \
  --output "$QUALITY_DIR/qoe_bandwidth_overview.png" --fps 30 \
  --title "500 ms $VARIANT: $TRACE_STEM"
echo "PASS: $RESULT_ROOT"
