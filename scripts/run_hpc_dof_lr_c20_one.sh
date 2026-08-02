#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TRACE="${TRACE:?TRACE is required}"
TRACE_STEM="$(basename "${TRACE%.csv}")"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_dof_lr_c20_bnq_v1}"
MODEL="${MODEL:-$BNQ_ROOT/00_dof_lr_model/dof_lr_model.npz}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/$USER/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_CircleTurns_LUT}"
GT_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"
RESULT="$BNQ_ROOT/dof_lr_c20/$TRACE_STEM"
POSE_DIR="$RESULT/00_pose"
PRED_VIS_DIR="$RESULT/00_predicted_visibility"
POLICY_DIR="$RESULT/01_policy"
QOE_DIR="$RESULT/02_bandwidth_qoe"
ACTUAL_VIS_DIR="$BNQ_ROOT/reference_visibility"
mkdir -p "$POSE_DIR" "$PRED_VIS_DIR" "$POLICY_DIR" "$QOE_DIR" "$ACTUAL_VIS_DIR/metadata"
for path in "$REPO" "$TRACE" "$MODEL" "$MODEL_ROOT" "$GT_ROOT"; do
  [[ -e "$path" ]] || { echo "Required path is missing: $path" >&2; exit 2; }
done
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
GSPLAT_LIBRARY_PATH="${GSPLAT_LIBRARY_PATH:-$(
  "$PYTHON" -c 'import pathlib, gsplat; print(pathlib.Path(gsplat.__file__).resolve().parent.parent)'
)}"

"$PYTHON" "$REPO/scripts/generate_dof_lr_pose_pair.py" \
  --trace "$TRACE" --model "$MODEL" \
  --predicted-trace "$POSE_DIR/predicted_trace.csv" \
  --actual-trace "$POSE_DIR/actual_trace.csv" \
  --metadata "$POSE_DIR/dof_prediction.json" \
  --fps 30 --history-ms 500 --horizon-ms 100

visibility() {
  local trace_path="$1" output="$2" metadata="$3"
  "$PYTHON" "$REPO/scripts/generate_standard_ply_visibility.py" \
    --trace "$trace_path" --gt-root "$GT_ROOT" --output "$output" \
    --metadata "$metadata" --gsplat-library-path "$GSPLAT_LIBRARY_PATH" \
    --start-mode beginning --fps 30 --width 1920 --height 1080 --hfov 77 \
    --cell-size-m 0.2 --visibility-weight-threshold 0.00392156862745098 \
    --skip-missing-assets --device cuda
}
visibility "$POSE_DIR/predicted_trace.csv" \
  "$PRED_VIS_DIR/cell_visibility.csv" "$PRED_VIS_DIR/metadata.json"
visibility "$POSE_DIR/actual_trace.csv" \
  "$ACTUAL_VIS_DIR/$TRACE_STEM.csv" "$ACTUAL_VIS_DIR/metadata/$TRACE_STEM.json"

"$PYTHON" "$REPO/scripts/generate_visibility_score_policy.py" \
  --visibility "$PRED_VIS_DIR/cell_visibility.csv" --output-dir "$POLICY_DIR" \
  --variant dof_lr_c20 --score contributing_fraction --threshold 0.20 \
  --predicted-pose-visibility

"$PYTHON" "$REPO/scripts/evaluate_standard_ply_qoe.py" \
  --trace "$POSE_DIR/actual_trace.csv" --decisions "$POLICY_DIR/cell_decisions.csv" \
  --model-root "$MODEL_ROOT" --gt-root "$GT_ROOT" \
  --gsplat-library-path "$GSPLAT_LIBRARY_PATH" --output-dir "$QOE_DIR" \
  --dataset-variant lut --width 1920 --height 1080 --hfov 77 \
  --cell-size-m 0.2 --sh-degree 3 --prefetch-workers 4 \
  --metric-batch-size 4 --save-every 250 --device cuda
echo "PASS: $RESULT"
