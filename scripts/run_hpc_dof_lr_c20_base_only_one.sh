#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TRACE_STEM="${TRACE_STEM:?TRACE_STEM is required}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_dof_lr_e3_c20_bnq_v1}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/$USER/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_CircleTurns_LUT}"
GT_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"

MATCHED="$BNQ_ROOT/dof_lr_e3_c20/$TRACE_STEM"
RESULT="$BNQ_ROOT/matched_base_only/$TRACE_STEM"
TRACE="$MATCHED/00_pose/actual_trace.csv"
SOURCE_DECISIONS="$MATCHED/01_policy/cell_decisions.csv"
POLICY_DIR="$RESULT/01_policy"
QOE_DIR="$RESULT/02_bandwidth_qoe"

for path in "$REPO" "$TRACE" "$SOURCE_DECISIONS" "$MODEL_ROOT" "$GT_ROOT"; do
  [[ -e "$path" ]] || { echo "Required path is missing: $path" >&2; exit 2; }
done
mkdir -p "$POLICY_DIR" "$QOE_DIR"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
GSPLAT_LIBRARY_PATH="${GSPLAT_LIBRARY_PATH:-$(
  "$PYTHON" -c 'import pathlib, gsplat; print(pathlib.Path(gsplat.__file__).resolve().parent.parent)'
)}"

"$PYTHON" "$REPO/scripts/generate_matched_base_only_policy.py" \
  --source-decisions "$SOURCE_DECISIONS" --output-dir "$POLICY_DIR" \
  --variant matched_base_only

"$PYTHON" "$REPO/scripts/evaluate_standard_ply_qoe.py" \
  --trace "$TRACE" --decisions "$POLICY_DIR/cell_decisions.csv" \
  --model-root "$MODEL_ROOT" --gt-root "$GT_ROOT" \
  --gsplat-library-path "$GSPLAT_LIBRARY_PATH" --output-dir "$QOE_DIR" \
  --dataset-variant lut --width 1920 --height 1080 --hfov 77 \
  --cell-size-m 0.2 --sh-degree 3 --prefetch-workers 4 \
  --metric-batch-size 4 --save-every 250 --device cuda
echo "PASS: $RESULT"
