#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TRACE="${TRACE:?TRACE is required}"
VARIANT="${VARIANT:?VARIANT is required}"
SCORE="${SCORE:?SCORE is required}"
RULE_KIND="${RULE_KIND:?RULE_KIND is required}"
RULE_VALUE="${RULE_VALUE:?RULE_VALUE is required}"
TRACE_STEM="$(basename "${TRACE%.csv}")"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/scratch/$USER/fov_visibility_lr}"
VISIBILITY="${VISIBILITY:-$EXPERIMENT_ROOT/visibility/$TRACE_STEM.csv}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/$USER/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_CircleTurns_LUT}"
GT_SEQUENCE_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_visibility_score_bnq_v1}"
RESULT_ROOT="$BNQ_ROOT/$VARIANT/$TRACE_STEM"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"
for path in "$REPO" "$TRACE" "$VISIBILITY" "$MODEL_ROOT" "$GT_SEQUENCE_ROOT"; do
  [[ -e "$path" ]] || { echo "Required path is missing: $path" >&2; exit 2; }
done
GSPLAT_LIBRARY_PATH="${GSPLAT_LIBRARY_PATH:-$(
  "$PYTHON" -c 'import pathlib, gsplat; print(pathlib.Path(gsplat.__file__).resolve().parent.parent)'
)}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
POLICY_DIR="$RESULT_ROOT/01_policy"
QUALITY_DIR="$RESULT_ROOT/02_bandwidth_qoe"
mkdir -p "$POLICY_DIR" "$QUALITY_DIR"
RULE_ARGS=()
if [[ "$RULE_KIND" == "threshold" ]]; then
  RULE_ARGS=(--threshold "$RULE_VALUE")
elif [[ "$RULE_KIND" == "coverage" ]]; then
  RULE_ARGS=(--cumulative-coverage "$RULE_VALUE")
else
  echo "Unknown RULE_KIND: $RULE_KIND" >&2; exit 2
fi
if [[ "${REUSE_POLICY:-0}" == "1" && -s "$POLICY_DIR/cell_decisions.csv" ]]; then
  echo "Reusing existing policy: $POLICY_DIR/cell_decisions.csv"
else
  "$PYTHON" "$REPO/scripts/generate_visibility_score_policy.py" \
    --visibility "$VISIBILITY" --output-dir "$POLICY_DIR" \
    --variant "$VARIANT" --score "$SCORE" "${RULE_ARGS[@]}"
fi
"$PYTHON" "$REPO/scripts/evaluate_standard_ply_qoe.py" \
  --trace "$TRACE" --decisions "$POLICY_DIR/cell_decisions.csv" \
  --model-root "$MODEL_ROOT" --gt-root "$GT_SEQUENCE_ROOT" \
  --gsplat-library-path "$GSPLAT_LIBRARY_PATH" --output-dir "$QUALITY_DIR" \
  --dataset-variant lut --width 1920 --height 1080 --hfov 77 \
  --cell-size-m 0.2 --sh-degree 3 --prefetch-workers 4 \
  --metric-batch-size 4 --save-every 250 --device cuda
echo "PASS: $RESULT_ROOT"
