#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TRACE="${TRACE:?TRACE is required}"
TRACE_STEM="$(basename "${TRACE%.csv}")"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/scratch/$USER/fov_visibility_lr}"
VISIBILITY="${VISIBILITY:-$EXPERIMENT_ROOT/visibility/$TRACE_STEM.csv}"
MODEL="${MODEL:-$EXPERIMENT_ROOT/linear_prediction_lr500_binary/visibility_model_500ms.npz}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/$USER/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_CircleTurns_LUT}"
GT_SEQUENCE_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/$USER/fov_lr500_bandwidth_qoe_v1}"
RESULT_ROOT="$OUTPUT_ROOT/$TRACE_STEM"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"

for path in "$REPO" "$TRACE" "$VISIBILITY" "$MODEL" "$MODEL_ROOT" "$GT_SEQUENCE_ROOT"; do
  [[ -e "$path" ]] || { echo "Required path is missing: $path" >&2; exit 2; }
done
GSPLAT_LIBRARY_PATH="${GSPLAT_LIBRARY_PATH:-$(
  "$PYTHON" -c 'import pathlib, gsplat; print(pathlib.Path(gsplat.__file__).resolve().parent.parent)'
)}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

POLICY_DIR="$RESULT_ROOT/01_lr500_policy"
QUALITY_DIR="$RESULT_ROOT/02_bandwidth_qoe"
mkdir -p "$POLICY_DIR" "$QUALITY_DIR"

echo "[1/3] 500 ms DoF-only LR -> cell decisions"
"$PYTHON" -m fovsim predict-linear-policy \
  --trace "$TRACE" --visibility "$VISIBILITY" --model "$MODEL" \
  --output-dir "$POLICY_DIR" --history-ms 500 --horizon-ms 500 \
  --model-root "$MODEL_ROOT" --gt-root "$GT_SEQUENCE_ROOT"

echo "[2/3] LR policy -> bandwidth and QoE"
"$PYTHON" "$REPO/scripts/evaluate_standard_ply_qoe.py" \
  --trace "$TRACE" --decisions "$POLICY_DIR/cell_decisions.csv" \
  --model-root "$MODEL_ROOT" --gt-root "$GT_SEQUENCE_ROOT" \
  --gsplat-library-path "$GSPLAT_LIBRARY_PATH" --output-dir "$QUALITY_DIR" \
  --dataset-variant lut --width 1920 --height 1080 --hfov 77 \
  --cell-size-m 0.2 --sh-degree 3 --prefetch-workers 4 \
  --metric-batch-size 4 --save-every 100 --device cuda

echo "[3/3] Curves"
"$PYTHON" "$REPO/scripts/plot_qoe_bandwidth.py" \
  --metrics "$QUALITY_DIR/per_frame_metrics.csv" \
  --output "$QUALITY_DIR/qoe_bandwidth_overview.png" --fps 30 \
  --title "500 ms DoF-only LR: $TRACE_STEM"
echo "PASS: $RESULT_ROOT"
