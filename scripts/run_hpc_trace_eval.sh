#!/usr/bin/env bash
# Run inside the project Apptainer container on one allocated GPU node.
set -euo pipefail

REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TRACE="${TRACE:-$REPO/trace_csvs/26_7_29_12_33_39.csv}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/$USER/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_CircleTurns_LUT}"
GT_ROOT="${GT_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
RESULT_ROOT="${RESULT_ROOT:-/scratch/$USER/fov_trace_eval/26_7_29_12_33_39}"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"

START_TIME="${START_TIME:-1.156988}"
DURATION="${DURATION:-57.292860}"
FPS="${FPS:-30}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"
HFOV="${HFOV:-77}"
CELL_SIZE_M="${CELL_SIZE_M:-0.2}"
POLICY_THRESHOLD="${POLICY_THRESHOLD:-0.5}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-4}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-4}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python is not executable: $PYTHON" >&2
  exit 2
fi
for path in "$REPO" "$TRACE" "$MODEL_ROOT" "$GT_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done

GSPLAT_LIBRARY_PATH="${GSPLAT_LIBRARY_PATH:-$(
  "$PYTHON" -c \
    'import pathlib, gsplat; print(pathlib.Path(gsplat.__file__).resolve().parent.parent)'
)}"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

VISIBILITY_DIR="$RESULT_ROOT/01_visibility"
POLICY_DIR="$RESULT_ROOT/02_policy_threshold050"
QUALITY_DIR="$RESULT_ROOT/03_quality_lut"
mkdir -p "$VISIBILITY_DIR" "$POLICY_DIR" "$QUALITY_DIR"

echo "Repository: $REPO"
echo "Trace:      $TRACE"
echo "Model root: $MODEL_ROOT"
echo "GT root:    $GT_ROOT"
echo "Results:    $RESULT_ROOT"
echo "Interval:   start=$START_TIME duration=$DURATION fps=$FPS"

echo "[1/4] GT PLY -> per-cell visibility"
"$PYTHON" "$REPO/scripts/generate_standard_ply_visibility.py" \
  --trace "$TRACE" \
  --gt-root "$GT_ROOT" \
  --require-model-assets-root "$MODEL_ROOT" \
  --output "$VISIBILITY_DIR/cell_visibility.csv" \
  --metadata "$VISIBILITY_DIR/run_metadata.json" \
  --gsplat-library-path "$GSPLAT_LIBRARY_PATH" \
  --start-time "$START_TIME" \
  --duration "$DURATION" \
  --fps "$FPS" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --hfov "$HFOV" \
  --cell-size-m "$CELL_SIZE_M" \
  --visibility-weight-threshold 0.00392156862745098 \
  --skip-missing-assets \
  --device cuda

echo "[2/4] Visibility -> threshold-0.5 decisions"
"$PYTHON" -m fovsim run \
  --trace "$TRACE" \
  --visibility "$VISIBILITY_DIR/cell_visibility.csv" \
  --output-dir "$POLICY_DIR" \
  --threshold "$POLICY_THRESHOLD" \
  --plot

echo "[3/4] LUT Base/E3/GT -> QoE and bandwidth"
"$PYTHON" "$REPO/scripts/evaluate_standard_ply_qoe.py" \
  --trace "$TRACE" \
  --decisions "$POLICY_DIR/cell_decisions.csv" \
  --model-root "$MODEL_ROOT" \
  --gt-root "$GT_ROOT" \
  --gsplat-library-path "$GSPLAT_LIBRARY_PATH" \
  --output-dir "$QUALITY_DIR" \
  --dataset-variant lut \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --hfov "$HFOV" \
  --cell-size-m "$CELL_SIZE_M" \
  --sh-degree 3 \
  --prefetch-workers "$PREFETCH_WORKERS" \
  --metric-batch-size "$METRIC_BATCH_SIZE" \
  --save-every 100 \
  --device cuda

echo "[4/4] Render bandwidth + PSNR/SSIM/LPIPS curves"
"$PYTHON" "$REPO/scripts/plot_qoe_bandwidth.py" \
  --metrics "$QUALITY_DIR/per_frame_metrics.csv" \
  --output "$QUALITY_DIR/qoe_bandwidth_overview.png" \
  --fps "$FPS" \
  --title "CircleTurns: 26_7_29_12_33_39 (LUT, threshold=0.5)"

echo "PASS: $RESULT_ROOT"
