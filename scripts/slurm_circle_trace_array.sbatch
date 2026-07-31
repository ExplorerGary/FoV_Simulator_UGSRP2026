#!/bin/bash
#SBATCH --job-name=fov-circle
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script with --array=0-3}"

TRACE_NAMES=(
  "26_7_29_12_33_39"
  "26_7_29_12_35_7"
  "26_7_29_12_37_21"
  "26_7_29_12_40_25"
)

if (( TASK_ID < 0 || TASK_ID >= ${#TRACE_NAMES[@]} )); then
  echo "Array index $TASK_ID is outside 0-$(( ${#TRACE_NAMES[@]} - 1 ))" >&2
  exit 2
fi

TRACE_STEM="${TRACE_NAMES[$TASK_ID]}"
TRACE="$REPO/trace_csvs/$TRACE_STEM.csv"
RESULT_ROOT="/scratch/$USER/fov_trace_eval/${TRACE_STEM}_full"
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/$USER/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_CircleTurns_LUT}"
GT_SEQUENCE_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"

for path in "$REPO" "$TRACE" "$IMAGE" "$OVERLAY" "$MODEL_ROOT" "$GT_SEQUENCE_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done
mkdir -p "$RESULT_ROOT"

echo "Array task: $TASK_ID"
echo "Trace:      $TRACE"
echo "Results:    $RESULT_ROOT"

apptainer exec --nv \
  --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" \
  --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /usr/bin/env \
    REPO="$REPO" \
    TRACE="$TRACE" \
    RESULT_ROOT="$RESULT_ROOT" \
    MODEL_ROOT="$MODEL_ROOT" \
    GT_SEQUENCE_ROOT="$GT_SEQUENCE_ROOT" \
    START_MODE=tracked-gaze \
    FPS=30 \
    WIDTH=1920 \
    HEIGHT=1080 \
    HFOV=77 \
    CELL_SIZE_M=0.2 \
    POLICY_THRESHOLD=0.5 \
    PREFETCH_WORKERS=4 \
    METRIC_BATCH_SIZE=4 \
    SEQUENCE_LABEL=CircleTurns \
    PYTHONUNBUFFERED=1 \
    /bin/bash "$REPO/scripts/run_hpc_trace_eval.sh"
