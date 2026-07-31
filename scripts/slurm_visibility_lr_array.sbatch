#!/bin/bash
#SBATCH --job-name=fov-vis-lr
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail

REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script with --array=0-9}"
TRACE_NAMES=(
  "26_7_29_12_33_39"
  "26_7_29_12_35_7"
  "26_7_29_12_37_21"
  "26_7_29_12_40_25"
  "26_7_31_14_59_37"
  "26_7_31_15_1_21"
  "26_7_31_15_3_19"
  "26_7_31_15_5_13"
  "26_7_31_15_6_30"
  "26_7_31_15_7_7"
)

if (( TASK_ID < 0 || TASK_ID >= ${#TRACE_NAMES[@]} )); then
  echo "Array index $TASK_ID is outside 0-$(( ${#TRACE_NAMES[@]} - 1 ))" >&2
  exit 2
fi

TRACE="$REPO/trace_csvs/${TRACE_NAMES[$TASK_ID]}.csv"
VISIBILITY_ROOT="${VISIBILITY_ROOT:-/scratch/$USER/fov_visibility_lr/visibility}"
GT_SEQUENCE_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"

for path in "$REPO" "$TRACE" "$GT_SEQUENCE_ROOT" "$IMAGE" "$OVERLAY"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done

apptainer exec --nv \
  --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" \
  --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /usr/bin/env \
    REPO="$REPO" \
    TRACE="$TRACE" \
    VISIBILITY_ROOT="$VISIBILITY_ROOT" \
    GT_SEQUENCE_ROOT="$GT_SEQUENCE_ROOT" \
    PYTHONUNBUFFERED=1 \
    /bin/bash "$REPO/scripts/run_hpc_visibility_one.sh"
