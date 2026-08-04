#!/bin/bash
#SBATCH --job-name=dof-trajectory
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail

REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with --array=0-9}"
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

TRACE_STEM="${TRACE_NAMES[$TASK_ID]}"
GT_TRACE="$REPO/trace_csvs/$TRACE_STEM.csv"
PREDICTION_ROOT="${PREDICTION_ROOT:-/scratch/$USER/fov_dof_lr_e3_c20_bnq_v1/dof_lr_e3_c20}"
PREDICTED_TRACE="$PREDICTION_ROOT/$TRACE_STEM/00_pose/predicted_trace.csv"
GT_PLY_ROOT="${GT_PLY_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
RESULT_ROOT="${RESULT_ROOT:-/scratch/$USER/fixed_view_dof_trajectory}"
OUTPUT_DIR="$RESULT_ROOT/$TRACE_STEM"
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"

for path in "$REPO" "$GT_TRACE" "$PREDICTED_TRACE" "$GT_PLY_ROOT" "$IMAGE" "$OVERLAY"; do
  if [[ ! -e "$path" ]]; then
    echo "Required existing path is missing: $path" >&2
    echo "This visualization job does not train a model or regenerate predicted traces." >&2
    exit 2
  fi
done
mkdir -p "$OUTPUT_DIR"

apptainer exec \
  --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" \
  --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" "$PYTHON" "$REPO/scripts/plot_fixed_view_camera_trajectory.py" \
    --gt-trace "$GT_TRACE" \
    --predicted-trace "$PREDICTED_TRACE" \
    --gt-ply-root "$GT_PLY_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --start-mode tracked-gaze \
    --particle-count 240 \
    --point-cloud-points 60000 \
    --camera-a-elevation 24 \
    --camera-a-azimuth -58 \
    --camera-b-elevation 18 \
    --camera-b-azimuth 122 \
    --dpi 180
