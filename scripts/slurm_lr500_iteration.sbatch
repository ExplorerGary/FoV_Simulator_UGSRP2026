#!/bin/bash
#SBATCH --job-name=fov-lr500
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/scratch/$USER/fov_visibility_lr}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"

for path in "$REPO" "$IMAGE" "$OVERLAY"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done

apptainer exec \
  --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" \
  --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /usr/bin/env \
    REPO="$REPO" \
    EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
    /bin/bash "$REPO/scripts/run_hpc_lr500_iteration.sh"
