#!/bin/bash
#SBATCH --job-name=lr500-v2-fit
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"
apptainer exec --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /usr/bin/env REPO="$REPO" PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
  MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
  OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
  /bin/bash "$REPO/scripts/run_hpc_lr500_bnq_train.sh"
