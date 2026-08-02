#!/bin/bash
#SBATCH --job-name=dof-lr-c20-bnq
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TRACES=("26_7_29_12_37_21" "26_7_31_15_1_21")
TRACE_STEM="${TRACES[${SLURM_ARRAY_TASK_ID:?Submit with --array=0-1%2}]}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"
apptainer exec --nv --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /usr/bin/env REPO="$REPO" \
  TRACE="$REPO/trace_csvs/$TRACE_STEM.csv" \
  BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_dof_lr_c20_bnq_v1}" \
  PYTHONUNBUFFERED=1 /bin/bash "$REPO/scripts/run_hpc_dof_lr_c20_one.sh"
