#!/bin/bash
#SBATCH --job-name=dof-lr-c20-train
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_dof_lr_e3_c20_bnq_v1}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"
mkdir -p "$BNQ_ROOT/00_dof_lr_model"
apptainer exec --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /usr/bin/env REPO="$REPO" BNQ_ROOT="$BNQ_ROOT" \
  PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1 \
  /ext3/envs/gsplat-hpc/bin/python "$REPO/scripts/run_dof_lr_experiment.py" \
  --trace-dir "$REPO/trace_csvs" --output-dir "$BNQ_ROOT/00_dof_lr_model" \
  --fps 30 --history-ms 500 --horizon-ms 100 --seed 20260731
