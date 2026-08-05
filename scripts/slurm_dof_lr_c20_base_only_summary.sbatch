#!/bin/bash
#SBATCH --job-name=c20-base-sum
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_dof_lr_e3_c20_bnq_v1}"
GT_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
SUMMARY_ROOT="$BNQ_ROOT/matched_base_only_aggregate"
mkdir -p "$SUMMARY_ROOT"
ln -sfn "$BNQ_ROOT/matched_base_only" "$SUMMARY_ROOT/matched_base_only"
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"
apptainer exec --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /ext3/envs/gsplat-hpc/bin/python \
  "$REPO/scripts/summarize_lr500_bnq.py" \
    --root "$SUMMARY_ROOT" --gt-root "$GT_ROOT" \
    --experiment "Matched Base-only for DoF-LR C20 frame set" \
    --variants matched_base_only
echo "Final CSV: $SUMMARY_ROOT/bnq_summary.csv"
