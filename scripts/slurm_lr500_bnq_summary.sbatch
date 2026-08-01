#!/bin/bash
#SBATCH --job-name=lr500-bnq-sum
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_lr500_bnq_v2}"
GT_SEQUENCE_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
python3 "$REPO/scripts/summarize_lr500_bnq.py" \
  --root "$BNQ_ROOT" --gt-root "$GT_SEQUENCE_ROOT"
