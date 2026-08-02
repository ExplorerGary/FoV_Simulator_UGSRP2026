#!/bin/bash
#SBATCH --job-name=dof-lr-c20-sum
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_dof_lr_c20_bnq_v1}"
GT_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
python3 "$REPO/scripts/summarize_lr500_bnq.py" \
  --root "$BNQ_ROOT" --gt-root "$GT_ROOT" \
  --visibility-root "$BNQ_ROOT/reference_visibility" \
  --experiment "500ms DoF OLS to 100ms pose, predicted-pose contribution C20" \
  --variants dof_lr_c20
