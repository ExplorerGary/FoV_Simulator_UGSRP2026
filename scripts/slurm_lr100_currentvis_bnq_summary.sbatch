#!/bin/bash
#SBATCH --job-name=lr100-cv-sum
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_lr100_currentvis_bnq_v1}"
GT_SEQUENCE_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
VISIBILITY_ROOT="${VISIBILITY_ROOT:-/scratch/$USER/fov_visibility_lr/visibility}"
python3 "$REPO/scripts/summarize_lr500_bnq.py" \
  --root "$BNQ_ROOT" --gt-root "$GT_SEQUENCE_ROOT" \
  --visibility-root "$VISIBILITY_ROOT" \
  --experiment "100ms Ridge LR with current cell visibility" \
  --variants base_only persistence gaze_f2 gaze_safe \
    current_f2 current_safe gaze_f2_guard6 current_f2_guard6
