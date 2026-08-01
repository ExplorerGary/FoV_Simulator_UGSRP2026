#!/bin/bash
#SBATCH --job-name=lr100-sum
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_lr100_bnq_v1}"
GT_SEQUENCE_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
VISIBILITY_ROOT="${VISIBILITY_ROOT:-/scratch/$USER/fov_visibility_lr/visibility}"
python3 "$REPO/scripts/summarize_lr500_bnq.py" \
  --root "$BNQ_ROOT" --gt-root "$GT_SEQUENCE_ROOT" \
  --visibility-root "$VISIBILITY_ROOT" \
  --experiment "LR-V2 candidate: 500ms history to 100ms visibility" \
  --variants base_only persistence head_f2 gaze_f2 \
    head_safe gaze_safe head_f2_guard6 gaze_f2_guard6
