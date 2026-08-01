#!/bin/bash
#SBATCH --job-name=lr500-gaze-sum
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_lr500_gaze_bnq_v1}"
GT_SEQUENCE_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
python3 "$REPO/scripts/summarize_lr500_bnq.py" \
  --root "$BNQ_ROOT" --gt-root "$GT_SEQUENCE_ROOT" \
  --experiment "500ms raw-head-history plus gaze Ridge LR matched sweep" \
  --variants base_only persistence \
    v1_t010 v1_t015 v1_t020 v1_t025 \
    gaze_t010 gaze_t015 gaze_t020 gaze_t025 \
    v1_t020_guard6 gaze_t020_guard6
