#!/bin/bash
#SBATCH --job-name=vis-score-sum
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_visibility_score_bnq_v1}"
GT_SEQUENCE_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
VISIBILITY_ROOT="${VISIBILITY_ROOT:-/scratch/$USER/fov_visibility_lr/visibility}"
python3 "$REPO/scripts/summarize_lr500_bnq.py" \
  --root "$BNQ_ROOT" --gt-root "$GT_SEQUENCE_ROOT" \
  --visibility-root "$VISIBILITY_ROOT" \
  --experiment "True-pose visibility score-definition and threshold oracle sweep" \
  --variants contrib_t010 contrib_t020 contrib_t030 contrib_t050 \
    raster_t010 raster_t020 raster_t030 raster_t050 \
    image_cov090 image_cov095 image_cov099

