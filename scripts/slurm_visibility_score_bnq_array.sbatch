#!/bin/bash
#SBATCH --job-name=vis-score-bnq
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with --array=0-21%2}"
TRACES=("26_7_29_12_37_21" "26_7_31_15_1_21")
VARIANTS=(
  "contrib_t010" "contrib_t020" "contrib_t030" "contrib_t050"
  "raster_t010" "raster_t020" "raster_t030" "raster_t050"
  "image_cov090" "image_cov095" "image_cov099"
)
TRACE_INDEX=$(( TASK_ID % 2 ))
VARIANT_INDEX=$(( TASK_ID / 2 ))
(( VARIANT_INDEX < ${#VARIANTS[@]} )) || exit 2
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
TRACE="$REPO/trace_csvs/${TRACES[$TRACE_INDEX]}.csv"
case "$VARIANT" in
  contrib_t*) SCORE="contributing_fraction"; RULE_KIND="threshold"; RULE_VALUE="0.${VARIANT##*t0}" ;;
  raster_t*) SCORE="rasterized_fraction"; RULE_KIND="threshold"; RULE_VALUE="0.${VARIANT##*t0}" ;;
  image_cov090) SCORE="image_share"; RULE_KIND="coverage"; RULE_VALUE="0.90" ;;
  image_cov095) SCORE="image_share"; RULE_KIND="coverage"; RULE_VALUE="0.95" ;;
  image_cov099) SCORE="image_share"; RULE_KIND="coverage"; RULE_VALUE="0.99" ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"
apptainer exec --nv --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /usr/bin/env REPO="$REPO" TRACE="$TRACE" VARIANT="$VARIANT" \
  SCORE="$SCORE" RULE_KIND="$RULE_KIND" RULE_VALUE="$RULE_VALUE" \
  BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_visibility_score_bnq_v1}" \
  PYTHONUNBUFFERED=1 /bin/bash "$REPO/scripts/run_hpc_visibility_score_bnq_one.sh"

