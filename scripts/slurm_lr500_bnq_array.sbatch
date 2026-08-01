#!/bin/bash
#SBATCH --job-name=lr500-bnq
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with --array=0-15%2}"
TRACES=("26_7_29_12_37_21" "26_7_31_15_1_21")
VARIANTS=(
  "base_only" "persistence" "lr_v1_t020" "lr_v2_t010"
  "lr_v2_t015" "lr_v2_t020" "lr_v2_t025" "lr_v2_t020_guard6"
)
TRACE_INDEX=$(( TASK_ID % 2 ))
VARIANT_INDEX=$(( TASK_ID / 2 ))
(( VARIANT_INDEX < ${#VARIANTS[@]} )) || exit 2
TRACE="$REPO/trace_csvs/${TRACES[$TRACE_INDEX]}.csv"
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
V1_MODEL="/scratch/$USER/fov_visibility_lr/linear_prediction_lr500_binary/visibility_model_500ms.npz"
V2_MODEL="/scratch/$USER/fov_visibility_lr/linear_prediction_lr500_motion_quadratic/visibility_model_500ms.npz"
MODEL="$V2_MODEL"; POLICY_MODE="linear"; DECISION_THRESHOLD=""; GUARD_BAND_STEPS=0
case "$VARIANT" in
  base_only) POLICY_MODE="base_only" ;;
  persistence) POLICY_MODE="persistence" ;;
  lr_v1_t020) MODEL="$V1_MODEL"; DECISION_THRESHOLD=0.20 ;;
  lr_v2_t010) DECISION_THRESHOLD=0.10 ;;
  lr_v2_t015) DECISION_THRESHOLD=0.15 ;;
  lr_v2_t020) DECISION_THRESHOLD=0.20 ;;
  lr_v2_t025) DECISION_THRESHOLD=0.25 ;;
  lr_v2_t020_guard6) DECISION_THRESHOLD=0.20; GUARD_BAND_STEPS=1 ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"
apptainer exec --nv --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /usr/bin/env REPO="$REPO" TRACE="$TRACE" VARIANT="$VARIANT" \
  MODEL="$MODEL" POLICY_MODE="$POLICY_MODE" \
  DECISION_THRESHOLD="$DECISION_THRESHOLD" GUARD_BAND_STEPS="$GUARD_BAND_STEPS" \
  BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_lr500_bnq_v2}" PYTHONUNBUFFERED=1 \
  /bin/bash "$REPO/scripts/run_hpc_lr500_bnq_one.sh"
