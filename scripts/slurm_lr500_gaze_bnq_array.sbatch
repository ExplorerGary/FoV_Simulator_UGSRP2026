#!/bin/bash
#SBATCH --job-name=lr500-gaze-bnq
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with --array=0-23%2}"
TRACES=("26_7_29_12_37_21" "26_7_31_15_1_21")
VARIANTS=(
  "base_only" "persistence"
  "v1_t010" "v1_t015" "v1_t020" "v1_t025"
  "gaze_t010" "gaze_t015" "gaze_t020" "gaze_t025"
  "v1_t020_guard6" "gaze_t020_guard6"
)
TRACE_INDEX=$(( TASK_ID % 2 )); VARIANT_INDEX=$(( TASK_ID / 2 ))
(( VARIANT_INDEX < ${#VARIANTS[@]} )) || exit 2
TRACE="$REPO/trace_csvs/${TRACES[$TRACE_INDEX]}.csv"
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
V1_MODEL="/scratch/$USER/fov_visibility_lr/linear_prediction_lr500_raw_gaze_window_baseline/visibility_model_500ms.npz"
GAZE_MODEL="/scratch/$USER/fov_visibility_lr/linear_prediction_lr500_raw_gaze/visibility_model_500ms.npz"
MODEL="$V1_MODEL"; POLICY_MODE="linear"; DECISION_THRESHOLD=""; GUARD_BAND_STEPS=0
case "$VARIANT" in
  base_only) POLICY_MODE="base_only" ;;
  persistence) POLICY_MODE="persistence" ;;
  v1_t010) DECISION_THRESHOLD=0.10 ;;
  v1_t015) DECISION_THRESHOLD=0.15 ;;
  v1_t020) DECISION_THRESHOLD=0.20 ;;
  v1_t025) DECISION_THRESHOLD=0.25 ;;
  gaze_t010) MODEL="$GAZE_MODEL"; DECISION_THRESHOLD=0.10 ;;
  gaze_t015) MODEL="$GAZE_MODEL"; DECISION_THRESHOLD=0.15 ;;
  gaze_t020) MODEL="$GAZE_MODEL"; DECISION_THRESHOLD=0.20 ;;
  gaze_t025) MODEL="$GAZE_MODEL"; DECISION_THRESHOLD=0.25 ;;
  v1_t020_guard6) DECISION_THRESHOLD=0.20; GUARD_BAND_STEPS=1 ;;
  gaze_t020_guard6) MODEL="$GAZE_MODEL"; DECISION_THRESHOLD=0.20; GUARD_BAND_STEPS=1 ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"
apptainer exec --nv --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /usr/bin/env REPO="$REPO" TRACE="$TRACE" VARIANT="$VARIANT" \
  MODEL="$MODEL" POLICY_MODE="$POLICY_MODE" \
  DECISION_THRESHOLD="$DECISION_THRESHOLD" GUARD_BAND_STEPS="$GUARD_BAND_STEPS" \
  REQUIRE_VALID_GAZE_HISTORY=1 \
  BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_lr500_gaze_bnq_v1}" PYTHONUNBUFFERED=1 \
  /bin/bash "$REPO/scripts/run_hpc_lr500_bnq_one.sh"
