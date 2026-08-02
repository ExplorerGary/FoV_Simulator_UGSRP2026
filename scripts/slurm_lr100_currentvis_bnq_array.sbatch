#!/bin/bash
#SBATCH --job-name=lr100-cv-bnq
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
  "base_only" "persistence" "gaze_f2" "gaze_safe"
  "current_f2" "current_safe" "gaze_f2_guard6" "current_f2_guard6"
)
TRACE_INDEX=$(( TASK_ID % 2 ))
VARIANT_INDEX=$(( TASK_ID / 2 ))
(( VARIANT_INDEX < ${#VARIANTS[@]} )) || exit 2
TRACE="$REPO/trace_csvs/${TRACES[$TRACE_INDEX]}.csv"
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
BASELINE_ROOT="/scratch/$USER/fov_visibility_lr/linear_prediction_lr100_currentvis_matched_gaze"
CURRENT_ROOT="/scratch/$USER/fov_visibility_lr/linear_prediction_lr100_gaze_current_visibility"
BASELINE_MODEL="$BASELINE_ROOT/visibility_model_100ms.npz"
CURRENT_MODEL="$CURRENT_ROOT/visibility_model_100ms.npz"
read_safe_threshold() {
  python3 - "$1" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
safe = summary["horizons"]["100ms"]["safe_threshold_calibration"]
if not safe["constraint_met"]:
    print(
        "WARNING: recall target was not met; using maximum-recall threshold",
        file=sys.stderr,
    )
print(safe["selected_decision_threshold"])
PY
}
MODEL="$BASELINE_MODEL"
POLICY_MODE="linear"
DECISION_THRESHOLD=""
GUARD_BAND_STEPS=0
case "$VARIANT" in
  base_only) POLICY_MODE="base_only" ;;
  persistence) POLICY_MODE="persistence" ;;
  gaze_f2) ;;
  gaze_safe) DECISION_THRESHOLD="$(read_safe_threshold "$BASELINE_ROOT/linear_visibility_summary.json")" ;;
  current_f2) MODEL="$CURRENT_MODEL" ;;
  current_safe) MODEL="$CURRENT_MODEL"; DECISION_THRESHOLD="$(read_safe_threshold "$CURRENT_ROOT/linear_visibility_summary.json")" ;;
  gaze_f2_guard6) GUARD_BAND_STEPS=1 ;;
  current_f2_guard6) MODEL="$CURRENT_MODEL"; GUARD_BAND_STEPS=1 ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"
apptainer exec --nv --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /usr/bin/env REPO="$REPO" TRACE="$TRACE" VARIANT="$VARIANT" \
  MODEL="$MODEL" POLICY_MODE="$POLICY_MODE" \
  DECISION_THRESHOLD="$DECISION_THRESHOLD" GUARD_BAND_STEPS="$GUARD_BAND_STEPS" \
  BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_lr100_currentvis_bnq_v1}" \
  PYTHONUNBUFFERED=1 /bin/bash "$REPO/scripts/run_hpc_lr100_bnq_one.sh"
