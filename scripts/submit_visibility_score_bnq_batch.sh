#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_visibility_score_bnq_v1}"
mkdir -p "$BNQ_ROOT/logs"
QOE_JOB=$(sbatch --parsable --array=0-21%2 \
  --export="ALL,BNQ_ROOT=$BNQ_ROOT" \
  --output="$BNQ_ROOT/logs/qoe-%A_%a.out" \
  --error="$BNQ_ROOT/logs/qoe-%A_%a.err" \
  "$REPO/scripts/slurm_visibility_score_bnq_array.sbatch")
SUMMARY_JOB=$(sbatch --parsable \
  --dependency="afterok:$QOE_JOB" --export="ALL,BNQ_ROOT=$BNQ_ROOT" \
  --output="$BNQ_ROOT/logs/summary-%j.out" \
  --error="$BNQ_ROOT/logs/summary-%j.err" \
  "$REPO/scripts/slurm_visibility_score_bnq_summary.sbatch")
echo "Visibility-score BNQ array job: $QOE_JOB"
echo "Visibility-score summary job: $SUMMARY_JOB"
echo "Final CSV: $BNQ_ROOT/bnq_summary.csv"

