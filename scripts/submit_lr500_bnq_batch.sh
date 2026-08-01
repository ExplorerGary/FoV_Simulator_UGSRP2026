#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_lr500_bnq_v2}"
mkdir -p "$BNQ_ROOT/logs"
TRAIN_JOB=$(sbatch --parsable \
  --output="$BNQ_ROOT/logs/train-%j.out" \
  --error="$BNQ_ROOT/logs/train-%j.err" \
  "$REPO/scripts/slurm_lr500_bnq_train.sbatch")
QOE_JOB=$(sbatch --parsable --array=0-15%2 \
  --dependency="afterok:$TRAIN_JOB" \
  --export="ALL,BNQ_ROOT=$BNQ_ROOT" \
  --output="$BNQ_ROOT/logs/qoe-%A_%a.out" \
  --error="$BNQ_ROOT/logs/qoe-%A_%a.err" \
  "$REPO/scripts/slurm_lr500_bnq_array.sbatch")
SUMMARY_JOB=$(sbatch --parsable \
  --dependency="afterok:$QOE_JOB" \
  --export="ALL,BNQ_ROOT=$BNQ_ROOT" \
  --output="$BNQ_ROOT/logs/summary-%j.out" \
  --error="$BNQ_ROOT/logs/summary-%j.err" \
  "$REPO/scripts/slurm_lr500_bnq_summary.sbatch")
echo "Training job: $TRAIN_JOB"
echo "BNQ array job: $QOE_JOB"
echo "Summary job: $SUMMARY_JOB"
echo "Final CSV: $BNQ_ROOT/bnq_summary.csv"
