#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_lr100_bnq_v1}"
mkdir -p "$BNQ_ROOT/logs"
TRAIN_JOB=$(sbatch --parsable \
  --output="$BNQ_ROOT/logs/train-%j.out" \
  --error="$BNQ_ROOT/logs/train-%j.err" \
  "$REPO/scripts/slurm_lr100_train.sbatch")
QOE_JOB=$(sbatch --parsable --array=0-15%2 \
  --dependency="afterok:$TRAIN_JOB" --export="ALL,BNQ_ROOT=$BNQ_ROOT" \
  --output="$BNQ_ROOT/logs/qoe-%A_%a.out" \
  --error="$BNQ_ROOT/logs/qoe-%A_%a.err" \
  "$REPO/scripts/slurm_lr100_bnq_array.sbatch")
SUMMARY_JOB=$(sbatch --parsable \
  --dependency="afterok:$QOE_JOB" --export="ALL,BNQ_ROOT=$BNQ_ROOT" \
  --output="$BNQ_ROOT/logs/summary-%j.out" \
  --error="$BNQ_ROOT/logs/summary-%j.err" \
  "$REPO/scripts/slurm_lr100_bnq_summary.sbatch")
echo "100 ms training job: $TRAIN_JOB"
echo "100 ms BNQ array job: $QOE_JOB"
echo "100 ms summary job: $SUMMARY_JOB"
echo "Final CSV: $BNQ_ROOT/bnq_summary.csv"
