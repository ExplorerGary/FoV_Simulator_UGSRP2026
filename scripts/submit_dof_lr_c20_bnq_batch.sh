#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_dof_lr_c20_bnq_v1}"
mkdir -p "$BNQ_ROOT/logs"
TRAIN_JOB=$(sbatch --parsable --export="ALL,REPO=$REPO,BNQ_ROOT=$BNQ_ROOT" \
  --output="$BNQ_ROOT/logs/train-%j.out" --error="$BNQ_ROOT/logs/train-%j.err" \
  "$REPO/scripts/slurm_dof_lr_c20_train.sbatch")
QOE_JOB=$(sbatch --parsable --array=0-1%2 --dependency="afterok:$TRAIN_JOB" \
  --export="ALL,REPO=$REPO,BNQ_ROOT=$BNQ_ROOT" \
  --output="$BNQ_ROOT/logs/qoe-%A_%a.out" --error="$BNQ_ROOT/logs/qoe-%A_%a.err" \
  "$REPO/scripts/slurm_dof_lr_c20_array.sbatch")
SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:$QOE_JOB" \
  --export="ALL,REPO=$REPO,BNQ_ROOT=$BNQ_ROOT" \
  --output="$BNQ_ROOT/logs/summary-%j.out" --error="$BNQ_ROOT/logs/summary-%j.err" \
  "$REPO/scripts/slurm_dof_lr_c20_summary.sbatch")
echo "DoF LR training job: $TRAIN_JOB"
echo "DoF LR C20 BNQ array job: $QOE_JOB"
echo "DoF LR C20 summary job: $SUMMARY_JOB"
echo "Final CSV: $BNQ_ROOT/bnq_summary.csv"
