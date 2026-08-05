#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
BNQ_ROOT="${BNQ_ROOT:-/scratch/$USER/fov_dof_lr_e3_c20_bnq_v1}"
mkdir -p "$BNQ_ROOT/logs"
QOE_JOB=$(sbatch --parsable --array=0-1%2 \
  --export="ALL,REPO=$REPO,BNQ_ROOT=$BNQ_ROOT" \
  --output="$BNQ_ROOT/logs/base-only-%A_%a.out" \
  --error="$BNQ_ROOT/logs/base-only-%A_%a.err" \
  "$REPO/scripts/slurm_dof_lr_c20_base_only_array.sbatch")
SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:$QOE_JOB" \
  --export="ALL,REPO=$REPO,BNQ_ROOT=$BNQ_ROOT" \
  --output="$BNQ_ROOT/logs/base-only-summary-%j.out" \
  --error="$BNQ_ROOT/logs/base-only-summary-%j.err" \
  "$REPO/scripts/slurm_dof_lr_c20_base_only_summary.sbatch")
echo "Matched Base-only QoE array job: $QOE_JOB"
echo "Matched Base-only summary job: $SUMMARY_JOB"
echo "Final CSV: $BNQ_ROOT/matched_base_only_aggregate/bnq_summary.csv"
