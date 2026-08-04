#!/usr/bin/env bash
# Read-only preflight for the two traces with existing DoF predictions.
set -euo pipefail

REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
PREDICTION_ROOT="${PREDICTION_ROOT:-/scratch/$USER/fov_dof_lr_e3_c20_bnq_v1/dof_lr_e3_c20}"
GT_PLY_ROOT="${GT_PLY_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
TRACE_NAMES=(
  "26_7_29_12_37_21"
  "26_7_31_15_1_21"
)

missing=0
if [[ ! -d "$GT_PLY_ROOT" ]]; then
  echo "MISSING GT PLY ROOT: $GT_PLY_ROOT" >&2
  missing=$((missing + 1))
fi
for trace_stem in "${TRACE_NAMES[@]}"; do
  gt_trace="$REPO/trace_csvs/$trace_stem.csv"
  predicted_trace="$PREDICTION_ROOT/$trace_stem/00_pose/predicted_trace.csv"
  if [[ ! -s "$gt_trace" ]]; then
    echo "MISSING GT:        $gt_trace" >&2
    missing=$((missing + 1))
  fi
  if [[ ! -s "$predicted_trace" ]]; then
    echo "MISSING PREDICTED: $predicted_trace" >&2
    missing=$((missing + 1))
  fi
done

if (( missing > 0 )); then
  echo "Preflight failed: $missing required input file(s) missing or empty." >&2
  echo "No model training or prediction was started." >&2
  exit 2
fi
echo "PASS: both GT traces and both existing predicted traces are present."
