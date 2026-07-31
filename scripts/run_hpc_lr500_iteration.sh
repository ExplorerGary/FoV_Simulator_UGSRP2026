#!/usr/bin/env bash
# Train and evaluate one simple binary-target 500 ms LR iteration.
set -euo pipefail

REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/scratch/$USER/fov_visibility_lr}"
VISIBILITY_ROOT="$EXPERIMENT_ROOT/visibility"
OUTPUT_ROOT="$EXPERIMENT_ROOT/linear_prediction_lr500_binary"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"

for path in "$REPO" "$VISIBILITY_ROOT" "$PYTHON"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done
if [[ "$(find "$VISIBILITY_ROOT" -maxdepth 1 -type f -name '*.csv' | wc -l)" -ne 10 ]]; then
  echo "Expected exactly 10 visibility CSVs in $VISIBILITY_ROOT" >&2
  exit 2
fi

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$OUTPUT_ROOT"

"$PYTHON" -m fovsim predict-linear \
  --trace-dir "$REPO/trace_csvs" \
  --visibility-dir "$VISIBILITY_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --history-ms 500 \
  --horizons-ms 500 \
  --target-mode binary \
  --visibility-threshold 0.5 \
  --decision-threshold-min 0.01 \
  --decision-threshold-max 0.5 \
  --decision-threshold-steps 50 \
  --test-fraction 0.2 \
  --seed 20260731 \
  --ridge-alpha 1.0 \
  --expected-traces 10

"$PYTHON" "$REPO/scripts/summarize_lr500_result.py" \
  --summary "$OUTPUT_ROOT/linear_visibility_summary.json" \
  --output "$OUTPUT_ROOT/lr500_evaluation.json"

echo "Per-trace metrics: $OUTPUT_ROOT/per_trace_metrics.csv"
echo "Evaluation report: $OUTPUT_ROOT/lr500_evaluation.json"
