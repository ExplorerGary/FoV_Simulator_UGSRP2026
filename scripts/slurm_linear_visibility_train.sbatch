#!/bin/bash
#SBATCH --job-name=fov-lr-train
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/scratch/$USER/fov_visibility_lr}"
VISIBILITY_ROOT="$EXPERIMENT_ROOT/visibility"
OUTPUT_ROOT="$EXPERIMENT_ROOT/linear_prediction"
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"

for path in "$REPO" "$VISIBILITY_ROOT" "$IMAGE" "$OVERLAY"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done

if [[ "$(find "$VISIBILITY_ROOT" -maxdepth 1 -type f -name '*.csv' | wc -l)" -ne 10 ]]; then
  echo "Expected exactly 10 visibility CSVs in $VISIBILITY_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
apptainer exec \
  --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" \
  --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /usr/bin/env \
    PYTHONPATH="$REPO/src" \
    OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
    MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
    OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
    /ext3/envs/gsplat-hpc/bin/python -m fovsim predict-linear \
      --trace-dir "$REPO/trace_csvs" \
      --visibility-dir "$VISIBILITY_ROOT" \
      --output-dir "$OUTPUT_ROOT" \
      --history-ms 500 \
      --horizons-ms 100 200 500 \
      --visibility-threshold 0.5 \
      --test-fraction 0.2 \
      --seed 20260731 \
      --ridge-alpha 1.0 \
      --expected-traces 10
