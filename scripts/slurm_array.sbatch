#!/bin/bash
#SBATCH --job-name=fovsim
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00

set -euo pipefail

MANIFEST="${1:?Usage: sbatch --array=0-N scripts/slurm_array.sbatch MANIFEST.jsonl}"
JOB_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"

# Each array task owns one independent trace job. Avoid nested thread
# oversubscription when many tasks share a compute node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

fovsim manifest-job \
  --manifest "${MANIFEST}" \
  --job-index "${JOB_INDEX}"
