#!/bin/bash
#SBATCH --job-name=lr500-qoe
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with --array=0-1%2}"
TRACE_NAMES=("26_7_29_12_37_21" "26_7_31_15_1_21")
(( TASK_ID >= 0 && TASK_ID < ${#TRACE_NAMES[@]} )) || exit 2
TRACE="$REPO/trace_csvs/${TRACE_NAMES[$TASK_ID]}.csv"
IMAGE="${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/$USER/gaussian_coding_env/gsplat-hpc.ext3}"
for path in "$REPO" "$TRACE" "$IMAGE" "$OVERLAY"; do
  [[ -e "$path" ]] || { echo "Required path is missing: $path" >&2; exit 2; }
done
apptainer exec --nv --overlay "${OVERLAY}:ro" \
  --bind "/home/$USER:/home/$USER" --bind "/scratch/$USER:/scratch/$USER" \
  "$IMAGE" /usr/bin/env REPO="$REPO" TRACE="$TRACE" PYTHONUNBUFFERED=1 \
  /bin/bash "$REPO/scripts/run_hpc_lr500_qoe_one.sh"
