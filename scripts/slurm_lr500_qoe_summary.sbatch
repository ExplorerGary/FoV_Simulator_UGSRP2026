#!/bin/bash
#SBATCH --job-name=lr500-sum
#SBATCH --account=torch_pr_1093_tandon_priority
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail
REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/$USER/fov_lr500_bandwidth_qoe_v1}"
python "$REPO/scripts/summarize_lr500_qoe.py" \
  --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/aggregate_summary.json"
