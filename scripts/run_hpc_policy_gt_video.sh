#!/usr/bin/env bash
# Run inside the project Apptainer container on one allocated GPU node.
set -euo pipefail

REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TRACE="${TRACE:?TRACE is required}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT is required}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/$USER/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_CircleTurns_LUT}"
GT_SEQUENCE_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"
DECISIONS="${DECISIONS:-$RESULT_ROOT/02_policy_threshold050/cell_decisions.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/04_policy_gt_video}"

for path in "$REPO" "$TRACE" "$MODEL_ROOT" "$GT_SEQUENCE_ROOT" "$DECISIONS"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done

GSPLAT_LIBRARY_PATH="${GSPLAT_LIBRARY_PATH:-$(
  "$PYTHON" -c \
    'import pathlib, gsplat; print(pathlib.Path(gsplat.__file__).resolve().parent.parent)'
)}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"$PYTHON" "$REPO/scripts/render_policy_gt_comparison_video.py" \
  --trace "$TRACE" \
  --decisions "$DECISIONS" \
  --model-root "$MODEL_ROOT" \
  --gt-root "$GT_SEQUENCE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --gsplat-library-path "$GSPLAT_LIBRARY_PATH" \
  --width "${WIDTH:-1920}" \
  --height "${HEIGHT:-1080}" \
  --hfov "${HFOV:-77}" \
  --cell-size-m "${CELL_SIZE_M:-0.2}" \
  --fps "${FPS:-30}" \
  --prefetch-workers "${PREFETCH_WORKERS:-4}" \
  --title-font-size "${TITLE_FONT_SIZE:-72}" \
  --detail-font-size "${DETAIL_FONT_SIZE:-34}" \
  --crf "${CRF:-18}" \
  --encoder-preset "${ENCODER_PRESET:-veryfast}" \
  --device cuda
