#!/usr/bin/env bash
# Generate complete GT-derived visibility for one CircleTurns trace.
set -euo pipefail

REPO="${REPO:-$HOME/FoV_Simulator_UGSRP2026}"
TRACE="${TRACE:?TRACE is required}"
VISIBILITY_ROOT="${VISIBILITY_ROOT:-/scratch/$USER/fov_visibility_lr/visibility}"
GT_SEQUENCE_ROOT="${GT_SEQUENCE_ROOT:-/scratch/$USER/DanceNet3D_GT/BiancaGolden_CircleTurns}"
PYTHON="${PYTHON:-/ext3/envs/gsplat-hpc/bin/python}"

for path in "$REPO" "$TRACE" "$GT_SEQUENCE_ROOT" "$PYTHON"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done

TRACE_STEM="$(basename "${TRACE%.csv}")"
mkdir -p "$VISIBILITY_ROOT" "$VISIBILITY_ROOT/metadata"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
GSPLAT_LIBRARY_PATH="${GSPLAT_LIBRARY_PATH:-$(
  "$PYTHON" -c \
    'import pathlib, gsplat; print(pathlib.Path(gsplat.__file__).resolve().parent.parent)'
)}"

echo "Trace:      $TRACE"
echo "GT:         $GT_SEQUENCE_ROOT"
echo "Visibility: $VISIBILITY_ROOT/$TRACE_STEM.csv"

"$PYTHON" "$REPO/scripts/generate_standard_ply_visibility.py" \
  --trace "$TRACE" \
  --gt-root "$GT_SEQUENCE_ROOT" \
  --output "$VISIBILITY_ROOT/$TRACE_STEM.csv" \
  --metadata "$VISIBILITY_ROOT/metadata/$TRACE_STEM.json" \
  --gsplat-library-path "$GSPLAT_LIBRARY_PATH" \
  --start-mode tracked-gaze \
  --fps 30 \
  --width 1920 \
  --height 1080 \
  --hfov 77 \
  --cell-size-m 0.2 \
  --visibility-weight-threshold 0.00392156862745098 \
  --device cuda

"$PYTHON" - "$VISIBILITY_ROOT/metadata/$TRACE_STEM.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = json.loads(path.read_text(encoding="utf-8"))
sampling = metadata["sampling"]
if metadata["pipeline_status"] != "PASS":
    raise SystemExit(f"Visibility failed: {path}")
if sampling["emitted_frames"] != sampling["requested_samples"]:
    raise SystemExit(
        "Incomplete visibility: "
        f"{sampling['emitted_frames']}/{sampling['requested_samples']}"
    )
print(
    "Visibility complete: "
    f"{sampling['emitted_frames']} frames, "
    f"{metadata['runtime']['frames_per_second']:.3f} fps"
)
PY
