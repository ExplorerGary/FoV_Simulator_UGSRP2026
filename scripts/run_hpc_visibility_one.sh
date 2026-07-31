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
  --skip-missing-assets \
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
accounted = sampling["emitted_frames"] + sampling["skipped_samples"]
if accounted != sampling["requested_samples"]:
    raise SystemExit(
        "Unaccounted visibility samples: "
        f"emitted={sampling['emitted_frames']}, "
        f"skipped={sampling['skipped_samples']}, "
        f"requested={sampling['requested_samples']}"
    )
missing_roles = set(sampling["missing_role_counts"])
if missing_roles - {"gt"}:
    raise SystemExit(f"Unexpected missing asset roles: {sorted(missing_roles)}")
skipped_fraction = sampling["skipped_samples"] / sampling["requested_samples"]
if skipped_fraction > 0.05:
    raise SystemExit(f"Too many missing GT samples: {skipped_fraction:.2%}")
print(
    "Visibility complete: "
    f"{sampling['emitted_frames']} emitted, "
    f"{sampling['skipped_samples']} missing-GT samples skipped, "
    f"{metadata['runtime']['frames_per_second']:.3f} fps"
)
PY
