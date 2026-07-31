# FoV Progressive Simulation

Portable infrastructure for offline Base/Enhancement-3 experiments on HPC.
It does not depend on Unreal Engine and intentionally does not version datasets,
GSV/PLY assets, videos, or generated results.

The initial policy is:

```text
contributing_gaussian_fraction >= 0.5 -> Base + Enhancement 3
contributing_gaussian_fraction <  0.5 -> Base only
```

## Inputs

Each simulation job takes:

1. A raw 6DoF trace CSV with the project columns `Location*`, `Rotation*`,
   `Frame`, and `Timestamp`.
2. A per-frame/per-cell visibility CSV exported by the validated POV renderer.

The simulator cross-checks `trace_source_row`, `trace_timestamp_s`, and
`gsv_frame` before applying the policy.
`scripts/generate_standard_ply_visibility.py` provides the formal HPC
visibility stage using DanceNet3D GT PLYs and gsplat CUDA. It computes
front-to-back per-Gaussian `T*alpha` attribution without reading Base or E3.
The original GSV renderer remains useful for local cross-validation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[plot]"
```

The policy core uses only the Python standard library. Matplotlib is optional
and needed only for PNG plots.

For EVOGS-v1 rendering and quality metrics, use an environment containing
PyTorch plus a compiled `gsplat`, then install:

```bash
python -m pip install -e ".[quality]"
```

Run the dependency-free test suite with:

```bash
python -m unittest discover -s tests -v
```

## Validate a trace

```bash
fovsim validate-trace --trace /data/user01.csv
```

## Run one simulation

```bash
fovsim run \
  --trace /data/user01.csv \
  --visibility /data/user01_cells.csv \
  --output-dir outputs/user01 \
  --threshold 0.5 \
  --first-frame 1 \
  --frame-count 30 \
  --plot
```

Outputs:

```text
cell_decisions.csv
frame_summary.csv
run_metadata.json
policy.png                 # only with --plot
```

`target_level` is `0` for Base and `3` for Base + Enhancement 3.

## Run many traces in parallel

Create a JSON Lines manifest with one object per job:

```json
{"name":"user01","trace":"/data/user01.csv","visibility":"/data/user01_cells.csv","output_dir":"outputs/user01","threshold":0.5,"first_frame":1,"frame_count":30,"plot":false}
{"name":"user02","trace":"/data/user02.csv","visibility":"/data/user02_cells.csv","output_dir":"outputs/user02","threshold":0.5,"first_frame":1,"frame_count":30,"plot":false}
```

Then:

```bash
fovsim batch --manifest jobs.jsonl --workers 8
```

Parallelism is job-level: each worker streams one visibility CSV in a single
pass and keeps only per-frame aggregates in memory. This avoids memory
duplication and gives predictable scaling on multi-core HPC nodes.

For a cluster, prefer one manifest row per Slurm array task:

```bash
sbatch --array=1-100 scripts/slurm_array.sbatch jobs.jsonl
```

The array index is one-based: task `1` executes the first non-empty JSONL row.

## EVOGS-v1 quality evaluation

The official HPC path uses only standard 3DGS PLY artifacts:

```text
scripts/evaluate_standard_ply_qoe.py
```

For each trace frame it loads the canonical `SINGLE` Base PLY, complete E3
PLY, and DanceNet3D GT PLY. The transmitted policy is constructed spatially:
Base Gaussians are kept in non-selected cells and E3 Gaussians are used in
selected cells. This directly models independently packaged Gaussian cell
chunks and does not require a training checkpoint.

```text
GT     = Render(DanceNet3D GT, same camera)
E3     = Render(complete E3, same camera)
Policy = Render(Base + FoV-selected E3, same camera)
```

Every frame reports E3-vs-GT, Policy-vs-GT, and Policy-vs-E3. Each pair has
both full-frame and foreground PSNR/SSIM/LPIPS-Alex. Foreground is defined from
the pair's reference alpha (`alpha >= 1/255`): GT is the reference for the
first two pairs and E3 for Policy-vs-E3. Foreground PSNR is normalized only
over mask pixels. For SSIM/LPIPS both images are masked to the common
background and evaluated on a tight reference-mask crop plus 24 pixels of
context.

Colors use camera-aware SH evaluation in the original DanceNet coordinates
before geometry is transformed into GSV/Unreal coordinates. Formal jobs must
use the `_LUT` model root and `--dataset-variant lut`; the CLI rejects an
accidental non-LUT path.

```bash
python scripts/evaluate_standard_ply_qoe.py \
  --trace /data/trace.csv \
  --decisions outputs/user01/cell_decisions.csv \
  --model-root \
    /scratch/zg2598/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_CircleTurns_LUT \
  --gt-root \
    /scratch/zg2598/DanceNet3D_GT/BiancaGolden_CircleTurns \
  --gsplat-library-path /opt/gsplat/build/lib.linux-x86_64-cpython-311 \
  --output-dir outputs/user01/quality \
  --dataset-variant lut \
  --sh-degree 3 \
  --prefetch-workers 2 \
  --metric-batch-size 4
```

Outputs are `per_frame_metrics.csv`, `summary.json`, and lossless visual sample
PNGs. `summary.json` includes throughput and a phase breakdown for asset
loading, cell-policy composition, GT/E3/policy rendering, metric computation,
and sample/output writes. CPU workers use bounded look-ahead so PLY parsing
overlaps GPU work without retaining the whole sequence in memory.

The metrics CSV also reports the simulated transmission payload:

```text
base_only_transmission_bytes
policy_transmission_bytes
full_progressive_transmission_bytes
policy_savings_vs_full_fraction
```

The model counts the complete Base PLY plus fixed-width E3 Gaussian records in
selected cells. It intentionally excludes unspecified network/container
headers. Generate the joint bandwidth and QoE figure with:

```bash
python scripts/plot_qoe_bandwidth.py \
  --metrics outputs/user01/quality/per_frame_metrics.csv \
  --output outputs/user01/quality/qoe_bandwidth_overview.png
```

## One-command HPC trace evaluation

Inside an allocated GPU node and the documented Apptainer environment:

```bash
REPO=$HOME/FoV_Simulator_UGSRP2026 \
bash scripts/run_hpc_trace_eval.sh
```

The defaults evaluate `trace_csvs/26_7_29_12_33_39.csv`, sample the tracked
interval at 30 fps, use the CircleTurns LUT and GT roots from `address.md`,
skip frame IDs without trained assets, apply threshold 0.5, and write results
below `/scratch/$USER/fov_trace_eval/26_7_29_12_33_39`.

`scripts/evaluate_evogs_v1.py` is retained only for legacy experiments that
need checkpoint/tree-lineage reconstruction.

See [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) for exact schemas and
[docs/PARALLELISM.md](docs/PARALLELISM.md) for HPC scaling guidance.
