# DanceNet3D / EvoGS address book

这是跨终端、跨对话使用的稳定地址簿。Slurm job ID、计算节点名和登录
节点会变化，因此不记录为固定地址。

## Canonical repository

```text
GitHub:
https://github.com/ExplorerGary/Gaussian_Coding_UGSRP2026

HPC checkout:
/home/zg2598/Gaussian_Coding_UGSRP2026
```

更新 HPC checkout：

```bash
cd ~/Gaussian_Coding_UGSRP2026
git switch main
git pull --ff-only origin main
git log -1 --oneline
```

`Dancenet3D_Data_Generator_UGSRP2026` 是数据生成仓库，不是训练脚本的
push 目标。

## Apptainer and Python environment

```text
CUDA image:
/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

Overlay:
/scratch/zg2598/gaussian_coding_env/gsplat-hpc.ext3

Conda activation script:
/ext3/miniforge3/etc/profile.d/conda.sh

Conda environment:
/ext3/envs/gsplat-hpc

Python:
/ext3/envs/gsplat-hpc/bin/python
```

标准进入方式：

```bash
export IMAGE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
export OVERLAY=/scratch/zg2598/gaussian_coding_env/gsplat-hpc.ext3

apptainer exec --nv \
  --overlay "${OVERLAY}:ro" \
  --bind /home/$USER:/home/$USER \
  --bind /scratch/$USER:/scratch/$USER \
  "$IMAGE" /bin/bash

source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate /ext3/envs/gsplat-hpc
cd ~/Gaussian_Coding_UGSRP2026
```

## LUT input datasets

### BiancaGolden CircleTurns

```text
/scratch/zg2598/PREPROCESSED_DATASETS_LUT_STAGING_14966059/DanceNet3D/BiancaGolden_CircleTurns/BiancaGolden_CircleTurns_res1
```

Nominal frame range is `0000001`–`0000433`. There are 426 actual numeric
input-frame directories; absent frame IDs are not training targets.

### BiancaGolden GrandPlies

```text
/scratch/zg2598/PREPROCESSED_DATASETS_GRANDPLIES_LUT/DanceNet3D/BiancaGolden_GrandPlies/BiancaGolden_GrandPlies_res1
```

Frame range is `0000001`–`0001061`, with 1061 input frames.

## Ground-truth PLY datasets

```text
GT root:
/scratch/zg2598/DanceNet3D_GT

CircleTurns:
/scratch/zg2598/DanceNet3D_GT/BiancaGolden_CircleTurns

GrandPlies:
/scratch/zg2598/DanceNet3D_GT/BiancaGolden_GrandPlies
```

GT filenames use seven-digit frame IDs, for example:

```text
/scratch/zg2598/DanceNet3D_GT/BiancaGolden_CircleTurns/0000001.ply
```

## LUT training outputs

### CircleTurns

```text
/scratch/zg2598/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_CircleTurns_LUT
```

### GrandPlies

```text
/scratch/zg2598/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_GrandPlies_LUT
```

Per-frame output layout:

```text
Base:
$OUTPUT_ROOT/SINGLE/<FRAME>_aggressive_base_random/ply/point_cloud_8999.ply

Level-1:
$OUTPUT_ROOT/EVOGS_V1/<FRAME>/enhancement_01_enhanced/ply/<FRAME>_enhancement.ply

Level-2:
$OUTPUT_ROOT/EVOGS_V1/<FRAME>/enhancement_02_enhanced/ply/<FRAME>_enhancement.ply

Level-3:
$OUTPUT_ROOT/EVOGS_V1/<FRAME>/enhancement_03_enhanced/ply/<FRAME>_enhancement.ply
```

## Evo evaluation outputs

CircleTurns evaluation root:

```text
/scratch/zg2598/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_CircleTurns_LUT/EVO_EVAL_28VIEW
```

Important files:

```text
metrics_28view.csv
metrics_28view_summary.json
metrics_28view.md
skipped_frames.json
logs/
frames/<FRAME>/<base|enhancement1|enhancement3|gt>/stats/train_all_step0000.json
```

`evo_eval.py` compares Base, Level-1, Level-3, and GT using the same captured
28-camera evaluation path. A frame is eligible only when all four PLY inputs
exist. Completed valid stats are reused by default with `--skip-completed`.

## Legacy non-LUT outputs

These directories are retained for reference but must not be mixed with LUT
training or LUT evaluation:

```text
/scratch/zg2598/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_CircleTurns
/scratch/zg2598/DanceNet3D_Out/WORLD_COORD_PROJ_e_BiancaGolden_GrandPlies
```

## Slurm account and safe starting points

```text
Account:
torch_pr_1093_tandon_priority

GPU request:
--gres=gpu:h100:1
```

Suggested starting concurrency on one H100:

```text
Base training:       12–16 parallel frame workers
Enhancement:         8–12 parallel frame workers
Evo evaluation:      4 parallel frame workers, then test 8
DataLoader workers:  1 per trainer
```

More processes are not automatically faster. Around 30 simultaneous Evo
trainers previously caused CPU/shared-filesystem contention and very low GPU
utilization.

## Files that must not be committed

Do not commit generated datasets, PLY files, checkpoints, evaluation stats,
logs, videos, Apptainer overlays, or environment binaries. Source scripts,
tests, `cmd_manual.md`, and this address book belong in Git.
