# Parallelism

## Chosen decomposition

The policy calculation is inexpensive compared with visibility rendering. Its
best parallel unit is an independent `(trace, visibility CSV)` job.

Within one job:

- the visibility CSV is read once;
- decisions are written as a stream;
- only one small accumulator per output frame is retained;
- output files are written atomically.

Across jobs:

- `fovsim batch` uses `ProcessPoolExecutor`;
- workers never share mutable state;
- each worker writes to a unique output directory.

This design avoids a large parent process, pandas copies, and inter-process
transfer of cell rows.

## Worker count

For a single node:

```text
workers = min(number of jobs, allocated CPU cores)
```

If plotting is enabled, leave at least one core free for scheduler and I/O.
For large CSVs on network storage, start with 4-8 workers and increase only
while aggregate read throughput continues to improve.

## Slurm

For hundreds of independent traces, a Slurm array is preferred over one large
local process pool. Each task reads exactly one JSONL manifest row.

```bash
sbatch --array=1-100%20 scripts/slurm_array.sbatch jobs.jsonl
```

The `%20` cap prevents all jobs from hitting shared storage simultaneously.

## Quality rendering stage

Mixed Base/E3 quality rendering uses a different parallel hierarchy:

1. one process per GPU;
2. a bounded CPU thread pool preloads and parses upcoming EVOGS states and GT
   PLYs while the current frame is rendered;
3. each Gaussian frame is uploaded once, then reused for its one reference or
   policy render;
4. equal-resolution full-frame metrics can be evaluated in small GPU batches;
5. variable-size foreground crops retain exact per-frame bounds and are
   evaluated independently;
6. multiple CUDA processes are not spawned on one GPU unless it is explicitly
   partitioned.

The defaults are `--prefetch-workers 2 --metric-batch-size 4`. Increase the
metric batch only while GPU memory and measured throughput improve. Use one
independent process per GPU for multi-GPU nodes.
