# Data contract

## Raw 6DoF trace

Required columns:

```text
FileName
LocationX LocationY LocationZ
RotationRoll RotationPitch RotationYaw
Frame Timestamp
```

Optional but preserved by the source system:

```text
GazeHitX GazeHitY GazeHitZ GazeConfidence
RoomX_m RoomY_m
```

Validation rules:

- numeric values must be finite;
- timestamps must be strictly increasing;
- repeated CSV headers inside a session are rejected;
- `Frame` must be a non-negative integer.

`source_row` is the original one-based CSV line number including the header.
The first data row therefore has `source_row = 2`.

## Cell visibility CSV

Required columns:

```text
output_frame
output_time_s
trace_source_row
trace_timestamp_s
gsv_frame
cell_id
active_gaussian_count
contributing_gaussian_count
contributing_gaussian_fraction
image_share
```

The policy uses only:

```text
contributing_gaussian_fraction >= threshold
```

The remaining fields enforce provenance and produce useful coverage summaries.

For each visibility row, the simulator checks:

```text
visibility.trace_source_row exists in the raw trace
visibility.trace_timestamp_s matches the trace row within tolerance
visibility.gsv_frame equals trace.Frame
```

## Decision output

`cell_decisions.csv` contains all input visibility fields plus:

```text
policy_threshold
enhancement_required    # 0 or 1
target_level            # 0 or 3
```

The threshold is inclusive:

```text
fraction == threshold -> target_level 3
```

## Standard-Ply QoE output

`scripts/evaluate_standard_ply_qoe.py` writes one
`per_frame_metrics.csv` row per camera/Gaussian frame. The three comparison
prefixes are:

```text
e3_vs_gt
policy_vs_gt
policy_vs_e3
```

Every prefix has:

```text
<pair>_full_mse
<pair>_full_psnr_db
<pair>_full_ssim
<pair>_full_lpips_alex
<pair>_fore_mse
<pair>_fore_psnr_db
<pair>_fore_ssim
<pair>_fore_lpips_alex
```

Full metrics use every RGB pixel. Foreground is reference-defined:

```text
E3 vs GT      -> GT alpha
Policy vs GT  -> GT alpha
Policy vs E3  -> E3 alpha
```

A foreground pixel satisfies `reference_alpha >= 1/255` by default.
Foreground PSNR normalizes MSE over mask pixels only. SSIM and LPIPS receive
masked images cropped to the tight reference-mask bounds plus configurable
context padding. The CSV also records the GT/E3 mask pixel counts and bounding
boxes, so every foreground result is auditable.

The standard-Ply policy is:

```text
Policy = Base Gaussians in non-selected cells
       + E3 Gaussians in selected cells
```

Cell assignment uses the same fixed GSV-local grid as visibility generation.

Per-frame transmission fields are:

```text
base_ply_file_bytes
e3_ply_file_bytes
base_gaussian_record_bytes
e3_gaussian_record_bytes
selected_e3_payload_bytes
base_only_transmission_bytes
policy_transmission_bytes
full_progressive_transmission_bytes
policy_savings_vs_full_bytes
policy_savings_vs_full_fraction
```

The simulated progressive payload is:

```text
Policy bytes = complete Base PLY bytes
             + selected E3 Gaussian count * E3 record bytes

Full bytes   = complete Base PLY bytes + complete E3 PLY bytes
```

This is an application-payload estimate. Per-cell packet headers, transport
headers, retransmission, and entropy coding are excluded until a concrete
container/protocol is specified.

## Frame summary

`frame_summary.csv` contains:

```text
display_frame
output_frame
output_time_s
trace_timestamp_s
gsv_frame
policy_threshold
occupied_cells
enhancement3_cells
base_only_cells
enhancement3_cell_fraction
active_gaussians
enhancement3_active_gaussians
enhancement3_gaussian_fraction
enhancement3_image_share
base_only_image_share
```

`enhancement3_image_share` is a visibility diagnostic, not PSNR/SSIM and not a
byte count.
