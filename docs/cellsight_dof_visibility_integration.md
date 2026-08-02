# CellSight DoF-to-cell visibility integration study

Date: 2026-08-02

## What CellSight actually does

For trajectory baselines, CellSight does not learn a second model that maps
6DoF to cells. It predicts a future camera pose and then applies deterministic
camera/scene geometry:

1. Convert predicted position and Euler orientation into a camera extrinsic
   matrix; combine it with the camera intrinsic matrix.
2. Project points into the image and retain points with positive depth and
   image coordinates inside the viewport.
3. Estimate each cell's viewport-overlap ratio by drawing 10,000 uniformly
   distributed points over the scene bounds and taking the per-cell fraction
   that projects inside the viewport.
4. For occlusion-aware visibility, crop the actual future point cloud to the
   viewport, apply Open3D hidden-point removal (HPR), and divide the surviving
   point count in each cell by that cell's original point count.
5. Derive angular features from overlap/visibility and distance:
   `2 * fraction * atan(cell_size / (2 * distance))`.

The future content is available at the server, so use of the future asset is
not future-user-pose leakage. The predicted pose remains the only future user
state used by the trajectory baseline.

## Difference from the current FoV simulator label

CellSight's `in_FoV_feature` is a geometric volume-overlap proxy. Its
`occlusion_feature` is an HPR-visible point fraction. Our maintained label,
`contributing_gaussian_fraction`, is stricter and renderer-specific: a Gaussian
counts only when at least one pixel has front-to-back compositing weight above
the configured threshold. These values are correlated but are not numerically
interchangeable, and a threshold of 0.5 has a different meaning for each.

## Can it connect to the current pipeline?

Yes. The clean boundary is:

```text
500 ms historical 6DoF
  -> LR future pose
  -> pose-to-cell geometry adapter
  -> per-cell predicted score
  -> Base/E3 decision CSV
  -> existing bandwidth and QoE evaluator
```

The existing BNQ evaluator already consumes `cell_decisions.csv`; it does not
require scores to have been produced by the current DoF-to-cell LR. Therefore
the final two stages need no conceptual change.

## Recommended two-stage implementation

### Stage A: CellSight-compatible viewport overlap

- Reuse the simulator's 0.2 m GSV-local cell IDs and cell origin.
- Generate one deterministic, seeded sample bank over the fixed scene bounds,
  then reuse it for every frame. Reusing the bank removes Monte Carlo jitter
  while preserving CellSight's definition.
- Use the simulator's existing camera convention, image size, horizontal FoV,
  and `camera_tensors`; do not copy CellSight's 8i coordinate scaling, Z flip,
  pitch offset, or `fx=fy=525` into the DanceNet3D coordinate system.
- Emit a distinct `viewport_overlap_fraction` field and calibrate its policy
  threshold on training traces. Do not relabel it as
  `contributing_gaussian_fraction`.

This stage needs no GT/E3 PLY during online prediction and should be cheap.

### Stage B: renderer-consistent predicted-pose visibility

- Add a predicted-pose override to `generate_standard_ply_visibility.py`.
- Keep its current future DanceNet3D GT PLY, cell partition, Gaussian
  rasterization, occlusion/compositing attribution, and CSV schema.
- Replace only the ground-truth camera pose with the LR-predicted future pose.

This directly produces the same `contributing_gaussian_fraction` used by the
current policy and gives the cleanest BNQ experiment, but it is more expensive
(the existing local run measured about 7.5 frames/s on the RTX 5070 Ti laptop).

## Validation gates before BNQ

1. Identity: true future pose passed through the adapter must reproduce the
   corresponding ground-truth geometry score within tolerance.
2. Coordinate convention: predicted camera renders must visually align with
   trace-camera renders on several frames, including Euler wrap crossings.
3. Cell schema: predicted and GT scores must use the same cell IDs and 0.2 m
   origin/size definition.
4. Cell metrics: report MSE and R2 over all cells and frames, matching the
   CellSight paper's aggregation formula.
5. Streaming metrics: only after the above, threshold/calibrate on training
   traces and run the existing point-count, bandwidth, PSNR, SSIM, and LPIPS
   evaluation on the two untouched test traces.

## Recommendation

Implement Stage A first as the literal CellSight reproduction and compare its
viewport-overlap R2 at 33 ms and 333 ms. Then implement Stage B as the
renderer-consistent version used for the actual DanceNet3D streaming claim.
Keeping both prevents a CellSight-methodology reproduction from being confused
with the simulator's Gaussian-contribution visibility definition.

