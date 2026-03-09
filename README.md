# Growth Alignment

Image registration and alignment pipelines for time-lapse plant root imagery (rhizotron tubes). The project aligns sequences of root images captured at multiple depth levels and timepoints so they can be composited into consistent mosaics for downstream analysis.

## Data

Images live in `slu_data/` and follow the naming convention:

```
{plant}_{tube}_{level}_{date}_TP{timepoint}.png
```

- **plant** — species/experiment identifier (e.g. `kura`)
- **tube** — rhizotron tube number
- **level** — depth level within the tube (1–7)
- **date** — acquisition date (`YYYY-MM-DD`)
- **timepoint** — longitudinal timepoint index (1–12)

## Core Notebooks

### `sift_feature_depths_pipeline.ipynb`

Stitches depth levels within a single timepoint into one tall composite image. For each tube and timepoint, adjacent depth-level images are registered using SIFT feature matching and the resulting translations (~800 px horizontally per level) are accumulated to place every depth tile onto a shared canvas. Output files are named `{plant}_{tube}_TP{timepoint}_sift_depths.png`.

### `sift_feature_multi_pipeline.ipynb`

Aligns the depth-stitched composites across timepoints. Takes the `*_sift_depths.png` outputs from the depths pipeline, groups them by tube, and registers consecutive timepoints to each other. Each aligned image is placed on a common canvas and written out with an `_all.png` suffix.

### `sift_feature_timepoints_pipeline.ipynb`

Aligns individual (non-stitched) images across timepoints for each tube and depth level. For every (tube, depth) group, consecutive timepoints are registered and the translations are accumulated into a global coordinate frame. Aligned images are saved with a `_sift.png` suffix, and red/blue overlay images (`_sift_overlay.png`) are produced for visual QA.

### Pipeline execution order

1. **Depths** — stitch depth levels per timepoint (`sift_feature_depths_pipeline`)
2. **Multi** — align the stitched composites across timepoints (`sift_feature_multi_pipeline`)

The timepoints pipeline is an alternative path that aligns across time *before* depth stitching.

## Registration Method

All three notebooks share the same SIFT-based registration workflow:

1. **Preprocessing** — Gaussian blur + CLAHE contrast enhancement.
2. **Keypoint detection** — OpenCV SIFT detector/descriptor.
3. **Matching** — Brute-force k-NN matching with Lowe's ratio test (threshold 0.75).
4. **Transform estimation** — Partial affine estimated via RANSAC; only the translation component (tx, ty) is kept.
5. **Sanity limits** — Translations exceeding a threshold are rejected (the threshold varies by pipeline to reflect expected motion between depth levels vs. timepoints).
6. **Fallback matching** — If a consecutive pair fails, the pipeline tries matching against every other image in the group before giving up.
7. **Accumulation** — Pairwise translations are chained recursively so every image is expressed relative to a single reference frame.
8. **Canvas compositing** — A bounding box is computed over all translated positions and each image is placed at its global offset on a zero-filled canvas.

## Other Notebooks

| Notebook | Description |
|---|---|
| `gpt_basic_phase.ipynb` | Phase correlation alignment experiments |
| `gpt_basic_phase-ecc.ipynb` | Phase correlation + ECC refinement |
| `gpt_basic_feature.ipynb` | Feature-based alignment prototyping |
| `gpt_feature_ecc.ipynb` | Feature matching followed by ECC refinement |
| `gpt_phase-ecc_bspline.ipynb` | Phase-ECC with B-spline non-rigid refinement |
| `correlation_pipeline.ipynb` | Correlation-based alignment pipeline |
| `optimize_correlation_pipeline_full.ipynb` | Optimization-based alignment over full images |
| `optimize_correlation_pipeline_masks.ipynb` | Optimization-based alignment using binary masks |

## Python Modules

| File | Purpose |
|---|---|
| `losses.py` | Loss/similarity functions: overlap, XOR, soft Dice, NCC, NGF, mutual information, L2 variants, Huber, weighted/masked L2 |
| `align_full.py` | Scipy-based multi-image alignment minimizing pairwise NCC loss with sub-pixel shifts |
| `align_masks.py` | Same optimization approach applied to binary masks |

## Dependencies

- Python 3
- OpenCV (`cv2`) with SIFT support
- NumPy
- SciPy (used by the optimization modules)
- Matplotlib (visualization)
