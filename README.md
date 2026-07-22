# Rice Tiller Pipeline

Inference-only pipeline for 3D rice point clouds. It loads a pretrained DGCNN,
segments a preprocessed single plant-pot point cloud into Pot and Plant, estimates
plant height from a 14.7 cm pot reference, and estimates tiller count using a
35 x 25 Canopy Height Model (CHM) with Gaussian smoothing and local maxima.

## Input contract

- One `.ply` or `.e57` file contains one rice plant and one pot.
- The ROI has already been cropped and denoised in CloudCompare.
- Height is aligned with the positive Z axis.
- The crop preserves the pot bottom, pot rim, and plant top.

The current workflow is therefore **semi-automatic**: manual ROI preparation is
followed by automatic inference and phenotype extraction.

## Default model contract

- Architecture: DGCNN segmentation from the research notebook
- Features: normalized XYZ + relative height + normals (7D)
- Graph neighbours: `k=20`
- Chunk size: 4096 points
- Classes: `0=Pot`, `1=Plant`

Copy the trained checkpoint to `weights/dgcnn_best/dg_best.pth`. The checkpoint
must match `weights/dgcnn_best/model.yaml`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[e57]"
```

For PLY-only use, `pip install -e .` is sufficient.

## Run

```bash
rice-tiller analyze sample.ply --config configs/default.yaml --output outputs
```

```bash
rice-tiller analyze sample.e57 \
  --scan-index 0 \
  --config configs/default.yaml \
  --output outputs
```

Inspect a checkpoint contract before analysis:

```bash
rice-tiller inspect-weight \
  weights/dgcnn_best/dg_best.pth \
  --metadata weights/dgcnn_best/model.yaml
```

## Outputs

Each sample receives an isolated output directory containing JSON/CSV results,
segmented PLY files, CHM arrays, quality-control figures, and a pipeline log. See
`docs/output-format.md` for the complete layout.

## Scientific reproducibility

Parameters are loaded from YAML and recorded in every `result.json`. Do not tune
preprocessing, percentile, Gaussian, or peak-distance parameters on the test set.
See `docs/methodology-notes.md` for known notebook/method differences.

