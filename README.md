# RiceST25-3D extraction pipeline

Code for the manuscript *RiceST25-3D: A Handheld 3D Point Cloud Dataset and Deep Learning Pipeline for
Non-Destructive Rice Phenotyping*. Given a handheld scan of one potted ST25 rice plant it

1. labels every point as **Pot** or **Plant** with a DGCNN (7D input: normalised XYZ, relative height, normals),
2. measures **plant height** from the pot rim, scaled by the known 14.7 cm pot height,
3. estimates **tiller number** from local maxima of an 8 mm canopy height model (CHM).

It runs locally from plain Python files (no Kaggle, no notebook). You supply the trained weights and,
to reproduce the paper, the labelled dataset.

```
ricest25/            package: io, features, models (DGCNN), inference, phenotypes, pipeline,
                     evaluation, reproduce, train, cli
configs/default.yaml every parameter of the manuscript (height, DBSCAN, CHM, voting, training)
configs/split_potwise.yaml  train / val / test pots of Section III-E
data/ground_truth.csv       manual height (cm) and tiller count for the 260 scheduled scans
data/paper_reference.json   numbers reported in the manuscript, used by the report
weights/             put checkpoints here (see weights/README.md)
docs/                pipeline description and reproduction guide
tests/               unit tests on synthetic clouds
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # numpy, scipy, scikit-learn, pandas, plyfile, torch, pytest
pip install -e ".[open3d]"         # optional: normals computed exactly as in the notebook
pip install -e ".[e57]"            # optional: read raw .e57 files
pytest                             # 21 tests, ~10 s, no data or GPU needed
```

A CUDA GPU is used automatically (`inference.device: auto`). On CPU a 274k-point scan takes minutes.

## Weights

Copy the checkpoints you were given:

```
weights/lopo/DGCNN_C1.1.pth ... DGCNN_C3.4.pth     13 leave-one-pot-out models (trait evaluation)
weights/potwise/DGCNN_seed42.pth                    optional: pot-wise-split model
```

All are plain `state_dict` files of the 7D DGCNN (`k=20`, 2 classes: 0 = Pot, 1 = Plant).
Check one with `python -m ricest25 inspect weights/lopo/DGCNN_C1.1.pth`.

## 1. Reproduce the manuscript (255 scans, out-of-fold)

Put the labelled PLY files (`<pot>_<daycode>.ply`, e.g. `C1.4_26422.ply`, with a `scalar_Plant` field) in
`data/RiceST25-3D/`, then:

```bash
python -m ricest25 reproduce --data data/RiceST25-3D --weights weights/lopo --output outputs/reproduce
```

Each scan is segmented by the checkpoint that did **not** see its pot, height and tillers are extracted,
and `outputs/reproduce/` receives `scan_results.csv`, `summary.json`, `report.md` (manuscript vs. this run,
with 95 % pot-bootstrap intervals), Table VI / VII CSVs and out-of-fold offset-calibrated heights. It resumes
per pot if interrupted. Rebuild the tables from an existing CSV without data or GPU:

```bash
python -m ricest25 report outputs/reproduce/scan_results.csv
```

Segmentation scores on 4096-point samples (Table III / IV style):

```bash
python -m ricest25 eval-seg --data data/RiceST25-3D --checkpoint weights/potwise/DGCNN_seed42.pth   # test pots
python -m ricest25 eval-seg --data data/RiceST25-3D --checkpoint weights/lopo/DGCNN_C1.3.pth --pots C1.3
```

## 2. Analyse your own scans

Input: one `.ply` (or `.e57`) containing one pot and one plant, already cropped and denoised, with the
vertical axis along +z and the pot bottom, rim and plant top included, in metric units.

```bash
python -m ricest25 analyze scan.ply --checkpoint weights/lopo/DGCNN_C1.1.pth --output outputs/my_scan --save-segmented
```

Writes `analysis.csv` (plant height in cm, tiller count, point counts), `<name>_segmented.ply` (per-point
`label`) and `<name>_chm.npy`.

## 3. Retrain (optional)

```bash
python -m ricest25 train --data data/RiceST25-3D --output outputs/train            # pot-wise split
python -m ricest25 train --data data/RiceST25-3D --output outputs/lopo --lopo      # 13 folds
```

Settings follow Table II (Adam 1e-3, weight decay 1e-4, cosine schedule, batch 8 x 4096 points, early stopping
on validation mIoU). Only the DGCNN is implemented; the five comparison networks are not part of this repo.

## Reproducibility notes

- Voting uses random 4096-point groups, so predictions are deterministic for a given seed
  (`inference.seed`) but not bit-identical to the Kaggle runs; expect small differences in the third digit.
- Height is measured from the **pot rim**; the manual tape starts at the substrate, which gives the constant
  ~3 cm underestimate discussed in the paper. Nothing is added to hide it; the report shows the raw and the
  out-of-fold offset-corrected error.
- Tiller counting is a baseline and is known to overcount in May and on lodged plants (R² ≈ 0.10).
- Do not tune parameters on the evaluation scans; see `docs/reproducing.md` for where code and notebook differ.

Citation and licence: add after acceptance (the manuscript states a Zenodo DOI and data licence will follow).
