# Reproducing the numbers

1. Install (`README.md`), copy the 13 LOPO checkpoints to `weights/lopo/`, the labelled PLY scans to
   `data/RiceST25-3D/` (file names `<pot>_<daycode>.ply`; day code `26422` = 22 April 2026, `2658` = 8 May).
2. `python -m ricest25 reproduce --data data/RiceST25-3D --weights weights/lopo`
3. Open `outputs/reproduce/report.md`.

Reference values (255 scans): complete-cloud mIoU 93.83 %; height MAE 3.41 cm, RMSE 4.90, RRMSE 6.31 %,
R² 0.93, bias -3.08; MAE 2.69 cm after the out-of-fold constant offset; tiller MAE 6.19, bias +3.37, R² 0.10.
They are stored in `data/paper_reference.json`.

Ground truth: `data/ground_truth.csv` is the table embedded in the research notebook (260 rows, pot x date).
Five scans have no point cloud and are ignored; lodging is defined as pots C1.3 and C3.2 from 10 May.

## Where this code and the notebook differ on purpose

| Topic | Notebook (run20) | This repo |
|---|---|---|
| Random numbers | unseeded `np.random` in inference | seeded `numpy.random.Generator` (`inference.seed`) |
| DBSCAN | comment says "normalised xyz", but it is applied to the metric cloud (eps 0.02 = 2 cm) | documented as metres; same behaviour |
| Plant apex | text says 99.9th percentile, code uses 100 | 100 (max), as in the manuscript |
| Tiller input | raw Plant points | same (no DBSCAN), as in the manuscript's evaluation |
| CHM cell fill | Python loop | vectorised `np.maximum.at`, identical result |
| Training/analysis | one Kaggle notebook | `train`, `reproduce`, `analyze` commands |
| Other architectures, input ablation, bootstrap baselines | in the notebook | not ported |

The previous version of this repository used a fixed 35 x 25 CHM raster and a CloudCompare/E57 preprocessing
contract. That contradicts the manuscript (8 mm cells) and the evaluated notebook, so it was replaced.

## Verifying a change did not alter results

`pytest` checks height geometry, DBSCAN, the CHM, voting determinism, the leave-one-pot-out fold logic and the
report tables on synthetic clouds. It cannot check the manuscript numbers: that needs the real data and weights.
