# Pipeline: manuscript section -> code

| Manuscript | Code | Parameter (configs/default.yaml) |
|---|---|---|
| IV-A point representation, Eq. 1-2 | `features.normalize_xyz`, `build_features`, `estimate_normals` | `features.mode: xyz_h_n`, `normal_k: 20` |
| IV-A inference: shuffled 4096-point groups + 3 random groups, mean probability | `inference.segment_cloud` | `inference.num_points`, `extra_passes: 3` |
| IV-B DGCNN | `models.DGCNNSeg` | `model.k: 20` |
| IV-C DBSCAN clean-up | `phenotypes.filter_outliers` | `height.dbscan`: 2 cm, 10 neighbours, clusters < 100 points dropped, 6 cm radius |
| IV-C Eq. 4-5 height | `phenotypes.plant_height` | rim = z99(Pot), base = z1(Pot), apex = max z(Plant), `pot_height_cm: 14.7` |
| IV-D Algorithm 1 tillers | `phenotypes.count_tillers` | `tillers.cell_size_m: 0.008`, `sigma_cells: 1`, `window: 3`, `min_height_ratio: 0.25` |
| V-C metrics, VI-D offset, bootstrap | `evaluation.py` | `evaluation.n_bootstrap` |
| Table II training | `train.py` | `training.*` |

## Per-scan flow

```
read PLY/E57 -> normals (file normals if present, else Open3D / NumPy PCA, k = 20, oriented to +z)
  -> normalise on the FULL cloud (centroid, max radius) and compute relative height on the full cloud
  -> DGCNN votes: one covering pass of shuffled 4096-point groups + 3 random groups
  -> per-point class = argmax of summed softmax
  -> height: DBSCAN-clean Pot and Plant, rim/base percentiles of Pot, apex of Plant, scale by 14.7 cm
  -> tillers: CHM of all Plant points (no DBSCAN), Gaussian sigma = 1 cell, 3x3 maxima above
     tau = min + 0.25 (max - min) and inside an occupied cell
```

Height is `None` (NaN in the CSV) when either class has fewer than 50 points, when the pot has no height,
or when the apex stays below even the 95th-percentile rim; the scan is then reported, not silently dropped.
