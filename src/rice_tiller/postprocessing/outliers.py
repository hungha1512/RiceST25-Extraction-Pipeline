from __future__ import annotations

from typing import Any

import numpy as np


def filter_large_clusters(
    xyz_normalized: np.ndarray,
    config: dict[str, Any],
    *,
    seed: int = 42,
) -> np.ndarray:
    points = np.asarray(xyz_normalized, dtype=np.float32)
    min_cluster_size = int(config.get("min_cluster_size", 100))
    if len(points) < min_cluster_size or not bool(config.get("enabled", True)):
        return np.ones(len(points), dtype=bool)

    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN

    max_fit_points = int(config.get("max_fit_points", 50000))
    rng = np.random.default_rng(seed)
    if len(points) > max_fit_points:
        sample_indices = rng.choice(len(points), max_fit_points, replace=False)
        fit_points = points[sample_indices]
    else:
        fit_points = points

    eps = float(config.get("eps_normalized", 0.02))
    labels = DBSCAN(
        eps=eps,
        min_samples=int(config.get("min_samples", 10)),
    ).fit_predict(fit_points)

    valid_labels = [
        label
        for label in np.unique(labels)
        if label != -1 and int((labels == label).sum()) >= min_cluster_size
    ]
    if not valid_labels:
        return np.ones(len(points), dtype=bool)

    valid_fit_points = fit_points[np.isin(labels, valid_labels)]
    tree = cKDTree(valid_fit_points)
    distances, _ = tree.query(points, k=1, workers=-1)
    multiplier = float(config.get("map_radius_multiplier", 3.0))
    return distances < eps * multiplier

