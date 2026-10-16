"""Plant height from the pot rim (manuscript Section IV-C)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def filter_outliers(xyz: np.ndarray, eps: float = 0.02, min_samples: int = 10, min_cluster_size: int = 100,
                    radius_factor: float = 3.0, max_fit_points: int = 50000,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Boolean mask that drops stray points.

    DBSCAN links points with >= ``min_samples`` neighbours within ``eps``; clusters smaller than
    ``min_cluster_size`` are stray. Points farther than ``radius_factor * eps`` from a kept cluster
    are discarded. Large clouds are fitted on a random subset of ``max_fit_points`` points.
    """
    keep_all = np.ones(len(xyz), dtype=bool)
    if len(xyz) < min_cluster_size:
        return keep_all
    rng = rng or np.random.default_rng(0)
    sub = xyz if len(xyz) <= max_fit_points else xyz[rng.choice(len(xyz), max_fit_points, replace=False)]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(sub).labels_
    sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.array([])
    big = np.flatnonzero(sizes >= min_cluster_size)
    if len(big) == 0:
        return keep_all
    dist, _ = cKDTree(sub[np.isin(labels, big)]).query(xyz, k=1)
    return dist < eps * radius_factor


def plant_height(xyz: np.ndarray, label: np.ndarray, cfg: dict, rng: np.random.Generator | None = None) -> dict | None:
    """Plant height in cm: (apex - rim) * pot_height_cm / (rim - base).

    rim = ``pot_percentile`` of Pot z, base = 100 - ``pot_percentile`` of Pot z (1st percentile),
    apex = ``plant_percentile`` of Plant z (100 = max). Returns None if the segmentation is unusable.
    """
    h = cfg["height"]
    pot, plant = xyz[label == 0], xyz[label == 1]
    if len(pot) < h["min_class_points"] or len(plant) < h["min_class_points"]:
        return None
    rng = rng or np.random.default_rng(0)
    db = h["dbscan"]
    if db["enabled"]:
        args = dict(eps=db["eps_m"], min_samples=db["min_samples"], min_cluster_size=db["min_cluster_size"],
                    radius_factor=db["map_radius_factor"], max_fit_points=db["max_fit_points"], rng=rng)
        keep_pot, keep_plant = filter_outliers(pot, **args), filter_outliers(plant, **args)
        pot = pot[keep_pot] if keep_pot.sum() > h["min_class_points"] else pot
        plant = plant[keep_plant] if keep_plant.sum() > h["min_class_points"] else plant

    z_rim = np.percentile(pot[:, 2], h["pot_percentile"])
    z_base = np.percentile(pot[:, 2], 100 - h["pot_percentile"])
    z_apex = np.percentile(plant[:, 2], h["plant_percentile"])
    pot_h = z_rim - z_base
    if pot_h <= 0:
        return None
    if z_apex - z_rim <= 0:  # stray Pot points above the canopy: retry with a stricter rim
        strict = np.percentile(pot[:, 2], h["strict_pot_percentile"])
        if z_apex - strict <= 0:
            return None
        z_rim = strict
    scale = h["pot_height_cm"] / pot_h
    return {
        "height_cm": float((z_apex - z_rim) * scale), "cm_per_unit": float(scale),
        "z_base": float(z_base), "z_rim": float(z_rim), "z_apex": float(z_apex),
        "pot_height_units": float(pot_h), "n_pot_used": int(len(pot)), "n_plant_used": int(len(plant)),
    }
