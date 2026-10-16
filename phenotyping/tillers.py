"""Tiller count from local maxima of a canopy height model (manuscript Section IV-D, Algorithm 1)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def canopy_height_model(plant_xyz: np.ndarray, cell: float) -> np.ndarray:
    """2D map of the highest Plant z per ``cell``-sized (x, y) cell; empty cells are NaN."""
    x, y, z = plant_xyz[:, 0], plant_xyz[:, 1], plant_xyz[:, 2]
    nx, ny = int((x.max() - x.min()) / cell) + 1, int((y.max() - y.min()) / cell) + 1
    xi = np.clip(((x - x.min()) / cell).astype(int), 0, nx - 1)
    yi = np.clip(((y - y.min()) / cell).astype(int), 0, ny - 1)
    flat = np.full(ny * nx, -np.inf)
    np.maximum.at(flat, yi * nx + xi, z)
    flat[np.isinf(flat)] = np.nan
    return flat.reshape(ny, nx)


def count_tillers(plant_xyz: np.ndarray, cfg: dict) -> dict:
    """Tiller count = number of tall local maxima of the smoothed canopy height model (Algorithm 1)."""
    t = cfg["tillers"]
    if len(plant_xyz) == 0:
        raise ValueError("no Plant points")
    chm = canopy_height_model(plant_xyz, t["cell_size_m"])
    filled = np.where(np.isnan(chm), np.nanmin(chm), chm)          # empty cells take the lowest height
    smooth = gaussian_filter(filled, sigma=t["sigma_cells"])
    is_max = smooth == maximum_filter(smooth, size=max(3, t["window"]))
    tau = np.nanmin(chm) + t["min_height_ratio"] * (np.nanmax(chm) - np.nanmin(chm))
    peaks = is_max & (smooth > tau) & ~np.isnan(chm)
    return {"n_tillers": int(peaks.sum()), "chm": chm, "chm_smooth": smooth,
            "peak_rc": np.argwhere(peaks), "tau": float(tau)}
