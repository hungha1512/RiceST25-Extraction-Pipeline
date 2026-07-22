from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class CHMResult:
    raw: np.ndarray
    smoothed: np.ndarray
    valid_mask: np.ndarray
    bounds: tuple[float, float, float, float]


def build_chm(plant_xyz: np.ndarray, *, rows: int = 35, cols: int = 25) -> tuple[
    np.ndarray, np.ndarray, tuple[float, float, float, float]
]:
    points = np.asarray(plant_xyz, dtype=np.float32)
    if len(points) == 0:
        raise ValueError("Cannot build a CHM from an empty Plant cloud")
    x, y, z = points.T
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Plant cloud must have positive X and Y extent")

    col_indices = np.floor((x - x_min) / (x_max - x_min) * cols).astype(np.int64)
    row_indices = np.floor((y - y_min) / (y_max - y_min) * rows).astype(np.int64)
    col_indices = np.clip(col_indices, 0, cols - 1)
    row_indices = np.clip(row_indices, 0, rows - 1)

    flattened = np.full(rows * cols, -np.inf, dtype=np.float32)
    flat_indices = row_indices * cols + col_indices
    np.maximum.at(flattened, flat_indices, z)
    chm = flattened.reshape(rows, cols)
    valid_mask = np.isfinite(chm)
    chm[~valid_mask] = np.nan
    return chm, valid_mask, (x_min, x_max, y_min, y_max)


def smooth_chm(chm: np.ndarray, *, sigma: float = 1.0) -> np.ndarray:
    if not np.isfinite(chm).any():
        raise ValueError("CHM has no valid cells")
    filled = np.where(np.isfinite(chm), chm, np.nanmin(chm))
    return gaussian_filter(filled, sigma=float(sigma)).astype(np.float32)


def make_chm(
    plant_xyz: np.ndarray,
    *,
    rows: int = 35,
    cols: int = 25,
    sigma: float = 1.0,
) -> CHMResult:
    raw, valid_mask, bounds = build_chm(plant_xyz, rows=rows, cols=cols)
    return CHMResult(
        raw=raw,
        smoothed=smooth_chm(raw, sigma=sigma),
        valid_mask=valid_mask,
        bounds=bounds,
    )

