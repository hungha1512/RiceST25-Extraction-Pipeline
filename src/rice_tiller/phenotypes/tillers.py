from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter

from rice_tiller.phenotypes.chm import make_chm
from rice_tiller.types import TillerResult


def detect_local_maxima(
    smoothed_chm: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_distance: int = 3,
    exclude_border: bool = False,
) -> np.ndarray:
    distance = max(1, int(min_distance))
    footprint = 2 * distance + 1
    local_max = maximum_filter(smoothed_chm, size=footprint, mode="nearest")
    candidate_mask = np.isclose(smoothed_chm, local_max) & valid_mask
    if valid_mask.any():
        candidate_mask &= smoothed_chm > float(smoothed_chm[valid_mask].min())
    if exclude_border:
        candidate_mask[:distance, :] = False
        candidate_mask[-distance:, :] = False
        candidate_mask[:, :distance] = False
        candidate_mask[:, -distance:] = False

    candidates = np.argwhere(candidate_mask)
    if len(candidates) == 0:
        return np.empty((0, 2), dtype=np.int64)
    values = smoothed_chm[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(values)[::-1]
    selected: list[np.ndarray] = []
    for candidate in candidates[order]:
        if all(np.linalg.norm(candidate - item) >= distance for item in selected):
            selected.append(candidate)
    return np.asarray(selected, dtype=np.int64).reshape(-1, 2)


def peaks_to_xyz(
    peaks: np.ndarray,
    smoothed_chm: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    rows, cols = smoothed_chm.shape
    x_min, x_max, y_min, y_max = bounds
    if len(peaks) == 0:
        return np.empty((0, 3), dtype=np.float32)
    peak_y = y_min + (peaks[:, 0] + 0.5) * (y_max - y_min) / rows
    peak_x = x_min + (peaks[:, 1] + 0.5) * (x_max - x_min) / cols
    peak_z = smoothed_chm[peaks[:, 0], peaks[:, 1]]
    return np.column_stack([peak_x, peak_y, peak_z]).astype(np.float32)


def count_tillers_chm(plant_xyz: np.ndarray, config: dict[str, Any]) -> TillerResult:
    raster = config.get("raster", {})
    gaussian = config.get("gaussian", {})
    maxima = config.get("local_maxima", {})
    rows = int(raster.get("rows", 35))
    cols = int(raster.get("cols", 25))
    sigma = float(gaussian.get("sigma_px", 1.0))
    min_distance = int(maxima.get("min_distance_px", 3))
    exclude_border = bool(maxima.get("exclude_border", False))

    try:
        chm = make_chm(plant_xyz, rows=rows, cols=cols, sigma=sigma)
        peaks = detect_local_maxima(
            chm.smoothed,
            chm.valid_mask,
            min_distance=min_distance,
            exclude_border=exclude_border,
        )
        peak_xyz = peaks_to_xyz(peaks, chm.smoothed, chm.bounds)
    except ValueError as exc:
        return TillerResult(
            valid=False,
            raster_shape=(rows, cols),
            gaussian_sigma_px=sigma,
            min_distance_px=min_distance,
            warning=str(exc),
        )

    return TillerResult(
        valid=True,
        n_tillers=int(len(peaks)),
        raster_shape=(rows, cols),
        gaussian_sigma_px=sigma,
        min_distance_px=min_distance,
        peak_coordinates=peaks,
        peak_xyz=peak_xyz,
        chm_raw=chm.raw,
        chm_smoothed=chm.smoothed,
        valid_mask=chm.valid_mask,
    )
