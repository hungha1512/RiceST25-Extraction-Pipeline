from __future__ import annotations

from pathlib import Path

import numpy as np

from rice_tiller.types import TillerResult


def plot_tillers(
    plant_xyz: np.ndarray,
    result: TillerResult,
    destination: str | Path,
    *,
    max_points: int = 10000,
) -> Path | None:
    if not result.valid or result.chm_raw is None or result.chm_smoothed is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(18, 5))
    raw_axis = figure.add_subplot(1, 3, 1)
    image = raw_axis.imshow(result.chm_raw, origin="lower", cmap="viridis")
    raw_axis.set_title("Raw CHM (35 x 25)")
    figure.colorbar(image, ax=raw_axis, label="Z")

    peak_axis = figure.add_subplot(1, 3, 2)
    peak_axis.imshow(result.chm_smoothed, origin="lower", cmap="viridis")
    if result.peak_coordinates is not None and len(result.peak_coordinates):
        peak_axis.scatter(
            result.peak_coordinates[:, 1],
            result.peak_coordinates[:, 0],
            c="red",
            s=100,
            marker="*",
            edgecolors="white",
            label=f"{result.n_tillers} peaks",
        )
        peak_axis.legend()
    peak_axis.set_title(f"Gaussian + local maxima: {result.n_tillers} tillers")

    cloud_axis = figure.add_subplot(1, 3, 3, projection="3d")
    rng = np.random.default_rng(42)
    indices = (
        rng.choice(len(plant_xyz), max_points, replace=False)
        if len(plant_xyz) > max_points
        else np.arange(len(plant_xyz))
    )
    points = plant_xyz[indices]
    cloud_axis.scatter(points[:, 0], points[:, 1], points[:, 2], c="#33CC33", s=1, alpha=0.3)
    if result.peak_xyz is not None and len(result.peak_xyz):
        cloud_axis.scatter(
            result.peak_xyz[:, 0],
            result.peak_xyz[:, 1],
            result.peak_xyz[:, 2],
            c="red",
            s=110,
            marker="*",
            edgecolors="black",
        )
    cloud_axis.set_title("Plant cloud and CHM peaks")

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path

