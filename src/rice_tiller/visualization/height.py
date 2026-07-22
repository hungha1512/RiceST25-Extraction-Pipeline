from __future__ import annotations

from pathlib import Path

import numpy as np

from rice_tiller.types import HeightResult, SegmentationResult


def plot_height(
    xyz: np.ndarray,
    segmentation: SegmentationResult,
    result: HeightResult,
    destination: str | Path,
    *,
    max_points: int = 30000,
) -> Path | None:
    if not result.valid:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    indices = (
        rng.choice(len(xyz), max_points, replace=False) if len(xyz) > max_points else np.arange(len(xyz))
    )
    colors = {0: "#FF8C00", 1: "#33CC33"}
    figure, axis = plt.subplots(figsize=(9, 6))
    for class_id, name in segmentation.class_names.items():
        mask = segmentation.labels[indices] == class_id
        points = xyz[indices][mask]
        if len(points):
            axis.scatter(
                points[:, 0],
                points[:, 2],
                s=1,
                alpha=0.45,
                color=colors.get(class_id),
                label=name,
            )
    axis.axhline(result.z_pot_bottom, color="#8B4513", linestyle="--", label="Pot bottom")
    axis.axhline(result.z_pot_top, color="red", linestyle="--", label="Pot top")
    axis.axhline(result.z_plant_top, color="green", linestyle="--", label="Plant top")
    axis.set_xlabel("X")
    axis.set_ylabel("Z")
    axis.set_title(f"Plant height = {result.plant_height_cm:.2f} cm")
    axis.grid(alpha=0.25)
    axis.legend()
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path

