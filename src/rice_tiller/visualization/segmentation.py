from __future__ import annotations

from pathlib import Path

import numpy as np

from rice_tiller.types import SegmentationResult


def plot_segmentation(
    xyz: np.ndarray,
    segmentation: SegmentationResult,
    destination: str | Path,
    *,
    max_points: int = 20000,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    indices = (
        rng.choice(len(xyz), max_points, replace=False) if len(xyz) > max_points else np.arange(len(xyz))
    )
    colors = {0: "#FF8C00", 1: "#33CC33"}
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    for class_id, name in segmentation.class_names.items():
        mask = segmentation.labels[indices] == class_id
        points = xyz[indices][mask]
        if len(points):
            axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=1,
                alpha=0.55,
                color=colors.get(class_id),
                label=name,
            )
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title("DGCNN Pot/Plant segmentation")
    axis.legend()
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path

