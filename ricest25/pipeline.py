"""End-to-end analysis of one scan: segmentation -> plant height -> tiller count."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .inference import segment_cloud
from .io import read_cloud, write_ply
from .phenotypes import count_tillers, plant_height


def pooled_miou(pred: np.ndarray, target: np.ndarray, num_classes: int = 2) -> tuple[float, list[float]]:
    """mIoU over the points of one cloud (classes absent from both pred and target are skipped)."""
    ious = []
    for c in range(num_classes):
        tp = int(((pred == c) & (target == c)).sum())
        denom = tp + int(((pred == c) & (target != c)).sum()) + int(((pred != c) & (target == c)).sum())
        ious.append(tp / denom if denom else float("nan"))
    return float(np.nanmean(ious)), ious


def analyze_cloud(model, cloud: dict, cfg: dict, device) -> dict:
    """Segment a cloud and extract both traits. ``cloud`` is the dict returned by ``io.read_cloud``."""
    inf, feat = cfg["inference"], cfg["features"]
    xyz = cloud["xyz"]
    normal = cloud["normal"] if feat["use_file_normals"] else None
    pred = segment_cloud(
        model, xyz, normal, mode=feat["mode"], num_points=inf["num_points"], extra_passes=inf["extra_passes"],
        batch_size=inf["batch_size"], normal_k=feat["normal_k"], seed=inf["seed"], device=device,
    )
    out = {"pred": pred, "height": None, "tillers": None}
    out["height"] = plant_height(xyz, pred, cfg, np.random.default_rng(inf["seed"]))
    try:
        out["tillers"] = count_tillers(xyz[pred == 1], cfg)   # all Plant points, no DBSCAN (as evaluated)
    except ValueError:
        pass
    if cloud.get("label") is not None:
        out["miou_full"], out["iou_per_class"] = pooled_miou(pred, cloud["label"])
    return out


def analyze_file(model, path: str | Path, cfg: dict, device, scan_index: int = 0,
                 save_dir: str | Path | None = None) -> dict:
    """Analyse a PLY/E57 file; optionally write the labelled cloud and a CHM figure to ``save_dir``."""
    cloud = read_cloud(path, scan_index)
    result = analyze_cloud(model, cloud, cfg, device)
    result["n_points"] = len(cloud["xyz"])
    if save_dir is not None:
        stem = Path(path).stem
        write_ply(Path(save_dir) / f"{stem}_segmented.ply", cloud["xyz"], result["pred"])
    return result
