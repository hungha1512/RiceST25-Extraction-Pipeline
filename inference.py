"""Whole-cloud inference (votes of 4096-point groups) and the model -> height / tillers step."""
from __future__ import annotations

import numpy as np
import torch

from phenotyping.height import plant_height
from phenotyping.tillers import count_tillers
from utils.features import build_features, estimate_normals, normalize_xyz


@torch.no_grad()
def segment_cloud(
    model: torch.nn.Module,
    xyz: np.ndarray,
    normal: np.ndarray | None = None,
    *,
    mode: str = "xyz_h_n",
    num_points: int = 4096,
    extra_passes: int = 3,
    batch_size: int = 8,
    normal_k: int = 20,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Per-point class (0=Pot, 1=Plant) for a complete cloud.

    1. The points are shuffled and cut into consecutive groups of ``num_points`` (the last group is
       padded with random repeats), so every point is voted on at least once.
    2. ``extra_passes`` further random groups of ``num_points`` are added.
    3. Each point takes the class with the highest summed softmax probability.

    Normalisation, relative height and normals are computed once on the *full* cloud, then indexed.
    """
    rng = np.random.default_rng(seed)
    n_total = len(xyz)
    if n_total <= model.ec1.k:
        raise ValueError(f"cloud has only {n_total} points")
    feats = build_features(
        normalize_xyz(xyz),
        (normal if normal is not None else estimate_normals(xyz, normal_k)) if mode == "xyz_h_n" else None,
        mode,
    ).astype(np.float32)

    groups = []
    order = rng.permutation(n_total)
    for start in range(0, n_total, num_points):
        chunk = order[start:start + num_points]
        if len(chunk) < num_points:
            chunk = np.concatenate([chunk, rng.choice(chunk, num_points - len(chunk))])
        groups.append(chunk)
    for _ in range(extra_passes):
        groups.append(rng.choice(n_total, num_points, replace=n_total < num_points))

    votes = np.zeros((n_total, model.seg[-1].out_channels), dtype=np.float32)
    model.eval()
    for i in range(0, len(groups), batch_size):
        batch = np.stack(groups[i:i + batch_size])                                   # (B, P)
        x = torch.from_numpy(feats[batch]).permute(0, 2, 1).to(device)               # (B, C, P)
        prob = model(x).exp().cpu().numpy()                                          # (B, P, classes)
        np.add.at(votes, batch.reshape(-1), prob.reshape(-1, prob.shape[-1]))
    return votes.argmax(-1).astype(np.int64)


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
