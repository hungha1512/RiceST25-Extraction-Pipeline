"""Segmentation scores: mIoU pooled over points, and mean / SD of per-scan mIoU."""
from __future__ import annotations

import numpy as np
import torch


def miou_from_labels(pred, tgt, num_classes=2):
    ious = []
    for c in range(num_classes):
        tp = ((pred == c) & (tgt == c)).sum().item()
        d = tp + ((pred == c) & (tgt != c)).sum().item() + ((pred != c) & (tgt == c)).sum().item()
        ious.append(tp / d if d else float("nan"))
    return float(np.nanmean(ious)), ious


@torch.no_grad()
def evaluate(model, loader, device, num_classes=2) -> dict:
    """mIoU pooled over all points of the loader, plus the mean/SD of per-scan mIoU."""
    model.eval()
    preds, tgts, per_scan = [], [], []
    for x, y in loader:
        p = model(x.to(device)).argmax(-1).cpu()
        per_scan += [miou_from_labels(p[b], y[b], num_classes)[0] for b in range(len(p))]
        preds.append(p); tgts.append(y)
    miou, ious = miou_from_labels(torch.cat(preds), torch.cat(tgts), num_classes)
    return {"miou": miou, "iou_per_class": ious, "miou_scan_mean": float(np.nanmean(per_scan)),
            "miou_scan_std": float(np.nanstd(per_scan, ddof=1)) if len(per_scan) > 1 else 0.0, "scan_mious": per_scan}
