"""Optional: retrain DGCNN (pot-wise split or leave-one-pot-out) with the manuscript's settings."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .features import INPUT_DIM, build_features, estimate_normals, normalize_xyz
from .io import read_ply
from .models import DGCNNSeg, resolve_device


def set_all_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def pot_of(path) -> str:
    return Path(path).stem.split("_")[0]


def cached_normals(path, xyz, k: int, cache_dir: Path | None) -> np.ndarray:
    """Normals of the full cloud (file normals if present), cached as .npy next to the run outputs."""
    cache = cache_dir / f"{Path(path).stem}_n{k}.npy" if cache_dir else None
    if cache is not None and cache.exists():
        return np.load(cache)
    normal = estimate_normals(xyz, k)
    if cache is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache, normal)
    return normal


class RiceDataset(Dataset):
    """Random (train) or fixed (val/test) subset of ``num_points`` points per labelled scan."""

    def __init__(self, files, mode="xyz_h_n", num_points=4096, augment=False, normal_k=20,
                 cache_dir=None, eval_seed=12345):
        self.files, self.mode, self.num_points, self.augment = list(files), mode, num_points, augment
        self.normal_k, self.cache_dir, self.eval_seed = normal_k, cache_dir, eval_seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = read_ply(self.files[idx])
        if data["label"] is None:
            raise ValueError(f"{self.files[idx]} has no Pot/Plant labels")
        n = len(data["xyz"])
        rng = np.random if self.augment else np.random.default_rng(self.eval_seed + idx)
        pick = rng.choice(n, self.num_points, replace=n < self.num_points)
        xyz, label = data["xyz"][pick], data["label"][pick]
        normal = None
        if self.mode == "xyz_h_n":
            full = data["normal"] if data["normal"] is not None else cached_normals(
                self.files[idx], data["xyz"], self.normal_k, self.cache_dir)
            normal = full[pick].copy()
        xyz = normalize_xyz(xyz)
        if self.augment:  # one rotation about z shared by coordinates and normals, then scale and jitter
            t = np.random.uniform(0, 2 * np.pi); c, s = np.cos(t), np.sin(t)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
            xyz = (xyz @ rot.T) * np.float32(np.random.uniform(0.85, 1.15))
            xyz = xyz + np.random.normal(0, 0.005, xyz.shape).astype(np.float32)
            normal = normal @ rot.T if normal is not None else None
        feat = build_features(xyz, normal, self.mode).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(feat)).T, torch.from_numpy(label).long()


def class_weights(files, num_classes=2) -> torch.Tensor:
    """Inverse-frequency weights from the training scans, clipped to [0.5, 10]."""
    counts = np.zeros(num_classes)
    for f in files:
        lab = read_ply(f)["label"]
        counts += np.bincount(lab, minlength=num_classes)
    counts = np.where(counts == 0, 1, counts)
    return torch.tensor(np.clip(counts.sum() / (num_classes * counts), 0.5, 10.0), dtype=torch.float32)


def _miou(pred, tgt, num_classes=2):
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
        per_scan += [_miou(p[b], y[b], num_classes)[0] for b in range(len(p))]
        preds.append(p); tgts.append(y)
    miou, ious = _miou(torch.cat(preds), torch.cat(tgts), num_classes)
    return {"miou": miou, "iou_per_class": ious, "miou_scan_mean": float(np.nanmean(per_scan)),
            "miou_scan_std": float(np.nanstd(per_scan, ddof=1)) if len(per_scan) > 1 else 0.0, "scan_mious": per_scan}


def train_model(model, train_loader, val_loader, weights, save_path, device, epochs=150, patience=20,
                lr=1e-3, weight_decay=1e-4, min_delta=1e-4, name="DGCNN", verbose=True) -> dict:
    """Adam + cosine annealing, weighted NLL; keeps the checkpoint with the best *validation* mIoU."""
    crit = nn.NLLLoss(weight=weights.to(device))
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_ep, bad, t0 = -1.0, 0, 0, time.time()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x).reshape(-1, weights.numel()), y.reshape(-1))
            loss.backward(); opt.step()
            loss_sum += loss.item()
        sched.step()
        val = evaluate(model, val_loader, device)
        if val["miou"] > best + min_delta:
            best, best_ep, bad = val["miou"], ep, 0
            torch.save(model.state_dict(), save_path)
        else:
            bad += 1
        if verbose:
            print(f"[{name}] epoch {ep:3d} loss {loss_sum / max(len(train_loader), 1):.4f} "
                  f"val mIoU {val['miou'] * 100:.2f}%" + ("  <= best" if bad == 0 else f"  (patience {bad}/{patience})"))
        if bad >= patience:
            break
    return {"best_val_miou": best, "best_epoch": best_ep, "epochs_run": ep, "train_minutes": (time.time() - t0) / 60}


def _loaders(train_f, val_f, test_f, cfg, cache_dir, mode="xyz_h_n"):
    t = cfg["training"]
    mk = lambda fs, aug: RiceDataset(fs, mode, t["num_points"], aug, cfg["features"]["normal_k"], cache_dir)
    dl = lambda ds, shuffle: DataLoader(ds, t["batch_size"], shuffle=shuffle, drop_last=shuffle,
                                        num_workers=t["num_workers"])
    return dl(mk(train_f, True), True), dl(mk(val_f, False), False), (dl(mk(test_f, False), False) if test_f else None)


def train_split(files, split: dict, out_dir, cfg, mode="xyz_h_n", seed=None) -> dict:
    """Train on ``split['train']`` pots, select on ``split['val']``, report ``split['test']`` once."""
    t, device = cfg["training"], resolve_device(cfg["inference"]["device"])
    seed = t["seed"] if seed is None else seed
    pick = lambda pots: sorted(f for f in files if pot_of(f) in set(pots))
    tr, va, te = (pick(split[k]) for k in ("train", "val", "test"))
    assert not (set(split["train"]) & set(split["val"]) | set(split["train"]) & set(split["test"])
                | set(split["val"]) & set(split["test"])), "a pot appears in two partitions"
    out_dir = Path(out_dir)
    tl, vl, el = _loaders(tr, va, te, cfg, out_dir / "normals_cache", mode)
    set_all_seeds(seed)
    model = DGCNNSeg(2, INPUT_DIM[mode], cfg["model"]["k"]).to(device)
    ckpt = out_dir / f"DGCNN_{mode}_seed{seed}.pth"
    hist = train_model(model, tl, vl, class_weights(tr), ckpt, device, t["epochs"], t["patience"], t["lr"], t["weight_decay"])
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    res = {"seed": seed, "mode": mode, **hist, "test": {k: v for k, v in evaluate(model, el, device).items() if k != "scan_mious"}}
    (out_dir / f"DGCNN_{mode}_seed{seed}.json").write_text(json.dumps(res, indent=2))
    return res


def train_lopo(files, out_dir, cfg, only: list[str] | None = None) -> list[dict]:
    """13-fold leave-one-pot-out: fold i tests pot i, validates on the next pot in sorted order."""
    t, device = cfg["training"], resolve_device(cfg["inference"]["device"])
    pots = sorted({pot_of(f) for f in files})
    by_pot = {p: sorted(f for f in files if pot_of(f) == p) for p in pots}
    out_dir, results = Path(out_dir), []
    for i, test_pot in enumerate(pots):
        if only and test_pot not in only:
            continue
        ckpt, meta = out_dir / f"DGCNN_{test_pot}.pth", out_dir / f"DGCNN_{test_pot}.json"
        if meta.exists():
            print(f"[{i + 1}/{len(pots)}] {test_pot}: done, skipping"); continue
        val_pot = pots[(i + 1) % len(pots)]
        tr = [f for p in pots if p not in (test_pot, val_pot) for f in by_pot[p]]
        tl, vl, el = _loaders(tr, by_pot[val_pot], by_pot[test_pot], cfg, out_dir / "normals_cache")
        set_all_seeds(t["seed"])
        model = DGCNNSeg(2, 7, cfg["model"]["k"]).to(device)
        hist = train_model(model, tl, vl, class_weights(tr), ckpt, device, t["lopo_epochs"], t["lopo_patience"],
                           t["lr"], t["weight_decay"], name=f"DGCNN-{test_pot}", verbose=False)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        res = {"test_pot": test_pot, "val_pot": val_pot, **hist,
               "test": {k: v for k, v in evaluate(model, el, device).items() if k != "scan_mious"}}
        meta.write_text(json.dumps(res, indent=2)); results.append(res)
        print(f"[{i + 1}/{len(pots)}] {test_pot}: test mIoU {res['test']['miou'] * 100:.2f}% ({hist['train_minutes']:.0f} min)")
    return results
