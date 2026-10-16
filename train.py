"""Train DGCNN for plant-pot segmentation (Table II settings).

    python train.py --data data/RiceST25-3D --output outputs/train            # pot-wise split
    python train.py --data data/RiceST25-3D --output outputs/lopo --lopo      # 13 leave-one-pot-out folds
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from datasets.rice import RiceDataset, class_weights, pot_of
from models.dgcnn import DGCNNSeg, resolve_device
from utils.config import load_config
from utils.features import INPUT_DIM
from utils.seg_metrics import evaluate


def set_all_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder of labelled <pot>_<date>.ply scans")
    ap.add_argument("--output", default="outputs/train")
    ap.add_argument("--split", default="configs/split_potwise.yaml")
    ap.add_argument("--mode", default="xyz_h_n", choices=list(INPUT_DIM), help="input features: 3D, 4D or 7D")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--lopo", action="store_true", help="leave-one-pot-out instead of the pot-wise split")
    ap.add_argument("--pots", nargs="*", help="with --lopo: only these held-out pots")
    ap.add_argument("--config", help="YAML overriding configs/default.yaml")
    args = ap.parse_args()
    cfg, files = load_config(args.config), sorted(Path(args.data).rglob("*.ply"))
    if args.lopo:
        train_lopo(files, args.output, cfg, args.pots)
    else:
        print(json.dumps(train_split(files, yaml.safe_load(open(args.split)), args.output, cfg, args.mode, args.seed), indent=2))
