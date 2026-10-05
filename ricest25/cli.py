"""Command line interface:  python -m ricest25 <command> --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import load_config


def _cmd_analyze(a, cfg):
    from .models import load_dgcnn, resolve_device
    from .pipeline import analyze_file

    device = resolve_device(cfg["inference"]["device"])
    model = load_dgcnn(a.checkpoint, device, cfg["model"]["k"])
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in a.input:
        res = analyze_file(model, f, cfg, device, a.scan_index, out if a.save_segmented else None)
        h, t = res["height"], res["tillers"]
        row = {"file": str(f), "n_points": res["n_points"], "plant_height_cm": h and h["height_cm"],
               "tiller_count": t and t["n_tillers"], "n_plant_points": int((res["pred"] == 1).sum()),
               "n_pot_points": int((res["pred"] == 0).sum())}
        if "miou_full" in res:
            row["miou_full"] = res["miou_full"]
        rows.append(row)
        print(json.dumps(row, default=float))
        if a.save_segmented and t is not None:
            np.save(out / f"{Path(f).stem}_chm.npy", t["chm"])
    pd.DataFrame(rows).to_csv(out / "analysis.csv", index=False)


def _cmd_reproduce(a, cfg):
    from .reproduce import run_reproduction, write_report

    df = run_reproduction(a.data, a.weights, a.ground_truth, a.output, cfg, a.pots, a.limit)
    print(write_report(df, a.output, cfg))


def _cmd_report(a, cfg):
    from .reproduce import write_report

    print(write_report(pd.read_csv(a.results), Path(a.results).parent if a.output is None else a.output, cfg))


def _cmd_eval_seg(a, cfg):
    from .models import load_dgcnn, resolve_device
    from .train import RiceDataset, evaluate, pot_of
    from torch.utils.data import DataLoader

    device = resolve_device(cfg["inference"]["device"])
    files = sorted(Path(a.data).rglob("*.ply"))
    pots = set(a.pots) if a.pots else set(yaml.safe_load(open(a.split))["test"])
    files = [f for f in files if pot_of(f) in pots]
    model = load_dgcnn(a.checkpoint, device, cfg["model"]["k"])
    ds = RiceDataset(files, "xyz_h_n", 4096, False, cfg["features"]["normal_k"], Path(a.output) / "normals_cache")
    res = evaluate(model, DataLoader(ds, 8, num_workers=0), device)
    print(json.dumps({"n_scans": len(files), "pots": sorted(pots), "mIoU_%": res["miou"] * 100,
                      "IoU_pot_%": res["iou_per_class"][0] * 100, "IoU_plant_%": res["iou_per_class"][1] * 100,
                      "scan_mean_%": res["miou_scan_mean"] * 100, "scan_sd_%": res["miou_scan_std"] * 100}, indent=2))


def _cmd_train(a, cfg):
    from .train import train_lopo, train_split

    files = sorted(Path(a.data).rglob("*.ply"))
    if a.lopo:
        train_lopo(files, a.output, cfg, a.pots)
    else:
        split = yaml.safe_load(open(a.split))
        print(json.dumps(train_split(files, split, a.output, cfg, a.mode, a.seed), indent=2))


def _cmd_inspect(a, cfg):
    import torch

    from .models import load_dgcnn

    m = load_dgcnn(a.checkpoint)
    print(json.dumps({"checkpoint": a.checkpoint, "in_channels": m.ec1.net[0].in_channels // 2,
                      "num_classes": m.seg[-1].out_channels, "k": m.ec1.k,
                      "parameters": sum(p.numel() for p in m.parameters())}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("ricest25", description=__doc__)
    p.add_argument("--config", help="YAML overriding configs/default.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("analyze", help="segment cropped scan(s) and extract height + tillers")
    s.add_argument("input", nargs="+"); s.add_argument("--checkpoint", required=True)
    s.add_argument("--output", default="outputs/analyze"); s.add_argument("--scan-index", type=int, default=0)
    s.add_argument("--save-segmented", action="store_true", help="write <name>_segmented.ply and the CHM array")
    s.set_defaults(func=_cmd_analyze)

    s = sub.add_parser("reproduce", help="255-scan out-of-fold evaluation with the 13 LOPO checkpoints")
    s.add_argument("--data", required=True, help="folder of labelled <pot>_<date>.ply scans")
    s.add_argument("--weights", default="weights/lopo"); s.add_argument("--ground-truth", default="data/ground_truth.csv")
    s.add_argument("--output", default="outputs/reproduce"); s.add_argument("--pots", nargs="*")
    s.add_argument("--limit", type=int, help="max scans per pot (smoke test)")
    s.set_defaults(func=_cmd_reproduce)

    s = sub.add_parser("report", help="rebuild tables/report from an existing scan_results.csv (no GPU, no data)")
    s.add_argument("results"); s.add_argument("--output")
    s.set_defaults(func=_cmd_report)

    s = sub.add_parser("eval-seg", help="4096-point segmentation mIoU of one checkpoint on chosen pots")
    s.add_argument("--data", required=True); s.add_argument("--checkpoint", required=True)
    s.add_argument("--split", default="configs/split_potwise.yaml"); s.add_argument("--pots", nargs="*")
    s.add_argument("--output", default="outputs/eval_seg")
    s.set_defaults(func=_cmd_eval_seg)

    s = sub.add_parser("train", help="retrain DGCNN (pot-wise split, or --lopo for the 13 folds)")
    s.add_argument("--data", required=True); s.add_argument("--output", default="outputs/train")
    s.add_argument("--split", default="configs/split_potwise.yaml"); s.add_argument("--mode", default="xyz_h_n")
    s.add_argument("--seed", type=int); s.add_argument("--lopo", action="store_true"); s.add_argument("--pots", nargs="*")
    s.set_defaults(func=_cmd_train)

    s = sub.add_parser("inspect", help="print the architecture contract of a checkpoint")
    s.add_argument("checkpoint")
    s.set_defaults(func=_cmd_inspect)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args, load_config(args.config))
