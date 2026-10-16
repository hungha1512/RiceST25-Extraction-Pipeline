"""Segmentation mIoU of one checkpoint on 4096-point samples (Table III / IV style).

    python eval_seg.py --data data/RiceST25-3D --checkpoint weights/potwise/DGCNN_seed42.pth      # test pots
    python eval_seg.py --data data/RiceST25-3D --checkpoint weights/lopo/DGCNN_C1.3.pth --pots C1.3
"""
import argparse
import json
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from datasets.rice import RiceDataset, pot_of
from models.dgcnn import load_dgcnn, resolve_device
from utils.config import load_config
from utils.seg_metrics import evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="configs/split_potwise.yaml", help="its `test` pots are used unless --pots")
    ap.add_argument("--pots", nargs="*")
    ap.add_argument("--output", default="outputs/eval_seg", help="cache for estimated normals")
    ap.add_argument("--config", help="YAML overriding configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["inference"]["device"])
    pots = set(args.pots or yaml.safe_load(open(args.split))["test"])
    files = [f for f in sorted(Path(args.data).rglob("*.ply")) if pot_of(f) in pots]
    model = load_dgcnn(args.checkpoint, device, cfg["model"]["k"])
    ds = RiceDataset(files, "xyz_h_n", 4096, False, cfg["features"]["normal_k"], Path(args.output) / "normals_cache")
    res = evaluate(model, DataLoader(ds, 8, num_workers=0), device)
    print(json.dumps({"n_scans": len(files), "pots": sorted(pots), "mIoU_%": res["miou"] * 100,
                      "IoU_pot_%": res["iou_per_class"][0] * 100, "IoU_plant_%": res["iou_per_class"][1] * 100,
                      "scan_mean_%": res["miou_scan_mean"] * 100, "scan_sd_%": res["miou_scan_std"] * 100}, indent=2))
