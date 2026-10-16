"""Segment cropped scan(s) and extract plant height and tiller count.

    python extract.py scan.ply --checkpoint weights/lopo/DGCNN_C1.1.pth --output outputs/my_scan --save_segmented

Input: one pot + one plant per file (.ply or .e57), vertical axis +z, metric units, already cropped and denoised.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from inference import analyze_cloud
from models.dgcnn import load_dgcnn, resolve_device
from utils.config import load_config
from utils.pointcloud import read_cloud, write_ply

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="+", help=".ply / .e57 file(s)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="outputs/extract")
    ap.add_argument("--scan_index", type=int, default=0, help="scan inside an .e57 file")
    ap.add_argument("--save_segmented", action="store_true", help="write <name>_segmented.ply and <name>_chm.npy")
    ap.add_argument("--config", help="YAML overriding configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["inference"]["device"])
    model = load_dgcnn(args.checkpoint, device, cfg["model"]["k"])
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in args.input:
        cloud = read_cloud(f, args.scan_index)
        res = analyze_cloud(model, cloud, cfg, device)
        h, t = res["height"], res["tillers"]
        row = {"file": str(f), "n_points": len(cloud["xyz"]), "plant_height_cm": h and h["height_cm"],
               "tiller_count": t and t["n_tillers"], "n_plant_points": int((res["pred"] == 1).sum()),
               "n_pot_points": int((res["pred"] == 0).sum())}
        rows.append(row)
        print(json.dumps(row, default=float))
        if args.save_segmented:
            write_ply(out / f"{Path(f).stem}_segmented.ply", cloud["xyz"], res["pred"])
            if t is not None:
                np.save(out / f"{Path(f).stem}_chm.npy", t["chm"])
    pd.DataFrame(rows).to_csv(out / "extract.csv", index=False)
