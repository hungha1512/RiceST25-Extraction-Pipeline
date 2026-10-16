"""Reproduce the 255-scan trait evaluation of the manuscript.

Every scan is segmented by the leave-one-pot-out DGCNN that did not see its pot, height and
tillers are extracted, and the result is compared with the manual measurements.

    python reproduce.py --data data/RiceST25-3D --weights weights/lopo --output outputs/reproduce
    python reproduce.py --report_only outputs/reproduce/scan_results.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from inference import analyze_cloud
from models.dgcnn import load_dgcnn, resolve_device
from utils.config import load_config
from utils.metrics import parse_scan_id
from utils.pointcloud import read_ply
from utils.report import write_report


def find_checkpoint(weights_dir: Path, pot: str) -> Path:
    for name in (f"DGCNN_{pot}.pth", f"{pot}.pth"):
        if (weights_dir / name).is_file():
            return weights_dir / name
    raise FileNotFoundError(f"no leave-one-pot-out checkpoint for pot {pot} in {weights_dir} (expected DGCNN_{pot}.pth)")


def run_reproduction(data_dir, weights_dir, gt_csv, out_dir, cfg, pots=None, limit=None) -> pd.DataFrame:
    """Out-of-fold inference over a directory of labelled ``<pot>_<date>.ply`` scans (resumable per pot)."""
    data_dir, weights_dir, out_dir = Path(data_dir), Path(weights_dir), Path(out_dir)
    gt = pd.read_csv(gt_csv).set_index("scan")
    files = sorted(data_dir.rglob("*.ply"))
    if not files:
        raise FileNotFoundError(f"no .ply files under {data_dir}")
    by_pot: dict[str, list[Path]] = {}
    for f in files:
        by_pot.setdefault(f.stem.split("_")[0], []).append(f)
    device = resolve_device(cfg["inference"]["device"])
    (out_dir / "per_pot").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, (pot, scans) in enumerate(sorted(by_pot.items()), 1):
        if pots and pot not in pots:
            continue
        csv = out_dir / "per_pot" / f"{pot}.csv"
        if csv.exists():
            print(f"[{i}/{len(by_pot)}] {pot}: cached"); continue
        model = load_dgcnn(find_checkpoint(weights_dir, pot), device, cfg["model"]["k"])
        rows = []
        for f in scans[:limit]:
            scan = f.stem
            cloud = read_ply(f)
            res = analyze_cloud(model, cloud, cfg, device)
            h, t = res["height"], res["tillers"]
            rows.append({"scan": scan, **{k: v for k, v in parse_scan_id(scan).items() if k != "date"},
                         "date": parse_scan_id(scan)["date"].date().isoformat(), "n_points": len(cloud["xyz"]),
                         "miou_full": res.get("miou_full", np.nan), "iou_plant": res.get("iou_per_class", [np.nan] * 2)[1],
                         "h_pred": h["height_cm"] if h else np.nan, "h_gt": gt.height_cm.get(scan, np.nan),
                         "t_pred": t["n_tillers"] if t else np.nan, "t_gt": gt.tillers.get(scan, np.nan)})
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"[{i}/{len(by_pot)}] {pot}: {len(rows)} scans, mIoU {np.nanmean([r['miou_full'] for r in rows]) * 100:.2f}% "
              f"({(time.time() - t0) / 60:.1f} min)")
    df = pd.concat([pd.read_csv(p) for p in sorted((out_dir / "per_pot").glob("*.csv"))], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(out_dir / "scan_results.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder of labelled <pot>_<date>.ply scans")
    ap.add_argument("--weights", default="weights/lopo", help="folder with DGCNN_<pot>.pth (13 LOPO checkpoints)")
    ap.add_argument("--ground_truth", default="data/ground_truth.csv")
    ap.add_argument("--output", default="outputs/reproduce")
    ap.add_argument("--pots", nargs="*", help="only these pots")
    ap.add_argument("--limit", type=int, help="max scans per pot (smoke test)")
    ap.add_argument("--config", help="YAML overriding configs/default.yaml")
    ap.add_argument("--report_only", metavar="SCAN_RESULTS_CSV", help="rebuild the report from an existing CSV")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.report_only:
        df, out = pd.read_csv(args.report_only), Path(args.report_only).parent
    else:
        if not args.data:
            ap.error("--data is required unless --report_only is given")
        df, out = run_reproduction(args.data, args.weights, args.ground_truth, args.output, cfg, args.pots, args.limit), args.output
    print(write_report(df, out, cfg))
