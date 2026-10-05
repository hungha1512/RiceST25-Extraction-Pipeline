"""Reproduce the manuscript's 255-scan evaluation: every scan is processed by the leave-one-pot-out
DGCNN whose training excluded that scan's pot, then compared with the manual measurements."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import parse_scan_id, summarize
from .io import read_ply
from .models import load_dgcnn, resolve_device
from .pipeline import analyze_cloud

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "paper_reference.json"


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


def _md(table: pd.DataFrame) -> str:
    """Minimal markdown table (avoids the optional tabulate dependency)."""
    t = table.round(2).reset_index().rename(columns={"index": "group"})
    t["n"] = t["n"].astype(int)
    lines = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def write_report(df: pd.DataFrame, out_dir, cfg) -> str:
    """Write summary.json, the two tables and report.md; returns the markdown."""
    out_dir = Path(out_dir)
    df = df.copy(); df["date"] = pd.to_datetime(df["date"])
    ev = cfg["evaluation"]
    summary, tables = summarize(df, ev["n_bootstrap"], ev["bootstrap_seed"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    tables["table_vi_height_by_group"].round(2).to_csv(out_dir / "table_vi_height_by_group.csv")
    tables["table_vii_tillers"].round(2).to_csv(out_dir / "table_vii_tillers.csv")
    tables["height_calibrated"].to_csv(out_dir / "height_oof_calibrated.csv", index=False)

    ref = json.loads(REFERENCE.read_text())
    h, t, ho = summary["height"], summary["tillers"], summary["height_oof_offset"]
    ci = lambda key, d: f"[{d[key][0]:.2f}, {d[key][1]:.2f}]" if key in d else ""
    rows = [
        ("mIoU, complete clouds (%)", ref["mIoU_full_cloud_mean_%"], summary["mIoU_full_cloud_mean_%"], ""),
        ("Height MAE (cm)", ref["height"]["MAE"], h["MAE"], ci("MAE", summary["height_ci95"])),
        ("Height RMSE (cm)", ref["height"]["RMSE"], h["RMSE"], ci("RMSE", summary["height_ci95"])),
        ("Height RRMSE (%)", ref["height"]["RRMSE_%"], h["RRMSE_%"], ""),
        ("Height R²", ref["height"]["R2"], h["R2"], ci("R2", summary["height_ci95"])),
        ("Height bias (cm)", ref["height"]["Bias"], h["Bias"], ci("Bias", summary["height_ci95"])),
        ("Height MAE after out-of-fold offset (cm)", ref["height_oof_offset"]["MAE"], ho["MAE"], ""),
        ("Height MAE, lodged scans (cm)", ref["height_lodged_MAE"], summary["height_lodged_MAE"], ""),
        ("Tiller MAE", ref["tillers"]["MAE"], t["MAE"], ci("MAE", summary["tillers_ci95"])),
        ("Tiller RMSE", ref["tillers"]["RMSE"], t["RMSE"], ""),
        ("Tiller bias", ref["tillers"]["Bias"], t["Bias"], ci("Bias", summary["tillers_ci95"])),
        ("Tiller R²", ref["tillers"]["R2"], t["R2"], ""),
    ]
    fmt = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.2f}"
    md = ["# RiceST25-3D reproduction report", "",
          f"{summary['n_scans']} scans, {summary['n_pots']} pots, each scan processed by the model that did not see its pot.",
          "", "| Metric | Manuscript | This run | 95% CI (pot bootstrap) |", "|---|---:|---:|---|"]
    md += [f"| {n} | {fmt(a)} | {fmt(b)} | {c} |" for n, a, b, c in rows]
    md += ["", "## Plant height by measured-height group (Table VI)", "", _md(tables["table_vi_height_by_group"]),
           "", "## Tiller counting by subset (Table VII)", "", _md(tables["table_vii_tillers"]), ""]
    text = "\n".join(md)
    (out_dir / "report.md").write_text(text)
    return text
