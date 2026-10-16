"""Markdown / JSON report that compares a reproduction run with the numbers in the manuscript."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.metrics import summarize

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "paper_reference.json"


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
