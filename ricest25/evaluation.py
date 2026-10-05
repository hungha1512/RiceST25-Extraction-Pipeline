"""Accuracy metrics, hierarchical bootstrap over pots, and out-of-fold height offset calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

LODGED_POTS, LODGING_DATE = ("C1.3", "C3.2"), pd.Timestamp("2026-05-10")


def parse_scan_id(scan: str) -> dict:
    """'C1.4_26422' -> pot C1.4, density C1, date 2026-04-22 (day code = 2, month digit, day)."""
    pot, code = scan.split("_")
    date = pd.Timestamp(f"2026-{code[2:3].zfill(2)}-{code[3:].zfill(2)}")
    return {"pot": pot, "density": pot.split(".")[0], "date": date,
            "lodged": pot in LODGED_POTS and date >= LODGING_DATE}


def trait_metrics(pred, gt) -> dict:
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    err = pred - gt
    r = np.corrcoef(pred, gt)[0, 1] if len(pred) > 2 and pred.std() > 0 and gt.std() > 0 else np.nan
    rmse = float(np.sqrt((err ** 2).mean()))
    return {"n": len(pred), "MAE": float(np.abs(err).mean()), "RMSE": rmse,
            "RRMSE_%": rmse / float(gt.mean()) * 100, "Bias": float(err.mean()), "R2": float(r ** 2), "r": float(r)}


def bootstrap_ci(df: pd.DataFrame, pred: str, gt: str, n_boot: int = 10000, seed: int = 0,
                 keys=("MAE", "RMSE", "Bias", "R2")) -> dict:
    """95% CI by resampling pots, then scans within each drawn pot (hierarchical bootstrap)."""
    rng = np.random.default_rng(seed)
    p, g = df[pred].to_numpy(float), df[gt].to_numpy(float)
    by_pot = [np.flatnonzero((df["pot"] == k).to_numpy()) for k in df["pot"].unique()]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(by_pot[j], len(by_pot[j])) for j in rng.integers(0, len(by_pot), len(by_pot))])
        m = trait_metrics(p[idx], g[idx])
        for k in keys:
            if np.isfinite(m[k]):
                draws[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items() if v}


def oof_height_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``h_off`` / ``h_cal``: height corrected with an offset / linear fit estimated on the other pots."""
    out = df.copy()
    out["h_off"], out["h_cal"] = np.nan, np.nan
    for pot in out["pot"].unique():
        tr, te = out["pot"] != pot, out["pot"] == pot
        offset = float((out.loc[tr, "h_gt"] - out.loc[tr, "h_pred"]).mean())
        a, b = np.polyfit(out.loc[tr, "h_pred"], out.loc[tr, "h_gt"], 1)
        out.loc[te, "h_off"] = out.loc[te, "h_pred"] + offset
        out.loc[te, "h_cal"] = a * out.loc[te, "h_pred"] + b
    return out


def height_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Table VI: height error by measured-height group, half-open intervals [a, b)."""
    bins, names = [0, 60, 75, 90, np.inf], ["<60", "[60,75)", "[75,90)", ">=90"]
    grp = pd.cut(df["h_gt"], bins, labels=names, right=False)
    rows = {n: {"n": int(len(g)), "GT_mean": g.h_gt.mean(), **{k: v for k, v in trait_metrics(g.h_pred, g.h_gt).items()
                                                               if k in ("MAE", "RMSE", "Bias")}}
            for n, g in df.groupby(grp, observed=True)}
    rows["Overall"] = {"n": int(len(df)), "GT_mean": df.h_gt.mean(),
                       **{k: v for k, v in trait_metrics(df.h_pred, df.h_gt).items() if k in ("MAE", "RMSE", "Bias")}}
    return pd.DataFrame(rows).T


def tillers_by_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Table VII: tiller error by planting density and by period (April / May upright / May lodged)."""
    period = np.where(df["date"] < pd.Timestamp("2026-05-01"), "April",
                      np.where(df["lodged"], "May, lodged", "May, not lodged"))
    subsets = [(f"{d} ({n} plant{'s' if n > 1 else ''}/pot)", df["density"] == d)
               for d, n in (("C1", 1), ("C2", 2), ("C3", 3))]
    subsets += [(p, pd.Series(period == p, index=df.index)) for p in ("April", "May, not lodged", "May, lodged")]
    subsets.append(("Overall", pd.Series(True, index=df.index)))
    rows = {}
    for name, mask in subsets:
        g = df[mask]
        if len(g):
            rows[name] = {"n": int(len(g)), "GT_mean": g.t_gt.mean(), "Pred_mean": g.t_pred.mean(),
                          "MAE": float((g.t_pred - g.t_gt).abs().mean()), "Bias": float((g.t_pred - g.t_gt).mean())}
    return pd.DataFrame(rows).T


def summarize(df: pd.DataFrame, n_boot: int = 10000, seed: int = 0) -> tuple[dict, dict]:
    """Everything reported for the 255-scan out-of-fold evaluation."""
    dh, dt = df.dropna(subset=["h_pred", "h_gt"]), df.dropna(subset=["t_pred", "t_gt"])
    if len(dh) < 3 or len(dt) < 3:
        raise ValueError(f"only {len(dh)} scans have a valid height and {len(dt)} a tiller count; "
                         "check the checkpoints (a model that labels everything Plant or Pot gives no height)")
    cal = oof_height_calibration(dh)
    res = {
        "n_scans": int(len(df)), "n_pots": int(df["pot"].nunique()),
        "mIoU_full_cloud_mean_%": float(df["miou_full"].mean() * 100) if "miou_full" in df else None,
        "height": trait_metrics(dh.h_pred, dh.h_gt),
        "height_ci95": bootstrap_ci(dh, "h_pred", "h_gt", n_boot, seed),
        "height_oof_offset": trait_metrics(cal.h_off, cal.h_gt),
        "height_oof_linear": trait_metrics(cal.h_cal, cal.h_gt),
        "height_lodged_MAE": float((dh[dh.lodged].h_pred - dh[dh.lodged].h_gt).abs().mean()) if dh.lodged.any() else None,
        "tillers": trait_metrics(dt.t_pred, dt.t_gt),
        "tillers_ci95": bootstrap_ci(dt, "t_pred", "t_gt", n_boot, seed),
    }
    return res, {"table_vi_height_by_group": height_by_group(dh), "table_vii_tillers": tillers_by_subset(dt),
                 "height_calibrated": cal}
