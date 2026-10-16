import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement

from utils.config import load_config
from datasets.rice import RiceDataset, class_weights
from reproduce import run_reproduction
from train import train_lopo

from conftest import synthetic_scan

POTS = ["C1.1", "C2.1", "C3.1"]


def _dataset(root):
    rows = []
    for pot in POTS:
        for day in ("26410", "26422"):
            xyz, lab = synthetic_scan(n_pot=900, n_plant=600, seed=len(rows))
            v = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("scalar_Plant", "f4")])
            v["x"], v["y"], v["z"] = xyz.T
            v["scalar_Plant"] = np.where(lab == 1, 1.0, np.nan)   # CloudCompare convention: NaN = not Plant
            PlyData([PlyElement.describe(v, "vertex")]).write(str(root / f"{pot}_{day}.ply"))
            rows.append((f"{pot}_{day}", 10, 55.0))
    gt = root / "gt.csv"
    pd.DataFrame(rows, columns=["scan", "tillers", "height_cm"]).to_csv(gt, index=False)
    return gt


def test_dataset_sample_shapes_and_weights(tmp_path):
    _dataset(tmp_path)
    files = sorted(tmp_path.glob("*.ply"))
    x, y = RiceDataset(files, num_points=256, augment=True, cache_dir=tmp_path / "c")[0]
    assert x.shape == (7, 256) and y.shape == (256,)
    w = class_weights(files)
    assert w[1] > w[0]  # Plant is the minority class


def test_train_lopo_then_reproduce(tmp_path):
    gt = _dataset(tmp_path)
    cfg = load_config()
    cfg["training"].update(lopo_epochs=1, lopo_patience=1, num_workers=0, batch_size=2, num_points=256)
    cfg["inference"].update(num_points=256, device="cpu")
    files = sorted(tmp_path.glob("*.ply"))
    res = train_lopo(files, tmp_path / "w", cfg)
    assert [r["test_pot"] for r in res] == POTS and [r["val_pot"] for r in res] == ["C2.1", "C3.1", "C1.1"]
    df = run_reproduction(tmp_path, tmp_path / "w", gt, tmp_path / "out", cfg, limit=1)
    assert len(df) == 3 and {"h_pred", "t_pred", "h_gt", "t_gt", "miou_full", "lodged"} <= set(df.columns)
    assert df.h_gt.eq(55.0).all()
