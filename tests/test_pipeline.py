import numpy as np
import pandas as pd
import torch

from utils.config import DEFAULTS, load_config
from utils.metrics import bootstrap_ci, height_by_group, oof_height_calibration, parse_scan_id, tillers_by_subset, trait_metrics
from utils.features import build_features, estimate_normals, normalize_xyz
from inference import segment_cloud
from utils.pointcloud import read_ply, write_ply
from models.dgcnn import DGCNNSeg, load_dgcnn

from conftest import synthetic_scan


def test_default_yaml_matches_defaults():
    assert load_config("configs/default.yaml") == DEFAULTS


def test_dgcnn_shapes_and_state_dict_keys():
    m = DGCNNSeg(2, 7, 20)
    out = m.eval()(torch.randn(2, 7, 128))
    assert out.shape == (2, 128, 2)
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, 128), atol=1e-5)
    keys = set(m.state_dict())
    assert {"ec1.net.0.weight", "ec4.net.1.running_mean", "cg.0.weight", "seg.7.bias"} <= keys


def test_checkpoint_roundtrip_infers_channels(tmp_path):
    torch.save(DGCNNSeg(2, 4, 20).state_dict(), tmp_path / "m.pth")
    m = load_dgcnn(tmp_path / "m.pth")
    assert m.ec1.net[0].in_channels == 8 and m.seg[-1].out_channels == 2


def test_features_shapes_and_normals(scan):
    xyz, _ = scan
    n = estimate_normals(xyz, 20)
    assert n.shape == xyz.shape and (n[:, 2] >= -1e-6).all()
    assert np.allclose(np.linalg.norm(n, axis=1), 1, atol=1e-3)
    f = build_features(normalize_xyz(xyz), n, "xyz_h_n")
    assert f.shape == (len(xyz), 7)
    assert np.linalg.norm(f[:, :3], axis=1).max() == np.float32(1.0) or abs(np.linalg.norm(f[:, :3], axis=1).max() - 1) < 1e-5


def test_segment_cloud_labels_every_point_and_is_deterministic():
    torch.manual_seed(0)
    xyz, _ = synthetic_scan(n_pot=1500, n_plant=900)
    m = DGCNNSeg(2, 7, 20).eval()
    a = segment_cloud(m, xyz, seed=1, num_points=512, extra_passes=2)
    b = segment_cloud(m, xyz, seed=1, num_points=512, extra_passes=2)
    assert a.shape == (len(xyz),) and set(np.unique(a)) <= {0, 1}
    assert (a == b).all()


def test_batch_size_does_not_change_votes():
    torch.manual_seed(0)
    xyz, _ = synthetic_scan(n_pot=1500, n_plant=900)
    m = DGCNNSeg(2, 7, 20).eval()
    kw = dict(seed=3, num_points=512, extra_passes=2)
    assert (segment_cloud(m, xyz, batch_size=1, **kw) == segment_cloud(m, xyz, batch_size=4, **kw)).all()


def test_ply_roundtrip_with_labels(tmp_path, scan):
    xyz, label = scan
    write_ply(tmp_path / "a.ply", xyz, label)
    d = read_ply(tmp_path / "a.ply")
    assert np.allclose(d["xyz"], xyz) and (d["label"] == label).all()


def test_parse_scan_id_and_lodging():
    a, b = parse_scan_id("C1.4_26422"), parse_scan_id("C3.2_26510")
    assert str(a["date"].date()) == "2026-04-22" and not a["lodged"] and a["density"] == "C1"
    assert str(b["date"].date()) == "2026-05-10" and b["lodged"]
    assert str(parse_scan_id("C1.1_2658")["date"].date()) == "2026-05-08"


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        pot = f"C{1 + i % 3}.{1 + i % 4}"
        scan = f"{pot}_{rng.choice(['26410', '26422', '26511'])}"
        h = rng.uniform(45, 105)
        rows.append({"scan": scan, **parse_scan_id(scan), "h_gt": h, "h_pred": h - 3 + rng.normal(0, 2),
                     "t_gt": rng.integers(4, 30), "t_pred": rng.integers(4, 30)})
    return pd.DataFrame(rows)


def test_trait_metrics_known_values():
    m = trait_metrics([1, 2, 3, 4], [2, 3, 4, 5])
    assert m["MAE"] == 1 and m["Bias"] == -1 and abs(m["R2"] - 1) < 1e-12 and abs(m["RMSE"] - 1) < 1e-12


def test_offset_calibration_is_out_of_fold_and_removes_bias():
    df = oof_height_calibration(_frame())
    assert abs((df.h_off - df.h_gt).mean()) < 1.0  # ~3 cm bias removed
    assert abs((df.h_pred - df.h_gt).mean()) > 2.0


def test_tables_and_bootstrap_run():
    df = _frame()
    assert height_by_group(df).loc["Overall", "n"] == len(df)
    t7 = tillers_by_subset(df)
    assert t7.loc["Overall", "n"] == len(df)
    ci = bootstrap_ci(df, "h_pred", "h_gt", n_boot=50)
    assert ci["MAE"][0] <= ci["MAE"][1]
