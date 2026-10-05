import numpy as np

from ricest25.config import load_config
from ricest25.phenotypes import canopy_height_model, count_tillers, filter_outliers, plant_height

from conftest import synthetic_scan


def test_height_matches_construction():
    xyz, label = synthetic_scan(plant_h=0.7)
    res = plant_height(xyz, label, load_config())
    # Pot z is uniform on [0, 0.147]: rim = z99, base = z1, apex = 0.7 m
    rim, base = 0.99 * 0.147, 0.01 * 0.147
    expected = (0.7 - rim) / (rim - base) * 14.7
    assert abs(res["height_cm"] - expected) < 0.3


def test_height_is_scale_invariant():
    xyz, label = synthetic_scan()
    cfg = load_config()
    a = plant_height(xyz, label, cfg)["height_cm"]
    cfg["height"]["dbscan"]["eps_m"] = 0.04  # scale cloud x2 -> same eps in cloud units
    b = plant_height(xyz * 2, label, cfg)["height_cm"]
    assert abs(a - b) < 0.5


def test_dbscan_removes_floating_points():
    xyz, label = synthetic_scan()
    stray = np.array([[0.0, 0.0, 2.0]] * 3, dtype=np.float32) + np.random.default_rng(1).normal(0, 1e-4, (3, 3))
    xyz2, label2 = np.vstack([xyz, stray]), np.r_[label, np.ones(3, dtype=np.int64)]
    cfg = load_config()
    assert plant_height(xyz2, label2, cfg)["height_cm"] < 60
    cfg["height"]["dbscan"]["enabled"] = False
    assert plant_height(xyz2, label2, cfg)["height_cm"] > 150  # without the filter the stray apex wins


def test_filter_outliers_small_cloud_kept():
    assert filter_outliers(np.zeros((10, 3), np.float32)).all()


def test_height_none_when_class_missing():
    xyz, label = synthetic_scan()
    assert plant_height(xyz, np.zeros_like(label), load_config()) is None


def test_chm_grid_and_max():
    pts = np.array([[0, 0, 1.0], [0.001, 0.001, 2.0], [0.02, 0.0, 0.5]], np.float32)
    chm = canopy_height_model(pts, 0.008)
    assert chm.shape == (1, 3)
    assert chm[0, 0] == 2.0 and chm[0, 2] == 0.5 and np.isnan(chm[0, 1])


def test_tiller_count_finds_separate_stems():
    xyz, label = synthetic_scan(n_stems=3, n_plant=9000)
    n = count_tillers(xyz[label == 1], load_config())["n_tillers"]
    assert n >= 3  # one peak per stem tip (plus possible noise peaks)


def test_tiller_threshold_excludes_low_peaks():
    cfg = load_config()
    low = np.array([[x * 0.02, 0, 0.05 + 0.001 * x] for x in range(20)] + [[0.0, 0.5, 1.0]], np.float32)
    # the single tall point defines tau = 0.25 of the range, so low-lying maxima must not be counted
    assert count_tillers(low, cfg)["n_tillers"] == 1
