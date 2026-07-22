import numpy as np
import unittest

from rice_tiller.phenotypes.chm import build_chm
from rice_tiller.phenotypes.tillers import detect_local_maxima, peaks_to_xyz


class CHMTillerTest(unittest.TestCase):
    def test_has_publication_raster_shape(self) -> None:
        rng = np.random.default_rng(42)
        xyz = rng.uniform(0, 1, size=(1000, 3)).astype(np.float32)
        chm, valid, bounds = build_chm(xyz, rows=35, cols=25)
        self.assertEqual(chm.shape, (35, 25))
        self.assertEqual(valid.shape, (35, 25))
        self.assertEqual(
            bounds,
            (xyz[:, 0].min(), xyz[:, 0].max(), xyz[:, 1].min(), xyz[:, 1].max()),
        )

    def test_local_maxima_respects_minimum_distance(self) -> None:
        image = np.zeros((35, 25), dtype=np.float32)
        image[5, 5] = 10
        image[6, 6] = 9
        image[20, 18] = 8
        valid = np.ones_like(image, dtype=bool)
        peaks = detect_local_maxima(image, valid, min_distance=3)
        self.assertEqual({tuple(item) for item in peaks}, {(5, 5), (20, 18)})
        xyz = peaks_to_xyz(peaks, image, (0.0, 25.0, 0.0, 35.0))
        self.assertEqual(xyz.shape, (2, 3))
