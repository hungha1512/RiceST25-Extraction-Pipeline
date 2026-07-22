import numpy as np
import unittest

from rice_tiller.inference.chunking import make_covering_chunks


class ChunkingTest(unittest.TestCase):
    def test_visits_every_point_once_before_padding(self) -> None:
        chunks = make_covering_chunks(103, 32, np.random.default_rng(42))
        visited = np.concatenate([chunk.valid_indices for chunk in chunks])
        self.assertEqual(len(visited), 103)
        np.testing.assert_array_equal(np.sort(visited), np.arange(103))
        self.assertTrue(all(len(chunk.indices) == 32 for chunk in chunks))
