import unittest


class PLYIntegrationTest(unittest.TestCase):
    @unittest.skip("Requires a redistributable golden PLY and trained checkpoint")
    def test_golden_ply_pipeline(self) -> None:
        pass
