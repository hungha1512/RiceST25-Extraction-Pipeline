import unittest


class E57IntegrationTest(unittest.TestCase):
    @unittest.skip("Requires a redistributable golden E57 and trained checkpoint")
    def test_golden_e57_pipeline(self) -> None:
        pass
