"""One synthetic film through every stage."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opcfilm.synthetic import run


class SyntheticFilm(unittest.TestCase):
    def test_every_stage_passes_the_acceptance_checks(self):
        result = run()
        self.assertTrue(all(result["checks"].values()), result["checks"])


if __name__ == "__main__":
    unittest.main()
