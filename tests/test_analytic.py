"""Analytic checks that do not need a trajectory."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opcfilm.dynamics.hbonds import directed_hydrogen_bonds
from opcfilm.structure.order import orientational_tetrahedral_q
from opcfilm.thermo.arrest import fit_arrhenius


class Analytic(unittest.TestCase):
    def test_perfect_tetrahedron_has_q_equal_to_one(self):
        vectors = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
        self.assertAlmostEqual(float(orientational_tetrahedral_q(vectors)), 1.0)

    def test_linear_hydrogen_bond_is_one_directed_pair(self):
        oxygen = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        hydrogen = np.zeros((2, 2, 3))
        hydrogen[0, 0] = (1.0, 0.0, 0.0)
        hydrogen[0, 1] = (0.0, 0.8, 0.0)
        hydrogen[1, 0] = (2.0, 0.6, 0.0)
        hydrogen[1, 1] = (3.2, 0.6, 0.0)
        pairs = directed_hydrogen_bonds(oxygen, hydrogen, (20.0, 20.0, 20.0), [0])
        self.assertIn((0, 1), pairs)

    def test_known_arrhenius_slope(self):
        temperature = np.array([200.0, 220.0, 240.0])
        fit = fit_arrhenius(temperature, np.exp(800.0 / temperature))
        self.assertAlmostEqual(fit["B_K"], 800.0, places=6)


if __name__ == "__main__":
    unittest.main()
