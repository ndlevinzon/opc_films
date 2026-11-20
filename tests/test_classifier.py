"""Synthetic checks for the classifier and the project manifest.

No trajectory is read. The Gaussians are separable on purpose so a failed
leave-one-replica-out score is a bug in the discriminant, not in the film.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import numpy as np

from common.classifier import (
    CLASSIFIER_ID,
    leave_one_replica_out,
    normalized_fraction,
    selections,
)
from common.constants import BULK_HALFWIDTH_A
from common.manifest import cutoff_note, output_directory, read_cut
from common.slab import slab_geometry


class ClassifierTests(unittest.TestCase):
    def test_leave_one_replica_out_on_separated_gaussians(self):
        rng = np.random.default_rng(0)
        by_replica = {}
        for replica, shift in (("r1", 0.0), ("r2", 0.05), ("r3", -0.05)):
            by_replica[replica] = {
                "300": rng.normal(scale=0.25, size=(60, 3)) + shift,
                "78": rng.normal(loc=3.0, scale=0.25, size=(60, 3)) + shift,
            }
        rows = leave_one_replica_out(by_replica)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["misclassification_rate"] < 0.05 for row in rows))
        self.assertTrue(all(row["classifier_id"] == CLASSIFIER_ID for row in rows))

    def test_normalized_fraction_is_per_region(self):
        values = normalized_fraction([0.2, 0.6, 1.0], 0.2, 1.0)
        self.assertTrue(np.allclose(values, [0.0, 0.5, 1.0]))

    def test_bulk_mask_is_15_angstroms(self):
        self.assertEqual(BULK_HALFWIDTH_A, 15.0)
        z_values = np.linspace(80.0, 220.0, 80)
        points = np.array([
            (x, y, z)
            for z in z_values
            for x in np.linspace(1.0, 29.0, 4)
            for y in np.linspace(1.0, 29.0, 4)
        ])
        lengths = [30.0, 30.0, 300.0]
        chosen = selections(points, lengths)
        interior = chosen["masks"]["interior"]
        relative, _depth, _bounds, _center = slab_geometry(points, lengths)
        self.assertTrue(np.any(interior))
        self.assertLessEqual(np.max(np.abs(relative[interior])), 15.0 + 1e-6)
        self.assertFalse(np.any(interior & chosen["masks"]["awi_3_12"]))
        self.assertFalse(np.any(interior & chosen["masks"]["awi_2_10"]))
        self.assertFalse(np.any(interior & chosen["masks"]["awi_4_14"]))

    def test_output_path_encodes_kind(self):
        path = output_directory(Path("outputs"), "hold", "r2", "4.44", "78", "chunk")
        self.assertEqual(path.parts[-5:], ("hold", "r2", "4.44Kns", "78K", "chunk"))

    def test_cut_is_the_last_header_value(self):
        with tempfile.TemporaryDirectory() as directory:
            mdout = Path(directory) / "hold.mdout"
            mdout.write_text("cut = 8.0\ncut = 9.0\n   A V E R A G E S\ncut = 1.0\n")
            self.assertEqual(read_cut(mdout), 9.0)

    def test_missing_hold_mdout_is_reported(self):
        note = cutoff_note([
            {"kind": "ramp", "cut_A": "9"},
            {"kind": "hold", "cut_A": ""},
        ])
        self.assertIn("not", note)
        self.assertIn("8.0", note)
        self.assertIn("9.0", note)


if __name__ == "__main__":
    unittest.main()
