"""Synthetic slab, q* file, and peak fit. No trajectory is required."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import numpy as np

from common.constants import fft_cell_counts, load_adopted_qstar
from common.rates import parse_rate
from common.slab import region_masks, slab_geometry


def _sq():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "analysis" / "glass_transition" / "02_sq2_fft.py"
    spec = importlib.util.spec_from_file_location("sq2_fft_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticFilm(unittest.TestCase):
    def test_three_layer_slab_has_a_center_and_both_bands(self):
        xs = np.arange(1.5, 30.0, 3.0)
        ys = np.arange(1.5, 30.0, 3.0)
        zs = np.arange(80.0, 220.0, 3.1)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        oxygen = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        z_relative, depth, bounds, center = slab_geometry(oxygen, (30.0, 30.0, 300.0))
        self.assertGreater(center, 140.0)
        self.assertLess(center, 160.0)
        self.assertLess(bounds[0], 0.0)
        self.assertGreater(bounds[1], 0.0)
        masks = region_masks(z_relative, depth)
        self.assertTrue(masks["interior"].any())
        self.assertTrue(masks["awi"].any())
        self.assertTrue(masks["awi_outer"].any())
        self.assertFalse(np.any(masks["interior"] & (masks["awi"] | masks["awi_outer"])))

    def test_fft_grid_uses_round(self):
        self.assertEqual(fft_cell_counts(194.567, 0.5), 389)
        self.assertEqual(fft_cell_counts(188.781, 0.5), 378)

    def test_quadratic_peak_and_form_factor(self):
        module = _sq()
        q = np.linspace(1.5, 2.3, 81)
        intensity = 1.0 - (q - 1.925) ** 2
        self.assertAlmostEqual(module.quadratic_peak(q, intensity, 0.15, 1.5, 2.3), 1.925, places=3)
        wave = np.zeros((2, 2))
        corrected = module.divide_histogram_form_factor(np.ones((2, 2)), wave, wave, 0.5, 0.5)
        self.assertTrue(np.allclose(corrected, 1.0))

    def test_qstar_file_is_required_and_must_agree(self):
        with self.assertRaises(FileNotFoundError):
            load_adopted_qstar(Path("outputs/missing-qstar.json"))
        path = Path(tempfile.mkdtemp()) / "qstar.json"
        path.write_text(json.dumps({"qstar_adopted_Ainv": 1.925}))
        self.assertEqual(load_adopted_qstar(path), 1.925)
        self.assertEqual(load_adopted_qstar(path, 1.925), 1.925)
        with self.assertRaises(ValueError):
            load_adopted_qstar(path, 1.90)

    def test_directory_code_is_not_a_cooling_rate(self):
        with self.assertRaises(ValueError):
            parse_rate(4)

    def test_hold_restart_continues_the_cooling_window(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "MD" / "hold_from_cooling.py"
        spec = importlib.util.spec_from_file_location("hold_from_cooling_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            module.cooling_restart(250, 4.44),
            "RST/cool_50ns_chunk_04_260_to_250K.rst7",
        )


if __name__ == "__main__":
    unittest.main()
