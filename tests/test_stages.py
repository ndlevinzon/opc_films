"""One synthetic check for each analysis stage. No trajectory is read."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import pandas as pd

from common.classifier import sampled_frames
from common.constants import HBOND_COMPARISON_LAG_PS, HBOND_LAGS_PS, HOLD_DISCARD_FRACTION
from common.manifest import discard_fraction_for
from common.mdout import time_span_ps
from common.trajectory import hold_lag_times_ps


def _load(relative):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(path.stem.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GlassTransition(unittest.TestCase):
    def test_each_temperature_keeps_its_own_spacing(self):
        fit = _load("analysis/glass_transition/05_fit_relaxation_tg.py")
        frame = pd.DataFrame({
            "T_K": [230.0, 230.0, 220.0],
            "frame_ps_selected": [1.0, 1.0, 10.0],
        })
        spacings = fit.spacing_by_temperature(frame)
        self.assertEqual(fit.spacing_at(spacings, 230.0), 1.0)
        self.assertEqual(fit.spacing_at(spacings, 220.0), 10.0)
        self.assertAlmostEqual(0.5 * fit.spacing_at(spacings, 230.0) * 1e-3, 0.0005)
        mixed = frame.copy()
        mixed.loc[0, "frame_ps_selected"] = 2.0
        with self.assertRaises(RuntimeError):
            fit.spacing_by_temperature(mixed)


class CoolingManifest(unittest.TestCase):
    def test_hold_length_is_the_time_span_and_the_discard_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            mdout = Path(directory) / "hold.mdout"
            mdout.write_text(
                "NSTEP = 1 TIME(PS) = 0.0\n"
                "NSTEP = 2 TIME(PS) = 20000.0\n"
                "   A V E R A G E S\n"
                "NSTEP = 3 TIME(PS) = 99999.0\n"
            )
            self.assertEqual(time_span_ps(mdout), 20000.0)
        self.assertEqual(discard_fraction_for("hold"), HOLD_DISCARD_FRACTION)
        self.assertEqual(discard_fraction_for("endpoint"), HOLD_DISCARD_FRACTION)
        self.assertEqual(discard_fraction_for("ramp"), 0.0)


class ClassifierFrames(unittest.TestCase):
    def test_hold_frames_start_after_the_discard(self):
        frames = sampled_frames(100, 20, "hold")
        self.assertEqual(int(frames[0]), 10)
        self.assertEqual(len(frames), 20)
        ramp = sampled_frames(100, 20, "ramp")
        self.assertEqual(int(ramp[0]), 0)


class RegionalDynamics(unittest.TestCase):
    def test_hold_lags_reach_a_quarter_of_the_trajectory(self):
        lags = hold_lag_times_ps(20_000.0, 10.0)
        self.assertGreater(max(lags), 4096.0)
        self.assertAlmostEqual(max(lags), 0.25 * 20_000.0)
        self.assertEqual(HBOND_COMPARISON_LAG_PS, 320.0)
        self.assertEqual(HBOND_LAGS_PS[-1], 640.0)


class HoldRestart(unittest.TestCase):
    def test_low_temperature_restarts_use_three_digits(self):
        driver = _load("MD/hold_from_cooling.py")
        self.assertEqual(
            driver.cooling_restart(90, 4.44),
            "RST/cool_50ns_chunk_20_100_to_090K.rst7",
        )
        self.assertEqual(
            driver.cooling_restart(80, 4.44),
            "RST/cool_50ns_chunk_21_090_to_080K.rst7",
        )
        self.assertEqual(
            driver.cooling_restart(78, 4.44),
            "RST/cool_50ns_chunk_22_080_to_078K.rst7",
        )


if __name__ == "__main__":
    unittest.main()
