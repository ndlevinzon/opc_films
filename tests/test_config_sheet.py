"""Every sheet field used by the analysis is read from the sheet."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

READS = (
    "qstar.peak_min_Ainv",
    "qstar.peak_max_Ainv",
    "qstar.peak_window_Ainv",
    "qstar.adopt_from",
    "dynamics.hold_lag_fraction",
    "dynamics.cooling_lag_cap_K",
    "dynamics.cooling_lags_ps",
    "dynamics.origins",
    "dynamics.references_per_region",
    "dynamics.event_window_ps",
    "hbond.origins",
    "classifier.scale_mad",
    "classifier.covariance_jitter",
    "classifier.frames",
    "classifier.threshold",
    "probe.oxygen_exclusion_A",
    "probe.radius_A",
    "probe.sensitivity_radii_A",
    "probe.n_probes",
    "arrhenius.window_K",
    "arrhenius.sensitivity_lower_K",
    "energy.bin_K",
    "energy.f_scale_kcal_mol_per_water",
    "energy.fit_min_K",
    "energy.fit_max_K",
    "energy.glass_min_K",
    "energy.glass_max_K",
    "energy.liquid_min_K",
    "energy.liquid_max_K",
    "energy.report_K",
    "uncertainty.paired_confidence",
    "discard.ramp_fraction",
    "trajectory.production_cut_A",
    "trajectory.equilibration_cut_A",
    "trajectory.cooling_ntwx",
    "trajectory.hold_ntwx",
    "trajectory.cooling_frame_ps",
    "trajectory.hold_frame_ps",
)


class ConfigSheetTest(unittest.TestCase):
    def test_each_field_is_read_outside_the_loader(self):
        text = []
        for folder in ("src", "analysis", "MD"):
            for path in (ROOT / folder).rglob("*.py"):
                if path.name == "config.py":
                    continue
                text.append(path.read_text(encoding="utf-8"))
        blob = "\n".join(text)
        missing = [name for name in READS if name not in blob]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
