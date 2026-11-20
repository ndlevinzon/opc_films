"""The Phase A graph calls the existing scripts."""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

ROOT = pathlib.Path(__file__).resolve().parents[1]


class PhaseA(unittest.TestCase):
    def test_snakefile_calls_the_scripts(self):
        text = (ROOT / "workflow" / "Snakefile").read_text(encoding="utf-8")
        text += (ROOT / "workflow" / "reduce" / "fs_tau_row.sh").read_text(encoding="utf-8")
        self.assertIn("03_run_row.py", text)
        self.assertIn("03_fs_alpha.py", text)
        self.assertIn("opcfilm qstar adopt", text)
        self.assertNotIn("opcfilm s00 --", text)
        self.assertNotIn("opcfilm {stage}", text)

    def test_figure_table_names_each_script(self):
        text = (ROOT / "docs" / "figures.md").read_text(encoding="utf-8")
        for script in (
            "figure1_from_v2.py",
            "figure2_from_v3.py",
            "06_plot_energetic_stabilization.py",
            "05b_plot_tg_vs_cooling_rate.py",
            "08_replot_core_awi_black.py",
            "plot_local_structure.py",
            "06_plot_hypothesis.py",
        ):
            self.assertIn(script, text)

    def test_adopt_files_writes_the_shared_peak(self):
        from opcfilm.qstar import adopt_files

        directory = pathlib.Path(tempfile.mkdtemp())
        row = directory / "qstar_row.json"
        row.write_text(json.dumps({"qstar_measured_Ainv": 1.91}), encoding="utf-8")
        destination = directory / "qstar.json"
        payload = adopt_files([row], destination)
        self.assertEqual(payload["qstar_adopted_Ainv"], 1.925)
        self.assertTrue(destination.is_file())

    def test_set_rebinds_the_sheet(self):
        import opcfilm.config as config

        original = config.load_config()
        try:
            updated = config.apply_cli_overrides(["qstar.tolerance_Ainv=0.04"])
            self.assertEqual(updated.qstar.tolerance_Ainv, 0.04)
            self.assertEqual(config.CONFIG.qstar.tolerance_Ainv, 0.04)
            self.assertEqual(updated.overrides[-1][0], "qstar.tolerance_Ainv")
        finally:
            config._bind(original)


if __name__ == "__main__":
    unittest.main()
