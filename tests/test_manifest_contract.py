"""A script that takes the manifest must not crash on a renamed column.

The failure, when there is one, has to be about a missing trajectory or a
missing table. ``KeyError`` and ``AttributeError`` on ``outdir``,
``trajectory``, ``rate_code``, ``rate_K_ns``, or ``temp_mid_K`` are the
failures this test exists to catch.
"""

import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from opcfilm.catalog import discover_rows, load_manifest, trajectory_list, write_manifest


FORBIDDEN = re.compile(
    r"KeyError: ['\"]?(outdir|trajectory|rate_code|rate_K_ns|temp_mid_K)"
    r"|AttributeError:.*'(outdir|trajectory|rate_code|rate_K_ns|temp_mid_K)'"
)


def _mdout(start, end, restart):
    return (
        "File Assignments:\n"
        f"|  INPCRD: {restart}\n"
        " &cntrl\n"
        "  cut = 9.0,\n"
        "  ntwx = 5000,\n"
        "  dt = 0.002,\n"
        " /\n"
        f" NSTEP = 1 TIME(PS) = {start}\n"
        " EPtot = -1.0\n"
        f" NSTEP = 2 TIME(PS) = {end}\n"
        " EPtot = -1.0\n"
        "      A V E R A G E S\n"
        " NSTEP = 2 TIME(PS) = 99999\n"
    )


def _layout(root):
    hold = root / "opc" / "r1" / "hold" / "250K"
    (hold / "TRAJ").mkdir(parents=True)
    (hold / "OUT").mkdir()
    (hold / "TRAJ" / "traj.1").write_bytes(b"")
    (hold / "TRAJ" / "traj.2").write_bytes(b"")
    (hold / "OUT" / "mdout.1").write_text(_mdout(0, 10000, "RST/cool_50ns_chunk_04_260_to_250K.rst7"))
    (hold / "OUT" / "mdout.2").write_text(_mdout(0, 5000, "RST/rst.1"))
    ramp = root / "opc" / "r1" / "ramp_nvt" / "4K_ns"
    (ramp / "TRAJ").mkdir(parents=True)
    (ramp / "OUT").mkdir()
    name = "cool_50ns_chunk_00_300_to_290K"
    (ramp / "TRAJ" / f"{name}.nc").write_bytes(b"")
    (ramp / "OUT" / f"{name}.mdout").write_text(_mdout(0, 2000, "min/step9.rst7"))
    (root / "top.prmtop").write_bytes(b"")


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManifestContract(unittest.TestCase):
    def test_profile_uses_the_slurm_executor(self):
        text = (ROOT / "workflow" / "profiles" / "slurm" / "config.yaml").read_text()
        self.assertIn("executor: slurm", text)
        self.assertNotIn("\ncluster:", "\n" + text)

    def test_hold_outputs_are_split_by_row_and_by_segment(self):
        defects = _load(ROOT / "analysis" / "interface_dynamics" / "hydrogen_bonds" / "02_run_row.py")
        cold = defects.output_directory(Path("results/defects"), {"row_id": "hold-r1-4.44-150"})
        hot = defects.output_directory(Path("results/defects"), {"row_id": "hold-r1-4.44-250"})
        self.assertNotEqual(cold, hot)
        self.assertEqual(hot.name, "hold-r1-4.44-250")
        runner = _load(ROOT / "analysis" / "cooling_path" / "03_run_row.py")
        destination = Path(tempfile.mkdtemp())
        row = {
            "topology": "top.prmtop",
            "temperature_start_K": "250",
            "temperature_end_K": "250",
            "files": "traj.1;traj.2",
        }
        original = subprocess.run
        runner.subprocess.run = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0)
        try:
            runner.write_capwave_input(row, destination, 1, "cpptraj", 10.0, 100.0)
        finally:
            subprocess.run = original
        text = (destination / "capwave.cpptraj.in").read_text()
        self.assertIn("trajin traj.1 1 last 1", text)
        self.assertIn("trajin traj.2 1 last 1", text)
        self.assertEqual(len(trajectory_list(row)), 2)
        from opcfilm.io.trajectory import frames_skipped, trajin_lines
        self.assertEqual(frames_skipped("1500", "10"), 150)
        self.assertEqual(frames_skipped("", "10"), 0)
        kept = trajin_lines(["traj.1", "traj.2"], 150, 10, frame_counts=[1000, 500])
        self.assertEqual(kept, ["trajin traj.1 151 last 10", "trajin traj.2 1 last 10"])
        later = trajin_lines(["traj.1", "traj.2"], 1100, 1, frame_counts=[1000, 500])
        self.assertEqual(later, ["trajin traj.2 101 last 1"])
        defects_script = _load(
            ROOT / "analysis" / "interface_dynamics" / "hydrogen_bonds" / "03_analyze_free_oh.py"
        )
        seen = []
        defects_script.analyze_trajectory = lambda args: seen.append(list(args.traj)) or Path("out")
        defects_script.main([
            "--top", "top.prmtop", "--traj", "traj.1", "--traj", "traj.2",
            "--out", "out", "--start-frame", "150",
        ])
        self.assertEqual([path.name for path in seen[0]], ["traj.1", "traj.2"])
        chosen = defects_script.selected_frames(1500, 4, 150)
        self.assertEqual(int(chosen[0]), 150)
        self.assertEqual(int(chosen[-1]), 1499)

    def test_scripts_fail_on_missing_data_rather_than_a_column(self):
        root = Path(tempfile.mkdtemp())
        _layout(root)
        rows = discover_rows(root, root, root / "top.prmtop", ("r1",), (4,))
        manifest = root / "manifest.tsv"
        write_manifest(manifest, rows)
        loaded = load_manifest(manifest)
        hold = next(index for index, row in enumerate(loaded, start=1) if row["kind"] == "hold")
        ramp = next(index for index, row in enumerate(loaded, start=1) if row["kind"] == "ramp")
        out = root / "out"
        out.mkdir()
        energy = root / "energy_map.tsv"
        energy.write_text("trajectory\tmdout\n")
        commands = [
            ["analysis/cooling_path/03_run_row.py", "--manifest", manifest, "--row", ramp],
            ["analysis/cooling_path/03_run_row.py", "--manifest", manifest, "--row", hold, "--capwave"],
            ["analysis/cooling_path/04_summarize.py", "--manifest", manifest, "--out", out / "summary"],
            ["analysis/interface_dynamics/hydrogen_bonds/02_run_row.py", "--manifest", manifest, "--row", hold, "--outroot", out / "defects"],
            ["analysis/interface_dynamics/hydrogen_bonds/06_plot_hypothesis.py", "--manifest", manifest, "--outroot", out / "defects", "--out", out / "fig7", "--rate", "4.44"],
            ["analysis/cooling_path/opc_thermo_response/02_thermo_response.py", "--density-manifest", manifest, "--out", out / "thermo"],
            ["analysis/cooling_path/06_plot_energetic_stabilization.py", "--manifest", manifest, "--energy-map", energy, "--topology", root / "top.prmtop", "--out-prefix", out / "fig3", "--allow-incomplete"],
            ["analysis/cooling_path/opc_thermo_response/03_make_energy_map_template.py", "--density-manifest", manifest, "--out", out / "map.tsv"],
            ["analysis/interface_dynamics/figure1_from_v2.py", "--manifest", manifest, "--out", out / "fig1"],
            ["analysis/interface_dynamics/figure2_from_v3.py", "--manifest", manifest, "--out", out / "fig2"],
            ["analysis/interface_dynamics/analyze_v2.py", "--manifest", manifest, "--row", hold],
            ["analysis/interface_dynamics/run_hb_manifest.py", "--manifest", manifest, "--row", hold],
            ["analysis/glass_transition/03_fs_alpha.py", "--manifest", manifest, "--row", hold, "--curve-out", out / "fs.csv", "--summary-out", out / "tau.csv"],
            ["analysis/glass_structure/opc_vitrification_spatial_workflow/07_core_awi_lda_and_free_volume.py", "--manifest", manifest, "--outdir", out / "classifier", "--allow-missing"],
        ]
        env = dict(**{key: value for key, value in __import__("os").environ.items()})
        env["PYTHONPATH"] = str(ROOT / "src") + ";" + str(ROOT / "analysis")
        for command in commands:
            completed = subprocess.run(
                [sys.executable, str(ROOT / command[0]), *[str(part) for part in command[1:]]],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            text = completed.stdout + completed.stderr
            self.assertIsNone(
                FORBIDDEN.search(text),
                f"{command[0]} failed on a manifest column:\n{text[-2000:]}",
            )

    def test_snakemake_dry_run_does_not_treat_cluster_as_a_target(self):
        if shutil.which("snakemake") is None:
            self.skipTest("snakemake is not installed")
        completed = subprocess.run(
            ["snakemake", "-n", "--profile", "workflow/profiles/slurm"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        text = completed.stdout + completed.stderr
        self.assertNotIn("MissingRuleException", text)
        self.assertNotIn("--cluster", text)


if __name__ == "__main__":
    unittest.main()
