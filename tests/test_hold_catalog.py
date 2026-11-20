"""A hold of several continuations is one manifest row."""

import argparse
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opcfilm.catalog import discover_rows


def _mdout(cut, start, end, restart):
    return (
        "File Assignments:\n"
        f"|  INPCRD: {restart}\n"
        " &cntrl\n"
        f"  cut = {cut},\n"
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


def _tree(root, second_cut="9.0"):
    hold = root / "opc" / "r1" / "hold" / "250K"
    (hold / "TRAJ").mkdir(parents=True)
    (hold / "OUT").mkdir()
    (hold / "TRAJ" / "traj.1").write_bytes(b"")
    (hold / "TRAJ" / "traj.2").write_bytes(b"")
    (hold / "OUT" / "mdout.1").write_text(
        _mdout("9.0", 0, 10000, "RST/cool_50ns_chunk_04_260_to_250K.rst7")
    )
    (hold / "OUT" / "mdout.2").write_text(_mdout(second_cut, 0, 5000, "RST/rst.1"))
    ramp = root / "opc" / "r1" / "ramp_nvt" / "4K_ns"
    (ramp / "TRAJ").mkdir(parents=True)
    (ramp / "OUT").mkdir()
    name = "cool_50ns_chunk_00_300_to_290K"
    (ramp / "TRAJ" / f"{name}.nc").write_bytes(b"")
    (ramp / "OUT" / f"{name}.mdout").write_text(_mdout("9.0", 0, 2000, "min/step9.rst7"))


class HoldSegments(unittest.TestCase):
    def test_continuations_are_one_row_with_the_numbered_mdout(self):
        root = Path(tempfile.mkdtemp()) / "139K"
        _tree(root)
        rows = discover_rows(root, root, root / "top.prmtop", ("r1",), (4,))
        holds = [row for row in rows if row["kind"] == "hold"]
        ramps = [row for row in rows if row["kind"] == "ramp"]
        self.assertEqual(len(holds), 1)
        self.assertEqual(len(ramps), 1)
        hold = holds[0]
        files = hold["files"].split(";")
        self.assertEqual([Path(path).name for path in files], ["traj.1", "traj.2"])
        self.assertEqual(
            [Path(path).name for path in hold["mdout"].split(";")],
            ["mdout.1", "mdout.2"],
        )
        self.assertEqual(hold["cut_A"], "9")
        self.assertEqual(hold["temperature_K"], "250")
        self.assertEqual(hold["duration_ps"], "15000")
        self.assertEqual(hold["discard_ps"], "1500")
        self.assertEqual(hold["frame_ps"], "10")
        self.assertEqual(hold["ntwx"], "5000")
        self.assertEqual(hold["source_restart"], "RST/cool_50ns_chunk_04_260_to_250K.rst7")
        self.assertEqual(hold["rate_K_per_ns"], "")
        self.assertEqual(hold["source_rate_K_per_ns"], "4.44")
        self.assertEqual(hold["row_id"], "hold-r1-4.44-250")
        self.assertEqual(ramps[0]["source_restart"], "min/step9.rst7")
        self.assertEqual(ramps[0]["row_id"], "ramp-r1-4.44-295")
        self.assertEqual(ramps[0]["cut_A"], "9")
        self.assertNotEqual(ramps[0]["mdout"], "")

    def test_segments_that_disagree_on_cut_are_refused(self):
        root = Path(tempfile.mkdtemp())
        _tree(root, second_cut="8.0")
        with self.assertRaises(ValueError):
            discover_rows(root, root, root / "top.prmtop", ("r1",), (4,))

    def test_a_segment_without_mdout_is_refused(self):
        root = Path(tempfile.mkdtemp())
        _tree(root)
        (root / "opc" / "r1" / "hold" / "250K" / "OUT" / "mdout.2").unlink()
        with self.assertRaises(FileNotFoundError):
            discover_rows(root, root, root / "top.prmtop", ("r1",), (4,))

    def test_hold_json_supplies_the_rate_when_the_restart_name_does_not(self):
        root = Path(tempfile.mkdtemp())
        _tree(root)
        hold = root / "opc" / "r1" / "hold" / "250K"
        (hold / "OUT" / "mdout.1").write_text(_mdout("9.0", 0, 10000, "RST/plain.rst7"))
        (hold / "hold_250K.json").write_text(json.dumps({
            "source_restart": "RST/plain.rst7",
            "source_rate_K_per_ns": 4.44,
            "length_ns": 20,
            "cut_A": 9.0,
        }))
        rows = discover_rows(root, root, root / "top.prmtop", ("r1",), (4,))
        hold_row = next(row for row in rows if row["kind"] == "hold")
        self.assertEqual(hold_row["source_restart"], "RST/plain.rst7")
        self.assertEqual(hold_row["source_rate_K_per_ns"], "4.44")

    def test_catalog_build_prints_the_cutoff_note(self):
        from opcfilm.cli import _build

        root = Path(tempfile.mkdtemp())
        _tree(root)
        args = argparse.Namespace(
            root=str(root),
            topology=str(root / "top.prmtop"),
            workflow=str(root),
            out=str(root / "manifest.tsv"),
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = _build(args)
        self.assertEqual(status, 0)
        self.assertIn("cut = [9.0]", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
