"""Holds and endpoints under ramp_nvt/<code>K_ns/prop/<T>K are catalogued."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opcfilm.catalog import discover_rows


def _mdout(restart, temperature):
    return (
        "File Assignments:\n"
        f"|  INPCRD: {restart}\n"
        " &cntrl\n  cut = 9.0,\n"
        f"  temp0 = {temperature},\n"
        "  ntwx = 5000,\n  dt = 0.002,\n /\n"
        " NSTEP = 1 TIME(PS) = 0\n EPtot = -1.0\n"
        " NSTEP = 2 TIME(PS) = 10000\n EPtot = -1.0\n"
        "      A V E R A G E S\n"
    )


def _setpoint(root, temperature, n_segments, restart):
    base = root / "opc" / "r1" / "ramp_nvt" / "4K_ns" / "prop" / f"{temperature}K"
    (base / "TRAJ").mkdir(parents=True)
    (base / "OUT").mkdir()
    for index in range(1, n_segments + 1):
        (base / "TRAJ" / f"traj.{index}").write_bytes(b"")
        start = restart if index == 1 else f"RST/rst.{index - 1}"
        (base / "OUT" / f"mdout.{index}").write_text(_mdout(start, temperature))
    (base / "centered_slab.nc").write_bytes(b"")
    return base


class PropLayout(unittest.TestCase):
    def test_prop_directories_become_holds_and_endpoints(self):
        root = Path(tempfile.mkdtemp())
        _setpoint(root, 250, 2, "RST/cool_50ns_chunk_04_260_to_250K.rst7")
        _setpoint(root, 200, 11, "RST/cool_50ns_chunk_09_210_to_200K.rst7")
        _setpoint(root, 78, 1, "RST/cool_50ns_chunk_22_080_to_078K.rst7")
        _setpoint(root, 300, 1, "../../min/step9.rst7")
        rows = discover_rows(root, root, root / "top.prmtop", ("r1",), (4,))
        kinds = {(row["kind"], row["temperature_K"]) for row in rows}
        self.assertEqual(kinds, {("hold", "250"), ("hold", "200"), ("endpoint", "78"), ("endpoint", "300")})
        hold = next(row for row in rows if row["temperature_K"] == "200")
        files = hold["files"].split(";")
        self.assertEqual(len(files), 11)
        self.assertTrue(files[-1].endswith("traj.11"))
        self.assertNotIn("centered", hold["files"])
        self.assertEqual(hold["source_rate_K_per_ns"], "4.44")
        self.assertEqual(hold["cut_A"], "9")

    def test_same_setpoint_in_two_layouts_is_refused(self):
        root = Path(tempfile.mkdtemp())
        _setpoint(root, 250, 1, "RST/cool_50ns_chunk_04_260_to_250K.rst7")
        other = root / "opc" / "r1" / "hold" / "250K"
        (other / "TRAJ").mkdir(parents=True)
        (other / "OUT").mkdir()
        (other / "TRAJ" / "traj.1").write_bytes(b"")
        (other / "OUT" / "mdout.1").write_text(_mdout("RST/cool_50ns_chunk_04_260_to_250K.rst7", 250))
        with self.assertRaises(ValueError):
            discover_rows(root, root, root / "top.prmtop", ("r1",), (4,))


if __name__ == "__main__":
    unittest.main()
