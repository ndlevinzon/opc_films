"""Footer records must not enter an mdout energy mean."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from common.mdout import parse_file, summarize_mdout, time_span_ps


FOOTER = """
 NSTEP = 100   TIME(PS) = 2.000  TEMP(K) = 300.00
 Etot = -1248000.0  EKtot = 100.0  EPtot = -1248000.0
 NSTEP = 200   TIME(PS) = 4.000  TEMP(K) = 290.00
 Etot = -1248000.0  EKtot = 90.0  EPtot = -1248000.0
      A V E R A G E S
 NSTEP = 200   TIME(PS) = 4.000  TEMP(K) = 295.00
 Etot = -1248000.0  EKtot = 95.0  EPtot = -1248000.0
      R M S  F L U C T U A T I O N S
 NSTEP = 200   TIME(PS) = 4.000  TEMP(K) = 0.16
 Etot = 12.0  EKtot = 1.0  EPtot = 300.0
 FINAL RESULTS
 NSTEP = 999
 EPtot = 1.0
"""


class MdoutFooter(unittest.TestCase):
    def test_footer_is_not_a_time_step(self):
        path = Path(tempfile.mkdtemp()) / "run.mdout"
        path.write_text(FOOTER)
        rows = parse_file(path)
        energies = [row["EPtot"] for row in rows]
        self.assertEqual(energies, [-1248000.0, -1248000.0])
        self.assertAlmostEqual(sum(energies) / len(energies), -1248000.0)

    def test_duplicate_step_keeps_the_last_copy(self):
        text = """
 NSTEP = 100   TIME(PS) = 2.000  TEMP(K) = 300.00
 EPtot = -1.0
 NSTEP = 100   TIME(PS) = 2.000  TEMP(K) = 301.00
 EPtot = -2.0
"""
        path = Path(tempfile.mkdtemp()) / "dup.mdout"
        path.write_text(text)
        rows = parse_file(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["EPtot"], -2.0)
        self.assertEqual(rows[0]["temp_K"], 301.0)

    def test_span_reads_the_last_step_before_the_banner(self):
        path = Path(tempfile.mkdtemp()) / "long.mdout"
        body = [
            "File Assignments:\n",
            "|  INPCRD: RST/cool_50ns_chunk_04_260_to_250K.rst7\n",
            " &cntrl\n",
            "  cut = 8.0,\n",
            "  cut = 9.0,\n",
            "  ntwx = 5000,\n",
            "  dt = 0.002,\n",
            " /\n",
            " NSTEP = 1 TIME(PS) = 0.0\n",
            " EPtot = -1.0\n",
        ]
        body.append("NSTEP = 2 TIME(PS) = 1.0\n" * 15000)
        body.append(" NSTEP = 3 TIME(PS) = 12345.0\n")
        body.append("      A V E R A G E S\n")
        body.append(" NSTEP = 3 TIME(PS) = 99999.0\n")
        path.write_text("".join(body))
        self.assertGreater(path.stat().st_size, 262144)
        summary = summarize_mdout(path)
        self.assertEqual(summary["cut"], 9.0)
        self.assertEqual(summary["ntwx"], "5000")
        self.assertEqual(summary["frame_ps"], "10")
        self.assertEqual(summary["inpcrd"], "RST/cool_50ns_chunk_04_260_to_250K.rst7")
        self.assertEqual(summary["duration_ps"], 12345.0)
        self.assertEqual(time_span_ps(path), 12345.0)


if __name__ == "__main__":
    unittest.main()
