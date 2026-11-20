"""The shared q* file is written once, from the declared rows."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from opcfilm.catalog import load_manifest, write_manifest
from opcfilm.qstar import adopt


class QStarAdoptTest(unittest.TestCase):
    def test_mean_of_the_declared_holds_only(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                manifest = Path("manifest.tsv")
                write_manifest(manifest, [
                    {
                        "kind": "hold",
                        "replica": "r1",
                        "rate_K_per_ns": "",
                        "source_rate_K_per_ns": "4.44",
                        "temperature_K": "200",
                        "files": "a.nc",
                    },
                    {
                        "kind": "hold",
                        "replica": "r1",
                        "rate_K_per_ns": "",
                        "source_rate_K_per_ns": "4.44",
                        "temperature_K": "78",
                        "files": "b.nc",
                    },
                ])
                for row, value in zip(load_manifest(manifest), (1.91, 3.00)):
                    out = Path(row["outdir"])
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "qstar_row.json").write_text(
                        json.dumps({"qstar_measured_Ainv": value}), encoding="utf-8"
                    )
                destination = Path(directory) / "shared" / "qstar.json"
                payload = adopt(manifest, destination)
            finally:
                os.chdir(previous)
            self.assertEqual(payload["n_rows"], 1)
            self.assertAlmostEqual(payload["qstar_measured_mean_Ainv"], 1.91)
            self.assertAlmostEqual(payload["qstar_adopted_Ainv"], 1.925)
            self.assertTrue(destination.is_file())


if __name__ == "__main__":
    unittest.main()
