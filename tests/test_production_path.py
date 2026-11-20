"""Production commands do not emit the synthetic toys."""

import unittest

import pandas as pd

from opcfilm.plotting.figures import render
from opcfilm.stages import run


class ProductionPathTest(unittest.TestCase):
    def test_stage_names_the_real_script(self):
        with self.assertRaises(NotImplementedError) as caught:
            run("s06", {"kind": "hold", "replica": "r1", "temperature_K": "250"}, ".")
        self.assertIn("hbond_v3.py", str(caught.exception))

    def test_a_missing_metric_is_not_replaced(self):
        with self.assertRaises(ValueError):
            render.save(pd.DataFrame(columns=["metric", "temperature_K", "value"]), "rho_core_g_cm3", "unused.pdf", "title")


if __name__ == "__main__":
    unittest.main()
