"""Replica summaries keep the mean, the band, and the paired contrast."""

import unittest

import pandas as pd

from opcfilm.stages.reduce import contrast_table, summarize_replicas


def _frame(replicas):
    rows = []
    for replica, bulk, awi in replicas:
        for region, value in (("interior", bulk), ("awi", awi)):
            rows.append({
                "kind": "hold",
                "replica": replica,
                "rate_K_per_ns": "",
                "source_rate_K_per_ns": "4.44",
                "temperature_K": "250",
                "region": region,
                "metric": "tau_ps",
                "value": value,
            })
    return pd.DataFrame(rows)


class ReduceSummaryTest(unittest.TestCase):
    def test_three_replicas_have_a_band_and_a_contrast(self):
        frame = _frame((("r1", 10.0, 4.0), ("r2", 12.0, 5.0), ("r3", 14.0, 6.0)))
        summary = summarize_replicas(frame)
        interior = summary[summary.region == "interior"].iloc[0]
        self.assertEqual(int(interior.n_replicas), 3)
        self.assertAlmostEqual(interior["mean"], 12.0)
        self.assertLess(interior.band_lo, interior["mean"])
        contrast = contrast_table(frame)
        self.assertEqual(len(contrast), 1)
        self.assertAlmostEqual(contrast.iloc[0]["difference"], 7.0)

    def test_fewer_than_three_replicas_is_refused(self):
        frame = _frame((("r1", 10.0, 4.0), ("r2", 12.0, 5.0)))
        with self.assertRaises(ValueError):
            summarize_replicas(frame)
        summary = summarize_replicas(frame, allow_incomplete=True)
        self.assertEqual(int(summary.iloc[0].n_replicas), 2)


if __name__ == "__main__":
    unittest.main()
