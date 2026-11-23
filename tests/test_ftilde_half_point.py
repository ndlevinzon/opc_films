"""Half-point temperatures in 08_replot_core_awi_black.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def replot():
    path = ROOT / "analysis" / "glass_structure" / "opc_vitrification_spatial_workflow" / "08_replot_core_awi_black.py"
    spec = importlib.util.spec_from_file_location("replot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_half_point_interpolates_from_hot_to_cold(replot):
    assert replot.half_point([300, 250, 200, 150], [0.0, 0.2, 0.6, 1.0]) == pytest.approx(212.5)
    assert np.isnan(replot.half_point([300, 200], [0.0, 0.4]))


def test_ramps_are_not_mixed_into_the_hold_curve(replot):
    rows = []
    for replica in ("r1", "r2", "r3"):
        for kind, rate, temperatures, value in (("hold", "", (250, 200, 150), {250: 0.1, 200: 0.6, 150: 0.9}),
                                                ("ramp", "111", (255, 205, 155), {255: 0.0, 205: 0.0, 155: 0.0})):
            for t in temperatures:
                for region in ("interior", "awi_3_12"):
                    rows.append({"kind": kind, "rate_K_ns": rate, "temperature_K": t, "region": region,
                                 "replica": replica, "f_normalized": value[t], "free_volume_normalized": value[t]})
    per_replica = replot.replica_means(pd.DataFrame(rows), ("hold", "endpoint"))
    assert set(per_replica["kind"]) == {"hold"}
    _, summary = replot.half_points(per_replica)
    bulk = summary[(summary.metric == "f_normalized") & (summary.region == "interior")]
    assert bulk["T_half_mean_K"].iloc[0] == pytest.approx(210.0)
