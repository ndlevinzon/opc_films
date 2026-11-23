"""analysis/cooling_path/04c_capillary_table.py and opcfilm.io.cpptraj."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capillary_fixture import make_capillary_tree  # noqa: E402

from opcfilm.io.cpptraj import helfrich_fit, read_sets  # noqa: E402


@pytest.fixture(scope="module")
def reducer():
    spec = importlib.util.spec_from_file_location("capillary_table", ROOT / "analysis" / "cooling_path" / "04c_capillary_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_sets_names_columns_by_aspect(tmp_path):
    path = tmp_path / "s.dat"
    path.write_text("#q (Ang^-1) ST_00001[S] ST_00001[gammaq]\n0.1 2.0 70\n0.2 0.5 71\n")
    table = read_sets(path)
    assert list(table.columns) == ["x", "S", "gammaq"]
    assert table["gammaq"].tolist() == [70.0, 71.0]


def test_helfrich_fit_recovers_intercept():
    q = np.linspace(0.03, 0.3, 12)
    g0, kp, n = helfrich_fit(q, 55.0 + 300.0 * q ** 2, 0.03, 0.3)
    assert g0 == pytest.approx(55.0)
    assert kp == pytest.approx(300.0)
    assert n == 12


def test_capillary_table_end_to_end(tmp_path, reducer):
    truth = make_capillary_tree(tmp_path)
    out = tmp_path / "results" / "capillary_summary"
    reducer.main(["--manifest", str(tmp_path / "config" / "manifest.tsv"),
                  "--results", str(tmp_path / "results"), "--out", str(out)])
    by_replica = pd.read_csv(out / "capillary_by_replica.csv")
    for _, row in by_replica.iterrows():
        assert row["gamma0_mN_m"] == pytest.approx(truth[row["row_id"]], rel=1e-4)
        assert row["gamma0_log_mN_m"] == pytest.approx(truth[row["row_id"]], abs=1e-3)
    by_t = pd.read_csv(out / "capillary_by_temperature.csv").set_index("temperature_K")
    assert set(by_t.index[by_t["italic"]]) == {160.0, 150.0}
    assert by_t.loc[250.0, "w_CWT_A"] == pytest.approx(by_t.loc[250.0, "w_A_mean"])
    spectrum = pd.read_csv(out / "capillary_spectrum_by_temperature.csv")
    assert {"q_Ainv", "S_mean", "gammaq_mean", "gammaq_sd"} <= set(spectrum.columns)
    rows = (out / "capillary_table_rows.tex").read_text().splitlines()
    assert rows[0].startswith("250 &") and rows[0].endswith("--- \\\\")
    assert rows[-1].startswith("\\textit{150}")


def test_capillary_panels_render(tmp_path, reducer):
    make_capillary_tree(tmp_path)
    results = tmp_path / "results"
    reducer.main(["--manifest", str(tmp_path / "config" / "manifest.tsv"),
                  "--results", str(results), "--out", str(results / "capillary_summary")])
    spec = importlib.util.spec_from_file_location("panels", ROOT / "analysis" / "figures" / "panels.py")
    panels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panels)
    out = tmp_path / "fig"
    for name in ("SI_roughness", "SI_roughness_time", "SI_capillary", "SI_capillary_spectrum"):
        files = panels.PANELS[name](results.resolve(), out, name)
        assert all(path.is_file() for path in files), name
