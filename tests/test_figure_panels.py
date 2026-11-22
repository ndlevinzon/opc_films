"""Helpers of analysis/figures/panels.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def panels():
    spec = importlib.util.spec_from_file_location("panels", ROOT / "analysis" / "figures" / "panels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpptraj_table_uses_last_header(tmp_path, panels):
    path = tmp_path / "x.dat"
    path.write_text("#Frame  w2\n#q  h2  fit\n0.1 2.0 1.9\n0.2 0.5 0.6\n")
    table = panels.read_cpptraj_table(path)
    assert list(table.columns) == ["q", "h2", "fit"]
    assert table.shape == (2, 3)


def test_crossing_time_is_log_interpolated(panels):
    lags = np.array([10.0, 100.0])
    values = np.array([1.0, np.exp(-2.0)])
    level = np.exp(-1.0)
    expected = np.exp(np.log(10) + (level - 1.0) * np.log(10) / (np.exp(-2.0) - 1.0))
    assert panels._crossing_time(lags, values) == pytest.approx(expected)
    assert np.isnan(panels._crossing_time(lags, np.array([0.9, 0.8])))


def test_every_manuscript_panel_is_registered(panels):
    names = set(panels.PANELS)
    for stem in ("Fig1b", "Fig4a", "Fig4f", "Fig5a", "Fig5c", "Fig5e", "Fig6c", "Fig7a", "Fig7b", "Fig7e"):
        assert any(name.startswith(stem) for name in names), stem
