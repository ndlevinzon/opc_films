"""Cooling windows too short for the lag set are recorded, not failed."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", [
    "analysis/interface_dynamics/analyze_v2.py",
    "analysis/interface_dynamics/hbond_v3.py",
])
def test_short_ramp_is_recorded(tmp_path, script):
    module = load(script)
    out = tmp_path / "row"
    module.skip_short_window(Namespace(hold=False, out=str(out), traj=[tmp_path / "t.nc"]),
                             "Trajectory too short", nframes=12)
    record = json.loads((out / "skipped.json").read_text())
    assert record["reason"] == "Trajectory too short"
    assert record["nframes"] == 12


@pytest.mark.parametrize("script", [
    "analysis/interface_dynamics/analyze_v2.py",
    "analysis/interface_dynamics/hbond_v3.py",
])
def test_short_hold_is_still_an_error(tmp_path, script):
    module = load(script)
    with pytest.raises((RuntimeError, ValueError)):
        module.skip_short_window(Namespace(hold=True, out=str(tmp_path), traj=[]), "Trajectory too short")
    assert not (tmp_path / "skipped.json").exists()
