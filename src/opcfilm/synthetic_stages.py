"""Toy stage calculations for the 2,000-water film.

These functions are the synthetic acceptance checks. They are not the
production analysis. ``opcfilm s00`` does not call them. The production
scripts are named by :mod:`opcfilm.stages`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from opcfilm.catalog import result_directory
from opcfilm.config import CONFIG, QSTAR_AINV, SEEDS
from opcfilm.metadata import write_run_metadata

TABLE_COLUMNS = [
    "kind",
    "replica",
    "rate_K_per_ns",
    "temperature_K",
    "region",
    "band_A",
    "lag_ps",
    "metric",
    "value",
    "status",
    "bound_ps",
]


def _blank(row):
    return {
        "kind": row.get("kind", ""),
        "replica": row.get("replica", ""),
        "rate_K_per_ns": row.get("rate_K_per_ns", ""),
        "temperature_K": row.get("temperature_K", ""),
        "region": "",
        "band_A": "",
        "lag_ps": "",
        "metric": "",
        "value": "",
        "status": "",
        "bound_ps": "",
    }


def write_table(directory, records):
    """Write the shared column schema.

    Parameters
    ----------
    directory : path-like
        Stage directory for one manifest row.
    records : sequence of dict
        Rows. Missing columns are left blank.

    Returns
    -------
    pathlib.Path
        The CSV that was written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{**_blank({}), **record} for record in records])
    path = directory / "metrics.csv"
    frame.to_csv(path, index=False, columns=TABLE_COLUMNS)
    return path


def _coordinates(row):
    path = Path(str(row.get("files") or row.get("trajectory")).split(";")[0])
    sidecar = path.with_suffix(".npz")
    if sidecar.is_file():
        return np.load(sidecar)["oxygen"]
    from scipy.io import netcdf_file

    with netcdf_file(path, "r") as handle:
        return np.asarray(handle.variables["coordinates"][-1], dtype=float)


def run_s00(row, root):
    """Center the film and record the interior density.

    Parameters
    ----------
    row : dict
        Manifest row. ``files`` is a trajectory whose last frame is stored
        as ``coordinates`` or as a sibling ``.npz``.
    root : path-like
        Directory that contains ``results/``.

    Returns
    -------
    pathlib.Path
        The row directory.
    """
    from opcfilm.geometry.density import mass_density
    from opcfilm.geometry.slab import slab_geometry

    started = time.time()
    oxygen = _coordinates(row)
    lengths = np.load(Path(str(row["files"]).split(";")[0]).with_suffix(".npz"))["lengths"]
    relative, _depth, _bounds, _center = slab_geometry(oxygen, lengths)
    half = CONFIG.regions.bulk_halfwidth_A
    interior = np.abs(relative) <= half
    volume = lengths[0] * lengths[1] * (2.0 * half)
    rho = float(mass_density(np.count_nonzero(interior) / volume))
    out = result_directory(root, "s00", row)
    write_table(out, [{
        **_blank(row),
        "region": "interior",
        "band_A": f"{-half:g}:{half:g}",
        "metric": "rho_core_g_cm3",
        "value": rho,
        "status": "resolved",
    }])
    write_run_metadata(out, row, {"classifier": SEEDS["classifier"]}, started, root)
    return out


def run_s01(row, root):
    """Parse one mdout into the per-record energy table.

    Parameters
    ----------
    row : dict
        Manifest row with ``mdout``.
    root : path-like
        Directory that contains ``results/``.

    Returns
    -------
    pathlib.Path
        The row directory.
    """
    from opcfilm.thermo.energy import records

    started = time.time()
    frame = records(row["mdout"])
    out = result_directory(root, "s01", row)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "energy_records.csv", index=False)
    write_table(out, [{
        **_blank(row),
        "metric": "n_energy_records",
        "value": len(frame),
        "status": "resolved",
    }])
    write_run_metadata(out, row, {}, started, root)
    return out


def run_s02(row, root):
    """Fit an Arrhenius line on the tau table stored beside the trajectory.

    Parameters
    ----------
    row : dict
        Manifest row. A sibling ``tau.csv`` supplies ``T_K``, ``tau``, and
        ``frame_ps_selected``.
    root : path-like
        Directory that contains ``results/``.

    Returns
    -------
    pathlib.Path
        The row directory.
    """
    from opcfilm.thermo.arrest import fit_arrhenius, solve_tg_arrhenius, spacing_by_temperature

    started = time.time()
    path = Path(str(row["files"]).split(";")[0]).with_suffix(".tau.csv")
    frame = pd.read_csv(path)
    spacings = spacing_by_temperature(frame)
    fit = fit_arrhenius(frame["T_K"], frame["tau"])
    rate_text = str(row.get("rate_K_per_ns", "")).strip() or str(row.get("source_rate_K_per_ns", "")).strip()
    rate = float(rate_text)
    tg = solve_tg_arrhenius(fit, rate)
    out = result_directory(root, "s02", row)
    write_table(out, [{
        **_blank(row),
        "metric": "Tg_dyn_K",
        "value": tg,
        "status": "resolved",
    }, {
        **_blank(row),
        "metric": "frame_ps_selected",
        "value": spacings[float(frame["T_K"].iloc[0])],
        "status": "resolved",
    }])
    write_run_metadata(out, row, {}, started, root)
    return out


def run_s03(row, root):
    """Score a perfect tetrahedron and a separated two-class sample.

    Parameters
    ----------
    row : dict
        Manifest row. The tetrahedron does not depend on the trajectory.
    root : path-like
        Directory that contains ``results/``.

    Returns
    -------
    pathlib.Path
        The row directory.
    """
    from opcfilm.structure.classifier import fit_discriminant, score_discriminant
    from opcfilm.structure.order import orientational_tetrahedral_q

    started = time.time()
    tetrahedron = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    order = float(orientational_tetrahedral_q(tetrahedron))
    rng = np.random.default_rng(SEEDS["classifier"])
    liquid = rng.normal(0.0, 0.05, size=(40, 3))
    glass = rng.normal(3.0, 0.05, size=(40, 3))
    model = fit_discriminant(liquid, glass)
    scores = score_discriminant(np.vstack([liquid, glass]), model)
    accuracy = float(np.mean((scores >= CONFIG.classifier.threshold) == np.repeat([False, True], 40)))
    out = result_directory(root, "s03", row)
    write_table(out, [
        {**_blank(row), "metric": "tetrahedral_q", "value": order, "status": "resolved"},
        {**_blank(row), "metric": "classifier_self_accuracy", "value": accuracy, "status": "resolved"},
        {**_blank(row), "metric": "qstar_Ainv", "value": QSTAR_AINV, "status": "resolved"},
    ])
    write_run_metadata(out, row, {"classifier": SEEDS["classifier"]}, started, root)
    return out


def run_s04(row, root):
    """Fourier transform a one-mode height field.

    Parameters
    ----------
    row : dict
        Manifest row.
    root : path-like
        Directory that contains ``results/``.

    Returns
    -------
    pathlib.Path
        The row directory.
    """
    from opcfilm.geometry.capillary import block_count, height_spectrum

    started = time.time()
    x = np.arange(32)
    height = np.sin(2.0 * np.pi * x / 32.0)[None, :] * np.ones((32, 1))
    qx, _qy, power = height_spectrum(height, dx=1.0, dy=1.0)
    peak = float(qx[np.argmax(power[0])])
    out = result_directory(root, "s04", row)
    write_table(out, [
        {**_blank(row), "metric": "capillary_blocks", "value": block_count(), "status": "resolved"},
        {**_blank(row), "metric": "height_qx_peak", "value": peak, "status": "resolved"},
    ])
    write_run_metadata(out, row, {}, started, root)
    return out


def run_s05(row, root):
    """Cross ``1/e`` on an exponential and report the censoring status.

    Parameters
    ----------
    row : dict
        Manifest row.
    root : path-like
        Directory that contains ``results/``.

    Returns
    -------
    pathlib.Path
        The row directory.
    """
    from opcfilm.dynamics.correlators import self_intermediate
    from opcfilm.dynamics.relaxation import classify_tau

    started = time.time()
    lags = np.array([1.0, 2.0, 4.0, 8.0])
    status, tau, bound = classify_tau(lags, np.exp(-lags / 3.0))
    fs = self_intermediate(np.zeros((4, 2)), QSTAR_AINV)
    out = result_directory(root, "s05", row)
    write_table(out, [
        {**_blank(row), "region": "interior", "lag_ps": tau, "metric": "tau_ps", "value": tau, "status": status, "bound_ps": bound},
        {**_blank(row), "region": "interior", "metric": "Fs_zero_displacement", "value": fs, "status": "resolved"},
    ])
    write_run_metadata(out, row, {"regional_dynamics": SEEDS["regional_dynamics"]}, started, root)
    return out


def run_s06(row, root):
    """Normalize a donor–acceptor histogram.

    Parameters
    ----------
    row : dict
        Manifest row.
    root : path-like
        Directory that contains ``results/``.

    Returns
    -------
    pathlib.Path
        The row directory.
    """
    from opcfilm.dynamics.hbonds import class_fractions

    started = time.time()
    fractions = class_fractions({"D2A2": 50, "D2A1": 30, "D1A2": 20})
    out = result_directory(root, "s06", row)
    write_table(out, [
        {**_blank(row), "region": "awi_outer", "band_A": "0:6", "metric": name, "value": value, "status": "resolved"}
        for name, value in fractions.items()
    ])
    write_run_metadata(out, row, {}, started, root)
    return out


def run_s07(row, root):
    """Insertion fraction in an empty box, which is entirely free.

    Parameters
    ----------
    row : dict
        Manifest row.
    root : path-like
        Directory that contains ``results/``.

    Returns
    -------
    pathlib.Path
        The row directory.
    """
    from opcfilm.structure.free_volume import exclusion_distance, probe_fraction

    started = time.time()
    rng = np.random.default_rng(SEEDS["classifier"])
    oxygen = np.zeros((1, 3))
    fraction = probe_fraction(
        oxygen, (20.0, 20.0, 20.0), 10.0, ((-1.0, 1.0),), 200,
        exclusion_distance(0.0), rng,
    )
    out = result_directory(root, "s07", row)
    write_table(out, [{
        **_blank(row),
        "region": "interior",
        "metric": "free_volume_empty_box",
        "value": fraction,
        "status": "resolved",
    }])
    write_run_metadata(out, row, {"classifier": SEEDS["classifier"]}, started, root)
    return out


STAGES = {
    "s00": run_s00,
    "s01": run_s01,
    "s02": run_s02,
    "s03": run_s03,
    "s04": run_s04,
    "s05": run_s05,
    "s06": run_s06,
    "s07": run_s07,
}


