"""A 2,000-water film with a NetCDF time variable.

The film is uniform between two vacuum gaps, so the interior mass density is
the requested density. Two identical frames keep the water count fixed.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from opcfilm.catalog import write_manifest
from opcfilm.config import CONFIG, DT_PS, NUMBER_TO_MASS_DENSITY
from opcfilm.geometry.slab import region_masks, slab_geometry
from opcfilm.stages.reduce import reduce_stage
from opcfilm.synthetic_stages import STAGES


def film_coordinates(n_water=2000, density=0.99, thickness=80, lz=200, seed=0):
    """Place waters uniformly in a slab.

    Parameters
    ----------
    n_water : int, optional
        Number of waters.
    density : float, optional
        Requested mass density in g cm^-3.
    thickness : float, optional
        Slab thickness in angstroms. It is wider than the bulk plus the
        interface bands, so those regions do not overlap.
    lz : float, optional
        Box length along z, in angstroms.
    seed : int, optional
        Sampler seed.

    Returns
    -------
    oxygen : numpy.ndarray, shape (n_water, 3)
        Oxygen coordinates.
    lengths : numpy.ndarray, shape (3,)
        Box lengths.
    """
    number_density = float(density) / NUMBER_TO_MASS_DENSITY
    side = math.sqrt(n_water / (number_density * float(thickness)))
    rng = np.random.default_rng(seed)
    center = float(lz) / 2.0
    oxygen = np.column_stack([
        rng.uniform(0.0, side, n_water),
        rng.uniform(0.0, side, n_water),
        center + rng.uniform(-thickness / 2.0, thickness / 2.0, n_water),
    ])
    return oxygen, np.array([side, side, float(lz)], dtype=float)


def write_trajectory(directory, oxygen, lengths, n_frames=20, diffusion_A2_per_ps=0.01, seed=1):
    """Write a hold-spaced NetCDF and a coordinate sidecar.

    Parameters
    ----------
    directory : path-like
        Destination directory.
    oxygen : numpy.ndarray
        Oxygen coordinates of the first frame.
    lengths : numpy.ndarray
        Box lengths.
    n_frames : int, optional
        Number of frames. The spacing is the hold spacing from the sheet.
    diffusion_A2_per_ps : float, optional
        In-plane diffusion constant used for the random walk, in Å^2/ps.
    seed : int, optional
        Step sampler.

    Returns
    -------
    pathlib.Path
        The NetCDF path. Frame spacing is ``trajectory.hold_frame_ps``.
    """
    from scipy.io import netcdf_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "film.nc"
    frame_ps = float(CONFIG.trajectory.hold_frame_ps)
    step = math.sqrt(2.0 * float(diffusion_A2_per_ps) * frame_ps)
    rng = np.random.default_rng(seed)
    current = np.array(oxygen, dtype=float, copy=True)
    frames = [current.copy()]
    for _ in range(int(n_frames) - 1):
        current = current.copy()
        current[:, 0] += rng.normal(0.0, step, len(current))
        current[:, 1] += rng.normal(0.0, step, len(current))
        frames.append(current)
    stacked = np.stack(frames)
    with netcdf_file(path, "w") as handle:
        handle.createDimension("frame", stacked.shape[0])
        handle.createDimension("water", stacked.shape[1])
        handle.createDimension("xyz", 3)
        time = handle.createVariable("time", "d", ("frame",))
        time[:] = np.arange(stacked.shape[0]) * frame_ps
        coordinates = handle.createVariable("coordinates", "d", ("frame", "water", "xyz"))
        coordinates[:] = stacked
    sidecar = path.with_suffix(".npz")
    np.savez(
        sidecar,
        oxygen=stacked[-1],
        lengths=lengths,
        diffusion_A2_per_ps=float(diffusion_A2_per_ps),
    )
    return path


def write_mdout(path):
    """Write a two-record mdout with a footer that must be ignored.

    Parameters
    ----------
    path : path-like
        Destination.

    Returns
    -------
    pathlib.Path
        The file. ``nstlim / ntpr`` is 2, ``ntwx * dt`` is 2 ps, and ``cut`` is 9.
    """
    path = Path(path)
    path.write_text(
        " &cntrl\n"
        "  nstlim=4, ntpr=2, "
        f"ntwx={CONFIG.trajectory.cooling_ntwx:g}, dt={DT_PS:g}, "
        f"cut={CONFIG.trajectory.production_cut_A:g},\n"
        " /\n"
        " NSTEP = 2   TIME(PS) = 0.004  TEMP(K) = 300.00\n"
        " EPtot = -10.0\n"
        " NSTEP = 4   TIME(PS) = 0.008  TEMP(K) = 300.00\n"
        " EPtot = -11.0\n"
        "      A V E R A G E S\n"
        " NSTEP = 4   TIME(PS) = 999.0  TEMP(K) = 0.0\n"
        " EPtot = 0.0\n",
        encoding="utf-8",
    )
    return path


def write_tau(path):
    """Write an Arrhenius series that crosses the cooling time.

    Parameters
    ----------
    path : path-like
        Destination CSV.

    Returns
    -------
    pathlib.Path
        The CSV. Every temperature has the same ``frame_ps_selected``.
    """
    temperature = np.array([200.0, 220.0, 240.0, 260.0, 280.0])
    slope = 800.0
    tau = np.exp(slope / temperature)
    frame = pd.DataFrame({
        "T_K": temperature,
        "tau": tau,
        "frame_ps_selected": CONFIG.trajectory.cooling_frame_ps,
    })
    path = Path(path)
    frame.to_csv(path, index=False)
    return path


def build(directory):
    """Write the toy film, its mdout, and a one-row manifest.

    Parameters
    ----------
    directory : path-like
        Workspace for the toy.

    Returns
    -------
    dict
        ``manifest`` path, the row, and the trajectory path.
    """
    if abs(CONFIG.trajectory.cooling_ntwx * DT_PS - CONFIG.trajectory.cooling_frame_ps) > 1e-9:
        raise ValueError("The cooling frame spacing is not ntwx times dt.")
    if abs(CONFIG.trajectory.hold_ntwx * DT_PS - CONFIG.trajectory.hold_frame_ps) > 1e-9:
        raise ValueError("The hold frame spacing is not ntwx times dt.")
    directory = Path(directory)
    oxygen, lengths = film_coordinates()
    trajectory = write_trajectory(directory, oxygen, lengths)
    mdout = write_mdout(directory / "film.mdout")
    write_tau(trajectory.with_suffix(".tau.csv"))
    row = {
        "kind": "hold",
        "replica": "r1",
        "rate_K_per_ns": "",
        "source_rate_K_per_ns": "4.44",
        "temperature_K": "300",
        "t_start_K": "300",
        "t_end_K": "300",
        "files": str(trajectory),
        "topology": str(directory / "film.prmtop"),
        "frame_ps": f"{CONFIG.trajectory.hold_frame_ps:g}",
        "ntwx": f"{CONFIG.trajectory.hold_ntwx:g}",
        "cut_A": "9",
        "source_restart": "RST/cool_50ns_chunk_00_310_to_300K.rst7",
        "discard_ps": "0",
        "duration_ps": f"{19 * CONFIG.trajectory.hold_frame_ps:g}",
        "mdout": str(mdout),
    }
    manifest = directory / "manifest.tsv"
    write_manifest(manifest, [row])
    return {"manifest": manifest, "trajectory": trajectory, "directory": directory}


def run(directory=None):
    """Run every stage on the toy film and check the acceptance list.

    Parameters
    ----------
    directory : path-like, optional
        Workspace. A temporary directory is used when this is omitted.

    Returns
    -------
    dict
        The workspace path and the acceptance results.

    Raises
    ------
    AssertionError
        If an acceptance check fails.
    """
    if directory is None:
        directory = Path(tempfile.mkdtemp(prefix="opcfilm-"))
    built = build(directory)
    from opcfilm.catalog import load_manifest

    row = load_manifest(built["manifest"])[0]
    for stage in STAGES:
        STAGES[stage](row, directory)
        reduce_stage(stage, directory)
    checks = acceptance(directory, row)
    print(Path(directory))
    for name, ok in checks.items():
        print(f"{name}\t{ok}")
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"Synthetic acceptance failed: {failed}. {checks}")
    return {"directory": Path(directory), "checks": checks}


def _counts_match(path, lengths):
    """Return whether the saved profile contains every water.

    Parameters
    ----------
    path : path-like
        NetCDF trajectory.
    lengths : array_like
        Box lengths. The histogram covers the z length.

    Returns
    -------
    bool
        True when the first frame's z histogram sums to 2,000.
    """
    from scipy.io import netcdf_file

    with netcdf_file(path, "r") as handle:
        z = np.asarray(handle.variables["coordinates"][0, :, 2], dtype=float)
    counts, _edges = np.histogram(z, bins=80, range=(0.0, float(lengths[2])))
    return int(counts.sum()) == 2000


def _diffusion_matches(path):
    """Return whether F_s of the first lag matches the random-walk D.

    Parameters
    ----------
    path : path-like
        NetCDF trajectory written by :func:`write_trajectory`.

    Returns
    -------
    bool
        True when ``mean(J0(q* |dr|))`` is within 0.08 of ``exp(-D q*^2 t)``.
    """
    from scipy.io import netcdf_file

    from opcfilm.config import QSTAR_AINV
    from opcfilm.dynamics.correlators import self_intermediate

    with netcdf_file(path, "r") as handle:
        frames = np.asarray(handle.variables["coordinates"][:], dtype=float)
    diffusion = float(np.load(Path(path).with_suffix(".npz"))["diffusion_A2_per_ps"])
    frame_ps = float(CONFIG.trajectory.hold_frame_ps)
    measured = self_intermediate(frames[1, :, :2] - frames[0, :, :2], CONFIG.qstar.adopted_Ainv)
    expected = math.exp(-diffusion * QSTAR_AINV ** 2 * frame_ps)
    return abs(measured - expected) < 0.08


def acceptance(directory, row):
    """Pipeline checks that can be evaluated on the toy film.

    Parameters
    ----------
    directory : path-like
        Workspace written by :func:`run`.
    row : dict
        The toy manifest row.

    Returns
    -------
    dict
        Check name to boolean.
    """
    directory = Path(directory)
    rho = pd.read_csv(directory / "results" / "s00" / "reduced" / "summary.csv")
    rho_value = float(rho.loc[rho.metric == "rho_core_g_cm3", "value"].iloc[0])
    records = pd.read_csv(
        next((directory / "results" / "s01").rglob("energy_records.csv"))
    )
    classes = pd.read_csv(directory / "results" / "s06" / "reduced" / "summary.csv")
    order = pd.read_csv(directory / "results" / "s03" / "reduced" / "summary.csv")
    oxygen, lengths = film_coordinates()
    relative, depth, _bounds, _center = slab_geometry(oxygen, lengths)
    masks = region_masks(relative, depth)
    spacing = float(row["frame_ps"])
    return {
        "rho_core": abs(rho_value - 0.99) < 0.05,
        "mass_conservation": len(np.load(Path(row["files"]).with_suffix(".npz"))["oxygen"]) == 2000,
        "classes_sum": abs(float(classes["value"].sum()) - 1.0) < 1e-8,
        "no_overlap": not np.any(masks["interior"] & (masks["awi"] | masks["awi_outer"])),
        "frame_spacing": abs(spacing - float(row["ntwx"]) * DT_PS) < 1e-8,
        "one_qstar": float(order.loc[order.metric == "qstar_Ainv", "value"].iloc[0]) == float(
            order.loc[order.metric == "qstar_Ainv", "value"].iloc[-1]
        ),
        "mdout_count": len(records) == 4 // 2,
        "classifier_accuracy": float(order.loc[order.metric == "classifier_self_accuracy", "value"].iloc[0]) > 0.9,
        "tetrahedron": abs(float(order.loc[order.metric == "tetrahedral_q", "value"].iloc[0]) - 1.0) < 1e-8,
        "density_integrates": _counts_match(Path(row["files"]), lengths),
        "self_intermediate": _diffusion_matches(Path(row["files"])),
        "metadata": (directory / "results" / "s00" / row["row_id"] / "run_metadata.json").is_file(),
    }
