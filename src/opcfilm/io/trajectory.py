"""Frame timing and the OPC-film cross-check.

Lag analyses read the NetCDF ``time`` variable. A spacing passed on the
command line is accepted only when it agrees with that variable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from opcfilm.config import CONFIG


def netcdf_time_ps(path):
    """Read the trajectory time coordinate.

    Parameters
    ----------
    path : path-like
        Amber NetCDF trajectory.

    Returns
    -------
    numpy.ndarray
        Time of every stored frame, in picoseconds.

    Raises
    ------
    RuntimeError
        If the file has no ``time`` variable.
    """
    from netCDF4 import Dataset

    with Dataset(Path(path)) as dataset:
        if "time" not in dataset.variables:
            raise RuntimeError(f"{path} has no time variable.")
        return np.asarray(dataset.variables["time"][:], dtype=float)


def ntwx_from_frame_ps(frame_ps, dt_ps=None):
    """Amber coordinate-output interval that produces one frame spacing.

    Parameters
    ----------
    frame_ps : float
        Spacing of stored frames, in picoseconds.
    dt_ps : float, optional
        Integration step in picoseconds. Production runs use 2 fs.

    Returns
    -------
    int
        ``ntwx``.
    """
    if dt_ps is None:
        dt_ps = CONFIG.film.dt_ps
    ntwx = int(round(float(frame_ps) / float(dt_ps)))
    if ntwx < 1 or abs(ntwx * dt_ps - float(frame_ps)) > 0.05 * float(frame_ps):
        raise ValueError(f"Frame spacing {frame_ps:g} ps is not an integer number of {dt_ps:g} ps steps.")
    return ntwx


def cooling_window_lag_cap_ps(duration_ps, max_temperature_drop_K=None, window_temperature_drop_K=10):
    """Longest lag that stays inside a cooling-window temperature change.

    Parameters
    ----------
    duration_ps : float
        Length of the window, in picoseconds.
    max_temperature_drop_K : float, optional
        Temperature change allowed along one lag.
    window_temperature_drop_K : float, optional
        Temperature change across the whole window.

    Returns
    -------
    float
        Lag cap in picoseconds.
    """
    if window_temperature_drop_K <= 0:
        raise ValueError("The window temperature drop must be positive.")
    if max_temperature_drop_K is None:
        max_temperature_drop_K = CONFIG.dynamics.cooling_lag_cap_K
    return float(duration_ps) * float(max_temperature_drop_K) / float(window_temperature_drop_K)


def _step_tolerance(times, spacing, rtol, atol):
    """Tolerance on a frame step, in picoseconds.

    AMBER NetCDF stores time in single precision. Above 262144 ps one unit
    in the last place is 0.03125 ps, so a step between two stored times can
    differ from the spacing by that much. The tolerance is never smaller
    than two such units at the largest stored time.
    """
    largest = float(np.max(np.abs(times)))
    float32_step = float(np.spacing(np.float32(largest)))
    return max(float(atol), float(rtol) * float(spacing), 2.0 * float32_step)


def frame_spacing_ps(times, override=None, rtol=1e-3, atol=1e-3):
    """Return the uniform spacing of a time series.

    Parameters
    ----------
    times : array_like
        Frame times in picoseconds, in the order they will be analyzed.
    override : float, optional
        Spacing claimed by the caller. It is used only when it matches the
        file within the relative and absolute tolerances.
    rtol : float, optional
        Relative tolerance on the spacing.
    atol : float, optional
        Absolute tolerance in picoseconds.

    Returns
    -------
    float
        Spacing in picoseconds.

    Raises
    ------
    ValueError
        If fewer than two times are present, the spacing is not uniform, or
        ``override`` disagrees with the file.
    """
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        raise ValueError("At least two frames are required to read the spacing.")
    step = np.diff(times)
    if not np.all(np.isfinite(step)) or np.any(step <= 0.0):
        raise ValueError("Trajectory time is not strictly increasing.")
    spacing = float(np.median(step))
    tolerance = _step_tolerance(times, spacing, rtol, atol)
    if float(np.max(np.abs(step - spacing))) > tolerance:
        raise ValueError(
            f"Frame spacing is not uniform: median {spacing:g} ps, "
            f"range {float(step.min()):g} to {float(step.max()):g} ps."
        )
    if override is not None and abs(float(override) - spacing) > tolerance:
        raise ValueError(
            f"Requested frame spacing {override:g} ps disagrees with the "
            f"trajectory time spacing {spacing:g} ps."
        )
    return spacing


# A hold or endpoint is joined from restarted segments. Each restart begins
# from a restart file written at an arbitrary step, so the first frame of a
# segment is 1 to 10 ps later than the 10 ps grid of the previous segment.
# The dynamics are continuous; only the sampling grid shifts at the join.
# A lag pair that crosses a join is kept when its true time separation is
# within these limits of the nominal lag, and dropped otherwise.
JOIN_LAG_ATOL_PS = 0.5
JOIN_LAG_RTOL = 0.01


def continuation_frame_spacing_ps(times, override=None, rtol=1e-3, atol=1e-3):
    """Spacing of a trajectory joined from restarted segments.

    Steps within a segment must equal the spacing. A step longer than the
    spacing and shorter than twice the spacing is accepted as a segment
    join. Any other step is an error.

    Parameters
    ----------
    times : array_like
        Frame times in picoseconds, in the order they will be analyzed.
    override : float, optional
        Spacing claimed by the caller. It must match the file.
    rtol, atol : float, optional
        Tolerance on a regular step, as in :func:`frame_spacing_ps`.

    Returns
    -------
    spacing : float
        Spacing within segments, in picoseconds.
    joins : int
        Number of steps accepted as segment joins.

    Raises
    ------
    ValueError
        If fewer than two times are present, time does not increase, or a
        step is neither regular nor a segment join.
    """
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        raise ValueError("At least two frames are required to read the spacing.")
    step = np.diff(times)
    if not np.all(np.isfinite(step)) or np.any(step <= 0.0):
        raise ValueError("Trajectory time is not strictly increasing.")
    spacing = float(np.median(step))
    tolerance = _step_tolerance(times, spacing, rtol, atol)
    regular = np.abs(step - spacing) <= tolerance
    join = ~regular & (step > spacing) & (step < 2.0 * spacing + tolerance)
    bad = ~regular & ~join
    if np.any(bad):
        first = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"Frame spacing is not uniform: median {spacing:g} ps, step "
            f"{step[first]:g} ps after frame {first} is not a segment join."
        )
    if override is not None and abs(float(override) - spacing) > tolerance:
        raise ValueError(
            f"Requested frame spacing {override:g} ps disagrees with the "
            f"trajectory time spacing {spacing:g} ps."
        )
    return spacing, int(np.count_nonzero(join))


def lag_pair_ok(times, start, lag, spacing, rtol=JOIN_LAG_RTOL, atol=JOIN_LAG_ATOL_PS):
    """Whether frame pairs separated by ``lag`` frames have the nominal lag.

    Parameters
    ----------
    times : array_like
        Frame times in picoseconds.
    start : int or array_like of int
        First frame of each pair.
    lag : int
        Separation in frames.
    spacing : float
        Spacing within segments, in picoseconds.
    rtol, atol : float, optional
        Allowed error on the time separation, relative to the nominal lag
        and in picoseconds. The larger of the two applies.

    Returns
    -------
    bool or numpy.ndarray of bool
        True where ``times[start + lag] - times[start]`` is within the
        tolerance of ``lag * spacing``.
    """
    times = np.asarray(times, dtype=float)
    start = np.asarray(start, dtype=int)
    nominal = float(lag) * float(spacing)
    actual = times[start + int(lag)] - times[start]
    return np.abs(actual - nominal) <= max(float(atol), float(rtol) * nominal)


def count_waters(topology):
    """Count water residues in an Amber topology.

    Parameters
    ----------
    topology : path-like
        Amber prmtop.

    Returns
    -------
    int
        Number of residues named WAT, HOH, or OPC.
    """
    import MDAnalysis as mda

    universe = mda.Universe(str(topology))
    names = {residue.resname.upper() for residue in universe.residues}
    count = sum(
        1
        for residue in universe.residues
        if residue.resname.upper() in ("WAT", "HOH", "OPC")
    )
    if count == 0:
        raise ValueError(f"No water residues in {topology}; residue names: {sorted(names)}.")
    return int(count)


def require_film(n_water, box, tolerance_a=0.2):
    """Fail if a trajectory is not the OPC film used for this manuscript.

    Parameters
    ----------
    n_water : int
        Water count read from the topology or the atom selection.
    box : array_like, shape (3,)
        Orthorhombic lengths of one frame, in angstroms.
    tolerance_a : float, optional
        Accepted absolute difference from the Packmol box, in angstroms.

    Returns
    -------
    numpy.ndarray, shape (3,)
        The frame box that passed the check.

    Raises
    ------
    ValueError
        If the water count or any box length disagrees with the film.
    """
    box = np.asarray(box, dtype=float)[:3]
    if int(n_water) != int(CONFIG.film.n_water):
        raise ValueError(f"Found {n_water} waters; the OPC film has {CONFIG.film.n_water}.")
    expected = np.asarray(CONFIG.film.box_A, dtype=float)
    if np.max(np.abs(box - expected)) > tolerance_a:
        raise ValueError(
            f"Frame box {box.tolist()} A disagrees with the OPC film "
            f"{expected.tolist()} A."
        )
    return box


def hold_lag_times_ps(duration_ps, frame_ps, max_fraction=None):
    """Doubling lags from one frame up to a fraction of a hold.

    Parameters
    ----------
    duration_ps : float
        Hold length in picoseconds.
    frame_ps : float
        Spacing of the stored frames.
    max_fraction : float, optional
        Longest lag as a fraction of the hold. This is the same fraction
        ``03_fs_alpha.py`` uses.

    Returns
    -------
    list of float
        Lag times in picoseconds. The last entry is the fraction of the hold
        when that time is longer than the last doubled step.
    """
    if max_fraction is None:
        max_fraction = CONFIG.dynamics.hold_lag_fraction
    limit = float(max_fraction) * float(duration_ps)
    step = float(frame_ps)
    if limit < step:
        raise ValueError("The hold is shorter than one frame at the requested lag fraction.")
    lags = [step]
    while lags[-1] * 2.0 <= limit + 1e-9:
        lags.append(lags[-1] * 2.0)
    if lags[-1] < limit - 1e-6:
        lags.append(limit)
    return lags


def discard_frame_count(n_frames, kind, fraction):
    """Frames skipped at the start of a hold or an endpoint.

    Parameters
    ----------
    n_frames : int
        Frames in the trajectory.
    kind : str
        ``ramp``, ``hold``, or ``endpoint``.
    fraction : float
        Fraction dropped for a hold or an endpoint.

    Returns
    -------
    int
        First frame index that is analyzed. A ramp starts at zero.
    """
    if kind not in ("hold", "endpoint"):
        return 0
    skipped = int(np.floor(int(n_frames) * float(fraction)))
    return min(skipped, max(int(n_frames) - 2, 0))


def frames_skipped(discard_ps, frame_ps):
    """Stored frames covered by the manifest equilibration time.

    Parameters
    ----------
    discard_ps : float or str
        ``discard_ps`` from the manifest row. Blank or zero keeps every frame.
    frame_ps : float or str
        Spacing of the stored frames, from the same row.

    Returns
    -------
    int
        Frames to drop at the start of the concatenated trajectory.

    Raises
    ------
    ValueError
        If a positive discard has no positive frame spacing.
    """
    text = str(discard_ps).strip()
    if text in ("", "nan"):
        return 0
    discarded = float(text)
    if discarded == 0.0:
        return 0
    if discarded < 0.0:
        raise ValueError("discard_ps must be non-negative.")
    spacing = str(frame_ps).strip()
    if spacing == "":
        raise ValueError("A row with discard_ps needs frame_ps.")
    step = float(spacing)
    if step <= 0.0:
        raise ValueError("frame_ps must be positive.")
    return int(np.floor(discarded / step + 1e-9))


def trajin_lines(paths, skip, stride, frame_counts=None):
    """cpptraj ``trajin`` lines that drop ``skip`` frames once.

    Parameters
    ----------
    paths : sequence of path-like
        Segments in read order.
    skip : int
        Frames dropped from the start of the concatenated trajectory.
        Zero keeps the first frame of the first segment.
    stride : int
        cpptraj stride applied after that start.
    frame_counts : sequence of int, optional
        Frames in each segment. Required when ``skip`` is positive.
        When omitted and ``skip`` is positive, each NetCDF ``time``
        variable is counted.

    Returns
    -------
    list of str
        One ``trajin path start last stride`` line per segment that
        still contains a kept frame. ``start`` is 1-based.

    Raises
    ------
    ValueError
        If the discard removes every frame, or the counts do not match
        the segments.
    """
    paths = list(paths)
    if int(skip) <= 0:
        return [f"trajin {path} 1 last {stride}" for path in paths]
    if frame_counts is None:
        frame_counts = [len(netcdf_time_ps(path)) for path in paths]
    counts = [int(count) for count in frame_counts]
    if len(counts) != len(paths):
        raise ValueError("Each segment needs a frame count.")
    remaining = int(skip)
    lines = []
    for path, count in zip(paths, counts):
        if count < 1:
            raise ValueError(f"{path} contains no frames.")
        if remaining >= count:
            remaining -= count
            continue
        lines.append(f"trajin {path} {remaining + 1} last {stride}")
        remaining = 0
    if not lines:
        raise ValueError("The discard removes every frame.")
    return lines


def resolve_n_water(topology, claimed=None):
    """Read the water count and check it against the OPC film.

    Parameters
    ----------
    topology : path-like
        Amber prmtop. The count is never taken from a hardcoded default alone.
    claimed : int, optional
        Value passed as ``--n-water``. It must equal the topology count.

    Returns
    -------
    int
        Water count read from ``topology``.

    Raises
    ------
    ValueError
        If the topology count is not the OPC film, or if ``claimed`` differs.
    """
    if topology is None:
        raise ValueError("Pass --topology so the water count is read from the prmtop.")
    count = count_waters(topology)
    if claimed is not None and int(claimed) != count:
        raise ValueError(f"--n-water {claimed} disagrees with the topology count {count}.")
    if count != int(CONFIG.film.n_water):
        raise ValueError(f"Topology has {count} waters; the OPC film has {CONFIG.film.n_water}.")
    return count
