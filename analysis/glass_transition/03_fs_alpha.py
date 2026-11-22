#!/usr/bin/env python3
"""
In-plane self-intermediate scattering function for the OPC slab.

The in-plane step is the minimum-image displacement of the unwrapped
oxygen coordinate, so a trajectory that has been re-imaged is acceptable.
After unwrapping, coordinates are not imaged again. The oxygen centroid
is removed at each frame. The lag-zero value is exactly 1, and the 1/e
time is classified as resolved, fast-censored, or slow-censored.

Definition:
    Fs_parallel(q,t) = < J0(q |r_i_parallel(t0+t)-r_i_parallel(t0)|) >_{i,t0}

Coordinates are Angstrom and q is Angstrom^-1.

The lag is read from the NetCDF time variable. ``--frame-ps`` is optional and
is accepted only when it agrees with that variable. ``--stride`` selects
frames from the file that is read. A file that was already thinned must
not be strided a second time.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import HOLD_DISCARD_FRACTION, load_adopted_qstar
from common.slab import water_indices
from common.trajectory import continuation_frame_spacing_ps, lag_pair_ok

import MDAnalysis as mda
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import parmed as pmd
from scipy.special import j0
from scipy.optimize import curve_fit

LEVEL = math.exp(-1.0)


def oxygen_indices(prmtop, water_names, oxygen_stride=1, oxygen_offset=0):
    """All oxygen indices and the subsample used for the scattering function.

    Parameters
    ----------
    prmtop : 
        Amber topology path.
    water_names : 
        Residue names treated as water.
    oxygen_stride : 
        Stride through the oxygen list.
    oxygen_offset : 
        Starting offset of the oxygen subsample.

    Returns
    -------
    result
        All oxygen indices and the subsample used for the scattering function.
    """
    requested = {name.upper() for name in water_names}
    if not requested <= {"WAT", "HOH", "OPC"}:
        raise ValueError(
            "Water oxygens come from common.slab.water_indices (WAT, HOH, OPC)."
        )
    idx, _hydrogen = water_indices(mda.Universe(prmtop))
    if oxygen_stride < 1:
        raise ValueError("--oxygen-stride must be >= 1.")
    if oxygen_offset < 0 or oxygen_offset >= oxygen_stride:
        raise ValueError("--oxygen-offset must satisfy 0 <= offset < oxygen_stride.")
    selected = idx[oxygen_offset::oxygen_stride]
    if len(selected) == 0:
        raise RuntimeError("Oxygen subsampling selected zero atoms.")
    return idx, selected


def topology_box_xy(prmtop):
    """Lateral box lengths stored in the topology, or None.

    Parameters
    ----------
    prmtop : 
        Amber topology path.

    Returns
    -------
    result
        Lateral box lengths stored in the topology, or None.
    """
    top = pmd.load_file(prmtop)
    if top.box is None:
        return None
    box = np.asarray(top.box, dtype=float)
    return box[:2].copy() if box.size >= 2 else None


def _frame_box_xy(nc, frame_index, fallback_box_xy):
    """Lateral box lengths of one frame.

    Parameters
    ----------
    nc : 
        Open netCDF dataset.
    frame_index : 
        Zero-based frame index.
    fallback_box_xy : 
        Box lengths used when a frame does not store them.

    Returns
    -------
    result
        Lateral box lengths of one frame.
    """
    if "cell_lengths" in nc.variables:
        b = np.asarray(nc.variables["cell_lengths"][frame_index, :2], dtype=np.float64)
        if np.all(np.isfinite(b)) and np.all(b > 0):
            return b
    if fallback_box_xy is not None:
        return np.asarray(fallback_box_xy, dtype=np.float64)
    raise RuntimeError(
        "Could not obtain x/y box lengths from NetCDF cell_lengths or topology."
    )


def read_corrected_xy(
    trajs,
    oxygen_idx,
    frame_ps,
    stride=1,
    fallback_box_xy=None,
    discard_ps=0.0,
    discard_fraction=0.0,
):
    """Read native wrapped O coordinates, unwrap x/y, and remove slab drift.

    Parameters
    ----------
    trajs : 
        Trajectory paths, read in order.
    oxygen_idx : 
        Oxygen atom indices.
    frame_ps : 
        Saved-frame spacing in picoseconds.
    stride : 
        Frame stride.
    fallback_box_xy : 
        Box lengths used when a frame does not store them.
    discard_ps : 
        Equilibration time to drop, in picoseconds. Ignored when
        ``discard_fraction`` is positive.
    discard_fraction : 
        Fraction of the loaded trajectory dropped from the start. Holds use 0.10.

    Returns
    -------
    result
        Read native wrapped O coordinates, unwrap x/y, and remove slab drift.
    """
    if frame_ps is not None and frame_ps <= 0:
        raise ValueError("--frame-ps must be > 0.")
    if stride < 1:
        raise ValueError("--stride must be >= 1.")

    frames = []
    selected_times = []
    prev_wrapped = None
    unwrapped = None
    first_box = None
    last_box = None
    max_box_change = 0.0
    global_frame = 0

    for tr in trajs:
        print(f"Reading native trajectory: {tr}")
        with Dataset(tr) as nc:
            if "coordinates" not in nc.variables:
                raise RuntimeError(f"{tr}: no 'coordinates' variable.")
            xyz = nc.variables["coordinates"]
            if "time" not in nc.variables:
                raise RuntimeError(f"{tr} has no time variable.")
            time = np.asarray(nc.variables["time"][:], dtype=float)
            nframes = xyz.shape[0]

            # Stride over the joined trajectory, not per segment, so a
            # segment boundary never shortens a strided step.
            first_local = (-global_frame) % stride
            global_frame += nframes
            for f in range(first_local, nframes, stride):
                selected_times.append(float(time[f]))
                wrapped = np.asarray(xyz[f, oxygen_idx, :2], dtype=np.float64)
                box_xy = _frame_box_xy(nc, f, fallback_box_xy)

                if first_box is None:
                    first_box = box_xy.copy()
                if last_box is not None:
                    max_box_change = max(
                        max_box_change,
                        float(np.max(np.abs(box_xy - last_box))),
                    )
                last_box = box_xy.copy()

                if prev_wrapped is None:
                    unwrapped = wrapped.copy()
                else:
                    dr = wrapped - prev_wrapped
                    dr -= box_xy * np.rint(dr / box_xy)
                    unwrapped += dr

                prev_wrapped = wrapped

                # Internal in-plane coordinates: remove instantaneous
                # geometric-centroid translation of the selected oxygens.
                centered = unwrapped - unwrapped.mean(axis=0, keepdims=True)
                frames.append(centered.astype(np.float32))

    if not frames:
        raise RuntimeError("No trajectory frames were read.")

    xy = np.stack(frames, axis=0)
    dt_selected_ps, n_joins = continuation_frame_spacing_ps(selected_times, override=frame_ps)
    if discard_fraction:
        discard_ps = float(discard_fraction) * (len(xy) - 1) * dt_selected_ps

    discard_frames = int(math.ceil(discard_ps / dt_selected_ps - 1.0e-12))
    if discard_frames >= len(xy) - 1:
        raise RuntimeError("--discard-ps removes all or almost all frames.")
    if discard_frames > 0:
        xy = xy[discard_frames:]

    # True frame times, relative to the first kept frame. Segment joins make
    # them differ from a uniform grid; lag pairs are checked against them.
    times = np.asarray(selected_times[discard_frames:], dtype=np.float64)
    times = times - times[0]

    print()
    print("Trajectory preprocessing")
    print("------------------------")
    print(f"Selected frames loaded:       {len(xy)}")
    print(f"Selected oxygen atoms:        {len(oxygen_idx)}")
    print(f"Effective frame spacing:      {dt_selected_ps:.6f} ps")
    print(f"Segment joins accepted:       {n_joins}")
    print(f"Discarded initial time:       {discard_frames * dt_selected_ps:.3f} ps")
    print(f"Discard fraction requested:   {discard_fraction:g}")
    print(f"Usable duration:              {times[-1] / 1000.0:.6f} ns")
    if first_box is not None:
        print(f"Initial Lx,Ly:               {first_box[0]:.6f}, {first_box[1]:.6f} A")
        print(f"Maximum frame box change:    {max_box_change:.6e} A")
    print()

    return xy, times, dt_selected_ps, discard_frames * dt_selected_ps


def make_lags(nframes, nlag=110, max_fraction=0.250, linear_early=12):
    """Frame lags, dense at short time and logarithmic afterward.

    Parameters
    ----------
    nframes : 
        Number of frames.
    nlag : 
        Number of lag times.
    max_fraction : 
        Largest lag as a fraction of the trajectory.
    linear_early : 
        Number of early lags kept on a linear grid.

    Returns
    -------
    result
        Frame lags, dense at short time and logarithmic afterward.
    """
    if nframes < 3:
        raise RuntimeError("Trajectory block has too few frames.")
    max_lag = int(math.floor((nframes - 1) * max_fraction))
    max_lag = max(1, min(max_lag, nframes - 1))
    early = np.arange(1, min(linear_early, max_lag) + 1, dtype=int)
    if max_lag <= linear_early:
        positive = early
    else:
        late = np.unique(
            np.rint(np.geomspace(linear_early + 1, max_lag, nlag)).astype(int)
        )
        positive = np.unique(np.concatenate([early, late]))
    return np.concatenate([np.array([0], dtype=int), positive])


def choose_origins(nframes, lag, max_origins):
    """Evenly spaced time origins that leave room for this lag.

    Parameters
    ----------
    nframes : 
        Number of frames.
    lag : 
        Time lag in frames.
    max_origins : 
        Maximum number of time origins per lag.

    Returns
    -------
    result
        Evenly spaced time origins that leave room for this lag.
    """
    n_available = nframes - lag
    if n_available <= 0:
        return np.array([], dtype=int)
    n_use = min(int(max_origins), n_available)
    if n_use <= 1:
        return np.array([0], dtype=int)
    return np.unique(
        np.rint(np.linspace(0, n_available - 1, n_use)).astype(int)
    )


def fs_for_lag(xy, lag, q, max_origins, origin_batch, times=None, dt_ps=None):
    """Mean Fs and spread across per-origin particle averages.

    Parameters
    ----------
    xy : 
        In-plane coordinates with shape (n_frame, n_oxygen, 2).
    lag : 
        Time lag in frames.
    q : 
        Wavevector magnitude in inverse angstroms.
    max_origins : 
        Maximum number of time origins per lag.
    origin_batch : 
        Number of time origins evaluated at once.
    times : array_like, optional
        True frame times in picoseconds. When given, origins whose pair
        crosses a segment join with a time error beyond
        ``JOIN_LAG_ATOL_PS`` or ``JOIN_LAG_RTOL`` are dropped.
    dt_ps : float, optional
        Spacing within segments. Required with ``times``.

    Returns
    -------
    result
        Mean Fs and spread across per-origin particle averages.
    """
    nframes = len(xy)
    if lag == 0:
        return {
            "Fs_parallel": 1.0,
            "Fs_origin_sd": 0.0,
            "Fs_origin_sem_naive": 0.0,
            "n_origins": nframes,
        }

    origins = choose_origins(nframes, lag, max_origins)
    if times is not None and len(origins):
        # Drop pairs whose true separation across a segment join is off the lag.
        origins = origins[lag_pair_ok(times, origins, lag, dt_ps)]
    if len(origins) == 0:
        return {
            "Fs_parallel": np.nan,
            "Fs_origin_sd": np.nan,
            "Fs_origin_sem_naive": np.nan,
            "n_origins": 0,
        }

    per_origin = []
    for i0 in range(0, len(origins), origin_batch):
        ob = origins[i0:i0 + origin_batch]
        dr = xy[ob + lag].astype(np.float64) - xy[ob].astype(np.float64)
        rr = np.sqrt(np.sum(dr * dr, axis=2))
        per_origin.append(np.mean(j0(q * rr), axis=1))

    per_origin = np.concatenate(per_origin)
    mean = float(np.mean(per_origin))
    if len(per_origin) > 1:
        sd = float(np.std(per_origin, ddof=1))
        sem = float(sd / np.sqrt(len(per_origin)))
    else:
        sd = np.nan
        sem = np.nan

    return {
        "Fs_parallel": mean,
        "Fs_origin_sd": sd,
        "Fs_origin_sem_naive": sem,
        "n_origins": int(len(per_origin)),
    }


def classify_tau(t_ps, fs, level=LEVEL):
    """Return (status, tau_ps, bound_ps).

    Parameters
    ----------
    t_ps : 
        Time in picoseconds.
    fs : 
        Self-intermediate scattering function.
    level : 
        Threshold of the correlation function, usually 1/e.

    Returns
    -------
    result
        Return (status, tau_ps, bound_ps).
    """
    from opcfilm.dynamics.relaxation import classify_tau as shared

    return shared(t_ps, fs, level)


def kww(t, A, tau, beta):
    """Kohlrausch stretched exponential.

    Parameters
    ----------
    t : 
        Time.
    A : 
        Activation parameter or amplitude, in the units of the calling fit.
    tau : 
        Relaxation time.
    beta : 
        Stretching exponent of the Kohlrausch function.

    Returns
    -------
    result
        Kohlrausch stretched exponential.
    """
    return A * np.exp(-np.power(t / tau, beta))


def fit_kww(t_ps, fs, tau_guess_ps):
    """Fit a stretched exponential to one decay.

    Parameters
    ----------
    t_ps : 
        Time in picoseconds.
    fs : 
        Self-intermediate scattering function.
    tau_guess_ps : 
        Initial relaxation time for the stretched-exponential fit, in picoseconds.

    Returns
    -------
    result
        Fit a stretched exponential to one decay.
    """
    t_ps = np.asarray(t_ps, dtype=float)
    fs = np.asarray(fs, dtype=float)
    m = (
        np.isfinite(t_ps)
        & np.isfinite(fs)
        & (t_ps > 0)
        & (fs < 0.9)
        & (fs > 0.05)
    )
    if np.count_nonzero(m) < 6:
        return np.nan, np.nan, np.nan
    if not np.isfinite(tau_guess_ps):
        tau_guess_ps = float(np.median(t_ps[m]))
    p0 = [min(1.0, max(0.3, float(fs[m][0]))), tau_guess_ps, 0.7]
    try:
        popt, _ = curve_fit(
            kww,
            t_ps[m],
            fs[m],
            p0=p0,
            bounds=([0.2, 1.0e-9, 0.1], [1.2, np.inf, 1.0]),
            maxfev=30000,
        )
        return tuple(float(x) for x in popt)
    except Exception:
        return np.nan, np.nan, np.nan


def analyze_block(
    xy,
    dt_ps,
    q,
    nlag,
    max_fraction,
    linear_early,
    max_origins,
    origin_batch,
    times=None,
):
    """Scattering function and 1/e time of one trajectory block.

    Parameters
    ----------
    xy : 
        In-plane coordinates with shape (n_frame, n_oxygen, 2).
    dt_ps : 
        Time between saved frames, in picoseconds.
    q : 
        Wavevector magnitude in inverse angstroms.
    nlag : 
        Number of lag times.
    max_fraction : 
        Largest lag as a fraction of the trajectory.
    linear_early : 
        Number of early lags kept on a linear grid.
    max_origins : 
        Maximum number of time origins per lag.
    origin_batch : 
        Number of time origins evaluated at once.
    times : array_like, optional
        True frame times of the block, passed to :func:`fs_for_lag`.

    Returns
    -------
    result
        Scattering function and 1/e time of one trajectory block.
    """
    lags = make_lags(
        len(xy),
        nlag=nlag,
        max_fraction=max_fraction,
        linear_early=linear_early,
    )
    rows = []

    for j, lag in enumerate(lags):
        result = fs_for_lag(
            xy,
            int(lag),
            q,
            max_origins,
            origin_batch,
            times=times,
            dt_ps=dt_ps,
        )
        rows.append({
            "lag_frames": int(lag),
            "lag_ps": float(lag * dt_ps),
            **result,
        })
        if j == 0 or j == len(lags) - 1 or j % max(1, len(lags) // 10) == 0:
            print(
                f"  lag {j+1:4d}/{len(lags):4d}: "
                f"{lag * dt_ps / 1000.0:10.5f} ns, "
                f"Fs={result['Fs_parallel']:.6f}, origins={result['n_origins']}"
            )

    df = pd.DataFrame(rows)
    status, tau_ps, bound_ps = classify_tau(
        df["lag_ps"].to_numpy(),
        df["Fs_parallel"].to_numpy(),
    )
    A, tau_kww_ps, beta = fit_kww(
        df["lag_ps"].to_numpy(),
        df["Fs_parallel"].to_numpy(),
        tau_ps,
    )
    return df, status, tau_ps, bound_ps, A, tau_kww_ps, beta


def main():
    """Write the in-plane self-intermediate scattering function.

    Returns
    -------
    result
        Write the in-plane self-intermediate scattering function.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Corrected in-plane self-intermediate scattering function "
            "from native wrapped OPC trajectories."
        )
    )
    ap.add_argument("-p", "--prmtop", default=None)
    ap.add_argument(
        "-x", "--traj", action="append", default=None,
        help="Native trajectory; repeat -x for contiguous files in chronological order.",
    )
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--row", type=int, default=None, help="1-based row of --manifest.")
    ap.add_argument(
        "--allow-unlisted", action="store_true",
        help="Permit -p and -x when the trajectory is not in the manifest.",
    )
    ap.add_argument("-q", "--qstar", type=float, default=None, help="Must match qstar_adopted_Ainv in --qstar-json.")
    ap.add_argument("--qstar-json", type=Path, default=Path("outputs/qstar.json"))
    ap.add_argument(
        "--frame-ps",
        type=float,
        default=None,
        help="Optional override. It must agree with the NetCDF time spacing.",
    )
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--discard-ns", type=float, default=None,
        help="Time dropped at the start, in ns. Holds and endpoints default to the first 10%%.",
    )
    ap.add_argument("--nblocks", type=int, default=3)
    ap.add_argument("--nlag", type=int, default=110)
    ap.add_argument("--linear-early-lags", type=int, default=12)
    ap.add_argument("--max-fraction", type=float, default=0.250)
    ap.add_argument("--max-origins", type=int, default=120)
    ap.add_argument("--origin-batch", type=int, default=8)
    ap.add_argument(
        "--oxygen-stride", type=int, default=10,
        help="Use every Nth water oxygen; use 1 for all oxygens.",
    )
    ap.add_argument("--oxygen-offset", type=int, default=0)
    ap.add_argument(
        "--water-resnames", nargs="+",
        default=["WAT", "HOH", "OPC"],
        help="Must be a subset of WAT, HOH, OPC; the oxygens come from common.slab.water_indices.",
    )
    ap.add_argument("--temperature", type=float, default=np.nan)
    ap.add_argument("--rate", type=float, default=np.nan)
    ap.add_argument("--replica", default="NA")
    ap.add_argument("--curve-out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()
    args.trajectory_kind = "ramp"
    if args.manifest is not None:
        from common.manifest import load_manifest, select_row, trajectory_list
        row = select_row(load_manifest(args.manifest), args.row)
        args.prmtop = row["topology"]
        args.traj = [str(path) for path in trajectory_list(row)]
        args.trajectory_kind = row["kind"]
        args.temperature = float(row["temperature_K"]) if row["temperature_K"] else np.nan
        args.replica = row["replica"]
        if row["rate_K_ns"] not in ("", "unset"):
            args.rate = float(row["rate_K_ns"])
        if row["frame_ps"]:
            args.frame_ps = float(row["frame_ps"])
    elif not args.allow_unlisted:
        raise SystemExit(
            "Pass --manifest and --row. A path typed by hand requires --allow-unlisted."
        )
    elif not args.prmtop or not args.traj:
        raise SystemExit("--allow-unlisted still requires -p and at least one -x.")
    args.qstar = load_adopted_qstar(args.qstar_json, args.qstar)

    if not (0 < args.max_fraction < 1):
        raise ValueError("--max-fraction must be between 0 and 1.")

    all_oi, selected_oi = oxygen_indices(
        args.prmtop,
        args.water_resnames,
        oxygen_stride=args.oxygen_stride,
        oxygen_offset=args.oxygen_offset,
    )
    box_xy = topology_box_xy(args.prmtop)
    if args.discard_ns is None:
        discard_ps = 0.0
        discard_fraction = HOLD_DISCARD_FRACTION if args.trajectory_kind in ("hold", "endpoint") else 0.0
    else:
        discard_ps = args.discard_ns * 1000.0
        discard_fraction = 0.0
    xy, times, dt_selected_ps, discarded_ps = read_corrected_xy(
        args.traj,
        selected_oi,
        args.frame_ps,
        stride=args.stride,
        fallback_box_xy=box_xy,
        discard_ps=discard_ps,
        discard_fraction=discard_fraction,
    )

    n = len(xy)
    blocks = [("whole", 0, n)]
    edges = np.linspace(0, n, args.nblocks + 1, dtype=int)
    for i in range(args.nblocks):
        blocks.append((f"block{i+1}", int(edges[i]), int(edges[i+1])))

    curve_rows = []
    summary_rows = []

    for name, start, end in blocks:
        print()
        print("=" * 88)
        print(
            f"Analyzing {name}: frames {start} .. {end-1} "
            f"({(end-start-1)*dt_selected_ps/1000.0:.4f} ns)"
        )
        print("=" * 88)

        (
            cdf,
            tau_status,
            tau_ps,
            bound_ps,
            A,
            tau_kww_ps,
            beta,
        ) = analyze_block(
            xy[start:end],
            dt_selected_ps,
            args.qstar,
            args.nlag,
            args.max_fraction,
            args.linear_early_lags,
            args.max_origins,
            args.origin_batch,
            times=times[start:end],
        )

        for row in cdf.to_dict("records"):
            curve_rows.append({
                "block": name,
                **row,
                "T_K": args.temperature,
                "rate_K_per_ns": args.rate,
                "replica": args.replica,
                "qstar_Ainv": args.qstar,
                "frame_ps_selected": dt_selected_ps,
                "n_oxygens_total": len(all_oi),
                "n_oxygens_used": len(selected_oi),
                "oxygen_stride": args.oxygen_stride,
            })

        summary_rows.append({
            "block": name,
            "T_K": args.temperature,
            "rate_K_per_ns": args.rate,
            "replica": args.replica,
            "qstar_Ainv": args.qstar,
            "discard_ps": discarded_ps,
            "tau_status": tau_status,
            "tau_1e_ns": tau_ps / 1000.0 if np.isfinite(tau_ps) else np.nan,
            "tau_bound_ns": bound_ps / 1000.0 if np.isfinite(bound_ps) else np.nan,
            "A_kww": A,
            "tau_kww_ns": tau_kww_ps / 1000.0 if np.isfinite(tau_kww_ps) else np.nan,
            "beta_kww": beta,
            "duration_ns": (
                (end - start - 1) * dt_selected_ps / 1000.0
                if end - start > 1 else 0.0
            ),
            "n_frames": end - start,
            "frame_ps_selected": dt_selected_ps,
            "discard_ps": discarded_ps,
            "n_oxygens_total": len(all_oi),
            "n_oxygens_used": len(selected_oi),
            "oxygen_stride": args.oxygen_stride,
            "max_origins": args.max_origins,
            "max_fraction": args.max_fraction,
        })

    curve_df = pd.DataFrame(curve_rows)
    summary_df = pd.DataFrame(summary_rows)

    block_tau = summary_df.loc[
        (summary_df["block"] != "whole")
        & (summary_df["tau_status"] == "resolved"),
        "tau_1e_ns",
    ].dropna().to_numpy(dtype=float)

    tau_cv = np.nan
    if len(block_tau) >= 2 and np.mean(block_tau) > 0:
        tau_cv = float(np.std(block_tau, ddof=1) / np.mean(block_tau))

    summary_df["block_tau_cv_resolved"] = tau_cv
    summary_df["stationary_tau_cv_lt_0.25"] = bool(
        np.isfinite(tau_cv) and tau_cv < 0.25
    )

    curve_out = Path(args.curve_out)
    summary_out = Path(args.summary_out)
    curve_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(curve_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print()
    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(summary_df.to_string(index=False))
    print()
    print(f"Wrote: {curve_out}")
    print(f"Wrote: {summary_out}")


if __name__ == "__main__":
    main()
