"""Local order, CHILL+, and depth-resolved in-plane structure factors.

Outputs are written in ``--out``. The box is read from each frame and
checked against the OPC film. The half-density surface used for the outer
structure-factor slice is the mean surface of the frames being analyzed.
``q``, ``Sk``, and LSI use the nearest neighbors with no distance cutoff,
so a surface molecule is scored with distant neighbors. LSI is missing
when fewer than two neighbors lie within 3.7 Å. Interfacial order is
therefore low partly because of that geometry.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.signal import find_peaks, peak_widths

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import BULK_HALFWIDTH_A, QSTAR_AINV, SEEDS, SQ_GRID_SPACING_A, fft_cell_counts
from opcfilm.config import CONFIG
from common.order import local_structure_index as shared_lsi
from common.order import orientational_tetrahedral_q as shared_q
from common.order import translational_tetrahedral_sk as shared_sk
from common.slab import region_masks, slab_geometry
from common.trajectory import frame_spacing_ps, require_film

def orientational_tetrahedral_q(O_pos, i, nbrs4):
    """Orientational tetrahedral order q.
    q = 1 for perfect tetrahedral arrangement.

    Parameters
    ----------
    O_pos : 
        Oxygen coordinates.
    i : 
        Index of the central oxygen.
    nbrs4 : 
        Indices of the four nearest oxygen neighbors.

    Returns
    -------
    result
        Orientational tetrahedral order q.
    """
    if len(nbrs4) < 4:
        return np.nan
    return shared_q(O_pos[nbrs4] - O_pos[i])

def translational_tetrahedral_Sk(dist4):
    """
    Conventional translational tetrahedral order parameter S_k.
    S_k = 1 for four equidistant neighbors forming the
    radial component of a perfect tetrahedral environment.
    Parameters
    ----------
    dist4 : array-like
        Distances to the four nearest oxygen neighbors.
    Returns
    -------
    float
        Translational tetrahedral order S_k.
    """
    return shared_sk(dist4)

def local_structure_index(neighbor_distances, cutoff=3.7):
    """
    Standard Local Structure Index (LSI) for water.
    Parameters
    ----------
    neighbor_distances : array-like
        O-O distances from the central water to neighboring waters.
        Must include at least the first neighbor beyond `cutoff`.
    cutoff : float
        O-O cutoff in Angstrom. Commonly 3.7 A.
    Returns
    -------
    float
        LSI in Angstrom^2.
    """
    return shared_lsi(neighbor_distances, cutoff)

def angular_distribution_angles(O_pos, i, nbrs4):
    """Six O-O-O angles formed by the four nearest O neighbors around central oxygen.

    Parameters
    ----------
    O_pos : 
        Oxygen coordinates.
    i : 
        Index of the central oxygen.
    nbrs4 : 
        Indices of the four nearest oxygen neighbors.

    Returns
    -------
    result
        Six O-O-O angles formed by the four nearest O neighbors around central oxygen.
    """
    if len(nbrs4) < 4:
        return []
    vecs = O_pos[nbrs4] - O_pos[i]
    norms = np.linalg.norm(vecs, axis=1)
    if np.any(norms == 0):
        return []
    vecs = vecs / norms[:, None]
    angles = []
    for a in range(3):
        for b in range(a + 1, 4):
            c = np.clip(np.dot(vecs[a], vecs[b]), -1.0, 1.0)
            angles.append(np.degrees(np.arccos(c)))
    return angles

def chillplus_classify_oxygen_positions(O_pos, Lx, Ly, Lz=None, z_center=None):
    """Run OVITO CHILL+ on oxygen-only water positions.

    Parameters
    ----------
    O_pos : 
        Oxygen coordinates.
    Lx : 
        Box length along x, in angstroms.
    Ly : 
        Box length along y, in angstroms.
    Lz : 
        Box length along z, in angstroms.
    z_center : float
        Absolute ``z`` of the slab center from ``slab_geometry``.

    Returns
    -------
    result
        Run OVITO CHILL+ on oxygen-only water positions.
    """
    from opcfilm.structure.chill import chill_plus

    if Lz is None:
        Lz = float(CONFIG.film.box_A[2])
    return chill_plus(O_pos, Lx, Ly, Lz, z_center)

def make_sq_depths(
    gibbs_abs_z,
    depth_step,
):
    """Generate depth centers:
        0, 5, 10, ... , Gibbs surface
    The exact Gibbs surface is included even when it is not
    an integer multiple of depth_step.

    Parameters
    ----------
    gibbs_abs_z : 
        Absolute half-density surface, in angstroms.
    depth_step : 
        Spacing of structure-factor slices, in angstroms.

    Returns
    -------
    result
        Generate depth centers:
    """
    if gibbs_abs_z <= 0:
        raise ValueError(
            "gibbs_abs_z must be positive."
        )
    depths = np.arange(
        0.0,
        gibbs_abs_z,
        depth_step,
        dtype=float,
    )
    # Add exact Gibbs dividing surface
    if (
        len(depths) == 0
        or
        not np.isclose(
            depths[-1],
            gibbs_abs_z,
        )
    ):
        depths = np.append(
            depths,
            gibbs_abs_z,
        )
    return depths

def prepare_sq2d_fft_grid(
    Lx,
    Ly,
    grid_spacing,
    q_min,
    q_max,
    dq,
):
    """Precompute reciprocal-space information for the
    in-plane O-O static structure factor.
    Returns radial q bins and FFT-grid metadata.

    Parameters
    ----------
    Lx : 
        Box length along x, in angstroms.
    Ly : 
        Box length along y, in angstroms.
    grid_spacing : 
        Real-space FFT spacing in angstroms.
    q_min : 
        Smallest wavevector kept, in inverse angstroms.
    q_max : 
        Largest wavevector kept, in inverse angstroms.
    dq : 
        Radial wavevector bin width in inverse angstroms.

    Returns
    -------
    result
        Precompute reciprocal-space information for the
    """
    nx = fft_cell_counts(Lx, grid_spacing)
    ny = fft_cell_counts(Ly, grid_spacing)
    dx_grid = (
        Lx / nx
    )
    dy_grid = (
        Ly / ny
    )
    # Reciprocal-space components in A^-1
    qx = (
        2.0 * np.pi
        * np.fft.fftfreq(
            nx,
            d=dx_grid,
        )
    )
    qy = (
        2.0 * np.pi
        * np.fft.fftfreq(
            ny,
            d=dy_grid,
        )
    )
    QX, QY = np.meshgrid(
        qx,
        qy,
        indexing="ij",
    )
    qmag = np.sqrt(
        QX**2
        + QY**2
    )
    q_edges = np.arange(
        q_min,
        q_max + dq,
        dq,
        dtype=float,
    )
    q_centers = (
        0.5
        * (
            q_edges[:-1]
            + q_edges[1:]
        )
    )
    q_flat = qmag.ravel()
    q_bin = (
        np.digitize(
            q_flat,
            q_edges,
        )
        - 1
    )
    # q = 0 is intentionally excluded.
    valid_q = (
        (q_flat > 1.0e-12)
        &
        (q_bin >= 0)
        &
        (q_bin < len(q_centers))
    )
    q_bin_valid = (
        q_bin[valid_q]
    )
    n_modes = np.bincount(
        q_bin_valid,
        minlength=len(
            q_centers
        ),
    )
    return {
        "nx": nx,
        "ny": ny,
        "dx_grid": dx_grid,
        "dy_grid": dy_grid,
        "qx": qx,
        "qy": qy,
        "QX": QX,
        "QY": QY,
        "qmag": qmag,
        "q_edges": q_edges,
        "q_centers": q_centers,
        "valid_q": valid_q,
        "q_bin_valid": q_bin_valid,
        "n_modes": n_modes,
    }

def calculate_sq2d_slice(
    x,
    y,
    Lx,
    Ly,
    fft_info,
    min_waters=200,
    return_grid=False,
):
    """Calculate radially averaged in-plane static structure
    factor S_parallel(q) for one z slice in one frame.
    Uses oxygen positions only.
        rho(q) = sum_j exp(i q . r_j)
        S(q) = |rho(q)|^2 / N
    The 2D density field is evaluated with an FFT.

    Parameters
    ----------
    x : 
        Independent variable.
    y : 
        Dependent variable or profile values.
    Lx : 
        Box length along x, in angstroms.
    Ly : 
        Box length along y, in angstroms.
    fft_info : 
        Wavevector grids returned by the FFT setup.
    min_waters : 
        Minimum number of oxygens required in a slice.
    return_grid : 
        Also return the two-dimensional structure factor.

    Returns
    -------
    result
        Calculate radially averaged in-plane static structure
    """
    x = np.asarray(
        x,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )
    n_waters = len(x)
    if n_waters < min_waters:
        return None
    nx = fft_info[
        "nx"
    ]
    ny = fft_info[
        "ny"
    ]
    # Put coordinates in primary lateral box.
    x_wrapped = np.mod(
        x,
        Lx,
    )
    y_wrapped = np.mod(
        y,
        Ly,
    )
    # --------------------------------------------------------
    # 2D projected oxygen density
    # --------------------------------------------------------
    density_grid, _, _ = np.histogram2d(
        x_wrapped,
        y_wrapped,
        bins=[
            nx,
            ny,
        ],
        range=[
            [0.0, Lx],
            [0.0, Ly],
        ],
    )
    # --------------------------------------------------------
    # Fourier transform
    # --------------------------------------------------------
    rho_q = np.fft.fft2(
        density_grid
    )
    # Static structure factor for each reciprocal vector.
    Sq_grid = (
        np.abs(rho_q) ** 2
        / float(n_waters)
    )
    Sq_flat = (
        Sq_grid.ravel()
    )
    # --------------------------------------------------------
    # Radial average in q
    # --------------------------------------------------------
    values = Sq_flat[
        fft_info[
            "valid_q"
        ]
    ]
    Sq_sum = np.bincount(
        fft_info[
            "q_bin_valid"
        ],
        weights=values,
        minlength=len(
            fft_info[
                "q_centers"
            ]
        ),
    )
    n_modes = fft_info[
        "n_modes"
    ]
    Sq_radial = np.divide(
        Sq_sum,
        n_modes,
        out=np.full(
            len(Sq_sum),
            np.nan,
            dtype=float,
        ),
        where=n_modes > 0,
    )
    if return_grid:
        return Sq_radial, Sq_grid
    return Sq_radial

def analyze_radial_bragg_peaks(
    q,
    Sq,
    dq_resolution,
    q_min=1.0,
    q_max=5.0,
    min_prominence=0.05,
    resolution_multiple=2.5,
):
    """Peaks in the radial structure factor and a flag for peaks narrower than the resolution.

    Parameters
    ----------
    q : 
        Wavevector magnitude in inverse angstroms.
    Sq : 
        Radially averaged in-plane structure factor.
    dq_resolution : 
        Reciprocal-space resolution in inverse angstroms.
    q_min : 
        Smallest wavevector kept, in inverse angstroms.
    q_max : 
        Largest wavevector kept, in inverse angstroms.
    min_prominence : 
        Minimum peak prominence passed to the peak finder.
    resolution_multiple : 
        A peak narrower than this many resolution elements is flagged.

    Returns
    -------
    result
        Peaks in the radial structure factor and a flag for peaks narrower than the resolution.
    """
    q = np.asarray(q, dtype=float)
    Sq = np.asarray(Sq, dtype=float)
    good = (
        np.isfinite(q)
        & np.isfinite(Sq)
        & (q >= q_min)
        & (q <= q_max)
    )
    q_use = q[good]
    Sq_use = Sq[good]
    if len(q_use) < 5:
        return pd.DataFrame()
    dq_bin = np.median(
        np.diff(q_use)
    )
    peaks, props = find_peaks(
        Sq_use,
        prominence=min_prominence,
    )
    if len(peaks) == 0:
        return pd.DataFrame()
    widths = peak_widths(
        Sq_use,
        peaks,
        rel_height=0.5,
    )
    width_bins = widths[0]
    records = []
    for i, peak_index in enumerate(peaks):
        fwhm_q = (
            width_bins[i]
            * dq_bin
        )
        width_ratio = (
            fwhm_q
            / dq_resolution
        )
        xi_A = (
            2.0 * np.pi / fwhm_q
            if fwhm_q > 0
            else np.nan
        )
        bragg_like = bool(
            width_ratio
            <= resolution_multiple
        )
        records.append({
            "q_peak_Ainv":
                q_use[peak_index],
            "Sq_peak":
                Sq_use[peak_index],
            "prominence":
                props["prominences"][i],
            "FWHM_Ainv":
                fwhm_q,
            "dq_resolution_Ainv":
                dq_resolution,
            "FWHM_over_resolution":
                width_ratio,
            "correlation_length_A":
                xi_A,
            "narrow_peak_flag":
                bragg_like,
        })
    return pd.DataFrame(
        records
    )

def angular_order_on_ring(
    Sq_grid,
    fft_info,
    q0,
    dq_shell=0.08,
    orders=(2, 4, 6),
):
    """Angular Fourier amplitudes on one reciprocal-space ring.

    Parameters
    ----------
    Sq_grid : 
        Two-dimensional structure factor on the FFT grid.
    fft_info : 
        Wavevector grids returned by the FFT setup.
    q0 : 
        Center of a reciprocal-space ring, in inverse angstroms.
    dq_shell : 
        Half-width of the ring used for angular order, in inverse angstroms.
    orders : 
        Angular Fourier orders, typically 2, 4, and 6.

    Returns
    -------
    result
        Angular Fourier amplitudes on one reciprocal-space ring.
    """
    QX = fft_info["QX"]
    QY = fft_info["QY"]
    qmag = fft_info["qmag"]
    phi = np.arctan2(
        QY,
        QX
    )
    shell = (
        (qmag >= q0 - dq_shell)
        &
        (qmag <= q0 + dq_shell)
    )
    intensities = (
        Sq_grid[shell]
    )
    angles = (
        phi[shell]
    )
    good = (
        np.isfinite(intensities)
        &
        np.isfinite(angles)
    )
    intensities = (
        intensities[good]
    )
    angles = (
        angles[good]
    )
    if len(intensities) == 0:
        return {}
    denominator = np.sum(
        intensities
    )
    results = {}
    for m in orders:
        numerator = np.abs(
            np.sum(
                intensities
                * np.exp(
                    1j * m * angles
                )
            )
        )
        results[f"A{m}"] = (
            numerator
            / denominator
            if denominator > 0
            else np.nan
        )
    mean_intensity = np.mean(
        intensities
    )
    max_intensity = np.max(
        intensities
    )
    results["spot_to_mean_ratio"] = (
        max_intensity / mean_intensity
        if mean_intensity > 0.0
        else np.nan
    )
    return results

def mean_and_sd(sum_x, sum_x2, count):
    """Mean and standard deviation from the stored sums.

    Parameters
    ----------
    sum_x : 
        Sum of the observations in each bin.
    sum_x2 : 
        Sum of squared observations in each bin.
    count : 
        Number of observations in each bin.

    Returns
    -------
    result
        Mean and standard deviation from the stored sums.
    """
    mean = np.divide(
        sum_x,
        count,
        out=np.full_like(sum_x, np.nan, dtype=float),
        where=count > 0,
    )
    mean2 = np.divide(
        sum_x2,
        count,
        out=np.full_like(sum_x2, np.nan, dtype=float),
        where=count > 0,
    )
    var = mean2 - mean**2
    var = np.clip(var, 0.0, None)
    sd = np.sqrt(var)
    return mean, sd

def main(argv=None):
    """Run the local-order, CHILL+, and structure-factor analysis.

    Parameters
    ----------
    argv : 
        Arguments excluding the program name. Defaults to sys.argv.

    Returns
    -------
    result
        Run the local-order, CHILL+, and structure-factor analysis.
    """
    global topology, trajectory, water_resname, oxygen_name, Lx, Ly, Lz, z_min, z_max, z_window_width, frame_stride, max_waters_per_frame, write_per_water, nearest_k, lsi_cutoff, theta_bins, theta_centers, rng, write_per_water_distributions, max_distribution_records_per_metric, calculate_depth_resolved_sq, gibbs_abs_z, sq_depth_step, sq_slice_width, sq_grid_spacing, sq_q_min, sq_q_max, sq_dq, sq_min_waters_per_side, save_sq_side_specific, calculate_bragg_diagnostics, bragg_depth_A, bragg_q_min, bragg_q_max, bragg_min_prominence, bragg_resolution_multiple, bragg_ring_halfwidth, bragg_angular_orders, debug_sq, debug_sq_first_n_frames, u, O_atoms, O_indices, z_edges, z_centers, n_z, sum_q, sum_q2, sum_sk, sum_sk2, sum_lsi, sum_lsi2, counts_q, counts_sk, counts_lsi, counts, theta_hist_by_z, chill_counts_by_z, q_records, sk_records, lsi_records, n_frames_used, sq_depths, sq_fft_info, sq_q_centers, sq_symmetric_frames, sq_plus_frames, sq_minus_frames, sq_N_plus, sq_N_minus, depth, bragg_Sq2D_frames, dq_x_resolution, dq_y_resolution, dq_resolution, ts, O_pos, z, n_O, structure_type, box, O_pbc, tree, nbrs4, dist4, ref_pos, neighbor_pos, vecs, norms, vecs_unit, pairs, cosines, q_vals, rbar, sk_vals, d_all, lsi_vals, row, z_sample, z_bin, good, z_bin_all, good_all, label, theta_vals, k, x_all, y_all, z_rel, sq_half_width, sample_idx, dists, nearest, mask, arr, acc, acc2, cnt, valid_arr, z_frame_midpoint, z_midpoint_shifted, oxygen_index, z_coord, zb, zc, hist, _, center_mask, n_center, plus_mask, minus_mask, n_plus, n_minus, Sq_plus, Sq_minus, result_center, Sq_center, Sq_symmetric, a, b, reasons, Sq_grid_center, mean_q, sd_q, mean_sk, sd_sk, mean_lsi, sd_lsi, profile, theta_dist_df, w_idx, total, bin_width, p_theta, q_df, sk_df, lsi_df, total_chill, chill_df, sq_records, sq_df, sq_output_file, symmetric_curves, symmetric_array, Sq_mean, mean_N_plus, mean_N_minus, Sq2D_array, Sq2D_mean, zero_mask, center_curves, center_array, center_Sq_mean, bragg_peaks_df, angular_records, Sq_sd, plus_array, minus_array, Sq_plus_mean, Sq_minus_mean, Sq_plus_sd, Sq_minus_sd, i_q, q_value, Sq2D_sd, q0, angular_metrics, angular_df, m
    parser = argparse.ArgumentParser(
        description="Local order, CHILL+, and depth-resolved in-plane S(q)."
    )
    parser.add_argument("--topology", required=True, help="Amber prmtop.")
    parser.add_argument("--trajectory", required=True, help="Centered slab trajectory.")
    parser.add_argument("--out", default=".", help="Directory for CSV and NPZ outputs.")
    parser.add_argument("--water-resname", default="WAT")
    parser.add_argument("--oxygen-name", default="O")
    parser.add_argument("--z-min", type=float, default=-100.0)
    parser.add_argument("--z-max", type=float, default=100.0)
    parser.add_argument("--z-window", type=float, default=10)
    parser.add_argument(
        "--stride",
        type=int,
        default=50,
        help="Keep every Nth stored frame. 50 is 0.5 ns on a 10 ps hold. Consecutive frames are correlated.",
    )
    parser.add_argument("--max-waters", type=int, default=20000)
    parser.add_argument("--nearest-k", type=int, default=25)
    parser.add_argument("--lsi-cutoff", type=float, default=3.7)
    parser.add_argument("--seed", type=int, default=SEEDS["local_order"])
    parser.add_argument("--no-sq", action="store_true")
    parser.add_argument("--no-bragg", action="store_true")
    parser.add_argument("--debug-sq", action="store_true")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)
    topology = args.topology
    trajectory = args.trajectory
    water_resname = args.water_resname
    oxygen_name = args.oxygen_name
    Lx = Ly = Lz = None
    z_min = args.z_min
    z_max = args.z_max
    z_window_width = args.z_window
    frame_stride = args.stride
    max_waters_per_frame = args.max_waters
    write_per_water = False
    nearest_k = args.nearest_k
    lsi_cutoff = args.lsi_cutoff
    theta_bins = np.arange(0.0, 182.0, 2.0)
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    rng = np.random.default_rng(args.seed)
    write_per_water_distributions = True
    max_distribution_records_per_metric = None
    # ============================================================
    # DEPTH-RESOLVED IN-PLANE STRUCTURE FACTOR S_parallel(q)
    # ============================================================
    calculate_depth_resolved_sq = not args.no_sq
    gibbs_abs_z = None
    # Depths:
    #
    #   z = 0
    #   z = +/-5
    #   z = +/-10
    #   ...
    #   Gibbs dividing surface
    #
    sq_depth_step = 5.0       # Angstrom
    # Thickness of each physical slice.
    #
    # A 5 A slice means:
    #
    # z=0     -> -2.5 to +2.5 A
    # z=5     -> +2.5 to +7.5 A
    # z=-5    -> -7.5 to -2.5 A
    #
    sq_slice_width = 5.0      # Angstrom
    # FFT grid spacing in x/y.
    #
    # 0.5 A gives a Nyquist q of ~6.3 A^-1.
    # This comfortably covers the principal water structure peaks.
    #
    sq_grid_spacing = SQ_GRID_SPACING_A
    # Radial q range
    sq_q_min = 0.250           # A^-1
    sq_q_max = 5.50           # A^-1
    sq_dq = 0.0250              # A^-1
    # Require enough waters for a meaningful slice S(q)
    sq_min_waters_per_side = 200
    # Save side-specific curves in addition to +/- averaged curve.
    #
    # Strongly recommended as a symmetry diagnostic.
    save_sq_side_specific = True
    # ============================================================
    # BRAGG / CRYSTALLINITY DIAGNOSTICS
    # ============================================================
    calculate_bragg_diagnostics = not args.no_bragg
    # Analyze central slab only for primary crystallinity test.
    bragg_depth_A = 0.0
    # Ignore very-low-q density/interface fluctuations.
    bragg_q_min = 1.0
    bragg_q_max = 5.0
    # Peak-finding settings for radially averaged S(q).
    bragg_min_prominence = 0.05
    # A peak only a few reciprocal-space resolution elements wide
    # is potentially Bragg-like.
    bragg_resolution_multiple = 2.5
    # Shell half-width for angular anisotropy measurements.
    bragg_ring_halfwidth = 0.08
    # m=6 is particularly useful for hexagonal orientational order.
    bragg_angular_orders = [2, 4, 6]
    # Debug output only; does not change trajectory length or analysis logic.
    debug_sq = args.debug_sq
    debug_sq_first_n_frames = 1
    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================
    # ============================================================
    # LOAD TRAJECTORY
    # ============================================================
    u = mda.Universe(topology, trajectory)
    O_atoms = u.select_atoms(f"resname {water_resname} and name {oxygen_name}")
    O_indices = O_atoms.indices
    print(f"O atoms analyzed: {len(O_indices)}")
    stored_times = [float(ts.time) for ts in u.trajectory]
    stored_spacing = frame_spacing_ps(stored_times)
    print(
        f"Stored frame spacing {stored_spacing:g} ps; "
        f"stride {frame_stride} analyzes every {stored_spacing * frame_stride:g} ps."
    )
    surfaces = []
    for ts in u.trajectory[::frame_stride]:
        box = require_film(len(O_indices), ts.dimensions[:3])
        _, _, bounds, _ = slab_geometry(
            np.asarray(O_atoms.positions, dtype=float),
            box,
            bulk_halfwidth=BULK_HALFWIDTH_A,
        )
        surfaces.append(0.5 * (abs(bounds[0]) + abs(bounds[1])))
    if not surfaces:
        raise RuntimeError("Trajectory has no frames.")
    Lx, Ly, Lz = [float(value) for value in require_film(len(O_indices), u.trajectory[0].dimensions[:3])]
    gibbs_abs_z = float(np.mean(surfaces))
    print(f"Half-density surface from this trajectory: |z| = {gibbs_abs_z:.3f} A")
    # ============================================================
    # Z WINDOWS
    # ============================================================
    z_edges = np.arange(z_min, z_max + z_window_width, z_window_width)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    n_z = len(z_centers)
    sum_q = np.zeros(n_z)
    sum_q2 = np.zeros(n_z)
    sum_sk = np.zeros(n_z)
    sum_sk2 = np.zeros(n_z)
    sum_lsi = np.zeros(n_z)
    sum_lsi2 = np.zeros(n_z)
    counts_q = np.zeros(n_z)
    counts_sk = np.zeros(n_z)
    counts_lsi = np.zeros(n_z)
    counts = np.zeros(n_z)
    theta_hist_by_z = np.zeros((n_z, len(theta_centers)))
    chill_counts_by_z = {}
    band_names = ("interior", "awi", "awi_outer")
    band_sum = {name: {"q": 0.0, "Sk": 0.0, "LSI": 0.0} for name in band_names}
    band_count = {name: {"q": 0, "Sk": 0, "LSI": 0} for name in band_names}
    band_chill = {name: {} for name in band_names}
    q_records = []
    sk_records = []
    lsi_records = []
    n_frames_used = 0
    # ============================================================
    # STRUCTURE-FACTOR HELPER FUNCTIONS
    # ============================================================
    # ============================================================
    # INITIALIZE DEPTH-RESOLVED STRUCTURE FACTOR
    # ============================================================
    if calculate_depth_resolved_sq:
        sq_depths = make_sq_depths(
            gibbs_abs_z=gibbs_abs_z,
            depth_step=sq_depth_step,
        )
        sq_fft_info = prepare_sq2d_fft_grid(
            Lx=Lx,
            Ly=Ly,
            grid_spacing=sq_grid_spacing,
            q_min=sq_q_min,
            q_max=sq_q_max,
            dq=sq_dq,
        )
        sq_q_centers = sq_fft_info["q_centers"]
        sq_symmetric_frames = {float(depth): [] for depth in sq_depths}
        sq_plus_frames = {float(depth): [] for depth in sq_depths}
        sq_minus_frames = {float(depth): [] for depth in sq_depths}
        sq_N_plus = {float(depth): [] for depth in sq_depths}
        sq_N_minus = {float(depth): [] for depth in sq_depths}
        print("\nDepth-resolved S_parallel(q) enabled")
        print(f"Gibbs dividing surface: |z_G| = {gibbs_abs_z:.3f} A")
        print("S(q) depth centers:", sq_depths)
        print(f"FFT grid: {sq_fft_info['nx']} x {sq_fft_info['ny']}")
        print(f"q range: {sq_q_min:.2f} - {sq_q_max:.2f} A^-1")
        print(f"Minimum waters per slice: {sq_min_waters_per_side}")
    if calculate_depth_resolved_sq and calculate_bragg_diagnostics:
        bragg_Sq2D_frames = []
        dq_x_resolution = 2.0 * np.pi / Lx
        dq_y_resolution = 2.0 * np.pi / Ly
        dq_resolution = 0.5 * (dq_x_resolution + dq_y_resolution)
        print(
            f"Bragg reciprocal-space resolution: "
            f"dqx={dq_x_resolution:.5f}, "
            f"dqy={dq_y_resolution:.5f}, "
            f"mean={dq_resolution:.5f} A^-1"
        )
    # ============================================================
    # MAIN LOOP
    # ============================================================
    for ts in u.trajectory[::frame_stride]:
        Lx, Ly, Lz = [float(value) for value in require_film(len(O_indices), ts.dimensions[:3])]
        n_frames_used += 1
        O_pos = u.atoms.positions[O_indices]
        z = O_pos[:, 2]
        # ========================================================
        # DEPTH-RESOLVED IN-PLANE STRUCTURE FACTOR
        # ========================================================
        if calculate_depth_resolved_sq:
            x_all = O_pos[:, 0].astype(float)
            y_all = O_pos[:, 1].astype(float)
            # Preserve the intended full-run behavior: this trajectory is
            # already centered, so slice selection uses raw z coordinates.
            z_rel = z.astype(float)
            sq_half_width = 0.5 * sq_slice_width
            # DEBUG ONLY. These diagnostics do not alter coordinates,
            # frame selection, masks, or trajectory length.
            if debug_sq and n_frames_used <= debug_sq_first_n_frames:
                z_frame_midpoint = 0.5 * (np.min(z) + np.max(z))
                z_midpoint_shifted = z.astype(float) - z_frame_midpoint
                print("\n" + "=" * 70)
                print(f"S(q) DEBUG -- analyzed frame {n_frames_used}, trajectory frame {ts.frame}")
                print("=" * 70)
                print(f"Raw z min        = {np.min(z):.3f} A")
                print(f"Raw z max        = {np.max(z):.3f} A")
                print(f"Raw z mean       = {np.mean(z):.3f} A")
                print(f"Raw z median     = {np.median(z):.3f} A")
                print(f"min/max midpoint = {z_frame_midpoint:.3f} A")
                print(
                    f"N in raw center slice |z| <= {sq_half_width:.2f} A: "
                    f"{np.sum(np.abs(z_rel) <= sq_half_width)}"
                )
                print(
                    f"N in hypothetical midpoint-centered slice: "
                    f"{np.sum(np.abs(z_midpoint_shifted) <= sq_half_width)}"
                )
                print(
                    "NOTE: midpoint-centered count is diagnostic only; "
                    "analysis continues with raw centered z coordinates."
                )
            for depth in sq_depths:
                depth = float(depth)
                if np.isclose(depth, 0.0):
                    center_mask = np.abs(z_rel) <= sq_half_width
                    n_center = int(np.sum(center_mask))
                    if debug_sq and n_frames_used <= debug_sq_first_n_frames:
                        print(
                            f"S(q) depth {depth:6.2f} A: N_center={n_center} "
                            f"({'OK' if n_center >= sq_min_waters_per_side else 'BELOW MIN'})"
                        )
                    if calculate_bragg_diagnostics:
                        result_center = calculate_sq2d_slice(
                            x=x_all[center_mask],
                            y=y_all[center_mask],
                            Lx=Lx,
                            Ly=Ly,
                            fft_info=sq_fft_info,
                            min_waters=sq_min_waters_per_side,
                            return_grid=True,
                        )
                        if result_center is not None:
                            Sq_center, Sq_grid_center = result_center
                            bragg_Sq2D_frames.append(Sq_grid_center)
                        else:
                            Sq_center = None
                    else:
                        Sq_center = calculate_sq2d_slice(
                            x=x_all[center_mask],
                            y=y_all[center_mask],
                            Lx=Lx,
                            Ly=Ly,
                            fft_info=sq_fft_info,
                            min_waters=sq_min_waters_per_side,
                        )
                    if Sq_center is not None:
                        sq_symmetric_frames[depth].append(Sq_center)
                        sq_plus_frames[depth].append(Sq_center)
                        sq_minus_frames[depth].append(Sq_center)
                        sq_N_plus[depth].append(n_center)
                        sq_N_minus[depth].append(n_center)
                        if debug_sq and n_frames_used <= debug_sq_first_n_frames:
                            print("    -> center S(q) accepted and stored")
                    elif debug_sq and n_frames_used <= debug_sq_first_n_frames:
                        print("    -> center S(q) rejected: too few waters")
                else:
                    plus_mask = np.abs(z_rel - depth) <= sq_half_width
                    minus_mask = np.abs(z_rel + depth) <= sq_half_width
                    n_plus = int(np.sum(plus_mask))
                    n_minus = int(np.sum(minus_mask))
                    if debug_sq and n_frames_used <= debug_sq_first_n_frames:
                        print(
                            f"S(q) depth {depth:6.2f} A: N+={n_plus}, N-={n_minus} "
                            f"(+ {'OK' if n_plus >= sq_min_waters_per_side else 'BELOW MIN'}, "
                            f"- {'OK' if n_minus >= sq_min_waters_per_side else 'BELOW MIN'})"
                        )
                    Sq_plus = calculate_sq2d_slice(
                        x=x_all[plus_mask],
                        y=y_all[plus_mask],
                        Lx=Lx,
                        Ly=Ly,
                        fft_info=sq_fft_info,
                        min_waters=sq_min_waters_per_side,
                    )
                    Sq_minus = calculate_sq2d_slice(
                        x=x_all[minus_mask],
                        y=y_all[minus_mask],
                        Lx=Lx,
                        Ly=Ly,
                        fft_info=sq_fft_info,
                        min_waters=sq_min_waters_per_side,
                    )
                    if Sq_plus is not None and Sq_minus is not None:
                        Sq_symmetric = 0.5 * (Sq_plus + Sq_minus)
                        sq_symmetric_frames[depth].append(Sq_symmetric)
                        sq_plus_frames[depth].append(Sq_plus)
                        sq_minus_frames[depth].append(Sq_minus)
                        sq_N_plus[depth].append(n_plus)
                        sq_N_minus[depth].append(n_minus)
                        if debug_sq and n_frames_used <= debug_sq_first_n_frames:
                            print("    -> +/- S(q) pair accepted and stored")
                    elif debug_sq and n_frames_used <= debug_sq_first_n_frames:
                        reasons = []
                        if Sq_plus is None:
                            reasons.append("+ side below minimum")
                        if Sq_minus is None:
                            reasons.append("- side below minimum")
                        print("    -> pair rejected: " + ", ".join(reasons))
        n_O = len(O_pos)
        z_relative, depth, _bounds, center = slab_geometry(
            np.asarray(O_pos, dtype=float), (Lx, Ly, Lz)
        )
        structure_type = chillplus_classify_oxygen_positions(
            O_pos=O_pos,
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            z_center=center,
        )
        if max_waters_per_frame is not None and n_O > max_waters_per_frame:
            sample_idx = rng.choice(n_O, size=max_waters_per_frame, replace=False)
        else:
            sample_idx = np.arange(n_O)
    # ============================================================
    # PERIODIC NEIGHBOR SEARCH
    # ============================================================
        box = np.array(
            [Lx, Ly, Lz],
            dtype=float,
        )
        # cKDTree with boxsize requires coordinates in [0, L)
        O_pbc = np.mod(
            O_pos,
            box,
        )
        tree = cKDTree(
            O_pbc,
            boxsize=box,
        )
        dists, nearest = tree.query(
            O_pbc[sample_idx],
            k=nearest_k,
        )
        nbrs4 = nearest[:, 1:5]
        dist4 = dists[:, 1:5]
        ref_pos = O_pbc[sample_idx]
        neighbor_pos = O_pbc[nbrs4]
        vecs = (
            neighbor_pos
            - ref_pos[:, None, :]
        )
        # Minimum-image convention
        vecs -= (
            box[None, None, :]
            * np.round(
                vecs
                / box[None, None, :]
            )
        )
        norms = np.linalg.norm(vecs, axis=2)
        vecs_unit = vecs / norms[:, :, None]
        q_vals = shared_q(vecs_unit)
        sk_vals = shared_sk(dist4)
        # ============================================================
        # LOCAL STRUCTURE INDEX
        # ============================================================
        # Distances to all queried neighbors except self.
        # IMPORTANT: retain neighbors beyond 3.7 A because the
        # first neighbor outside the cutoff, r_(n+1), is required
        # for the standard LSI definition.
        d_all = dists[:, 1:]
        lsi_vals = np.full(
            len(sample_idx),
            np.nan,
            dtype=float,
        )
        for row in range(len(sample_idx)):
            lsi_vals[row] = local_structure_index(
                d_all[row],
                cutoff=lsi_cutoff,
            )
        frame_masks = region_masks(z_relative, depth)
        for band in band_names:
            chosen = frame_masks[band][sample_idx]
            for key, values in (("q", q_vals), ("Sk", sk_vals), ("LSI", lsi_vals)):
                finite = chosen & np.isfinite(values)
                band_sum[band][key] += float(np.sum(values[finite]))
                band_count[band][key] += int(np.count_nonzero(finite))
            for label in np.unique(structure_type):
                label = int(label)
                hits = int(np.count_nonzero(frame_masks[band] & (structure_type == label)))
                band_chill[band][label] = band_chill[band].get(label, 0) + hits
        z_sample = z[sample_idx]
        z_bin = np.digitize(z_sample, z_edges) - 1
        good = (z_bin >= 0) & (z_bin < n_z)
        if write_per_water_distributions:
            for k in np.where(good)[0]:
                oxygen_index = int(
                    O_indices[
                    sample_idx[k]
                    ]
                )
                z_coord = float(
                z_sample[k]
                )
                zb = z_bin[k]
                zc = z_centers[zb]
                if np.isfinite(q_vals[k]):
                    q_records.append({
                        "frame": ts.frame,
                        "oxygen_index": oxygen_index,
                        "z_coord": z_coord,
                        "z_center": zc,
                        "z_lower": z_edges[zb],
                        "z_upper": z_edges[zb + 1],
                        "q": q_vals[k],
                    })
                if np.isfinite(sk_vals[k]):
                    sk_records.append({
                        "frame": ts.frame,
                        "oxygen_index": oxygen_index,
                        "z_coord": z_coord,
                        "z_center": zc,
                        "z_lower": z_edges[zb],
                        "z_upper": z_edges[zb + 1],
                        "Sk": sk_vals[k],
                    })
                if np.isfinite(lsi_vals[k]):
                    lsi_records.append({
                        "frame": ts.frame,
                        "oxygen_index": oxygen_index,
                        "z_coord": z_coord,
                        "z_center": zc,
                        "z_lower": z_edges[zb],
                        "z_upper": z_edges[zb + 1],
                        "LSI": lsi_vals[k],
                    })
        z_bin_all = np.digitize(z, z_edges) - 1
        good_all = (z_bin_all >= 0) & (z_bin_all < n_z)
        if n_frames_used == 1:
            print("CHILL+ labels found:", np.unique(structure_type, return_counts=True))
        for label in np.unique(structure_type):
            if label not in chill_counts_by_z:
                chill_counts_by_z[label] = np.zeros(n_z)
            mask = good_all & (structure_type == label)
            chill_counts_by_z[label] += np.bincount(
                z_bin_all[mask],
                minlength=n_z
            )
        counts += np.bincount(z_bin[good], minlength=n_z)
        for arr, acc, acc2, cnt in [
            (q_vals, sum_q, sum_q2, counts_q),
            (sk_vals, sum_sk, sum_sk2, counts_sk),
            (lsi_vals, sum_lsi, sum_lsi2, counts_lsi),
        ]:
            valid_arr = good & np.isfinite(arr)
            acc += np.bincount(
                z_bin[valid_arr],
                weights=arr[valid_arr],
                minlength=n_z
            )
            acc2 += np.bincount(
                z_bin[valid_arr],
                weights=arr[valid_arr] ** 2,
                minlength=n_z
            )
            cnt += np.bincount(
                z_bin[valid_arr],
                minlength=n_z
            )
        theta_vals = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
        for k in range(len(sample_idx)):
            if not good[k]:
                continue
            hist, _ = np.histogram(theta_vals[k], bins=theta_bins, density=False)
            theta_hist_by_z[z_bin[k]] += hist
    # ============================================================
    # Z-RESOLVED PROFILES
    # ============================================================
    mean_q, sd_q = mean_and_sd(sum_q, sum_q2, counts_q)
    mean_sk, sd_sk = mean_and_sd(sum_sk, sum_sk2, counts_sk)
    mean_lsi, sd_lsi = mean_and_sd(sum_lsi, sum_lsi2, counts_lsi)
    profile = pd.DataFrame({
        "z_center": z_centers,
        "z_lower": z_edges[:-1],
        "z_upper": z_edges[1:],
        "count": counts,
        "mean_q_orientational_tetrahedral": mean_q,
        "sd_q_orientational_tetrahedral": sd_q,
        "mean_Sk_translational_tetrahedral": mean_sk,
        "sd_Sk_translational_tetrahedral": sd_sk,
        "mean_LSI": mean_lsi,
        "sd_LSI": sd_lsi,
    })
    profile.to_csv("water_local_order_z_profile_10A.csv", index=False)
    # Individual requested CSVs
    profile[[
        "z_center", "z_lower", "z_upper", "count",
        "mean_q_orientational_tetrahedral",
        "sd_q_orientational_tetrahedral",
    ]].to_csv("orientational_tetrahedral_q_z_profile_10A.csv", index=False)
    profile[[
        "z_center", "z_lower", "z_upper", "count",
        "mean_Sk_translational_tetrahedral",
        "sd_Sk_translational_tetrahedral",
    ]].to_csv("translational_tetrahedral_Sk_z_profile_10A.csv", index=False)
    profile[[
        "z_center", "z_lower", "z_upper", "count",
        "mean_LSI",
        "sd_LSI",
    ]].to_csv("LSI_z_profile_10A.csv", index=False)
    theta_dist_df = pd.DataFrame({"theta_deg": theta_centers})
    for w_idx in range(n_z):
        total = theta_hist_by_z[w_idx].sum()
        if total > 0:
            # Convert counts to probability density P(theta)
            bin_width = theta_bins[1] - theta_bins[0]
            p_theta = theta_hist_by_z[w_idx] / (total * bin_width)
        else:
            p_theta = np.full(len(theta_centers), np.nan)
        label = f"z_{z_edges[w_idx]:.0f}_to_{z_edges[w_idx + 1]:.0f}"
        theta_dist_df[label] = p_theta
    theta_dist_df.to_csv(
        "water_water_angular_distribution_Ptheta_by_z_window.csv",
        index=False
    )
    if write_per_water_distributions:
        q_df = pd.DataFrame(q_records)
        sk_df = pd.DataFrame(sk_records)
        lsi_df = pd.DataFrame(lsi_records)
        q_df.to_csv("q_per_water_distribution.csv", index=False)
        sk_df.to_csv("Sk_per_water_distribution.csv", index=False)
        lsi_df.to_csv("LSI_per_water_distribution.csv", index=False)
        print("  q_per_water_distribution.csv")
        print("  Sk_per_water_distribution.csv")
        print("  LSI_per_water_distribution.csv")
    # ============================================================
    # SAVE CHILL+ Z PROFILE
    # ============================================================
    total_chill = np.zeros(n_z)
    for label, arr in chill_counts_by_z.items():
        total_chill += arr
    chill_df = pd.DataFrame({
        "z_center": z_centers,
        "z_lower": z_edges[:-1],
        "z_upper": z_edges[1:],
        "total_waters": total_chill,
    })
    for label, arr in chill_counts_by_z.items():
        chill_df[f"count_type_{label}"] = arr
        chill_df[f"fraction_type_{label}"] = np.divide(
            arr,
            total_chill,
            out=np.full_like(arr, np.nan, dtype=float),
            where=total_chill > 0,
        )
    chill_df.to_csv("CHILLplus_z_profile_10A.csv", index=False)
    band_rows = []
    for band in band_names:
        row = {
            "region": band,
            "band_A": {"interior": "|z|<=15", "awi": "3-12", "awi_outer": "0-6"}[band],
        }
        for key in ("q", "Sk", "LSI"):
            count = band_count[band][key]
            row[f"n_{key}"] = count
            row[f"mean_{key}"] = band_sum[band][key] / count if count else np.nan
        total = int(sum(band_chill[band].values()))
        row["n_chill"] = total
        for label, count in sorted(band_chill[band].items()):
            row[f"chill_fraction_{label}"] = count / total if total else np.nan
        band_rows.append(row)
    pd.DataFrame(band_rows).to_csv("band_order_summary.csv", index=False)
    # ============================================================
    # SAVE DEPTH-RESOLVED STRUCTURE FACTOR
    # ============================================================
    if calculate_depth_resolved_sq:
        if debug_sq:
            print("\n" + "=" * 70)
            print("S(q) FRAME-LEVEL STORAGE SUMMARY")
            print("=" * 70)
            for depth in sq_depths:
                depth = float(depth)
                print(
                    f"|z|={depth:6.2f} A : "
                    f"symmetric={len(sq_symmetric_frames[depth])}, "
                    f"plus={len(sq_plus_frames[depth])}, "
                    f"minus={len(sq_minus_frames[depth])}"
                )
        sq_records = []
        for depth in sq_depths:
            depth = float(depth)
            symmetric_curves = sq_symmetric_frames[depth]
            if len(symmetric_curves) == 0:
                print(f"WARNING: no usable S(q) curves for |z| = {depth:.2f} A")
                continue
            symmetric_array = np.stack(symmetric_curves, axis=0)
            Sq_mean = np.nanmean(symmetric_array, axis=0)
            if symmetric_array.shape[0] > 1:
                Sq_sd = np.nanstd(symmetric_array, axis=0, ddof=1)
            else:
                Sq_sd = np.zeros_like(Sq_mean)
            if save_sq_side_specific:
                plus_array = np.stack(sq_plus_frames[depth], axis=0)
                minus_array = np.stack(sq_minus_frames[depth], axis=0)
                Sq_plus_mean = np.nanmean(plus_array, axis=0)
                Sq_minus_mean = np.nanmean(minus_array, axis=0)
                if plus_array.shape[0] > 1:
                    Sq_plus_sd = np.nanstd(plus_array, axis=0, ddof=1)
                    Sq_minus_sd = np.nanstd(minus_array, axis=0, ddof=1)
                else:
                    Sq_plus_sd = np.zeros_like(Sq_plus_mean)
                    Sq_minus_sd = np.zeros_like(Sq_minus_mean)
            else:
                Sq_plus_mean = np.full_like(Sq_mean, np.nan)
                Sq_minus_mean = np.full_like(Sq_mean, np.nan)
                Sq_plus_sd = np.full_like(Sq_mean, np.nan)
                Sq_minus_sd = np.full_like(Sq_mean, np.nan)
            mean_N_plus = np.mean(sq_N_plus[depth])
            mean_N_minus = np.mean(sq_N_minus[depth])
            for i_q, q_value in enumerate(sq_q_centers):
                sq_records.append({
                    "depth_A": depth,
                    "is_gibbs_surface": bool(np.isclose(depth, gibbs_abs_z)),
                    "slice_width_A": sq_slice_width,
                    "q_Ainv": q_value,
                    "Sq_mean": Sq_mean[i_q],
                    "Sq_sd": Sq_sd[i_q],
                    "Sq_plus_mean": Sq_plus_mean[i_q],
                    "Sq_plus_sd": Sq_plus_sd[i_q],
                    "Sq_minus_mean": Sq_minus_mean[i_q],
                    "Sq_minus_sd": Sq_minus_sd[i_q],
                    "n_frames": symmetric_array.shape[0],
                    "mean_N_plus": mean_N_plus,
                    "mean_N_minus": mean_N_minus,
                    "gibbs_abs_z_A": gibbs_abs_z,
                    "Lx_A": Lx,
                    "Ly_A": Ly,
                    "grid_spacing_A": sq_grid_spacing,
                })
        sq_df = pd.DataFrame(sq_records)
        sq_output_file = "water_Sq2D_by_depth.csv"
        sq_df.to_csv(sq_output_file, index=False)
        print(f"Saved: {sq_output_file}")
        if (
            calculate_bragg_diagnostics
            and len(bragg_Sq2D_frames) > 0
            and len(sq_symmetric_frames[float(bragg_depth_A)]) > 0
        ):
            if debug_sq:
                print(
                    f"Bragg diagnostics: {len(bragg_Sq2D_frames)} 2D center maps; "
                    f"{len(sq_symmetric_frames[float(bragg_depth_A)])} radial center curves."
                )
            Sq2D_array = np.stack(bragg_Sq2D_frames, axis=0)
            Sq2D_mean = np.nanmean(Sq2D_array, axis=0)
            if Sq2D_array.shape[0] > 1:
                Sq2D_sd = np.nanstd(Sq2D_array, axis=0, ddof=1)
            else:
                Sq2D_sd = np.zeros_like(Sq2D_mean)
            zero_mask = sq_fft_info["qmag"] < 1.0e-12
            Sq2D_mean[zero_mask] = np.nan
            np.savez_compressed(
                "water_Sq2D_center_bragg_map.npz",
                Sq_mean=Sq2D_mean,
                Sq_sd=Sq2D_sd,
                qx=sq_fft_info["qx"],
                qy=sq_fft_info["qy"],
                QX=sq_fft_info["QX"],
                QY=sq_fft_info["QY"],
                qmag=sq_fft_info["qmag"],
                Lx_A=Lx,
                Ly_A=Ly,
                dq_x_resolution_Ainv=2.0 * np.pi / Lx,
                dq_y_resolution_Ainv=2.0 * np.pi / Ly,
            )
            print("Saved: water_Sq2D_center_bragg_map.npz")
            center_curves = sq_symmetric_frames[float(bragg_depth_A)]
            center_array = np.stack(center_curves, axis=0)
            center_Sq_mean = np.nanmean(center_array, axis=0)
            bragg_peaks_df = analyze_radial_bragg_peaks(
                q=sq_q_centers,
                Sq=center_Sq_mean,
                dq_resolution=dq_resolution,
                q_min=bragg_q_min,
                q_max=bragg_q_max,
                min_prominence=bragg_min_prominence,
                resolution_multiple=bragg_resolution_multiple,
            )
            bragg_peaks_df.to_csv("water_bragg_peak_diagnostics.csv", index=False)
            print(
                f"Saved: water_bragg_peak_diagnostics.csv "
                f"({len(bragg_peaks_df)} detected radial peaks)"
            )
            angular_records = []
            for _, row in bragg_peaks_df.iterrows():
                q0 = float(row["q_peak_Ainv"])
                angular_metrics = angular_order_on_ring(
                    Sq_grid=Sq2D_mean,
                    fft_info=sq_fft_info,
                    q0=q0,
                    dq_shell=bragg_ring_halfwidth,
                    orders=bragg_angular_orders,
                )
                angular_records.append({"q_peak_Ainv": q0, **angular_metrics})
            if len(angular_records) > 0:
                angular_df = pd.DataFrame(angular_records)
            else:
                angular_df = pd.DataFrame(
                    columns=[
                        "q_peak_Ainv",
                        *[f"A{m}" for m in bragg_angular_orders],
                        "spot_to_mean_ratio",
                    ]
                )
            angular_df.to_csv("water_bragg_angular_anisotropy.csv", index=False)
            print("Saved: water_bragg_angular_anisotropy.csv")
        else:
            if calculate_bragg_diagnostics:
                print(
                    "WARNING: insufficient center-slab data for Bragg diagnostics. "
                    f"2D maps={len(bragg_Sq2D_frames)}, "
                    f"center radial curves="
                    f"{len(sq_symmetric_frames.get(float(bragg_depth_A), []))}."
                )
    Path("metadata.json").write_text(json.dumps({
        "seed": args.seed,
        "stride": args.stride,
        "stride_note": "Default stride 50. On a 10 ps hold that is one sample per 0.5 ns.",
        "sq_grid_spacing_A": sq_grid_spacing,
        "z_window_A": args.z_window,
        "qstar_reference_Ainv": QSTAR_AINV,
        "bulk_halfwidth_A": BULK_HALFWIDTH_A,
        "half_density_z_A": None if gibbs_abs_z is None else float(gibbs_abs_z),
        "frames": n_frames_used,
        "neighbor_note": "q, Sk, and LSI use nearest neighbors with no distance cutoff; LSI is missing when fewer than two neighbors are within 3.7 A",
    }, indent=2))
    print("=" * 70)
    print("Analysis complete")
    print("=" * 70)
    print(f"Frames used: {n_frames_used}")
    print("Saved:")
    print("  water_local_order_z_profile_10A.csv")
    print("  orientational_tetrahedral_q_z_profile_10A.csv")
    print("  translational_tetrahedral_Sk_z_profile_10A.csv")
    print("  LSI_z_profile_10A.csv")
    print("  water_water_angular_distribution_Ptheta_by_z_window.csv")
    print("  CHILL+ labels found:", np.unique(structure_type, return_counts=True))
    print("=" * 70)
    if calculate_depth_resolved_sq:
        print(
            "  water_Sq2D_by_depth.csv"
        )


if __name__ == "__main__":
    main()
