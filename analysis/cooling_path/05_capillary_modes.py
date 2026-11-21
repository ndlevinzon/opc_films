#!/usr/bin/env python3
"""Capillary-wave autocorrelation of a stationary slab trajectory.

The script builds a coarse oxygen-density grid, locates the upper and lower
half-density surfaces, and writes the normalized complex Fourier-mode
correlation. It does not estimate a surface tension. Use it on an
equilibrated hold, and check the grid spacing before interpreting the decay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import BULK_HALFWIDTH_A
from opcfilm.config import CONFIG
from common.slab import slab_geometry, water_indices
from common.trajectory import frame_spacing_ps


def half_density_height(smoothed, z_centered, side, threshold):
    """Locate one interface on every lateral grid column.

    Parameters
    ----------
    smoothed : numpy.ndarray, shape (nx, ny, nz)
        Smoothed oxygen counts.
    z_centered : numpy.ndarray, shape (nz,)
        Bin centers relative to the slab center, in angstroms.
    side : {"upper", "lower"}
        Which interface to follow.
    threshold : float
        Density level, typically half the core plateau.

    Returns
    -------
    height : numpy.ndarray, shape (nx, ny)
        Interface height in angstroms. Missing columns are NaN.
    valid_fraction : float
        Fraction of columns with a crossing.
    """
    nx, ny, _ = smoothed.shape
    height = np.full((nx, ny), np.nan)
    sequence = np.flatnonzero(z_centered >= 0.0) if side == "upper" else np.flatnonzero(z_centered <= 0.0)[::-1]
    for ix in range(nx):
        for iy in range(ny):
            line = smoothed[ix, iy, :]
            values = line[sequence]
            crossings = np.flatnonzero((values[:-1] >= threshold) & (values[1:] < threshold))
            if len(crossings) == 0:
                continue
            jump = int(crossings[0])
            left = int(sequence[jump])
            right = int(sequence[jump + 1])
            height[ix, iy] = z_centered[left] + (z_centered[right] - z_centered[left]) * (
                threshold - line[left]
            ) / (line[right] - line[left])
    valid = np.isfinite(height)
    return height, float(valid.mean())


def frame_modes(oxygen, lengths, grid, smooth_sigma, q_mask):
    """Fourier amplitudes of both interfaces in one frame.

    Parameters
    ----------
    oxygen : numpy.ndarray, shape (n, 3)
        Oxygen coordinates.
    lengths : array_like, shape (3,)
        Box lengths.
    grid : tuple
        ``(nx, ny, nz, dx, dy, dz)``.
    smooth_sigma : tuple of float
        Gaussian widths in bins, ordered ``(xy, xy, z)``.
    q_mask : numpy.ndarray of bool
        Reciprocal-space modes to keep.

    Returns
    -------
    modes : dict
        ``upper`` and ``lower`` complex amplitudes on ``q_mask``.
    valid_fraction : float
        Smaller of the two surface coverage fractions.
    """
    nx, ny, nz, dx, dy, dz = grid
    lx, ly, lz = lengths
    coordinates = oxygen.astype(float).copy()
    coordinates[:, 0] %= lx
    coordinates[:, 1] %= ly
    _relative, _depth, _bounds, center = slab_geometry(coordinates, lengths)
    coordinates[:, 2] = (coordinates[:, 2] - center + lz / 2.0) % lz
    histogram = np.histogramdd(
        coordinates,
        bins=(nx, ny, nz),
        range=((0.0, lx), (0.0, ly), (0.0, lz)),
    )[0]
    smoothed = gaussian_filter(
        histogram,
        sigma=smooth_sigma,
        mode=("wrap", "wrap", "constant"),
    )
    z_centered = (np.arange(nz) + 0.5) * dz - lz / 2.0
    bulk = np.mean(smoothed.mean(axis=(0, 1))[np.abs(z_centered) < BULK_HALFWIDTH_A])
    if bulk <= 0.0:
        raise RuntimeError("The core of this frame has no oxygen density.")
    modes = {}
    coverage = []
    for side in ("upper", "lower"):
        height, fraction = half_density_height(smoothed, z_centered, side, 0.5 * bulk)
        coverage.append(fraction)
        if fraction < 0.98:
            raise RuntimeError(
                f"{side} interface is only {fraction:.2%} complete. "
                "Increase the grid spacing or the smoothing length."
            )
        missing = ~np.isfinite(height)
        height[missing] = np.nanmean(height)
        height = height - height.mean()
        modes[side] = np.fft.fft2(height)[q_mask] / (nx * ny)
    return modes, min(coverage)


def correlate_modes(heights, times, q_values, frame_dt_ps, max_lag_frames):
    """Normalized autocorrelation of each retained Fourier mode.

    Parameters
    ----------
    heights : dict
        ``upper`` and ``lower`` lists of complex amplitude vectors.
    times : sequence of float
        Frame times in picoseconds. Used only to confirm the series is long
        enough; lags are computed from the frame stride.
    q_values : numpy.ndarray
        Wavevector magnitude of each retained mode, in inverse angstroms.
    frame_dt_ps : float
        Time between analyzed frames, in picoseconds.
    max_lag_frames : int
        Largest lag, also limited to one third of the series.

    Returns
    -------
    correlations, spectrum : tuple of pandas.DataFrame
        Lag correlations and the mean squared amplitude of each mode.
    """
    if len(times) < 4:
        raise SystemExit("Need at least four frames.")
    correlations = []
    spectrum = []
    for side in ("upper", "lower"):
        series = np.asarray(heights[side], dtype=complex)
        for index, wavevector in enumerate(q_values):
            amplitude = series[:, index]
            spectrum.append(
                {
                    "interface": side,
                    "q_Ainv": float(wavevector),
                    "mean_h2_A2": float(np.mean(np.abs(amplitude) ** 2)),
                }
            )
        max_lag = min(max_lag_frames, len(series) // 3)
        norm = np.mean(np.abs(series) ** 2, axis=0)
        for lag in range(max_lag + 1):
            product = np.mean(np.conj(series[: len(series) - lag]) * series[lag:], axis=0)
            for index, wavevector in enumerate(q_values):
                if norm[index] <= 0.0:
                    continue
                value = product[index] / norm[index]
                correlations.append(
                    {
                        "interface": side,
                        "lag_frames": lag,
                        "lag_ps": lag * frame_dt_ps,
                        "q_Ainv": float(wavevector),
                        "C_real": float(np.real(value)),
                        "C_imag": float(np.imag(value)),
                    }
                )
    return pd.DataFrame(correlations), pd.DataFrame(spectrum)


def main(argv=None):
    """Compute interface-mode correlations for one trajectory.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments, excluding the program name.

    Returns
    -------
    int
        Process status code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", required=True)
    parser.add_argument("--traj", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--grid-A", type=float, default=2.5)
    parser.add_argument("--zbin-A", type=float, default=1.0)
    parser.add_argument("--sigma-xy-A", type=float, default=2.5)
    parser.add_argument("--sigma-z-A", type=float, default=1.50)
    parser.add_argument("--qmax", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=CONFIG.classifier.liquid_K)
    parser.add_argument("--max-lag-frames", type=int, default=120)
    args = parser.parse_args(argv)

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    universe = mda.Universe(args.top, args.traj, format="NCDF")
    oxygen_index, _hydrogen = water_indices(universe)
    oxygen_atoms = universe.atoms[oxygen_index]
    first = universe.trajectory[0]
    lx, ly, lz = first.dimensions[:3].astype(float)
    nx = max(8, round(lx / args.grid_A))
    ny = max(8, round(ly / args.grid_A))
    nz = int(np.ceil(lz / args.zbin_A))
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    wavevector = np.hypot(kx[:, None], ky[None, :])
    mask = (wavevector > 1e-8) & (wavevector <= args.qmax)
    sigma = (args.sigma_xy_A / dx, args.sigma_xy_A / dy, args.sigma_z_A / dz)
    grid = (nx, ny, nz, dx, dy, dz)
    heights = {"upper": [], "lower": []}
    times = []
    coverage = []
    for frame in universe.trajectory[:: args.stride]:
        modes, fraction = frame_modes(
            oxygen_atoms.positions, (lx, ly, lz), grid, sigma, mask
        )
        for side in ("upper", "lower"):
            heights[side].append(modes[side])
        times.append(float(frame.time))
        coverage.append(fraction)
    print(
        f"Frames {len(times)}; grid {nx} {ny} {nz}; q modes {int(mask.sum())}",
        flush=True,
    )
    frame_dt = frame_spacing_ps(times)
    correlations, spectrum = correlate_modes(
        heights, times, wavevector[mask], frame_dt, args.max_lag_frames
    )
    correlations.to_csv(destination / "capillary_mode_autocorrelation.csv", index=False)
    spectrum.to_csv(destination / "capillary_mode_spectrum.csv", index=False)
    metadata = {
        "trajectory": args.traj,
        "temperature_K": args.temperature,
        "grid": [nx, ny, nz],
        "frame_dt_ps": frame_dt,
        "frames": len(times),
        "qmax_Ainv": args.qmax,
        "minimum_valid_fraction": min(coverage) if coverage else None,
        "warning": (
            "A relaxation time read from this correlation is meaningful only "
            "for a stationary liquid surface."
        ),
    }
    (destination / "capillary_modes_metadata.json").write_text(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
