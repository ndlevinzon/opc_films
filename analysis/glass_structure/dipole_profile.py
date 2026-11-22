#!/usr/bin/env python3
"""Dipole orientation of water across a free-standing film.

The molecular dipole points from the oxygen to the midpoint of the two
hydrogens. ``cos theta`` is its projection on the slab normal. ``P2`` is
the second Legendre polynomial of that cosine. Both are averaged in slabs
along z.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import MDAnalysis as mda
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import PROFILE_BIN_A
from opcfilm.config import CONFIG


def water_indices(universe, resname, oxygen_name):
    """Collect oxygen and the two hydrogen indices of each water.

    OPC is a four-site model. The selection is ``name H*``, so the
    massless site is not treated as a hydrogen.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Topology and trajectory.
    resname : str
        Water residue name.
    oxygen_name : str
        Oxygen atom name.

    Returns
    -------
    oxygen, hydrogen1, hydrogen2 : tuple of numpy.ndarray
        Atom indices. Waters that do not have one oxygen and two
        hydrogens are skipped.
    """
    oxygen, hydrogen1, hydrogen2 = [], [], []
    for residue in universe.select_atoms(f"resname {resname}").residues:
        oxygens = residue.atoms.select_atoms(f"name {oxygen_name}")
        hydrogens = residue.atoms.select_atoms("name H*")
        if len(oxygens) == 1 and len(hydrogens) == 2:
            oxygen.append(oxygens.indices[0])
            hydrogen1.append(hydrogens.indices[0])
            hydrogen2.append(hydrogens.indices[1])
    return (
        np.asarray(oxygen, dtype=int),
        np.asarray(hydrogen1, dtype=int),
        np.asarray(hydrogen2, dtype=int),
    )


WATER_MASS_G_MOL = CONFIG.film.water_molar_mass_g_mol
AVOGADRO = CONFIG.film.avogadro_per_mol


def slab_center(z, length):
    """Center of a slab that may cross the periodic boundary in z.

    Parameters
    ----------
    z : numpy.ndarray
        Oxygen z coordinates, in angstroms.
    length : float
        Box length along z.

    Returns
    -------
    float
        Circular mean of ``z`` on the periodic interval.
    """
    angle = 2.0 * np.pi * z / length
    center = np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle)))
    return float(center * length / (2.0 * np.pi))


def accumulate_orientation(universe, indices, z_min, z_max, dz, frames=None):
    """Bin the dipole projection along z.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory to average.
    indices : tuple of numpy.ndarray
        Oxygen and two hydrogen index arrays.
    z_min, z_max, dz : float
        Histogram limits and bin width, in angstroms. z is measured from
        the slab center of each frame.
    frames : iterable of int, optional
        Frames to use. Default every frame.

    Returns
    -------
    pandas.DataFrame
        Columns ``z``, ``mean_cos_theta``, ``mean_P2``, ``count``,
        ``rho_g_cm3`` (oxygen-derived water mass density) and ``n_frames``.
    """
    oxygen, hydrogen1, hydrogen2 = indices
    if frames is None:
        frames = range(len(universe.trajectory))
    edges = np.arange(z_min, z_max + dz, dz)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centers)
    sum_cos = np.zeros(n_bins)
    sum_p2 = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    volume = 0.0
    n_frames = 0
    for frame in frames:
        timestep = universe.trajectory[frame]
        positions = universe.atoms.positions
        dipole = 0.5 * (positions[hydrogen1] + positions[hydrogen2]) - positions[oxygen]
        norm = np.linalg.norm(dipole, axis=1)
        valid = norm > 0.0
        cosine = np.zeros(len(norm))
        cosine[valid] = dipole[valid, 2] / norm[valid]
        order = 0.5 * (3.0 * cosine**2 - 1.0)
        lx, ly, lz = (float(value) for value in timestep.dimensions[:3])
        # z is measured from the slab center, wrapped into the box.
        z = positions[oxygen, 2]
        z = (z - slab_center(z, lz) + 0.5 * lz) % lz - 0.5 * lz
        volume += lx * ly * dz
        n_frames += 1
        bins = np.digitize(z, edges) - 1
        keep = (bins >= 0) & (bins < n_bins)
        sum_cos += np.bincount(bins[keep], weights=cosine[keep], minlength=n_bins)
        sum_p2 += np.bincount(bins[keep], weights=order[keep], minlength=n_bins)
        counts += np.bincount(bins[keep], minlength=n_bins)
    return pd.DataFrame(
        {
            "z": centers,
            "mean_cos_theta": np.divide(sum_cos, counts, out=np.full_like(sum_cos, np.nan), where=counts > 0),
            "mean_P2": np.divide(sum_p2, counts, out=np.full_like(sum_p2, np.nan), where=counts > 0),
            "count": counts,
            "rho_g_cm3": counts / volume * WATER_MASS_G_MOL / AVOGADRO * 1.0e24 if volume else np.nan,
            "n_frames": n_frames,
        }
    )


def save_plots(profile, destination, show):
    """Write the cosine and P2 profiles.

    Parameters
    ----------
    profile : pandas.DataFrame
        Table from :func:`accumulate_orientation`.
    destination : pathlib.Path
        Output directory.
    show : bool
        If true, display the figures after saving them.

    Returns
    -------
    None
        ``water_cos_theta_z.png`` and ``water_P2_z.png`` are written in
        ``destination``.
    """
    for column, label, stem in (
        ("mean_cos_theta", r"$\langle \cos\theta \rangle$", "water_cos_theta_z"),
        ("mean_P2", r"$\langle P_2(\cos\theta) \rangle$", "water_P2_z"),
    ):
        plt.figure(figsize=(6, 8))
        plt.plot(profile[column], profile["z"], linewidth=2)
        plt.axvline(0, color="black", linestyle="--", alpha=0.5)
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.xlabel(label)
        plt.ylabel("z (angstrom)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(destination / f"{stem}.png", dpi=300)
        if show:
            plt.show()
        plt.close()


def main(argv=None):
    """Average the dipole orientation of one trajectory.

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
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True, nargs="+",
                        help="Trajectory, or the segments of one run in order.")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth stored frame.")
    parser.add_argument("--discard-ps", type=float, default=0.0, help="Leading time to skip, in ps.")
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--z-min", type=float, default=-100.0)
    parser.add_argument("--z-max", type=float, default=100.0)
    parser.add_argument("--dz", type=float, default=PROFILE_BIN_A, help="Profile bin width in angstroms.")
    parser.add_argument("--resname", default="WAT")
    parser.add_argument("--oxygen-name", default="O")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    universe = mda.Universe(str(args.topology), [str(path) for path in args.trajectory], format="NCDF")
    indices = water_indices(universe, args.resname, args.oxygen_name)
    print(f"Waters analyzed: {len(indices[0])}")
    times = np.array([float(ts.time) for ts in universe.trajectory])
    first = int(np.searchsorted(times - times[0], args.discard_ps - 1e-6))
    frames = range(first, len(times), max(1, args.stride))
    if not len(frames):
        raise SystemExit("--discard-ps removes every frame.")
    profile = accumulate_orientation(universe, indices, args.z_min, args.z_max, args.dz, frames)
    profile.to_csv(args.out / "water_orientation_z_profile.csv", index=False)
    (args.out / "orientation_metadata.json").write_text(json.dumps({
        "profile_bin_A": args.dz,
        "role": "profile",
    }, indent=2))
    save_plots(profile, args.out, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
