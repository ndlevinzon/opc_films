#!/usr/bin/env python3
"""Check that an Amber trajectory has a uniform NetCDF time step.

The spacing is read from the ``time`` variable. A value passed with
``--frame-ps`` is accepted only when it matches that spacing. With
``--topology``, the water count and the first-frame box are checked
against the OPC film.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.trajectory import (
    frame_spacing_ps,
    netcdf_time_ps,
    require_film,
    resolve_n_water,
)


def first_box(path):
    """Read the first orthorhombic cell of an Amber NetCDF trajectory.

    Parameters
    ----------
    path : path-like
        Trajectory with a ``cell_lengths`` variable.

    Returns
    -------
    numpy.ndarray, shape (3,)
        First-frame lengths in angstroms.
    """
    from netCDF4 import Dataset

    with Dataset(Path(path)) as dataset:
        if "cell_lengths" not in dataset.variables:
            raise RuntimeError(f"{path} has no cell_lengths variable.")
        return np.asarray(dataset.variables["cell_lengths"][0, :3], dtype=float)


def main(argv=None):
    """Report the frame spacing of each trajectory and stop on a mismatch.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments. ``None`` uses the process arguments.

    Returns
    -------
    int
        Zero when every trajectory passes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", nargs="+", type=Path, required=True)
    parser.add_argument("--topology", type=Path, default=None)
    parser.add_argument("--frame-ps", type=float, default=None,
                        help="Claimed spacing. It must agree with the NetCDF time.")
    args = parser.parse_args(argv)
    n_water = None
    if args.topology is not None:
        n_water = resolve_n_water(args.topology)
    for path in args.traj:
        times = netcdf_time_ps(path)
        spacing = frame_spacing_ps(times, override=args.frame_ps)
        print(f"{path}: {len(times)} frames, {spacing:g} ps")
        if n_water is not None:
            require_film(n_water, first_box(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
