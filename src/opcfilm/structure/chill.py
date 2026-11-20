"""OVITO CHILL+ identification.

The cell is periodic in x and y and open in z. Coordinates are shifted so the
slab center sits at the middle of the cell. OVITO is imported only when the
function runs.
"""

from __future__ import annotations

import numpy as np


def chill_plus(oxygen, lx, ly, lz, z_center):
    """Identify ice-like molecules with OVITO CHILL+.

    Parameters
    ----------
    oxygen : array_like, shape (n, 3)
        Oxygen coordinates in angstroms.
    lx : float
        Box length along x, in angstroms.
    ly : float
        Box length along y, in angstroms.
    lz : float
        Box length along z, in angstroms.
    z_center : float
        Absolute z of the slab center.

    Returns
    -------
    numpy.ndarray
        Integer structure type for each oxygen, using OVITO's CHILL+ coding.

    Raises
    ------
    ImportError
        If OVITO is not installed.
    ValueError
        If ``z_center`` is missing.
    """
    if z_center is None:
        raise ValueError("Pass the circular-mean slab center from slab_geometry.")
    try:
        from ovito.data import DataCollection, Particles, SimulationCell
        from ovito.modifiers import ChillPlusModifier
        from ovito.pipeline import Pipeline, StaticSource
    except ImportError as error:
        raise ImportError("OVITO is required for CHILL+ and is not installed.") from error
    pos = np.asarray(oxygen, dtype=float).copy()
    pos[:, 0] = pos[:, 0] % lx
    pos[:, 1] = pos[:, 1] % ly
    z_relative = (pos[:, 2] % lz) - float(z_center)
    z_relative -= lz * np.rint(z_relative / lz)
    pos[:, 2] = z_relative + 0.5 * lz
    data = DataCollection()
    particles = Particles()
    particles.create_property("Position", data=pos)
    data.objects.append(particles)
    cell = SimulationCell(pbc=(True, True, False))
    cell[:, 0] = [lx, 0.0, 0.0]
    cell[:, 1] = [0.0, ly, 0.0]
    cell[:, 2] = [0.0, 0.0, lz]
    data.objects.append(cell)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(ChillPlusModifier())
    output = pipeline.compute()
    return np.asarray(output.particles["Structure Type"])


def ovito_available():
    """Report whether OVITO can be imported.

    Returns
    -------
    bool
        True when ``ovito`` imports.
    """
    try:
        import ovito  # noqa: F401
    except ImportError:
        return False
    return True
