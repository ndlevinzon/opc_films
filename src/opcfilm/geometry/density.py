"""Mass density from a number-density profile."""

from __future__ import annotations

import numpy as np

from opcfilm.config import CONFIG
from opcfilm.geometry.slab import half_density_surfaces


def mass_density(number_density):
    """Convert a number density to g cm^-3.

    Parameters
    ----------
    number_density : array_like
        Molecules per cubic angstrom.

    Returns
    -------
    numpy.ndarray
        Mass density in g cm^-3.
    """
    factor = CONFIG.film.water_molar_mass_g_mol / CONFIG.film.number_density_divisor
    return np.asarray(number_density, dtype=float) * factor


def core_mass_density(z, oxygen_count_per_bin, lengths, bulk_halfwidth, bin_width=None):
    """Mass density inside the interior of one profile.

    Parameters
    ----------
    z : array_like
        Bin centers in angstroms.
    oxygen_count_per_bin : array_like
        Oxygen counts in those bins.
    lengths : array_like, shape (3,)
        Box lengths in angstroms.
    bulk_halfwidth : float
        Half-width of the interior, in angstroms.
    bin_width : float, optional
        Bin width in angstroms. The sheet value is the default.

    Returns
    -------
    float
        Interior mass density in g cm^-3. The half-density surfaces are
        returned only as a check that the profile has two sides.
    """
    if bin_width is None:
        bin_width = CONFIG.profile.bin_A
    z = np.asarray(z, dtype=float)
    counts = np.asarray(oxygen_count_per_bin, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    volume = lengths[0] * lengths[1] * float(bin_width)
    rho = counts / volume
    half_density_surfaces(z, rho, bulk_halfwidth=bulk_halfwidth)
    interior = np.abs(z) <= float(bulk_halfwidth)
    if not np.any(interior):
        raise ValueError("The profile has no interior bins.")
    return float(mass_density(np.mean(rho[interior])))
