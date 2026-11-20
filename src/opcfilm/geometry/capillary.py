"""Capillary-wave spectrum of a height field.

The production driver still lives with the cooling-path scripts. This module
is the shared definition: five blocks across a hold, and the squared 2D FFT
of a mean-subtracted height field.
"""

from __future__ import annotations

import numpy as np


def block_count(n_blocks=5):
    """Number of capillary blocks in one hold.

    Parameters
    ----------
    n_blocks : int, optional
        Blocks the hold is split into. The reported setting is 5, so each
        block is a fifth of the hold.

    Returns
    -------
    int
        The block count.
    """
    if int(n_blocks) < 1:
        raise ValueError("A hold needs at least one capillary block.")
    return int(n_blocks)


def height_spectrum(height, dx, dy):
    """Squared Fourier amplitude of a height field.

    Parameters
    ----------
    height : array_like, shape (ny, nx)
        Height in angstroms.
    dx : float
        Grid spacing along x, in angstroms.
    dy : float
        Grid spacing along y, in angstroms.

    Returns
    -------
    qx, qy, power : numpy.ndarray
        Wavevectors in inverse angstroms and the squared amplitude.
        The mean height is removed first.
    """
    field = np.asarray(height, dtype=float)
    if field.ndim != 2:
        raise ValueError("height must be a 2D grid.")
    field = field - np.mean(field)
    ny, nx = field.shape
    qx = 2.0 * np.pi * np.fft.fftfreq(nx, d=float(dx))
    qy = 2.0 * np.pi * np.fft.fftfreq(ny, d=float(dy))
    power = np.abs(np.fft.fft2(field)) ** 2
    return qx, qy, power
