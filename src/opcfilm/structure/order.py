"""Tetrahedral order parameters shared by the cooling path and the glass.

``q``, ``Sk``, and the local structure index are the definitions used for the
glass: Chau–Hardwick translational order and the standard LSI, which keeps
the first neighbor outside the cutoff. ``radial_regularity`` is a separate
Gaussian score of the four neighbor distances. It is not ``Sk``.
"""

from __future__ import annotations

import numpy as np


def orientational_tetrahedral_q(vectors):
    """Orientational tetrahedral order of one molecule or a batch.

    Parameters
    ----------
    vectors : array_like, shape (4, 3) or (n, 4, 3)
        Vectors from the central oxygen to its four nearest neighbors.
        Zero-length vectors make the result undefined.

    Returns
    -------
    float or numpy.ndarray
        ``1`` for a perfect tetrahedron. An array is returned when
        ``vectors`` has three dimensions.
    """
    vectors = np.asarray(vectors, dtype=float)
    single = vectors.ndim == 2
    if single:
        vectors = vectors[None, :, :]
    if vectors.shape[-2:] != (4, 3):
        raise ValueError("Expected vectors with shape (4, 3) or (n, 4, 3).")
    norms = np.linalg.norm(vectors, axis=2)
    undefined = ~np.isfinite(norms).all(axis=1) | np.any(norms == 0.0, axis=1)
    unit = np.divide(
        vectors,
        norms[:, :, None],
        out=np.zeros_like(vectors),
        where=norms[:, :, None] > 0.0,
    )
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    cosines = np.stack(
        [np.sum(unit[:, i] * unit[:, j], axis=1) for i, j in pairs],
        axis=1,
    )
    order = 1.0 - (3.0 / 8.0) * np.sum((cosines + 1.0 / 3.0) ** 2, axis=1)
    order[undefined] = np.nan
    return float(order[0]) if single else order


def translational_tetrahedral_sk(distances):
    """Chau–Hardwick translational tetrahedral order.

    Parameters
    ----------
    distances : array_like, shape (4,) or (n, 4)
        Distances from a molecule to its four nearest neighbors, in angstroms.

    Returns
    -------
    float or numpy.ndarray
        ``1`` when the four distances are equal. An array is returned when
        ``distances`` is two-dimensional.
    """
    distances = np.asarray(distances, dtype=float)
    single = distances.ndim == 1
    if single:
        distances = distances[None, :]
    if distances.shape[1] < 4:
        raise ValueError("Sk needs the four nearest-neighbor distances.")
    distances = distances[:, :4]
    undefined = ~np.isfinite(distances).all(axis=1)
    mean = np.mean(distances, axis=1)
    undefined |= mean <= 0.0
    deviation = (distances - mean[:, None]) ** 2 / (4.0 * mean[:, None] ** 2)
    order = 1.0 - (1.0 / 3.0) * np.sum(deviation, axis=1)
    order[undefined] = np.nan
    return float(order[0]) if single else order


def local_structure_index(neighbor_distances, cutoff=3.7):
    """Standard local structure index, including the first neighbor beyond the cutoff.

    Parameters
    ----------
    neighbor_distances : array_like
        Oxygen–oxygen distances from one molecule, excluding the self distance.
    cutoff : float, optional
        Shell cutoff in angstroms. The index uses every neighbor inside this
        distance and the next neighbor outside it.

    Returns
    -------
    float
        Mean squared deviation of those neighbor spacings, in angstroms
        squared. ``nan`` when fewer than two neighbors lie inside the cutoff
        or when no neighbor lies outside it.
    """
    distances = np.asarray(neighbor_distances, dtype=float)
    distances = distances[np.isfinite(distances) & (distances > 0.0)]
    distances = np.sort(distances)
    inside = int(np.sum(distances < cutoff))
    if inside < 2 or inside >= len(distances):
        return np.nan
    spacing = np.diff(distances[: inside + 1])
    return float(np.mean((spacing - np.mean(spacing)) ** 2))


def local_structure_index_batch(neighbor_distances, cutoff=3.7):
    """Standard local structure index for many molecules.

    Parameters
    ----------
    neighbor_distances : array_like, shape (n, k)
        Distances to neighbors other than the molecule itself. Each row may
        contain zeros or non-finite placeholders for missing neighbors.
    cutoff : float, optional
        Shell cutoff passed to :func:`local_structure_index`.

    Returns
    -------
    numpy.ndarray, shape (n,)
        One index per row.
    """
    distances = np.asarray(neighbor_distances, dtype=float)
    return np.array(
        [local_structure_index(row, cutoff) for row in distances],
        dtype=float,
    )


def radial_regularity(distances, center=2.76, width=0.150):
    """Gaussian score of the four nearest-neighbor distances.

    This is not the Chau–Hardwick ``Sk``. It is close to one when all four
    distances sit near ``center``.

    Parameters
    ----------
    distances : array_like, shape (4,) or (n, 4)
        Neighbor distances in angstroms.
    center : float, optional
        Distance, in angstroms, at which the score is one.
    width : float, optional
        Gaussian width in angstroms.

    Returns
    -------
    float or numpy.ndarray
        ``exp(-mean(((r - center) / width)^2))``.
    """
    distances = np.asarray(distances, dtype=float)
    single = distances.ndim == 1
    if single:
        distances = distances[None, :]
    score = np.exp(-np.mean(((distances[:, :4] - center) / width) ** 2, axis=1))
    score[~np.isfinite(distances[:, :4]).all(axis=1)] = np.nan
    return float(score[0]) if single else score
