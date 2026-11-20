"""One structural classifier for the OPC film.

The liquid reference is 300 K and the glass reference is 78 K. Both are
the bulk, ``|z| <= BULK_HALFWIDTH_A``. A core of 45 Å reaches the 3–12 Å
interface and is not used. The 250 K versus 150 K coloring in
``plot_color_LDA.py`` is a display illustration. It is not this classifier.

Features are ``q``, Chau–Hardwick ``Sk``, and the local structure index
from :mod:`common.order`. The center and the regions come from
:mod:`common.slab`. The box is the frame's own box.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from opcfilm.config import BULK_HALFWIDTH_A, CONFIG
from opcfilm.structure.order import (
    local_structure_index_batch,
    orientational_tetrahedral_q,
    translational_tetrahedral_sk,
)
from opcfilm.geometry.slab import region_masks, slab_geometry

CLASSIFIER_ID = "qsk_lsi_shared_covariance_300_vs_78_bulk15A_leave_one_replica"
DISPLAY_250_150 = "display_only_250K_vs_150K_one_frame_z45"
DISPLAY_MEDOID_Z45 = "display_only_300_78_one_medoid_z45_concatenated"
SENSITIVITY_BANDS = tuple(tuple(band) for band in CONFIG.regions.sensitivity_bands_A)
MAIN_AWI = tuple(CONFIG.regions.awi_A)


def write_panel_record(path, classifier_id, role, training, note):
    """Record which classifier produced a panel.

    Parameters
    ----------
    path : path-like
        JSON file written next to the figure.
    classifier_id : str
        Stable name of the classifier.
    role : str
        ``quantitative`` or ``display_only``.
    training : str
        Temperatures, frames, and region used to fit the score.
    note : str
        What the panel may and may not be used to claim.

    Returns
    -------
    pathlib.Path
        The file that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "classifier_id": classifier_id,
        "role": role,
        "training": training,
        "quantitative_classifier_id": CLASSIFIER_ID,
        "note": note,
    }, indent=2))
    return path


def sampled_frames(n_frames, count, kind):
    """Frame indices after the hold equilibration discard.

    Parameters
    ----------
    n_frames : int
        Frames in the trajectory.
    count : int
        Frames to keep, at least one.
    kind : str
        ``ramp`` uses the whole file. ``hold`` and ``endpoint`` drop the
        first ``HOLD_DISCARD_FRACTION``.

    Returns
    -------
    numpy.ndarray of int
        Increasing frame indices.
    """
    from opcfilm.config import HOLD_DISCARD_FRACTION
    from opcfilm.io.trajectory import discard_frame_count

    start = discard_frame_count(n_frames, kind, HOLD_DISCARD_FRACTION)
    return start + even_frame_indices(int(n_frames) - start, count)


def even_frame_indices(n_frames, count):
    """Evenly spaced frame indices, including both ends.

    Parameters
    ----------
    n_frames : int
        Number of frames in the trajectory.
    count : int
        Requested number of frames.

    Returns
    -------
    numpy.ndarray of int
        Unique indices in increasing order.
    """
    if int(n_frames) < 1:
        raise ValueError("The trajectory has no frames.")
    count = min(int(count), int(n_frames))
    if count < 1:
        raise ValueError("At least one frame is required.")
    return np.unique(np.rint(np.linspace(0, n_frames - 1, count)).astype(int))


def structure_features(oxygen, lengths, query):
    """``q``, ``Sk``, and LSI for the queried oxygens.

    Parameters
    ----------
    oxygen : numpy.ndarray, shape (n, 3)
        Oxygen coordinates in angstroms.
    lengths : array_like, shape (3,)
        Orthorhombic box lengths of this frame.
    query : array_like of int
        Row indices into ``oxygen``.

    Returns
    -------
    numpy.ndarray, shape (len(query), 3)
        Columns are ``q``, ``Sk``, and LSI. A row is non-finite when a
        neighbor shell is undefined. Neighbors have no distance cutoff.
    """
    oxygen = np.asarray(oxygen, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    query = np.asarray(query, dtype=int)
    if query.size == 0:
        return np.empty((0, 3))
    wrapped = np.mod(oxygen, lengths)
    tree = cKDTree(wrapped, boxsize=lengths)
    neighbors = min(25, len(wrapped))
    distances, indices = tree.query(wrapped[query], k=neighbors)
    if distances.ndim == 1:
        distances = distances[None, :]
        indices = indices[None, :]
    vectors = wrapped[indices[:, 1:5]] - wrapped[query, None, :]
    vectors -= lengths * np.rint(vectors / lengths)
    features = np.column_stack([
        orientational_tetrahedral_q(vectors),
        translational_tetrahedral_sk(distances[:, 1:5]),
        local_structure_index_batch(distances[:, 1:]),
    ])
    return features


def _finite(features):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != 3:
        raise ValueError("Features must have three columns: q, Sk, LSI.")
    return features[np.isfinite(features).all(axis=1)]


def robust_location_scale(features):
    """Median and MAD scale shared by the classifier and the display medoids.

    Parameters
    ----------
    features : array_like, shape (n, d)
        Feature rows.

    Returns
    -------
    center, scale : tuple of numpy.ndarray
        Column median and ``scale_mad`` times the median absolute deviation.
        A zero scale falls back to the standard deviation, then to 1.
    """
    features = np.asarray(features, dtype=float)
    center = np.median(features, axis=0)
    scale = CONFIG.classifier.scale_mad * np.median(np.abs(features - center), axis=0)
    std = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
    return center, scale


def fit_discriminant(liquid, glass):
    """Equal-prior shared-covariance score of glass versus liquid.

    Parameters
    ----------
    liquid, glass : array_like, shape (n, 3)
        Bulk waters at 300 K and at 78 K.

    Returns
    -------
    dict
        Location, scale, class means, and the inverse covariance. A high
        score is glass-like. It is not a calibrated phase probability.
    """
    liquid = _finite(liquid)
    glass = _finite(glass)
    if len(liquid) < 5 or len(glass) < 5:
        raise ValueError("Each reference class needs at least five finite waters.")
    pooled = np.vstack([liquid, glass])
    center, scale = robust_location_scale(pooled)
    liquid_z = (liquid - center) / scale
    glass_z = (glass - center) / scale
    covariance = (
        (len(liquid_z) - 1) * np.cov(liquid_z, rowvar=False)
        + (len(glass_z) - 1) * np.cov(glass_z, rowvar=False)
    ) / (len(liquid_z) + len(glass_z) - 2)
    covariance = covariance + CONFIG.classifier.covariance_jitter * np.eye(3)
    return {
        "classifier_id": CLASSIFIER_ID,
        "center": center,
        "scale": scale,
        "mu_liquid": liquid_z.mean(axis=0),
        "mu_glass": glass_z.mean(axis=0),
        "inv_cov": np.linalg.inv(covariance),
        "n_liquid": int(len(liquid_z)),
        "n_glass": int(len(glass_z)),
        "bulk_halfwidth_A": CONFIG.classifier.bulk_halfwidth_A,
        "liquid_K": CONFIG.classifier.liquid_K,
        "glass_K": CONFIG.classifier.glass_K,
        "scheme": CONFIG.classifier.scheme,
    }


def score_discriminant(features, model):
    """Glass-like score of each finite feature row.

    Parameters
    ----------
    features : array_like, shape (n, 3)
        ``q``, ``Sk``, and LSI.
    model : dict
        Result of :func:`fit_discriminant`.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Score in ``(0, 1)``. Non-finite input rows stay non-finite.
    """
    features = np.asarray(features, dtype=float)
    scores = np.full(len(features), np.nan)
    finite = np.isfinite(features).all(axis=1)
    if not np.any(finite):
        return scores
    standardized = (features[finite] - model["center"]) / model["scale"]
    liquid = standardized - model["mu_liquid"]
    glass = standardized - model["mu_glass"]
    liquid_distance = np.einsum("ij,jk,ik->i", liquid, model["inv_cov"], liquid)
    glass_distance = np.einsum("ij,jk,ik->i", glass, model["inv_cov"], glass)
    logit = np.clip(0.5 * (liquid_distance - glass_distance), -30.0, 30.0)
    scores[finite] = 1.0 / (1.0 + np.exp(-logit))
    return scores


def selections(oxygen, lengths):
    """Bulk and interface masks for one frame.

    Parameters
    ----------
    oxygen : numpy.ndarray, shape (n, 3)
        Oxygen coordinates.
    lengths : array_like, shape (3,)
        Box lengths of this frame.

    Returns
    -------
    dict
        ``interior`` is ``|z| <= 15`` Å. ``awi_3_12`` is the 3–12 Å band.
        ``awi_2_10`` and ``awi_4_14`` are the sensitivity bands. None of
        the interface bands overlap the bulk.

    Raises
    ------
    ValueError
        If a sensitivity band overlaps the bulk.
    """
    _relative, depth, bounds, _center = slab_geometry(oxygen, lengths)
    base = region_masks(_relative, depth)
    named = {"interior": base["interior"], "awi_3_12": base["awi"]}
    for inner, outer in SENSITIVITY_BANDS:
        band = (depth >= inner) & (depth < outer)
        if np.any(band & base["interior"]):
            raise ValueError(f"The {inner:g}–{outer:g} Å band overlaps the 15 Å bulk.")
        named[f"awi_{inner:g}_{outer:g}"] = band
    return {"masks": named, "depth": depth, "bounds": bounds}


def band_intervals(bounds, inner, outer):
    """Planar z intervals for a depth band.

    Parameters
    ----------
    bounds : sequence of float
        Lower and upper half-density surfaces, relative to the slab center.
    inner, outer : float
        Depth interval in angstroms, ``[inner, outer)``.

    Returns
    -------
    list of tuple
        Two ``(z_low, z_high)`` intervals, one on each side of the film.
    """
    lower, upper = float(bounds[0]), float(bounds[1])
    return [(lower + inner, lower + outer), (upper - outer, upper - inner)]


def leave_one_replica_out(by_replica):
    """Train on two replicas and score the third, at both endpoints.

    Parameters
    ----------
    by_replica : dict
        ``replica -> {"300": features, "78": features}``. Each value is an
        array of bulk waters from the frames of that replica.

    Returns
    -------
    list of dict
        Score mean, 10th and 90th percentiles, glass-like fraction, and
        misclassification rate for the held-out replica at 300 K and 78 K.
    """
    names = sorted(by_replica)
    if len(names) < 3:
        raise ValueError(
            "Leave-one-replica-out needs an endpoint trajectory for each of "
            "three replicas. A concatenated file with no replica label cannot be used."
        )
    rows = []
    for held in names:
        liquid = np.vstack([by_replica[name]["300"] for name in names if name != held])
        glass = np.vstack([by_replica[name]["78"] for name in names if name != held])
        model = fit_discriminant(liquid, glass)
        for temperature, block in (("300", by_replica[held]["300"]), ("78", by_replica[held]["78"])):
            scores = score_discriminant(block, model)
            scores = scores[np.isfinite(scores)]
            glasslike = scores >= 0.5
            mistakes = glasslike if temperature == "300" else ~glasslike
            rows.append({
                "classifier_id": CLASSIFIER_ID,
                "held_out_replica": held,
                "temperature_K": float(temperature),
                "n_waters": int(scores.size),
                "score_mean": float(np.mean(scores)),
                "score_p10": float(np.percentile(scores, 10)),
                "score_p90": float(np.percentile(scores, 90)),
                "glasslike_fraction": float(np.mean(glasslike)),
                "misclassification_rate": float(np.mean(mistakes)),
            })
    return rows


def normalized_fraction(value, liquid_fraction, glass_fraction):
    """Map a regional fraction onto the span of its own endpoints.

    Parameters
    ----------
    value : array_like
        ``f(T)`` in one region.
    liquid_fraction, glass_fraction : float
        The same region's fractions at 300 K and 78 K.

    Returns
    -------
    numpy.ndarray
        ``[f(T) - f(300)] / [f(78) - f(300)]``.
    """
    span = float(glass_fraction) - float(liquid_fraction)
    if not np.isfinite(span) or abs(span) < 1e-8:
        raise ValueError("The 300 K and 78 K fractions do not separate this region.")
    return (np.asarray(value, dtype=float) - float(liquid_fraction)) / span


def model_record(model):
    """JSON-ready copy of a fitted discriminant.

    Parameters
    ----------
    model : dict
        Result of :func:`fit_discriminant`.

    Returns
    -------
    dict
        Arrays replaced by lists.
    """
    record = {}
    for key, value in model.items():
        record[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return record
