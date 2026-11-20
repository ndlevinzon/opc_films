"""Slab centering, region masks, and geometric hydrogen bonds.

Distances use the minimum image in the supplied orthorhombic cell. The film
is assumed to be a single slab with vacuum on both sides along z, so the
periodic center of the oxygen coordinates is unambiguous.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from opcfilm.config import BULK_HALFWIDTH_A, CONFIG


def minimum_image(displacement, lengths):
    """Wrap a displacement into the primary cell.

    Parameters
    ----------
    displacement : array_like
        Coordinate difference. The last axis is ``x, y, z``.
    lengths : array_like
        Orthorhombic box lengths, broadcast against ``displacement``.

    Returns
    -------
    numpy.ndarray
        Displacement minus the nearest periodic image.
    """
    lengths = np.asarray(lengths, dtype=float)
    return np.asarray(displacement, dtype=float) - lengths * np.rint(
        np.asarray(displacement, dtype=float) / lengths
    )


def water_indices(universe):
    """Collect oxygen and hydrogen atom indices for each water residue.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Topology containing WAT, HOH, or OPC residues with one oxygen and
        two hydrogens.

    Returns
    -------
    oxygen : numpy.ndarray of int, shape (n_water,)
        Oxygen atom index of each water.
    hydrogen : numpy.ndarray of int, shape (n_water, 2)
        The two hydrogen atom indices of each water, in residue order.

    Raises
    ------
    ValueError
        If a water residue does not have one oxygen and two hydrogens, or
        if the topology contains no water.
    """
    oxygen = []
    hydrogen = []
    for residue in universe.residues:
        if residue.resname.upper() not in ("WAT", "HOH", "OPC"):
            continue
        atoms = list(residue.atoms)
        oxygens = [
            atom.index
            for atom in atoms
            if atom.name.upper() in ("O", "OW", "OH2")
        ]
        hydrogens = [
            atom.index
            for atom in atoms
            if atom.name.upper() in ("H1", "H2", "HW", "H")
            or atom.name.upper().startswith("H")
        ]
        if len(oxygens) != 1 or len(hydrogens) != 2:
            raise ValueError(
                f"Water residue {residue.resid} has {len(oxygens)} oxygens "
                f"and {len(hydrogens)} hydrogens; expected 1 and 2."
            )
        oxygen.append(oxygens[0])
        hydrogen.append(hydrogens)
    if not oxygen:
        raise ValueError("No WAT, HOH, or OPC residues in the topology.")
    return np.asarray(oxygen, dtype=int), np.asarray(hydrogen, dtype=int)


def _outward_crossing(z, rho, threshold, side):
    """First interpolated crossing of a threshold, walking out from the center.

    Parameters
    ----------
    z : numpy.ndarray
        Bin centers in angstroms, in increasing order.
    rho : numpy.ndarray
        Density or count on those bins.
    threshold : float
        Density the profile falls through.
    side : {"lower", "upper"}
        Which interface to walk toward.

    Returns
    -------
    float
        Interpolated ``z``. ``nan`` when the profile never falls through
        the threshold.
    """
    if side == "upper":
        order = np.where(z >= 0.0)[0]
    else:
        order = np.where(z <= 0.0)[0][::-1]
    zz = z[order]
    yy = rho[order]
    hits = np.flatnonzero((yy[:-1] >= threshold) & (yy[1:] < threshold))
    if len(hits) == 0:
        return np.nan
    index = int(hits[0])
    span = yy[index + 1] - yy[index]
    if span == 0.0:
        return float(zz[index])
    return float(
        zz[index] + (zz[index + 1] - zz[index]) * (threshold - yy[index]) / span
    )


def half_density_surfaces(z, rho, bulk_halfwidth=BULK_HALFWIDTH_A, fractions=(0.5,)):
    """Half-density surfaces from a slab-centered density profile.

    The bulk density is the mean of ``rho`` for ``|z| <= bulk_halfwidth``.
    On each side, a surface is the first place, walking outward from the
    center, where the profile falls through a fraction of that density.
    The location is linearly interpolated. Pass the 1 Å-smoothed profile
    when the caller has one.

    Parameters
    ----------
    z : array_like
        Bin centers in angstroms, measured from the slab center.
    rho : array_like
        Density or occupancy on those bins.
    bulk_halfwidth : float, optional
        Half-width of the bulk window, in angstroms.
    fractions : sequence of float, optional
        Fractions of the bulk density. ``0.5`` is always included.

    Returns
    -------
    dict
        ``rho_bulk``, ``z_lower``, ``z_upper``, and ``crossings``.
        ``crossings[fraction]`` is ``(z_lower, z_upper)``.

    Raises
    ------
    ValueError
        If the bulk window is empty or a half-density surface is missing.
    """
    z = np.asarray(z, dtype=float)
    rho = np.asarray(rho, dtype=float)
    bulk = np.abs(z) <= bulk_halfwidth
    if not np.any(bulk):
        raise ValueError("Bulk window contains no density bins.")
    rho_bulk = float(np.mean(rho[bulk]))
    if not np.isfinite(rho_bulk) or rho_bulk <= 0.0:
        raise ValueError("Bulk density is not positive.")
    requested = {float(fraction) for fraction in fractions}
    requested.add(0.5)
    crossings = {}
    for fraction in sorted(requested):
        lower = _outward_crossing(z, rho, fraction * rho_bulk, "lower")
        upper = _outward_crossing(z, rho, fraction * rho_bulk, "upper")
        crossings[fraction] = (lower, upper)
    lower, upper = crossings[0.5]
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Could not locate both half-density surfaces.")
    return {
        "rho_bulk": rho_bulk,
        "z_lower": float(lower),
        "z_upper": float(upper),
        "crossings": crossings,
    }


def region_masks(z_relative, depth, bulk_halfwidth=BULK_HALFWIDTH_A):
    """Bulk and the two named interface bands.

    The bands may overlap each other. Neither may overlap the bulk. The
    main interface for structure and dynamics is ``awi``. ``awi_outer`` is
    the outermost layer used for defects and as the sensitivity check.

    Parameters
    ----------
    z_relative : array_like
        ``z`` measured from the slab center, in angstroms.
    depth : array_like
        Depth inward from the half-density surface, in angstroms.
    bulk_halfwidth : float, optional
        Half-width of the bulk window.

    Returns
    -------
    dict of numpy.ndarray of bool
        ``interior``, ``awi`` (3–12 Å), and ``awi_outer`` (0–6 Å).

    Raises
    ------
    ValueError
        If any bulk molecule also falls in an interface band.
    """
    z_relative = np.asarray(z_relative, dtype=float)
    depth = np.asarray(depth, dtype=float)
    masks = {
        "interior": np.abs(z_relative) <= bulk_halfwidth,
        "awi": (depth >= CONFIG.regions.awi_A[0]) & (depth < CONFIG.regions.awi_A[1]),
        "awi_outer": (depth >= CONFIG.regions.awi_outer_A[0]) & (depth < CONFIG.regions.awi_outer_A[1]),
    }
    overlap = masks["interior"] & (masks["awi"] | masks["awi_outer"])
    if np.any(overlap):
        raise ValueError(
            f"The {bulk_halfwidth:g} Å bulk overlaps an interface band "
            f"for {int(np.count_nonzero(overlap))} molecules."
        )
    return masks


def slab_geometry(oxygen, lengths, dz=None, bulk_halfwidth=BULK_HALFWIDTH_A):
    """Center a slab and measure depth from the half-density surfaces.

    The center is the periodic circular mean of the oxygen ``z`` coordinates.
    The surfaces are the interpolated half-density crossings of the occupancy
    profile after 1 Å Gaussian smoothing, from :func:`half_density_surfaces`.

    Parameters
    ----------
    oxygen : numpy.ndarray, shape (n, 3)
        Oxygen coordinates in angstroms.
    lengths : array_like, shape (3,)
        Orthorhombic box lengths.
    dz : float, optional
        Histogram bin width along ``z``, in angstroms.
    bulk_halfwidth : float, optional
        Half-width of the central window used to estimate the plateau, in
        angstroms. Same name as :func:`half_density_surfaces`.

    Returns
    -------
    z_relative : numpy.ndarray, shape (n,)
        Oxygen ``z`` measured from the slab center, in angstroms.
    depth : numpy.ndarray, shape (n,)
        Distance inward from the nearer half-density surface. Negative
        values lie outside the liquid.
    bounds : list of float
        ``[z_lower, z_upper]`` of the half-density surfaces, relative to
        the slab center.
    center : float
        Absolute ``z`` of the slab center inside the periodic cell.
    """
    if dz is None:
        dz = CONFIG.profile.region_bin_A
    lengths = np.asarray(lengths, dtype=float)
    phase = np.exp(2j * np.pi * (oxygen[:, 2] % lengths[2]) / lengths[2])
    resultant = phase.mean()
    if abs(resultant) < 0.05:
        raise ValueError("Slab center is ambiguous; the z distribution is too uniform.")
    center = (np.angle(resultant) % (2.0 * np.pi)) * lengths[2] / (2.0 * np.pi)
    z_relative = minimum_image(oxygen[:, 2] - center, lengths[2])
    n_bins = int(np.ceil(lengths[2] / dz))
    edges = np.linspace(-lengths[2] / 2.0, lengths[2] / 2.0, n_bins + 1)
    z_mid = 0.5 * (edges[1:] + edges[:-1])
    counts = np.histogram(z_relative, bins=edges)[0].astype(float)
    binwidth = float(edges[1] - edges[0])
    smoothed = gaussian_filter1d(counts, sigma=CONFIG.profile.smooth_sigma_A / binwidth)
    surfaces = half_density_surfaces(z_mid, smoothed, bulk_halfwidth)
    bounds = [surfaces["z_lower"], surfaces["z_upper"]]
    depth = np.minimum(z_relative - bounds[0], bounds[1] - z_relative)
    return z_relative, depth, bounds, float(center)


def assign_regions(
    z_relative,
    depth,
    core_halfwidth,
    awi_inner,
    awi_outer,
    policy="error",
):
    """Assign waters to the bulk and one interface band.

    The bulk is ``|z| <= core_halfwidth``. The interface is
    ``awi_inner <= depth < awi_outer``. With the 15 Å bulk these windows
    do not overlap; ``policy="error"`` asserts that.

    Parameters
    ----------
    z_relative : numpy.ndarray
        Coordinate from :func:`slab_geometry`.
    depth : numpy.ndarray
        Inward depth from :func:`slab_geometry`.
    core_halfwidth : float
        Half-width of the bulk window, in angstroms.
    awi_inner, awi_outer : float
        Inner and outer bounds of the interface band, in angstroms,
        measured inward from the half-density surface.
    policy : {"clip-awi", "clip-core", "error"}, optional
        ``clip-awi`` keeps the core fixed and drops overlapping waters
        from the interface. ``clip-core`` does the reverse. ``error``
        raises if any water would fall in both windows.

    Returns
    -------
    regions : dict
        ``interior`` and ``awi`` map to integer water indices.
    qc : dict
        Counts of the overlap and the depth range that survived clipping.
    """
    core_mask = np.abs(z_relative) <= core_halfwidth
    awi_mask = (depth >= awi_inner) & (depth < awi_outer)
    overlap = core_mask & awi_mask
    n_overlap = int(np.count_nonzero(overlap))
    n_awi_before = int(np.count_nonzero(awi_mask))
    if n_overlap:
        if policy == "error":
            raise ValueError(
                f"Core and interface overlap for {n_overlap} waters. "
                "Choose --overlap-policy clip-awi or clip-core."
            )
        if policy == "clip-awi":
            awi_mask = awi_mask & ~core_mask
        elif policy == "clip-core":
            core_mask = core_mask & ~awi_mask
        else:
            raise ValueError(f"Unknown overlap policy {policy!r}.")
    if np.any(core_mask & awi_mask):
        raise AssertionError("Region assignment left an overlapping water.")
    awi_depths = depth[awi_mask]
    qc = {
        "overlap_waters": n_overlap,
        "awi_eligible_before_clipping": n_awi_before,
        "awi_eligible_after_clipping": int(np.count_nonzero(awi_mask)),
        "awi_fraction_removed_due_to_overlap": (
            n_overlap / n_awi_before if n_awi_before else 0.0
        ),
        "core_eligible_after_clipping": int(np.count_nonzero(core_mask)),
        "awi_actual_min_depth_A": (
            float(np.min(awi_depths)) if len(awi_depths) else np.nan
        ),
        "awi_actual_max_depth_A": (
            float(np.max(awi_depths)) if len(awi_depths) else np.nan
        ),
    }
    return {
        "interior": np.flatnonzero(core_mask),
        "awi": np.flatnonzero(awi_mask),
    }, qc


def _angle_below_cutoff(oh_vectors, oxygen_oxygen, cos_cut):
    """Return which hydrogens form an angle steeper than the cutoff.

    The angle is the O-H···O angle: the angle at H between the OH bond
    and the vector from that hydrogen to the acceptor oxygen.

    Parameters
    ----------
    oh_vectors : numpy.ndarray, shape (n_pair, 2, 3)
        Minimum-image OH vectors of the donor water.
    oxygen_oxygen : numpy.ndarray, shape (n_pair, 3)
        Minimum-image vector from donor oxygen to acceptor oxygen.
    cos_cut : float
        Cosine of the minimum accepted O-H···O angle. An angle larger
        than the cutoff has a smaller cosine.

    Returns
    -------
    numpy.ndarray of bool, shape (n_pair, 2)
        True where that hydrogen meets the angular criterion.
    """
    hydrogen_to_acceptor = oxygen_oxygen[:, None, :] - oh_vectors
    denominator = np.linalg.norm(oh_vectors, axis=2) * np.linalg.norm(
        hydrogen_to_acceptor, axis=2
    )
    cosine = np.sum(-oh_vectors * hydrogen_to_acceptor, axis=2) / np.maximum(
        denominator, 1e-15
    )
    return cosine < cos_cut


def hydrogen_bond_counts(
    oxygen,
    hydrogen,
    lengths,
    references,
    cutoff=None,
    angle_deg=None,
    batch=512,
):
    """Count donated and accepted hydrogen bonds for a set of waters.

    A bond requires an oxygen-oxygen distance below ``cutoff`` and an
    O-H···O angle above ``angle_deg``. Two hydrogens of the same donor
    that reach the same acceptor add one to the donor count. Every water
    in the slab can be a partner; only ``references`` are scored.

    Parameters
    ----------
    oxygen : numpy.ndarray, shape (n_water, 3)
        Oxygen coordinates.
    hydrogen : numpy.ndarray, shape (n_water, 2, 3)
        Hydrogen coordinates in the same water order as ``oxygen``.
    lengths : array_like, shape (3,)
        Orthorhombic box lengths.
    references : numpy.ndarray of int
        Water indices to score.
    cutoff : float, optional
        Maximum oxygen-oxygen distance, in angstroms.
    angle_deg : float, optional
        Minimum O-H···O angle, in degrees.
    batch : int, optional
        Number of reference waters per neighbor-list query.

    Returns
    -------
    n_donor : numpy.ndarray of int, shape (n_ref,)
        Bonds donated by each reference water.
    n_acceptor : numpy.ndarray of int, shape (n_ref,)
        Bonds accepted by each reference water.
    occupied : numpy.ndarray of bool, shape (n_ref, 2)
        True when that hydrogen of the reference donates at least one bond.
    """
    if cutoff is None:
        cutoff = CONFIG.hbond.oo_cutoff_A
    if angle_deg is None:
        angle_deg = CONFIG.hbond.angle_deg
    lengths = np.asarray(lengths, dtype=float)
    wrapped = oxygen % lengths
    tree = cKDTree(wrapped, boxsize=lengths)
    n_donor = np.zeros(len(references), dtype=np.int16)
    n_acceptor = np.zeros(len(references), dtype=np.int16)
    occupied = np.zeros((len(references), 2), dtype=bool)
    cos_cut = np.cos(np.deg2rad(angle_deg))
    oh_vectors = minimum_image(hydrogen - oxygen[:, None, :], lengths)
    for offset in range(0, len(references), batch):
        part = references[offset : offset + batch]
        neighbors = tree.query_ball_point(wrapped[part], cutoff)
        sizes = np.fromiter((len(item) for item in neighbors), dtype=int, count=len(neighbors))
        local_row = np.repeat(np.arange(len(part)), sizes)
        if len(local_row) == 0:
            continue
        other = np.concatenate([np.asarray(item, dtype=int) for item in neighbors])
        water = part[local_row]
        keep = other != water
        local_row = local_row[keep]
        other = other[keep]
        water = water[keep]
        if len(water) == 0:
            continue
        delta = minimum_image(oxygen[other] - oxygen[water], lengths)
        distance = np.linalg.norm(delta, axis=1)
        keep = (distance > 0.0) & (distance < cutoff)
        local_row = local_row[keep]
        other = other[keep]
        water = water[keep]
        delta = delta[keep]
        if len(water) == 0:
            continue
        donated = _angle_below_cutoff(oh_vectors[water], delta, cos_cut)
        accepted = _angle_below_cutoff(oh_vectors[other], -delta, cos_cut)
        donor_rows = local_row[np.any(donated, axis=1)]
        acceptor_rows = local_row[np.any(accepted, axis=1)]
        if len(donor_rows):
            np.add.at(n_donor, offset + donor_rows, 1)
        if len(acceptor_rows):
            np.add.at(n_acceptor, offset + acceptor_rows, 1)
        np.logical_or.at(occupied, (offset + local_row, np.zeros(len(local_row), dtype=int)), donated[:, 0])
        np.logical_or.at(occupied, (offset + local_row, np.ones(len(local_row), dtype=int)), donated[:, 1])
    return n_donor, n_acceptor, occupied


def directed_hydrogen_bonds(
    oxygen,
    hydrogen,
    lengths,
    donors,
    cutoff=None,
    angle_deg=None,
):
    """Directed donor-acceptor pairs with the same geometry as the bond counts.

    Parameters
    ----------
    oxygen : numpy.ndarray, shape (n_water, 3)
        Oxygen coordinates.
    hydrogen : numpy.ndarray, shape (n_water, 2, 3)
        Hydrogen coordinates in the same water order as ``oxygen``.
    lengths : array_like, shape (3,)
        Orthorhombic box lengths.
    donors : array_like of int
        Waters tested as donors. Any other water inside the cutoff can accept.
    cutoff : float, optional
        Maximum oxygen-oxygen distance, in angstroms.
    angle_deg : float, optional
        Minimum O-H···O angle, in degrees.

    Returns
    -------
    set of tuple of int
        ``(donor, acceptor)``. Either hydrogen of the donor is enough for one pair.
    """
    if cutoff is None:
        cutoff = CONFIG.hbond.oo_cutoff_A
    if angle_deg is None:
        angle_deg = CONFIG.hbond.angle_deg
    lengths = np.asarray(lengths, dtype=float)
    donors = np.asarray(donors, dtype=int)
    wrapped = oxygen % lengths
    tree = cKDTree(wrapped, boxsize=lengths)
    cos_cut = np.cos(np.deg2rad(angle_deg))
    oh_vectors = minimum_image(hydrogen - oxygen[:, None, :], lengths)
    pairs = set()
    neighbors = tree.query_ball_point(wrapped[donors], cutoff)
    for donor, near in zip(donors, neighbors):
        others = np.asarray([index for index in near if index != int(donor)], dtype=int)
        if len(others) == 0:
            continue
        delta = minimum_image(oxygen[others] - oxygen[int(donor)], lengths)
        distance = np.linalg.norm(delta, axis=1)
        keep = (distance > 0.0) & (distance < cutoff)
        others = others[keep]
        delta = delta[keep]
        if len(others) == 0:
            continue
        donor_oh = np.broadcast_to(oh_vectors[int(donor)], (len(others), 2, 3))
        donated = _angle_below_cutoff(donor_oh, delta, cos_cut)
        for acceptor in others[np.any(donated, axis=1)]:
            pairs.add((int(donor), int(acceptor)))
    return pairs
