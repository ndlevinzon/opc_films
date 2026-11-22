#!/usr/bin/env python3
"""Hydrogen-bond coordination and free OH groups in the bulk and at the surface.

A hydrogen bond is an oxygen-oxygen contact closer than 3.5 angstroms with
an O-H···O angle greater than 150 degrees. A directed donor-acceptor pair
is counted once even if both hydrogens of the donor meet the angle. An OH
is free when that hydrogen donates no geometric bond.

The bulk is ``|z| <= 15`` Å. Both interface bands are written, under
``awi_0-6/`` and ``awi_3-12/``. Overlap between the bulk and either band
is an error. The same pass writes the free-OH tables and the
donor-acceptor tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import (
    AWI_OUTER_A,
    AWI_SUB_A,
    BULK_HALFWIDTH_A,
    HBOND_ANGLE_DEG,
    HBOND_OO_CUTOFF_A,
    REGION_BIN_A,
    SEEDS,
)
from common.slab import (
    assign_regions,
    hydrogen_bond_counts,
    minimum_image,
    slab_geometry,
    water_indices,
)


def band_directory(inner, outer):
    """Directory name that records an interface band.

    Parameters
    ----------
    inner, outer : float
        Inward depth bounds in angstroms.

    Returns
    -------
    str
        ``awi_<inner>-<outer>``.
    """
    return f"awi_{inner:g}-{outer:g}"


def donor_acceptor_metrics():
    """Load the donor-acceptor summary from the hypothesis module.

    Returns
    -------
    callable
        ``frame_metrics(oxygen, hydrogen, lengths, references, z, depth, ...)``.
    """
    import importlib.util

    path = Path(__file__).with_name("04_analyze_hypothesis.py")
    spec = importlib.util.spec_from_file_location("opc_hb_hypothesis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.frame_metrics


def summarize_frame(
    oxygen,
    hydrogen,
    lengths,
    references,
    z_relative,
    cutoff,
    angle,
    outward_cos,
    batch,
):
    """Summarize coordination and free-OH orientation for one region.

    Parameters
    ----------
    oxygen, hydrogen : numpy.ndarray
        Coordinates of every water, with hydrogens shaped ``(n_water, 2, 3)``.
    lengths : array_like
        Orthorhombic box lengths.
    references : numpy.ndarray of int
        Waters assigned to this region.
    z_relative : numpy.ndarray
        Slab-centered ``z`` of every water.
    cutoff : float
        Oxygen-oxygen cutoff in angstroms.
    angle : float
        Minimum O-H···O angle in degrees.
    outward_cos : float
        Minimum cosine between an OH bond and the outward surface normal.
    batch : int
        Neighbor-list batch size.

    Returns
    -------
    metrics : dict
        Region-level fractions. Free-OH fractions use ``2 * n_water`` as
        the denominator unless the name says they are per water.
    coordination : list of dict
        Joint donor/acceptor histogram for this frame.
    orientation : list of dict
        Histogram of the outward cosine of free OH groups.
    """
    n_donor, n_acceptor, occupied = hydrogen_bond_counts(
        oxygen, hydrogen, lengths, references, cutoff, angle, batch
    )
    oh_vectors = minimum_image(
        hydrogen[references] - oxygen[references, None, :], lengths
    )
    unit_z = oh_vectors[:, :, 2] / np.maximum(np.linalg.norm(oh_vectors, axis=2), 1e-12)
    outward = np.where(z_relative[references, None] >= 0.0, unit_z, -unit_z)
    free = ~occupied
    pointing = outward >= outward_cos
    coordination_number = n_donor + n_acceptor
    n_ref = len(references)
    defective = coordination_number != 4
    three_fold = coordination_number == 3
    metrics = {
        "n_waters": n_ref,
        "mean_hb": float(np.mean(coordination_number)),
        "mean_donor": float(np.mean(n_donor)),
        "mean_acceptor": float(np.mean(n_acceptor)),
        "p_defect": float(np.mean(defective)),
        "p_under": float(np.mean(coordination_number < 4)),
        "p_over": float(np.mean(coordination_number > 4)),
        "p_n3": float(np.mean(three_fold)),
        "p_n4": float(np.mean(coordination_number == 4)),
        "p_n5": float(np.mean(coordination_number == 5)),
        "p_donor_defect": float(np.mean(n_donor != 2)),
        "p_acceptor_defect": float(np.mean(n_acceptor != 2)),
        "p_donor_missing": float(np.mean(n_donor < 2)),
        "p_acceptor_missing": float(np.mean(n_acceptor < 2)),
        "p_free_oh": float(np.mean(free)),
        "p_free_oh_outward": float(np.mean(free & pointing)),
        "p_free_oh_pointing_given_free": float(
            np.sum(free & pointing) / max(int(np.sum(free)), 1)
        ),
        "p_water_any_free_oh": float(np.mean(free.any(axis=1))),
        "p_water_any_outward_free_oh": float(np.mean((free & pointing).any(axis=1))),
        "p_defect_with_any_free_oh": (
            float(np.mean(free.any(axis=1)[defective])) if np.any(defective) else np.nan
        ),
        "p_defect_with_outward_free_oh": (
            float(np.mean((free & pointing).any(axis=1)[defective]))
            if np.any(defective)
            else np.nan
        ),
        "p_3fold_with_outward_free_oh": (
            float(np.mean((free & pointing).any(axis=1)[three_fold]))
            if np.any(three_fold)
            else np.nan
        ),
        "n_free_oh": int(np.sum(free)),
        "n_outward_free_oh": int(np.sum(free & pointing)),
        "n_defect_waters": int(np.sum(defective)),
        "n_defect_with_outward_free_oh": int(
            np.sum(defective & (free & pointing).any(axis=1))
        ),
        "n_threefold": int(np.sum(three_fold)),
        "n_threefold_with_outward_free_oh": int(
            np.sum(three_fold & (free & pointing).any(axis=1))
        ),
    }
    pairs = np.stack([n_donor, n_acceptor], axis=1)
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    coordination = []
    for (donated, accepted), count in zip(unique, counts):
        coordination.append(
            {
                "n_donor": int(donated),
                "n_acceptor": int(accepted),
                "n_hb": int(donated + accepted),
                "count": int(count),
                "prob": float(count / n_ref),
            }
        )
    edges = np.linspace(-1.0, 1.0, 21)
    free_orientation = outward[free]
    histogram = np.histogram(free_orientation, bins=edges)[0]
    orientation = [
        {
            "cos_outward_mid": float(0.5 * (left + right)),
            "count": int(count),
            "prob_given_free": float(count / max(len(free_orientation), 1)),
        }
        for left, right, count in zip(edges[:-1], edges[1:], histogram)
    ]
    return metrics, coordination, orientation


def selected_frames(n_frames, n_requested, first=0):
    """Return unique, evenly spaced frame indices.

    Parameters
    ----------
    n_frames : int
        Length of the concatenated trajectory.
    n_requested : int
        Maximum number of frames to keep.
    first : int, optional
        First frame that may be selected. A hold passes the frame index
        of its manifest ``discard_ps``. The index is into the concatenated
        trajectory, so a discard that ends in a later segment is applied once.

    Returns
    -------
    numpy.ndarray of int
        Increasing frame indices, including the last frame when
        ``n_requested`` is at least 1.

    Raises
    ------
    ValueError
        If the trajectory is empty or ``first`` leaves no frame.
    """
    if n_frames < 1:
        raise ValueError("Trajectory contains no frames.")
    if first < 0 or first >= n_frames:
        raise ValueError("The start frame removes every frame.")
    count = min(n_requested, n_frames - first)
    return np.unique(np.linspace(first, n_frames - 1, count, dtype=int))


def analyze_trajectory(args):
    """Sample one trajectory and write the free-OH tables.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    pathlib.Path
        Directory that received the CSV and JSON outputs.
    """
    universe = mda.Universe(str(args.top), [str(path) for path in args.traj], format="NCDF")
    oxygen_index, hydrogen_index = water_indices(universe)
    if len(oxygen_index) < 2:
        raise ValueError("Topology contains fewer than two waters.")
    rng = np.random.default_rng(args.seed)
    frame_ids = selected_frames(
        len(universe.trajectory), args.frames, getattr(args, "start_frame", 0)
    )
    frame_metrics = donor_acceptor_metrics()
    stored = {
        band: {"rows": [], "coordination": [], "orientation": [], "qc": [],
               "hypothesis": [], "classes": [], "angles": []}
        for band in args.bands
    }
    for frame_id in frame_ids:
        frame = universe.trajectory[int(frame_id)]
        lengths = np.asarray(frame.dimensions[:3], dtype=float)
        if not np.allclose(frame.dimensions[3:], 90.0, atol=0.05):
            raise ValueError("This analysis requires an orthorhombic cell.")
        if np.any(lengths <= 0.0):
            raise ValueError("Simulation box lengths must be positive.")
        oxygen = np.asarray(frame.positions[oxygen_index], dtype=float) % lengths
        hydrogen = np.asarray(frame.positions[hydrogen_index], dtype=float)
        z_relative, depth, bounds, center = slab_geometry(
            oxygen, lengths, args.dz, args.plateau_halfwidth
        )
        for inner, outer in args.bands:
            regions, overlap_qc = assign_regions(
                z_relative,
                depth,
                args.core_halfwidth,
                inner,
                outer,
                policy="error",
            )
            bucket = stored[(inner, outer)]
            for region, eligible in regions.items():
                if len(eligible) < args.min_region_waters:
                    raise ValueError(
                        f"Frame {frame_id}, {inner:g}-{outer:g} A, {region}: "
                        f"only {len(eligible)} eligible waters."
                    )
                references = np.sort(
                    rng.choice(eligible, size=min(args.nref, len(eligible)), replace=False)
                )
                metrics, coordination, orientation = summarize_frame(
                    oxygen,
                    hydrogen,
                    lengths,
                    references,
                    z_relative,
                    args.distance,
                    args.angle,
                    args.outward_cos,
                    args.batch,
                )
                basic = {
                    "frame": int(frame_id),
                    "region": region,
                    "awi_inner_A": inner,
                    "awi_outer_A": outer,
                }
                bucket["rows"].append(
                    {**basic, "eligible_waters": int(len(eligible)), **metrics}
                )
                bucket["coordination"].extend({**basic, **item} for item in coordination)
                bucket["orientation"].extend({**basic, **item} for item in orientation)
                hypothesis, classes, angles = frame_metrics(
                    oxygen,
                    hydrogen,
                    lengths,
                    references,
                    z_relative,
                    depth,
                    args.distance,
                    args.angle,
                    args.outward_cos,
                    args.batch,
                )
                bucket["hypothesis"].append({**basic, **hypothesis})
                bucket["classes"].extend({**basic, **item} for item in classes)
                bucket["angles"].extend({**basic, **item} for item in angles)
            bucket["qc"].append(
                {
                    "frame": int(frame_id),
                    "awi_inner_A": inner,
                    "awi_outer_A": outer,
                    "z_center_A": float(center),
                    "gds_lower_A": float(bounds[0]),
                    "gds_upper_A": float(bounds[1]),
                    "n_waters": len(oxygen_index),
                    **overlap_qc,
                }
            )
        print(f"frame {frame_id} / {len(universe.trajectory) - 1}", flush=True)

    root = Path(args.out)
    for (inner, outer), bucket in stored.items():
        destination = root / band_directory(inner, outer)
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(bucket["rows"]).to_csv(destination / "free_oh_frame_summary.csv", index=False)
        pd.DataFrame(bucket["coordination"]).to_csv(
            destination / "free_oh_joint_coordination.csv", index=False
        )
        pd.DataFrame(bucket["orientation"]).to_csv(
            destination / "free_oh_orientation.csv", index=False
        )
        pd.DataFrame(bucket["qc"]).to_csv(destination / "free_oh_qc.csv", index=False)
        pd.DataFrame(bucket["hypothesis"]).to_csv(destination / "hypotheses_frame.csv", index=False)
        pd.DataFrame(bucket["classes"]).to_csv(destination / "hypotheses_classes.csv", index=False)
        pd.DataFrame(bucket["angles"]).to_csv(
            destination / "hypotheses_acceptor_orientation.csv", index=False
        )
        metadata = {}
        for key, value in vars(args).items():
            if isinstance(value, Path):
                metadata[key] = str(value)
            elif isinstance(value, tuple):
                metadata[key] = [list(item) if isinstance(item, tuple) else item for item in value]
            else:
                metadata[key] = value
        metadata.update(
            awi_inner_A=inner,
            awi_outer_A=outer,
            nwaters=len(oxygen_index),
            nframes=len(universe.trajectory),
            frames_selected=[int(item) for item in frame_ids],
            contrast="bulk_minus_awi",
            overlap_policy="error",
            hydrogen_bond="O-O < cutoff and O-H...O angle > threshold; one count per directed water pair.",
            region_assignment="Bulk |z|<=15 A and one depth band. Overlap is an error.",
        )
        (destination / "free_oh_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        print(f"Wrote {destination}", flush=True)
    return root


def main(argv=None):
    """Parse arguments and analyze one trajectory.

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
    parser.add_argument("--top", type=Path, required=True)
    parser.add_argument("--traj", type=Path, nargs="+", action="extend", required=True)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First concatenated frame kept. A hold passes its manifest discard.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--nref", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=SEEDS["hydrogen_bond_free_oh"])
    parser.add_argument("--core-halfwidth", type=float, default=BULK_HALFWIDTH_A)
    parser.add_argument("--plateau-halfwidth", type=float, default=BULK_HALFWIDTH_A)
    parser.add_argument("--distance", type=float, default=HBOND_OO_CUTOFF_A)
    parser.add_argument("--angle", type=float, default=HBOND_ANGLE_DEG)
    parser.add_argument("--outward-cos", type=float, default=0.5)
    parser.add_argument("--dz", type=float, default=REGION_BIN_A)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--min-region-waters", type=int, default=20)
    args = parser.parse_args(argv)
    args.bands = (AWI_OUTER_A, AWI_SUB_A)
    if args.frames < 1 or args.nref < 1:
        parser.error("--frames and --nref must be positive.")
    if not -1.0 <= args.outward_cos <= 1.0:
        parser.error("--outward-cos must lie in [-1, 1].")
    analyze_trajectory(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
