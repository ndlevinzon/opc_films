#!/usr/bin/env python3
"""Score bulk and interface water with the one 300 K versus 78 K classifier.

Training waters are every oxygen inside ``|z| <= 15`` Å, on evenly spaced
frames of a per-replica endpoint trajectory. A concatenated endpoint file
with no replica label is refused. Each replica is scored by a model trained
on the other two. The reported order parameter is

    f~(T) = [f(T) - f(300 K)] / [f(78 K) - f(300 K)]

computed separately in the bulk and in each interface band. Raw ``f`` is
written as well. A larger raw ``f`` in the bulk is not evidence that the
bulk vitrifies first.

The geometric probe uses radius 0.80 Å and an oxygen exclusion radius of
1.52 Å. It is an insertion fraction, not a pore volume.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.classifier import (
    CLASSIFIER_ID,
    band_intervals,
    fit_discriminant,
    leave_one_replica_out,
    model_record,
    normalized_fraction,
    sampled_frames,
    score_discriminant,
    selections,
    structure_features,
    write_panel_record,
)
from common.constants import BULK_HALFWIDTH_A, HOLD_DISCARD_FRACTION, SEEDS
from common.rates import REPLICAS
from common.manifest import load_manifest, trajectory_list
from common.slab import water_indices
from opcfilm.config import CONFIG
from opcfilm.structure.free_volume import probe_fraction


OXYGEN_EXCLUSION_A = CONFIG.probe.oxygen_exclusion_A
PROBE_RADIUS_A = CONFIG.probe.radius_A
PROBE_SENSITIVITY_A = CONFIG.probe.sensitivity_radii_A


def bulk_features(topology, trajectories, n_frames):
    """Bulk ``q``, ``Sk``, and LSI on evenly spaced frames.

    Parameters
    ----------
    topology : path-like
        Amber topology.
    trajectories : path-like or sequence of path-like
        Endpoint segments for one replica. A continuation is one row.
    n_frames : int
        Number of frames, including both ends when the trajectory is longer.

    Returns
    -------
    numpy.ndarray, shape (n, 3)
        Features of every bulk oxygen on the selected frames.
    """
    import MDAnalysis as mda

    if isinstance(trajectories, (str, Path)):
        trajectories = [trajectories]
    universe = mda.Universe(str(topology), [str(path) for path in trajectories], format="NCDF")
    oxygen_index, _hydrogens = water_indices(universe)
    chosen = sampled_frames(len(universe.trajectory), n_frames, "endpoint")
    blocks = []
    for frame in chosen:
        universe.trajectory[int(frame)]
        oxygen = np.asarray(universe.atoms[oxygen_index].positions, dtype=float)
        lengths = np.asarray(universe.trajectory.ts.dimensions[:3], dtype=float)
        mask = selections(oxygen, lengths)["masks"]["interior"]
        blocks.append(structure_features(oxygen, lengths, np.flatnonzero(mask)))
    if not blocks:
        raise ValueError(f"{trajectory} produced no training frames.")
    return np.vstack(blocks)


def score_frame(oxygen, lengths, model, n_probes, rng):
    """Regional scores and probe fractions for one frame.

    Parameters
    ----------
    oxygen : numpy.ndarray, shape (n, 3)
        Oxygen coordinates.
    lengths : array_like, shape (3,)
        Box lengths of this frame.
    model : dict
        Discriminant trained without this replica.
    n_probes : int
        Probe centers per region.
    rng : numpy.random.Generator
        Sampler for the probe.

    Returns
    -------
    list of dict
        One record per region. ``f_lda`` is the glass-like fraction.
    """
    from common.slab import slab_geometry

    _relative, _depth, bounds, center = slab_geometry(oxygen, lengths)
    named = selections(oxygen, lengths)["masks"]
    intervals = {
        "interior": [(-BULK_HALFWIDTH_A, BULK_HALFWIDTH_A)],
        "awi_3_12": band_intervals(bounds, 3.0, 12.0),
        "awi_2_10": band_intervals(bounds, 2.0, 10.0),
        "awi_4_14": band_intervals(bounds, 4.0, 14.0),
    }
    rows = []
    for name, mask in named.items():
        scores = score_discriminant(structure_features(oxygen, lengths, np.flatnonzero(mask)), model)
        finite = scores[np.isfinite(scores)]
        record = {
            "region": name,
            "n_waters": int(finite.size),
            "f_lda": float(np.mean(finite >= 0.5)) if finite.size else np.nan,
            "score_mean": float(np.mean(finite)) if finite.size else np.nan,
            "free_volume_fraction": probe_fraction(
                oxygen, lengths, center, intervals[name], n_probes,
                OXYGEN_EXCLUSION_A + PROBE_RADIUS_A, rng,
            ),
        }
        for radius in PROBE_SENSITIVITY_A:
            record[f"free_volume_r{radius:g}"] = probe_fraction(
                oxygen, lengths, center, intervals[name], n_probes,
                OXYGEN_EXCLUSION_A + radius, rng,
            )
        rows.append(record)
    return rows


def _write_csv(path, rows):
    """Write dictionaries as a tab-separated table.

    Parameters
    ----------
    path : pathlib.Path
        Destination. Parent directories are created.
    rows : sequence of dict
        Records. An empty sequence writes an empty file.

    Returns
    -------
    None
        The table is written to ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _endpoint_features(rows, temperature, n_frames):
    """Bulk features at one endpoint, grouped by replica.

    Parameters
    ----------
    rows : sequence of dict
        Project-manifest rows.
    temperature : float
        Endpoint temperature in kelvin.
    n_frames : int
        Frames taken from each endpoint trajectory.

    Returns
    -------
    dict
        Replica name to a feature array. A concatenated file with no
        replica is not accepted.
    """
    grouped = {}
    matched = [
        row for row in rows
        if row["kind"] == "endpoint" and row["temperature_K"] != ""
        and abs(float(row["temperature_K"]) - temperature) < 0.5
    ]
    if not matched:
        raise SystemExit(
            f"The manifest has no per-replica endpoint at {temperature:g} K. "
            "A concatenated 300K_prop_analysis or 78K_prop_analysis file is not a training set."
        )
    for row in matched:
        paths = trajectory_list(row)
        if not paths:
            raise SystemExit(f"Endpoint row {row['label']} names no trajectory.")
        grouped.setdefault(row["replica"], []).append(bulk_features(row["topology"], paths, n_frames))
    return {replica: np.vstack(blocks) for replica, blocks in grouped.items()}


def _models(features_300, features_78):
    """Fit one discriminant per held-out replica.

    Parameters
    ----------
    features_300, features_78 : dict
        Replica name to bulk features at 300 K and at 78 K.

    Returns
    -------
    tuple
        The paired feature dict and ``{held_out_replica: model}``.
    """
    names = sorted(set(features_300) & set(features_78))
    if len(names) < 3:
        raise SystemExit(
            "Leave-one-replica training needs 300 K and 78 K endpoints for three replicas. "
            f"Found {names}."
        )
    by_replica = {name: {"300": features_300[name], "78": features_78[name]} for name in names}
    models = {}
    for held in names:
        liquid = np.vstack([features_300[name] for name in names if name != held])
        glass = np.vstack([features_78[name] for name in names if name != held])
        models[held] = fit_discriminant(liquid, glass)
    return by_replica, models


def _normalized(frame_rows):
    """Attach per-region endpoint-normalized fractions.

    Parameters
    ----------
    frame_rows : list of dict
        Scored frames, including the 300 K and 78 K endpoints.

    Returns
    -------
    list of dict
        The same rows, with ``f_normalized`` and
        ``free_volume_normalized``. A region whose endpoints do not
        separate is left blank.
    """
    endpoints = {}
    for row in frame_rows:
        if row["kind"] != "endpoint":
            continue
        endpoints.setdefault((row["replica"], row["region"]), {})[round(float(row["temperature_K"]))] = row
    for row in frame_rows:
        pair = endpoints.get((row["replica"], row["region"]), {})
        if 300 not in pair or 78 not in pair:
            row["f_normalized"] = ""
            row["free_volume_normalized"] = ""
            continue
        try:
            row["f_normalized"] = float(normalized_fraction(
                float(row["f_lda"]), pair[300]["f_lda"], pair[78]["f_lda"]
            ))
            row["free_volume_normalized"] = float(normalized_fraction(
                float(row["free_volume_fraction"]),
                pair[300]["free_volume_fraction"],
                pair[78]["free_volume_fraction"],
            ))
        except (TypeError, ValueError):
            row["f_normalized"] = ""
            row["free_volume_normalized"] = ""
    return frame_rows


def _summary(frame_rows):
    """Replica means of the raw and normalized regional fractions.

    Parameters
    ----------
    frame_rows : sequence of dict
        Frame records that already contain ``f_normalized``.

    Returns
    -------
    list of dict
        One row per kind, temperature, and region, with the replica mean
        and sample standard deviation.
    """
    per_replica = {}
    fields = ("f_lda", "f_normalized", "free_volume_fraction", "free_volume_normalized")
    for row in frame_rows:
        if row.get("f_normalized", "") == "":
            continue
        key = (row["kind"], row["temperature_K"], row["region"], row["replica"])
        per_replica.setdefault(key, []).append(row)
    replica_rows = []
    for key, items in per_replica.items():
        replica_rows.append({
            "kind": key[0],
            "temperature_K": float(key[1]),
            "region": key[2],
            "replica": key[3],
            **{name: float(np.nanmean([float(item[name]) for item in items])) for name in fields},
            "f_normalized_frame_width": _frame_width(items, "f_normalized"),
            "free_volume_fraction_frame_width": _frame_width(items, "free_volume_fraction"),
        })
    grouped = {}
    for row in replica_rows:
        grouped.setdefault((row["kind"], row["temperature_K"], row["region"]), []).append(row)
    summary = []
    for key, items in grouped.items():
        record = {
            "classifier_id": CLASSIFIER_ID,
            "kind": key[0],
            "temperature_K": key[1],
            "region": key[2],
            "n_replicas": len({item["replica"] for item in items}),
        }
        for name in fields:
            values = np.array([item[name] for item in items], dtype=float)
            record[f"{name}_mean"] = float(np.nanmean(values))
            record[f"{name}_sd"] = float(np.nanstd(values, ddof=1)) if np.sum(np.isfinite(values)) > 1 else np.nan
        record["f_normalized_frame_width"] = float(np.nanmean(
            [item["f_normalized_frame_width"] for item in items]
        ))
        record["free_volume_fraction_frame_width"] = float(np.nanmean(
            [item["free_volume_fraction_frame_width"] for item in items]
        ))
        summary.append(record)
    return summary


def _frame_width(items, name):
    """Sample standard deviation of one column across frames of one replica.

    Parameters
    ----------
    items : sequence of dict
        Frames of one replica and region.
    name : str
        Column to summarize.

    Returns
    -------
    float
        Frame-to-frame width. This is not a replica uncertainty.
    """
    values = np.array([float(item[name]) for item in items], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return np.nan
    return float(np.std(finite, ddof=1))


def _require_replicas(frame_rows, allow_missing):
    """Stop when a temperature is missing a replica.

    Parameters
    ----------
    frame_rows : sequence of dict
        Scored frames.
    allow_missing : bool
        When true, a missing replica is not an error.

    Returns
    -------
    None
        Raises ``SystemExit`` when a scored temperature lacks a replica.
    """
    if allow_missing:
        return
    present = {}
    for row in frame_rows:
        present.setdefault((row["kind"], row["temperature_K"], row["rate_K_ns"]), set()).add(row["replica"])
    for key, replicas in present.items():
        missing = [name for name in REPLICAS if name not in replicas]
        if missing:
            raise SystemExit(f"Missing replicas {missing} for {key}. Refusing to write a partial figure.")


def main(argv=None):
    """Fit the leave-one-replica classifier and score the manifest.

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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=CONFIG.classifier.frames)
    parser.add_argument("--allow-few-frames", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--probes", type=int, default=CONFIG.probe.n_probes)
    parser.add_argument("--seed", type=int, default=SEEDS["classifier"])
    args = parser.parse_args(argv)
    if args.frames < 20 and not args.allow_few_frames:
        parser.error("--frames must be at least 20. The frame-to-frame width is not an error bar.")
    rows = load_manifest(args.manifest)
    liquid = _endpoint_features(rows, 300.0, args.frames)
    glass = _endpoint_features(rows, 78.0, args.frames)
    by_replica, models = _models(liquid, glass)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_panel_record(
        args.outdir / "classifier_panel.json",
        CLASSIFIER_ID,
        "quantitative",
        "300 K versus 78 K, all waters with |z| <= 15 Å, "
        f"{args.frames} frames after dropping the first {HOLD_DISCARD_FRACTION:g} of each endpoint, "
        "leave one replica out",
        "Bulk-first claims use f_normalized in each region, not raw f_lda. "
        "The 250 K versus 150 K coloring is not this classifier.",
    )
    (args.outdir / "classifier_model.json").write_text(
        __import__("json").dumps({
            "classifier_id": CLASSIFIER_ID,
            "by_held_out_replica": {name: model_record(model) for name, model in models.items()},
        }, indent=2)
    )
    _write_csv(args.outdir / "leave_one_replica_out.tsv", leave_one_replica_out(by_replica))
    import MDAnalysis as mda

    rng = np.random.default_rng(args.seed)
    scored = []
    for row in rows:
        if row["temperature_K"] == "":
            continue
        if row["replica"] not in models:
            raise SystemExit(f"{row['replica']} has no leave-one-replica model.")
        universe = mda.Universe(row["topology"], [str(path) for path in trajectory_list(row)], format="NCDF")
        oxygen_index, _hydrogens = water_indices(universe)
        for frame in sampled_frames(len(universe.trajectory), args.frames, row["kind"]):
            universe.trajectory[int(frame)]
            oxygen = np.asarray(universe.atoms[oxygen_index].positions, dtype=float)
            lengths = np.asarray(universe.trajectory.ts.dimensions[:3], dtype=float)
            for region in score_frame(oxygen, lengths, models[row["replica"]], args.probes, rng):
                scored.append({
                    "classifier_id": CLASSIFIER_ID,
                    "kind": row["kind"],
                    "replica": row["replica"],
                    "temperature_K": row["temperature_K"],
                    "rate_K_ns": row["rate_K_ns"],
                    "label": row["label"],
                    "frame": int(frame),
                    **region,
                })
    scored = _normalized(scored)
    _require_replicas(scored, args.allow_missing)
    _write_csv(args.outdir / "frame_region_metrics.tsv", scored)
    _write_csv(args.outdir / "region_summary.tsv", _summary(scored))
    print(f"Wrote {args.outdir} with classifier {CLASSIFIER_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
