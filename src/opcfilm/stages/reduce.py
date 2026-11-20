"""Concatenate per-row tables and summarize them across replicas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opcfilm.stats.contrast import bulk_minus_awi
from opcfilm.stats.uncertainty import curve_band, paired_student_interval, replica_mean_sd

_GROUP = (
    "kind",
    "rate_K_per_ns",
    "source_rate_K_per_ns",
    "temperature_K",
    "region",
    "metric",
)


def reduce_stage(stage, root):
    """Concatenate one stage's per-row tables.

    Parameters
    ----------
    stage : str
        Stage name, for example ``s00``.
    root : path-like
        Directory that contains ``results/``.

    Returns
    -------
    pathlib.Path
        ``results/<stage>/reduced/summary.csv``.

    Raises
    ------
    FileNotFoundError
        If the stage has no ``metrics.csv``.
    """
    paths = sorted(Path(root).joinpath("results", stage).rglob("metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No metrics.csv under results/{stage}.")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    destination = Path(root) / "results" / stage / "reduced"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "summary.csv"
    frame.to_csv(path, index=False)
    return path


def _prepared(frame):
    """Copy a metrics table and fill the grouping columns.

    Parameters
    ----------
    frame : pandas.DataFrame
        Concatenated metrics.

    Returns
    -------
    pandas.DataFrame
        Numeric ``value`` and blank strings for missing group columns.
    """
    frame = frame.copy()
    for column in (*_GROUP, "replica"):
        if column not in frame.columns:
            frame[column] = ""
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def summarize_replicas(frame, allow_incomplete=False):
    """Replica mean and ± 1 SD band for each group.

    Parameters
    ----------
    frame : pandas.DataFrame
        Concatenated metrics with ``replica`` and ``value``.
    allow_incomplete : bool, optional
        When false, a group with fewer than three replicas is an error.

    Returns
    -------
    pandas.DataFrame
        One row per group, with ``n_replicas``, ``mean``, ``sd``,
        ``band_lo``, and ``band_hi``.

    Raises
    ------
    ValueError
        If a group has fewer than three replicas and incomplete groups
        are not allowed.
    """
    frame = _prepared(frame)
    rows = []
    for key, part in frame.groupby(list(_GROUP), dropna=False):
        per_replica = part.groupby("replica", dropna=False)["value"].mean()
        mean, sd, count = replica_mean_sd(per_replica.to_numpy())
        if count < 3 and not allow_incomplete:
            raise ValueError(f"{dict(zip(_GROUP, key))} has {count} replicas.")
        low, high = curve_band([mean], [sd])
        record = dict(zip(_GROUP, key))
        record.update({
            "n_replicas": count,
            "mean": mean,
            "sd": sd,
            "band_lo": float(low[0]),
            "band_hi": float(high[0]),
        })
        rows.append(record)
    return pd.DataFrame(rows)


def contrast_table(frame):
    """Paired bulk minus AWI, then the Student-t interval across replicas.

    Parameters
    ----------
    frame : pandas.DataFrame
        Metrics with regions ``interior`` and ``awi``.

    Returns
    -------
    pandas.DataFrame
        One row per group. ``difference`` is the replica mean of
        bulk minus AWI. ``halfwidth`` is the Student-t half-width.
    """
    frame = _prepared(frame)
    keys = [column for column in _GROUP if column != "region"]
    paired = []
    chosen = frame[frame["region"].isin(["interior", "awi"])]
    for key, part in chosen.groupby(["replica", *keys], dropna=False):
        bulk = part.loc[part["region"] == "interior", "value"]
        awi = part.loc[part["region"] == "awi", "value"]
        if bulk.empty or awi.empty:
            continue
        difference = float(bulk_minus_awi([bulk.mean()], [awi.mean()])[0])
        record = dict(zip(["replica", *keys], key))
        record["difference"] = difference
        paired.append(record)
    if not paired:
        return pd.DataFrame(columns=[*keys, "n_replicas", "difference", "sd", "halfwidth"])
    paired = pd.DataFrame(paired)
    rows = []
    for key, part in paired.groupby(keys, dropna=False):
        mean, sd, halfwidth, count = paired_student_interval(part["difference"])
        record = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        record.update({
            "n_replicas": count,
            "difference": mean,
            "sd": sd,
            "halfwidth": halfwidth,
        })
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_stage(stage, root, allow_incomplete=False):
    """Write the replica summary and the paired contrast.

    Parameters
    ----------
    stage : str
        Stage name.
    root : path-like
        Directory that contains ``results/``.
    allow_incomplete : bool, optional
        Allow a group with fewer than three replicas.

    Returns
    -------
    tuple of pathlib.Path
        ``summary_replicas.csv`` and ``contrast.csv``.
    """
    summary = reduce_stage(stage, root)
    frame = pd.read_csv(summary)
    destination = summary.parent
    replicas = destination / "summary_replicas.csv"
    contrast = destination / "contrast.csv"
    summarize_replicas(frame, allow_incomplete=allow_incomplete).to_csv(replicas, index=False)
    contrast_table(frame).to_csv(contrast, index=False)
    return replicas, contrast
