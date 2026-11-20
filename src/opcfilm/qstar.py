"""Adopt one in-plane peak from the row measurements.

``02_sq2_fft.py`` writes ``qstar_row.json`` beside its own table. This module
is the only writer of ``outputs/qstar.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from opcfilm.catalog import load_manifest
from opcfilm.config import CONFIG, adopted_qstar


def selected_rows(rows):
    """Rows declared in ``qstar.adopt_from``.

    Parameters
    ----------
    rows : sequence of dict
        Manifest rows.

    Returns
    -------
    list of dict
        Rows whose kind and temperature are inside the declared set.
    """
    spec = CONFIG.qstar.adopt_from
    kinds = set(spec["kinds"])
    low = float(spec["temperature_min_K"])
    high = float(spec["temperature_max_K"])
    chosen = []
    for row in rows:
        if row.get("kind") not in kinds:
            continue
        text = str(row.get("temperature_K", "")).strip()
        if text == "":
            continue
        temperature = float(text)
        if low <= temperature <= high:
            chosen.append(row)
    return chosen


def _row_peak(row):
    path = Path(row["outdir"]) / "qstar_row.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run 02_sq2_fft.py for this row before adopting q*."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "qstar_measured_Ainv" not in payload:
        raise ValueError(f"{path} has no qstar_measured_Ainv.")
    return path, float(payload["qstar_measured_Ainv"])


def adopt_files(paths, destination):
    """Write the one adopted peak from per-row JSON files.

    Parameters
    ----------
    paths : sequence of path-like
        ``qstar_row.json`` files. Each must contain ``qstar_measured_Ainv``.
    destination : path-like
        ``outputs/qstar.json``. This function is the only writer of that file.

    Returns
    -------
    dict
        The payload that was written.

    Raises
    ------
    ValueError
        If ``paths`` is empty or a file has no measured peak.
    """
    records = []
    for path in paths:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "qstar_measured_Ainv" not in payload:
            raise ValueError(f"{path} has no qstar_measured_Ainv.")
        records.append({
            "kind": "",
            "replica": "",
            "temperature_K": payload.get("temperature_K", ""),
            "rate_K_per_ns": "",
            "qstar_measured_Ainv": float(payload["qstar_measured_Ainv"]),
            "source": str(path),
        })
    if not records:
        raise ValueError("No per-row q* files were given.")
    return _write(records, destination, {"source": "files"})


def adopt(manifest, destination):
    """Write the one adopted peak.

    Parameters
    ----------
    manifest : path-like
        Project manifest.
    destination : path-like
        ``outputs/qstar.json``, or another path for a test.

    Returns
    -------
    dict
        The payload that was written. ``qstar_adopted_Ainv`` is
        :func:`opcfilm.config.adopted_qstar` applied once to the mean of
        the selected measured peaks.

    Raises
    ------
    ValueError
        If the declared set contains no row, or a selected row has no
        ``qstar_row.json``.
    """
    rows = selected_rows(load_manifest(manifest))
    if not rows:
        raise ValueError("qstar.adopt_from matched no manifest rows.")
    records = []
    for row in rows:
        path, measured = _row_peak(row)
        records.append({
            "kind": row.get("kind", ""),
            "replica": row.get("replica", ""),
            "temperature_K": float(row["temperature_K"]),
            "rate_K_per_ns": row.get("rate_K_per_ns", ""),
            "qstar_measured_Ainv": measured,
            "source": str(path),
        })
    return _write(records, destination, CONFIG.qstar.adopt_from)


def _write(records, destination, adopt_from):
    """Apply ``adopted_qstar`` once and write the shared file.

    Parameters
    ----------
    records : sequence of dict
        One measured peak per selected row.
    destination : path-like
        Shared JSON path.
    adopt_from : dict
        The selection that produced ``records``.

    Returns
    -------
    dict
        The payload that was written.
    """
    measured = np.array([record["qstar_measured_Ainv"] for record in records], dtype=float)
    by_temperature = []
    for temperature in sorted({record["temperature_K"] for record in records}):
        values = measured[[record["temperature_K"] == temperature for record in records]]
        by_temperature.append({
            "temperature_K": temperature,
            "mean_Ainv": float(np.mean(values)),
            "n": int(len(values)),
        })
    mean = float(np.mean(measured))
    spread = float(np.std(measured, ddof=1)) if len(measured) > 1 else float("nan")
    payload = {
        "qstar_adopted_Ainv": adopted_qstar(mean, tolerance=CONFIG.qstar.tolerance_Ainv),
        "qstar_measured_mean_Ainv": mean,
        "qstar_measured_sd_Ainv": spread,
        "n_rows": len(records),
        "adopt_from": adopt_from,
        "per_temperature": by_temperature,
        "rows": records,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def default_destination(root):
    """Path of the shared peak file.

    Parameters
    ----------
    root : path-like
        Repository directory.

    Returns
    -------
    pathlib.Path
        ``<root>/<QSTAR_JSON>``.
    """
    return Path(root) / CONFIG.qstar.json
