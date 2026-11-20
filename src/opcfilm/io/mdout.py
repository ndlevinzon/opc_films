"""Parse Amber ``mdout`` thermodynamic records into plain Python rows."""

from __future__ import annotations

import re

import pandas as pd

_FLOAT = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"
_KV_RE = re.compile(r"([A-Za-z][A-Za-z0-9_()/.-]*)\s*=\s*(" + _FLOAT + r")")

_SUMMARY = re.compile(
    r"A\s+V\s+E\s+R\s+A\s+G\s+E\s+S|R\s+M\s+S\s+F\s+L\s+U\s+C",
    re.IGNORECASE,
)
_FINAL = re.compile(r"\bFINAL RESULTS\b", re.IGNORECASE)

_KEY_MAP = {
    "TIME(PS)": "time_ps",
    "TEMP(K)": "temp_K",
    "PRESS": "press_bar",
    "Etot": "Etot",
    "EKtot": "EKtot",
    "EPtot": "EPtot",
    "VOLUME": "volume_A3",
    "Density": "density_g_cm3",
    "DENSITY": "density_g_cm3",
    "NSTEP": "nstep",
}


def fortran_float(text):
    """Parse a Fortran-style float, accepting ``D`` exponents.

    Parameters
    ----------
    text : str
        Numeric token from an mdout line.

    Returns
    -------
    float
        Parsed value.
    """
    return float(text.replace("D", "E").replace("d", "e"))


def normalize_key(key):
    """Map an Amber column label onto a stable CSV column name.

    Parameters
    ----------
    key : str
        Label as printed before ``=`` in the mdout file.

    Returns
    -------
    str
        Column name. Unrecognized labels are returned unchanged.
    """
    return _KEY_MAP.get(key.strip(), key.strip())


def parse_file(path):
    """Read one Amber mdout file into a list of thermodynamic records.

    A new record starts on every line that contains ``NSTEP``. Later
    key/value pairs on following lines update that record until the next
    ``NSTEP``. Parsing stops at ``AVERAGES``, ``RMS FLUCTUATIONS``, or
    ``FINAL RESULTS``. Those blocks are not time steps: the RMS potential
    is a fluctuation. Duplicate ``(nstep, time_ps)`` rows keep the last copy.

    Parameters
    ----------
    path : path-like
        Amber mdout path.

    Returns
    -------
    list of dict
        One dictionary per printed step. Each dictionary includes
        ``source`` set to ``path``.
    """
    rows = []
    current = None
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            if _FINAL.search(line) or _SUMMARY.search(line):
                break
            pairs = _KV_RE.findall(line)
            if not pairs:
                continue
            values = {normalize_key(key): fortran_float(raw) for key, raw in pairs}
            if "nstep" in values:
                if current is not None:
                    rows.append(current)
                current = {"source": str(path)}
            if current is not None:
                current.update(values)
        if current is not None:
            rows.append(current)
    return _drop_duplicate_steps(rows)


def _drop_duplicate_steps(rows):
    """Keep the last copy of each ``(nstep, time_ps)`` pair.

    Parameters
    ----------
    rows : list of dict
        Records in file order.

    Returns
    -------
    list of dict
        Records with repeated steps collapsed.
    """
    index = {}
    unique = []
    for row in rows:
        key = (row.get("nstep"), row.get("time_ps"))
        if None in key:
            unique.append(row)
            continue
        if key in index:
            unique[index[key]] = row
        else:
            index[key] = len(unique)
            unique.append(row)
    return unique


def nstep_span_matches(nstep_first, nstep_last, n_frames, ntwx, tolerance_frames=1):
    """Whether an mdout step range covers one trajectory.

    Parameters
    ----------
    nstep_first, nstep_last : int
        First and last ``NSTEP`` among the time-step records.
    n_frames : int
        Number of frames in the trajectory.
    ntwx : int
        Steps between stored frames.
    tolerance_frames : int, optional
        Accepted difference, in frames.

    Returns
    -------
    bool
        True when the step span is ``(n_frames - 1) * ntwx`` within tolerance.
    """
    expected = (int(n_frames) - 1) * int(ntwx)
    return abs(int(nstep_last) - int(nstep_first) - expected) <= int(tolerance_frames) * int(ntwx)


def time_span_ps(path):
    """Length of a finished run from its printed ``TIME(PS)`` values.

    Parameters
    ----------
    path : path-like
        Amber mdout. Averages and the RMS block are not included.

    Returns
    -------
    float or None
        Last printed time minus the first, in picoseconds. None when the
        file has fewer than two time steps.
    """
    times = [row["time_ps"] for row in parse_file(path) if "time_ps" in row]
    if len(times) < 2:
        return None
    return float(times[-1] - times[0])


def parse_dataframe(path):
    """Parse one mdout file into a table.

    Parameters
    ----------
    path : path-like
        Amber mdout path.

    Returns
    -------
    pandas.DataFrame
        One row per thermodynamic record. Empty if the file has no
        ``NSTEP`` lines.
    """
    return pd.DataFrame(parse_file(path))
