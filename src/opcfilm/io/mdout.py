"""Parse Amber ``mdout`` thermodynamic records into plain Python rows."""

from __future__ import annotations

import re
from pathlib import Path

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


_BANNER_BYTES = (b"A V E R A G E S", b"FINAL RESULTS", b"F L U C T")
_HEAD_BYTES = 262144
_TAIL_BYTES = 65536
_CHUNK_BYTES = 8 * 1024 * 1024
_TIME_RE = re.compile(r"TIME\(PS\)\s*=\s*(" + _FLOAT + r")", re.IGNORECASE)
_CUT_RE = re.compile(r"^\s*cut\s*=\s*([0-9]*\.?[0-9]+)")
_NTWX_RE = re.compile(r"^\s*ntwx\s*=\s*([0-9]+)")
_DT_RE = re.compile(r"^\s*dt\s*=\s*([0-9]*\.?[0-9]+)")
_INPCRD_RE = re.compile(r"INPCRD:\s*(\S+)")


def _banner_offset(data):
    """Byte offset of the first averages, RMS, or final-results banner.

    Parameters
    ----------
    data : bytes
        A slice of an mdout file.

    Returns
    -------
    int or None
        Offset of the earliest banner in ``data``, or None when the slice
        has no banner.
    """
    found = None
    for needle in _BANNER_BYTES:
        index = data.find(needle)
        if index != -1 and (found is None or index < found):
            found = index
    return found


def _summary_from_text(text):
    """Header fields and the time span in one mdout text slice.

    Parameters
    ----------
    text : str
        Mdout text that ends at the first banner. Lines after a banner in
        the slice are ignored.

    Returns
    -------
    dict
        ``cut`` is a float or None. ``ntwx``, ``frame_ps``, and ``inpcrd``
        are strings. ``duration_ps`` is a float or None. ``first_time``,
        ``last_time``, and ``time_count`` describe the printed steps.
    """
    cut = None
    ntwx = None
    dt = None
    inpcrd = None
    first = None
    last = None
    count = 0
    for line in text.splitlines():
        if _SUMMARY.search(line) or _FINAL.search(line):
            break
        if "cut" in line:
            found = _CUT_RE.search(line)
            if found:
                cut = float(found.group(1))
        if "ntwx" in line:
            found = _NTWX_RE.search(line)
            if found:
                ntwx = found.group(1)
        if "dt" in line:
            found = _DT_RE.search(line)
            if found:
                dt = found.group(1)
        if "INPCRD" in line:
            found = _INPCRD_RE.search(line)
            if found:
                inpcrd = found.group(1)
        if "TIME(PS)" in line:
            found = _TIME_RE.search(line)
            if found:
                value = fortran_float(found.group(1))
                if count == 0:
                    first = value
                last = value
                count += 1
    duration = None
    if count >= 2 and first is not None and last is not None:
        duration = float(last - first)
    frame_ps = ""
    if ntwx is not None and dt is not None:
        frame_ps = f"{float(ntwx) * float(dt):g}"
    return {
        "cut": cut,
        "ntwx": "" if ntwx is None else ntwx,
        "frame_ps": frame_ps,
        "inpcrd": "" if inpcrd is None else inpcrd,
        "duration_ps": duration,
        "first_time": first,
        "last_time": last,
        "time_count": count,
    }


def _public_summary(summary, duration):
    """Fields the catalog stores, without the scan bookkeeping.

    Parameters
    ----------
    summary : dict
        Header summary from the start of the file.
    duration : float or None
        Printed time span in picoseconds.

    Returns
    -------
    dict
        ``cut``, ``ntwx``, ``frame_ps``, ``inpcrd``, and ``duration_ps``.
    """
    return {
        "cut": summary["cut"],
        "ntwx": summary["ntwx"],
        "frame_ps": summary["frame_ps"],
        "inpcrd": summary["inpcrd"],
        "duration_ps": duration,
    }


def summarize_mdout(path):
    """Read the mdout fields the catalog needs without parsing every step.

    The cutoff, ``ntwx``, ``dt``, and restart are in the header. The duration
    is the last ``TIME(PS)`` before the first averages, RMS, or final-results
    banner, minus the first ``TIME(PS)``. A long run is scanned for that
    banner instead of turning every energy line into a record.

    Parameters
    ----------
    path : path-like
        Amber mdout.

    Returns
    -------
    dict
        ``cut`` is a float or None. ``ntwx``, ``frame_ps``, and ``inpcrd``
        are strings. ``duration_ps`` is a float or None when fewer than two
        steps were printed.
    """
    path = Path(path)
    with path.open("rb") as handle:
        head = handle.read(_HEAD_BYTES)
        banner = _banner_offset(head)
        if banner is not None:
            text = head[:banner].decode("utf-8", errors="replace")
            summary = _summary_from_text(text)
            return _public_summary(summary, summary["duration_ps"])
        newline = head.rfind(b"\n")
        if newline != -1:
            head_text = head[: newline + 1]
            resume = newline + 1
        else:
            head_text = head
            resume = len(head)
        summary = _summary_from_text(head_text.decode("utf-8", errors="replace"))
        handle.seek(resume)
        overlap = head[max(0, resume - 32) : resume]
        position = resume
        found = None
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            data = overlap + chunk
            relative = _banner_offset(data)
            if relative is not None:
                found = position - len(overlap) + relative
                break
            overlap = data[-32:]
            position += len(chunk)
        if found is None:
            handle.seek(0, 2)
            size = handle.tell()
            start = max(resume, size - _TAIL_BYTES)
            handle.seek(start)
            window = handle.read().decode("utf-8", errors="replace")
        else:
            start = max(resume, found - _TAIL_BYTES)
            handle.seek(start)
            window = handle.read(found - start).decode("utf-8", errors="replace")
        tail = _summary_from_text(window)
        if tail["time_count"]:
            last = tail["last_time"]
            total = summary["time_count"] + tail["time_count"]
        else:
            last = summary["last_time"]
            total = summary["time_count"]
        duration = None
        if total >= 2 and summary["first_time"] is not None and last is not None:
            duration = float(last - summary["first_time"])
        return _public_summary(summary, duration)


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
    return summarize_mdout(path)["duration_ps"]


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
