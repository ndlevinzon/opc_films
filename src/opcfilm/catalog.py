"""Trajectory catalog and the only reader of cooling-rate directory codes.

Physical rates are what every table stores. Directory codes appear in the
path patterns in this module and nowhere else.
"""

from __future__ import annotations

import numpy as np

RATE_CODES = (4, 11, 22, 44, 111)

RATE_CODE_TO_K_PER_NS = {
    4: 4.44,
    11: 11.1,
    22: 22.2,
    44: 44.4,
    111: 111.0,
}

RATE_LABEL = {code: f"{RATE_CODE_TO_K_PER_NS[code]:g}" for code in RATE_CODES}

RATE_COLOR = {
    4: "#1f77b4",
    11: "#ff7f0e",
    22: "#2ca02c",
    44: "#d62728",
    111: "#9467bd",
}

RATE_COLOR_BY_VALUE = {
    RATE_CODE_TO_K_PER_NS[code]: RATE_COLOR[code] for code in RATE_CODES
}

INTERVAL_COLOR = {
    (4, 11): "#1f77b4",
    (11, 22): "#ff7f0e",
    (22, 44): "#2ca02c",
    (44, 111): "#d62728",
}

PHYSICAL_RATES = tuple(RATE_CODE_TO_K_PER_NS[code] for code in RATE_CODES)

_CODE_BY_RATE = {rate: code for code, rate in RATE_CODE_TO_K_PER_NS.items()}


def parse_rate(value):
    """Return the physical cooling rate, in K/ns.

    Parameters
    ----------
    value : float or str
        A physical rate. Directory codes and file tags are rejected.

    Returns
    -------
    float
        One of 4.44, 11.1, 22.2, 44.4, or 111.0.

    Raises
    ------
    ValueError
        If ``value`` is not one of those five rates.
    """
    number = float(value)
    for rate in PHYSICAL_RATES:
        if abs(number - rate) <= 1e-6:
            return float(rate)
    raise ValueError(
        f"{value} is not a physical cooling rate. Pass one of {PHYSICAL_RATES} "
        "K/ns. Directory codes and file tags are not accepted on the command line."
    )


def directory_code(rate):
    """Directory code for a physical cooling rate.

    Parameters
    ----------
    rate : float
        Physical rate in K/ns.

    Returns
    -------
    int
        ``4``, ``11``, ``22``, ``44``, or ``111``.
    """
    return _CODE_BY_RATE[parse_rate(rate)]


def file_tag(rate):
    """File-name tag for a physical cooling rate.

    Parameters
    ----------
    rate : float
        Physical rate in K/ns.

    Returns
    -------
    str
        ``4p44``, ``11p1``, ``22p2``, ``44p4``, or ``111``.
    """
    rate = parse_rate(rate)
    if rate == 111.0:
        return "111"
    return f"{rate:g}".replace(".", "p")


def trajectory_directory(rate):
    """Trajectory-directory name for a physical cooling rate.

    Parameters
    ----------
    rate : float
        Physical rate in K/ns.

    Returns
    -------
    str
        ``4K_ns``, ``11K_ns``, ``22K_ns``, ``44K_ns``, or ``111K_ns``.
    """
    return f"{directory_code(rate)}K_ns"


def color_for(rate):
    """Color used for a physical cooling rate.

    Parameters
    ----------
    rate : float
        Physical rate in K/ns.

    Returns
    -------
    str
        Hex color.
    """
    return RATE_COLOR[directory_code(rate)]


def as_physical_rate(value):
    """Interpret a stored rate that may be a physical rate or a directory code.

    Parameters
    ----------
    value : float or str
        ``4.44`` or the old directory code ``4``. The code ``111`` is the
        physical rate 111 K/ns.

    Returns
    -------
    float
        Physical rate in K/ns.

    Raises
    ------
    ValueError
        If the value is neither a physical rate nor a known directory code.
    """
    number = float(value)
    for rate in PHYSICAL_RATES:
        if abs(number - rate) <= 1e-6:
            return float(rate)
    code = int(round(number))
    if abs(number - code) <= 1e-6 and code in RATE_CODE_TO_K_PER_NS and code != 111:
        return float(RATE_CODE_TO_K_PER_NS[code])
    raise ValueError(f"Unrecognized cooling rate {value!r}.")


def with_physical_rate(frame):
    """Copy a table with every rate column stored as a physical rate.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table that may contain ``rate``, ``rate_K_ns``, or ``rate_K_per_ns``.

    Returns
    -------
    pandas.DataFrame
        Copy whose rate columns are physical rates in K/ns.
    """
    frame = frame.copy()
    for column in ("rate", "rate_K_ns", "rate_K_per_ns", "cooling_rate_K_per_ns"):
        if column in frame.columns:
            frame[column] = frame[column].map(as_physical_rate)
    return frame


def canonicalize_rate(value):
    """Format a command-line rate as the physical rate string.

    Parameters
    ----------
    value : float or str
        Physical cooling rate.

    Returns
    -------
    str
        ``4.44``, ``11.1``, ``22.2``, ``44.4``, or ``111``.
    """
    return f"{parse_rate(value):g}"


def stored_rate_label(value):
    """Format a stored rate, which may be a directory code, as a physical-rate string.

    Parameters
    ----------
    value : float or str
        Physical rate or an old directory code.

    Returns
    -------
    str
        ``4.44``, ``11.1``, ``22.2``, ``44.4``, or ``111``.
    """
    return f"{as_physical_rate(value):g}"


def canonicalize_frame(frame):
    """Store rate columns as physical-rate strings.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table that may contain a directory code in a rate column.

    Returns
    -------
    pandas.DataFrame
        Copy whose rate columns match :func:`canonicalize_rate`.
    """
    frame = with_physical_rate(frame)
    for column in ("rate", "rate_K_ns", "rate_K_per_ns", "cooling_rate_K_per_ns"):
        if column in frame.columns:
            frame[column] = frame[column].map(lambda item: f"{float(item):g}")
    return frame


def select_rate(frame, rate):
    """Rows for one physical cooling rate.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table with a rate column. Directory codes in that column are converted.
    rate : float
        Physical rate in K/ns.

    Returns
    -------
    pandas.DataFrame
        Matching rows. The rate columns contain physical rates.

    Raises
    ------
    ValueError
        If the table has no rate column.
    """
    target = parse_rate(rate)
    converted = with_physical_rate(frame)
    column = next(
        (
            name
            for name in ("rate_K_ns", "rate", "rate_K_per_ns", "cooling_rate_K_per_ns")
            if name in converted.columns
        ),
        None,
    )
    if column is None:
        raise ValueError("No cooling-rate column.")
    return converted[np.isclose(converted[column].astype(float), target)].copy()


import csv
import json
import re
from pathlib import Path

from opcfilm.config import CONFIG, HOLD_DISCARD_FRACTION
from opcfilm.io.mdout import time_span_ps

REPLICAS = tuple(str(name) for name in CONFIG.film.replicas)

COLUMNS = [
    "row_id",
    "kind",
    "replica",
    "rate_K_per_ns",
    "source_rate_K_per_ns",
    "temperature_K",
    "t_start_K",
    "t_end_K",
    "files",
    "topology",
    "frame_ps",
    "ntwx",
    "cut_A",
    "source_restart",
    "discard_ps",
    "duration_ps",
    "mdout",
]
_NTWX = re.compile(r"(?m)^\s*ntwx\s*=\s*([0-9]+)")
_DT = re.compile(r"(?m)^\s*dt\s*=\s*([0-9]*\.?[0-9]+)")
KINDS = ("ramp", "hold", "endpoint")
_WINDOW = re.compile(r"(\d{2,3})_to_(\d{2,3})K")
_SETPOINT = re.compile(r"(\d{2,3})K")
_CUT = re.compile(r"(?m)^\s*cut\s*=\s*([0-9]*\.?[0-9]+)")



def _frame_from_mdout(path):
    """Saved-frame spacing echoed by an Amber header.

    Parameters
    ----------
    path : path-like or None
        Mdout. ``None`` returns blanks.

    Returns
    -------
    ntwx, frame_ps : tuple of str
        Step interval and spacing in picoseconds. Either is blank when the
        header does not echo it.
    """
    if path is None:
        return "", ""
    text = Path(path).read_text(errors="replace")
    head = re.split(r"A\s+V\s+E\s+R\s+A\s+G\s+E\s+S", text, maxsplit=1, flags=re.I)[0]
    ntwx = _NTWX.findall(head)
    dt = _DT.findall(head)
    if not ntwx:
        return "", ""
    if not dt:
        return ntwx[-1], ""
    return ntwx[-1], f"{float(ntwx[-1]) * float(dt[-1]):g}"


def _with_aliases(row):
    """Fill the canonical columns and the names older scripts still read.

    Parameters
    ----------
    row : dict
        A manifest row in either naming.

    Returns
    -------
    dict
        The same row plus ``rate_K_ns``, ``trajectory``, the temperature
        window names, ``discard_fraction``, ``label``, and ``outdir``.
    """
    row = dict(row)
    if not str(row.get("rate_K_per_ns", "")).strip():
        row["rate_K_per_ns"] = row.get("rate_K_ns", "")
    row["rate_K_ns"] = row.get("rate_K_per_ns", "")
    if not str(row.get("t_start_K", "")).strip():
        row["t_start_K"] = row.get("temperature_start_K", "")
    row["temperature_start_K"] = row.get("t_start_K", "")
    if not str(row.get("t_end_K", "")).strip():
        row["t_end_K"] = row.get("temperature_end_K", "")
    row["temperature_end_K"] = row.get("t_end_K", "")
    if not str(row.get("files", "")).strip():
        row["files"] = row.get("trajectory", "")
    row["trajectory"] = row.get("files", "")
    kind = row.get("kind", "")
    if kind in KINDS and not str(row.get("discard_fraction", "")).strip():
        row["discard_fraction"] = f"{discard_fraction_for(kind):g}"
    if not str(row.get("discard_ps", "")).strip():
        duration = str(row.get("duration_ps", "")).strip()
        fraction = str(row.get("discard_fraction", "")).strip()
        if duration and fraction:
            row["discard_ps"] = f"{float(duration) * float(fraction):g}"
    if not str(row.get("label", "")).strip():
        files = str(row.get("files", ""))
        row["label"] = Path(files.split(";")[0]).stem if files else ""
    if not str(row.get("row_id", "")).strip():
        row["row_id"] = row_id(row)
    if not str(row.get("outdir", "")).strip():
        row["outdir"] = str(result_directory(Path("."), row.get("stage") or "shared", row))
    if not str(row.get("rate_code", "")).strip() and str(row.get("rate_K_per_ns", "")).strip():
        try:
            row["rate_code"] = str(directory_code(float(row["rate_K_per_ns"])))
        except (TypeError, ValueError):
            row["rate_code"] = ""
    else:
        row.setdefault("rate_code", "")
    row.setdefault("temp_mid_K", row.get("temperature_K", ""))
    return row


def row_id(row):
    """Stable identity of one manifest row.

    Parameters
    ----------
    row : dict
        Manifest row. A hold or an endpoint uses ``source_rate_K_per_ns``.
        A ramp uses ``rate_K_per_ns``. ``rate_K_per_ns`` stays empty for a
        hold, so the id cannot be mistaken for a cooling window.

    Returns
    -------
    str
        ``hold-r2-4.44-250`` or ``ramp-r1-4.44-295``.
    """
    kind = str(row.get("kind", "unset"))
    replica = str(row.get("replica", "unset"))
    if kind == "ramp":
        rate = str(row.get("rate_K_per_ns", "")).strip() or "none"
    else:
        rate = str(row.get("source_rate_K_per_ns", "")).strip() or "none"
    temperature = str(row.get("temperature_K", "")).strip() or "none"
    return f"{kind}-{replica}-{rate}-{temperature}"


def result_directory(root, stage, row):
    """Per-row output directory.

    Parameters
    ----------
    root : path-like
        Directory that contains ``results/``.
    stage : str
        Stage name, for example ``s00``.
    row : dict
        Manifest row.

    Returns
    -------
    pathlib.Path
        ``results/<stage>/<row_id>``.
    """
    return Path(root) / "results" / str(stage) / row_id(row)


def output_directory(root, kind, replica, rate_K_ns, temperature_K, label):
    """Directory for one manifest row.

    Parameters
    ----------
    root : path-like
        Directory that will contain the outputs.
    kind : str
        ``ramp``, ``hold``, or ``endpoint``.
    replica : str
        Replica name, for example ``r1``.
    rate_K_ns : str
        Physical cooling rate. Empty for a trajectory that is not on a ramp.
    temperature_K : str
        Midpoint or setpoint in kelvin.
    label : str
        Trajectory stem or another short name.

    Returns
    -------
    pathlib.Path
        ``<root>/<kind>/<replica>/<rate>Kns/<temperature>K/<label>``.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}.")
    rate = rate_K_ns if str(rate_K_ns) not in ("", "nan") else "unset"
    return (
        Path(root) / kind / str(replica) / f"{rate}Kns"
        / f"{temperature_K}K" / str(label)
    )


def discard_fraction_for(kind):
    """Equilibration fraction dropped for one trajectory kind.

    Parameters
    ----------
    kind : str
        ``ramp``, ``hold``, or ``endpoint``.

    Returns
    -------
    float
        ``HOLD_DISCARD_FRACTION`` for a hold or an endpoint, and the ramp
        fraction from the parameter sheet for a cooling window.
    """
    if kind in ("hold", "endpoint"):
        return float(CONFIG.discard.hold_fraction)
    if kind == "ramp":
        return float(CONFIG.discard.ramp_fraction)
    raise ValueError(f"kind must be one of {KINDS}.")


def read_cut(path):
    """Real-space cutoff echoed in an Amber mdout header.

    Parameters
    ----------
    path : path-like
        Finished mdout. The value is the last ``cut =`` before the averages.

    Returns
    -------
    float
        Cutoff in angstroms.

    Raises
    ------
    ValueError
        If the header does not contain ``cut``.
    """
    text = Path(path).read_text(errors="replace")
    head = re.split(r"A\s+V\s+E\s+R\s+A\s+G\s+E\s+S", text, maxsplit=1, flags=re.I)[0]
    found = _CUT.findall(head)
    if not found:
        raise ValueError(f"{path} does not echo a cut value.")
    return float(found[-1])


def cutoff_note(rows):
    """State whether holds and ramps used the same cutoff.

    Parameters
    ----------
    rows : sequence of dict
        Manifest rows with ``kind`` and ``cut_A``.

    Returns
    -------
    str
        A methods sentence. A blank cutoff is reported as not yet read.
    """
    cuts = {}
    missing = []
    for row in rows:
        kind = row.get("kind", "")
        if kind not in KINDS:
            continue
        value = str(row.get("cut_A", "")).strip()
        if value == "":
            missing.append(kind)
            continue
        cuts.setdefault(kind, set()).add(float(value))
    if "hold" in missing or "endpoint" in missing:
        return (
            "No finished hold mdout was available, so the hold cutoff was not "
            f"read. The archived prop_trajectory.mdin says cut = {CONFIG.trajectory.equilibration_cut_A:.1f} and the "
            f"cooling input says cut = {CONFIG.trajectory.production_cut_A:.1f}. If the holds used "
            f"{CONFIG.trajectory.equilibration_cut_A:.0f} Å, the 78 K "
            "and 150–250 K data are not at the ramp cutoff."
        )
    ramp = cuts.get("ramp", set())
    other = cuts.get("hold", set()) | cuts.get("endpoint", set())
    if ramp and other and ramp != other:
        return (
            f"Cooling windows used cut = {sorted(ramp)} Å and holds or "
            f"endpoints used cut = {sorted(other)} Å. The 78 K and 150–250 K "
            "analyses are then at a different real-space cutoff from the ramps."
        )
    if ramp and other:
        return f"Holds, endpoints, and cooling windows all echo cut = {sorted(ramp)} Å."
    return "Cutoff values are only present for some manifest kinds."


_INPCRD = re.compile(r"INPCRD:\s*(\S+)")
_SEGMENT = re.compile(r"\.(\d+)$")
_SETPOINT_DIR = re.compile(r"(\d{2,3})K$")
_RESTART_PREFIX = re.compile(r"cool_(\d+)ns_chunk_")
_PREFIX_NS_TO_RATE = {50: 4.44, 20: 11.1, 10: 22.2, 5: 44.4, 2: 111.0}


def read_inpcrd(path):
    """Restart named by ``-c``, as echoed in the mdout file assignments.

    Parameters
    ----------
    path : path-like
        Amber mdout. The value is the last ``INPCRD`` before the averages.

    Returns
    -------
    str
        The restart path, or ``""`` when the header does not echo one.
    """
    text = Path(path).read_text(errors="replace")
    head = re.split(r"A\s+V\s+E\s+R\s+A\s+G\s+E\s+S", text, maxsplit=1, flags=re.I)[0]
    found = _INPCRD.findall(head)
    return found[-1] if found else ""


def _segment_index(path):
    """Continuation index of ``traj.N`` or ``mdout.N``.

    Parameters
    ----------
    path : pathlib.Path
        Trajectory or mdout path.

    Returns
    -------
    int
        The numeric suffix, or zero when the name has none.
    """
    match = _SEGMENT.search(path.name)
    return int(match.group(1)) if match else 0


def source_rate_from_restart(path):
    """Physical cooling rate encoded by a cooling-restart file name.

    Parameters
    ----------
    path : str
        Restart path. ``cool_50ns_chunk_...`` is 4.44 K/ns. A name that does
        not carry the cooling prefix returns an empty string.

    Returns
    -------
    str
        ``4.44``, ``11.1``, ``22.2``, ``44.4``, ``111``, or ``""``.
    """
    match = _RESTART_PREFIX.search(str(path).replace("\\", "/"))
    if match is None:
        return ""
    rate = _PREFIX_NS_TO_RATE.get(int(match.group(1)))
    return "" if rate is None else f"{rate:g}"


def setpoint_temperature(trajectory, base):
    """Hold or endpoint temperature from the first directory under ``base``.

    Parameters
    ----------
    trajectory : pathlib.Path
        One segment, usually ``opc/<replica>/hold/<T>K/TRAJ/traj.N``.
    base : pathlib.Path
        ``<root>/opc/<replica>/hold`` or the endpoint directory.

    Returns
    -------
    str
        The temperature in kelvin. A first component that is not
        ``<T>K`` returns ``""``. The match is not applied to the absolute
        path, so a parent such as ``139K`` is not a setpoint.
    """
    relative = Path(trajectory).resolve().relative_to(Path(base).resolve())
    match = _SETPOINT_DIR.fullmatch(relative.parts[0])
    return match.group(1) if match else ""


def _hold_sidecar(trajectory, temperature):
    """Read ``hold_<T>K.json`` written beside the hold input.

    Parameters
    ----------
    trajectory : pathlib.Path
        One segment of the hold.
    temperature : str
        Setpoint in kelvin.

    Returns
    -------
    dict
        ``source_restart``, ``source_rate_K_per_ns``, ``length_ns``, and
        ``cut_A`` when the file exists. An empty dict when it does not.
    """
    path = _setpoint_directory(trajectory) / f"hold_{temperature}K.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _setpoint_directory(trajectory):
    """Directory named ``<temperature>K`` that contains one hold or endpoint.

    Parameters
    ----------
    trajectory : pathlib.Path
        One segment, usually ``.../<T>K/TRAJ/traj.N``.

    Returns
    -------
    pathlib.Path
        The setpoint directory. The trajectory's parent is used when no
        ``<T>K`` directory is above it.
    """
    for parent in trajectory.parents:
        if _SETPOINT_DIR.fullmatch(parent.name):
            return parent
    return trajectory.parent


def _trajectories(directory):
    if not directory.is_dir():
        return []
    found = []
    for path in directory.rglob("*"):
        if path.is_file() and (path.suffix == ".nc" or path.name.startswith("traj.")):
            found.append(path)
    return sorted(found)


def _mdout_for(trajectory):
    """Mdout that belongs to one trajectory segment.

    Cooling windows are ``OUT/<stage>.mdout``. Holds and endpoints written
    by ``prop_trajectory.gpu`` are ``OUT/mdout.N`` beside ``TRAJ/traj.N``.

    Parameters
    ----------
    trajectory : pathlib.Path
        One trajectory file.

    Returns
    -------
    pathlib.Path or None
        The mdout, when a candidate exists.
    """
    index = _segment_index(trajectory)
    candidates = []
    if index:
        candidates.extend([
            trajectory.parent.parent / "OUT" / f"mdout.{index}",
            trajectory.parent / "OUT" / f"mdout.{index}",
        ])
    candidates.extend([
        trajectory.with_suffix(".mdout"),
        trajectory.parent / "OUT" / f"{trajectory.stem}.mdout",
        trajectory.parent.parent / "OUT" / f"{trajectory.stem}.mdout",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _row(kind, replica, trajectory, topology, workflow, rate_code, start, end, temperature, restart=""):
    rate = ""
    if rate_code != "":
        rate = f"{as_physical_rate(int(rate_code)):g}"
    mdout = _mdout_for(trajectory)
    cut = ""
    duration = ""
    if mdout is not None:
        try:
            cut = f"{read_cut(mdout):g}"
        except ValueError:
            cut = ""
        span = time_span_ps(mdout)
        if span is not None:
            duration = f"{span:g}"
    label = trajectory.stem
    if not restart and mdout is not None:
        restart = read_inpcrd(mdout)
    ntwx, frame_ps = _frame_from_mdout(mdout)
    fraction = discard_fraction_for(kind)
    discard_ps = ""
    if duration != "":
        discard_ps = f"{float(duration) * fraction:g}"
    resolved = str(trajectory.resolve())
    return _with_aliases({
        "kind": kind,
        "replica": replica,
        "rate_K_per_ns": rate,
        "source_rate_K_per_ns": "" if kind == "ramp" else source_rate_from_restart(restart),
        "temperature_K": temperature,
        "t_start_K": start,
        "t_end_K": end,
        "files": resolved,
        "topology": str(Path(topology).resolve()) if topology is not None else "",
        "frame_ps": frame_ps,
        "ntwx": ntwx,
        "cut_A": cut,
        "source_restart": restart,
        "discard_ps": discard_ps,
        "duration_ps": duration,
        "mdout": "" if mdout is None else str(mdout.resolve()),
        "label": label,
    })


def _continuation_row(kind, replica, trajectories, topology, workflow, temperature):
    """One hold or endpoint, including every continuation segment.

    Parameters
    ----------
    kind : str
        ``hold`` or ``endpoint``.
    replica : str
        Replica name.
    trajectories : sequence of pathlib.Path
        Segments in the order they were run.
    topology : pathlib.Path
        Amber topology.
    workflow : pathlib.Path
        Directory under which ``outputs/`` is created.
    temperature : str
        Setpoint in kelvin.

    Returns
    -------
    dict
        One manifest row. ``files`` and ``mdout`` join the segments with
        ``;``. ``duration_ps`` is the sum of the segment time spans.
        ``discard_ps`` is the equilibration fraction of that sum, applied
        once to the whole hold.

    Raises
    ------
    ValueError
        If the segments echo different real-space cutoffs or different
        frame spacings.
    """
    ordered = sorted(trajectories, key=lambda path: (_segment_index(path), path.name))
    mdouts = [_mdout_for(path) for path in ordered]
    missing = [str(path) for path, mdout in zip(ordered, mdouts) if mdout is None]
    if missing:
        raise FileNotFoundError(f"Segments without mdout: {missing}")
    cuts = []
    spacings = []
    spans = []
    for mdout in mdouts:
        if mdout is None:
            spans = None
            continue
        try:
            cuts.append(read_cut(mdout))
        except ValueError:
            cuts.append(None)
        ntwx, frame_ps = _frame_from_mdout(mdout)
        if ntwx or frame_ps:
            spacings.append((ntwx, frame_ps))
        if spans is not None:
            span = time_span_ps(mdout)
            if span is None:
                spans = None
            else:
                spans.append(span)
    present = [value for value in cuts if value is not None]
    if len(set(present)) > 1:
        raise ValueError(
            f"{kind} {replica} at {temperature} K has segments with different "
            f"cut values: {present}."
        )
    if len(set(spacings)) > 1:
        raise ValueError(
            f"{kind} {replica} at {temperature} K has segments with different "
            f"frame spacings: {spacings}."
        )
    cut = f"{present[0]:g}" if len(present) == len(mdouts) and present else ""
    ntwx, frame_ps = spacings[0] if len(set(spacings)) == 1 else ("", "")
    duration = "" if not spans else f"{sum(spans):g}"
    fraction = discard_fraction_for(kind)
    discard_ps = "" if duration == "" else f"{float(duration) * fraction:g}"
    first = next((path for path in mdouts if path is not None), None)
    restart = "" if first is None else read_inpcrd(first)
    sidecar = _hold_sidecar(ordered[0], temperature)
    if not restart:
        restart = str(sidecar.get("source_restart", ""))
    source_rate = source_rate_from_restart(restart) or str(sidecar.get("source_rate_K_per_ns", ""))
    label = _setpoint_directory(ordered[0]).name
    return _with_aliases({
        "kind": kind,
        "replica": replica,
        "rate_K_per_ns": "",
        "source_rate_K_per_ns": source_rate,
        "temperature_K": temperature,
        "t_start_K": temperature,
        "t_end_K": temperature,
        "files": ";".join(str(path.resolve()) for path in ordered),
        "topology": str(Path(topology).resolve()) if topology is not None else "",
        "frame_ps": frame_ps,
        "ntwx": ntwx,
        "cut_A": cut,
        "source_restart": restart,
        "discard_ps": discard_ps,
        "duration_ps": duration,
        "mdout": ";".join(str(path.resolve()) for path in mdouts if path is not None),
        "label": label,
    })


def discover_rows(root, workflow, topology, replicas, rate_codes):
    """Find ramp, hold, and per-replica endpoint trajectories.

    Parameters
    ----------
    root : pathlib.Path
        Data root containing ``opc/``.
    workflow : pathlib.Path
        Directory under which ``outputs/`` is created.
    topology : pathlib.Path
        Amber topology recorded on every row.
    replicas : sequence of str
        Replica directory names.
    rate_codes : sequence of int
        Cooling-rate directory codes.

    Returns
    -------
    list of dict
        Manifest rows. A ramp is one cooling window. A hold or an endpoint is
        one setpoint: every ``traj.N`` in that directory is one row, in
        numeric order. A ramp whose name has no temperature window keeps a
        blank ``temperature_K``.
    """
    root = Path(root)
    rows = []
    for replica in replicas:
        for rate in rate_codes:
            base = root / "opc" / replica / "ramp_nvt" / f"{rate}K_ns" / "TRAJ"
            for trajectory in _trajectories(base):
                match = _WINDOW.search(trajectory.name)
                if match:
                    start, end = match.group(1), match.group(2)
                    midpoint = f"{(int(start) + int(end)) / 2:g}"
                else:
                    start = end = midpoint = ""
                rows.append(_row("ramp", replica, trajectory, topology, workflow, rate, start, end, midpoint))
        for kind, folder in (("hold", "hold"), ("endpoint", "endpoint")):
            base = root / "opc" / replica / folder
            if not base.is_dir():
                continue
            grouped = {}
            for trajectory in _trajectories(base):
                temperature = setpoint_temperature(trajectory, base)
                grouped.setdefault(temperature, []).append(trajectory)
            for temperature, trajectories in sorted(grouped.items()):
                rows.append(_continuation_row(
                    kind, replica, trajectories, topology, workflow, temperature,
                ))
    return rows


def load_manifest(path):
    """Read the project manifest.

    Parameters
    ----------
    path : path-like
        Tab-separated manifest.

    Returns
    -------
    list of dict
        One row per trajectory.

    Raises
    ------
    ValueError
        If ``kind`` is missing or not one of the three allowed values.
    """
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path} has no rows.")
    if "kind" not in rows[0]:
        raise ValueError(f"{path} has no kind column. Rebuild it with opcfilm catalog build.")
    for row in rows:
        if row["kind"] not in KINDS:
            raise ValueError(f"Unknown kind {row['kind']!r} in {path}.")
    return [_with_aliases(row) for row in rows]


def plotted_rate(row):
    """Physical cooling rate used to place a row on a figure.

    Parameters
    ----------
    row : dict
        Manifest row. A hold keeps ``rate_K_per_ns`` empty and records the
        ramp it continued in ``source_rate_K_per_ns``.

    Returns
    -------
    str
        The ramp rate, or the source rate for a hold or an endpoint.
        An empty string when neither is set.
    """
    rate = str(row.get("rate_K_per_ns", "")).strip()
    if rate:
        return rate
    return str(row.get("source_rate_K_per_ns", "")).strip()


def trajectory_list(row):
    """Trajectory paths on one manifest row.

    Parameters
    ----------
    row : dict
        A manifest row. Several files are separated by ``;``.

    Returns
    -------
    list of pathlib.Path
        Files in the order they should be read.
    """
    text = row.get("files") or row.get("trajectory") or ""
    return [Path(part) for part in str(text).split(";") if part]


def write_manifest(path, rows):
    """Write manifest rows.

    Parameters
    ----------
    path : path-like
        Destination TSV.
    rows : sequence of dict
        Rows using :data:`COLUMNS`. Missing keys are written blank.

    Returns
    -------
    pathlib.Path
        The file that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            aliased = _with_aliases(row)
            writer.writerow({column: aliased.get(column, "") for column in COLUMNS})
    return path


def select_row(rows, index):
    """Return one 1-based manifest row.

    Parameters
    ----------
    rows : sequence of dict
        Manifest rows.
    index : int
        Row number starting at 1, matching the other runners.

    Returns
    -------
    dict
        The selected row.
    """
    if not 1 <= int(index) <= len(rows):
        raise ValueError(f"Manifest row {index} is outside 1..{len(rows)}.")
    return rows[int(index) - 1]


def select_row_id(rows, identifier):
    """Return the manifest row with this ``row_id``.

    Parameters
    ----------
    rows : sequence of dict
        Manifest rows.
    identifier : str
        Value of ``row_id``, for example ``hold-r2-4.44-250``.

    Returns
    -------
    dict
        The matching row.

    Raises
    ------
    KeyError
        If the identifier is missing or duplicated.
    """
    matched = [row for row in rows if row.get("row_id") == identifier]
    if len(matched) != 1:
        raise KeyError(f"row_id {identifier!r} matched {len(matched)} rows.")
    return matched[0]