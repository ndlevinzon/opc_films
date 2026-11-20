#!/usr/bin/env python3
"""Analyze one manifest row: density profile, and optionally capillary spectra.

``cpptraj`` is resolved from ``--cpptraj`` or from ``PATH``. The capillary
branch calls a custom ``surftension`` action and does nothing unless that
executable is supplied and ``--capwave`` is set.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import BULK_HALFWIDTH_A
from common.manifest import load_manifest, result_directory, select_row, trajectory_list
from common.trajectory import frame_spacing_ps, frames_skipped, netcdf_time_ps, trajin_lines


def run_density(script, row, destination, stride, nref):
    """Run the per-chunk density and local-order analysis.

    Parameters
    ----------
    script : pathlib.Path
        Path to ``01_analyze_chunk.py``.
    row : dict
        Manifest row containing ``topology`` and ``trajectory``.
    destination : pathlib.Path
        Chunk output directory.
    stride : int
        Trajectory stride passed through to the analyzer.
    nref : int
        Number of waters sampled for local order parameters.

    Returns
    -------
    None
        The analyzer is run in a subprocess. A nonzero status raises
        ``CalledProcessError``.
    """
    command = [
        sys.executable,
        str(script),
        "--top",
        row["topology"],
        "--out",
        str(destination),
        "--stride",
        str(stride),
        "--nref",
        str(nref),
        "--start-frame",
        str(frames_skipped(row.get("discard_ps", ""), row.get("frame_ps", ""))),
    ]
    for path in trajectory_list(row):
        command.extend(["--traj", str(path)])
    print("RUN", command, flush=True)
    subprocess.run(command, check=True)


def write_capwave_input(row, destination, stride, cpptraj, frame_ps, blocktime_ps):
    """Write and execute a cpptraj surface-spectrum input for one chunk.

    Parameters
    ----------
    row : dict
        Manifest row. Temperature fields must already be numeric strings.
    destination : pathlib.Path
        Chunk output directory.
    stride : int
        Frame stride applied to the raw trajectory.
    frame_ps : float
        Spacing of the selected frames, read from the NetCDF time variable.
    cpptraj : pathlib.Path
        cpptraj executable that provides the ``surftension`` action.
    blocktime_ps : float
        Block length in picoseconds. One fifth of the hold.

    Returns
    -------
    pathlib.Path
        Prefix of the spectrum, roughness, block, and summary files.
    """
    start = float(row["temperature_start_K"])
    end = float(row["temperature_end_K"])
    temperature = 0.5 * (start + end)
    prefix = destination / "capwave"
    lines = [f"parm {row['topology']}"]
    lines.extend(trajin_lines(
        trajectory_list(row),
        frames_skipped(row.get("discard_ps", ""), row.get("frame_ps", "")),
        stride,
    ))
    lines.extend([
        "autoimage",
        "center :WAT mass origin",
        "image origin center familiar",
        (
            f"surftension :WAT@O temp {temperature:g} gridspacing 2.5 dz 1.0 "
            f"sigmaxy 2.5 sigmaz 1.5 bulkhalfwidth {BULK_HALFWIDTH_A:g} threshold 0.5 qmax 0.30 "
            f"dt {frame_ps:g} blocktime {blocktime_ps:g} "
            f"spectrumout {prefix}_spectrum.dat roughout {prefix}_roughness.dat "
            f"blockout {prefix}_blocks.dat summaryout {prefix}_summary.dat"
        ),
        "run",
        "quit",
    ])
    script = destination / "capwave.cpptraj.in"
    script.write_text("\n".join(lines) + "\n")
    with (destination / "capwave.log").open("w") as log:
        subprocess.run(
            [str(cpptraj), "-i", str(script)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return prefix


def main(argv=None):
    """Analyze the requested 1-based manifest row.

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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--row", required=True, type=int)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--nref", type=int, default=4000)
    parser.add_argument(
        "--cpptraj",
        default=shutil.which("cpptraj") or "cpptraj",
        help="cpptraj executable. Defaults to the first cpptraj on PATH.",
    )
    parser.add_argument("--capwave", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    row = select_row(load_manifest(args.manifest), args.row)
    stage = "capillary" if args.capwave else "profiles"
    destination = result_directory(Path(__file__).resolve().parents[2], stage, row)
    destination.mkdir(parents=True, exist_ok=True)
    paths = trajectory_list(row)
    if not paths:
        raise SystemExit(f"Row {args.row} names no trajectory.")
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Missing trajectory: {path}")
    if not Path(row["topology"]).is_file():
        raise SystemExit(f"Missing topology: {row['topology']}")
    if args.overwrite or not (destination / "frame_geometry.csv").exists():
        run_density(
            Path(__file__).resolve().parent / "01_analyze_chunk.py",
            row,
            destination,
            args.stride,
            args.nref,
        )
    else:
        print("Skipping existing density/structure output", flush=True)
    if not args.capwave:
        return 0
    if row.get("kind") != "hold":
        raise SystemExit(
            "--capwave is only for kind=hold rows. A cooling window is shorter "
            "than one 10 ns block and is not stationary. List the holds with "
            "02_build_manifest.py --holds-tsv."
        )
    cpptraj = Path(args.cpptraj)
    if not cpptraj.is_file():
        raise SystemExit(f"cpptraj not found: {cpptraj}")
    if row["temperature_start_K"] == "":
        print("Skipping capwave: temperature is not in the file name.", flush=True)
        return 0
    if str(row.get("duration_ps", "")).strip():
        duration_ps = float(row["duration_ps"])
    else:
        selected = []
        for path in paths:
            selected.extend(netcdf_time_ps(path)[:: args.stride])
        duration_ps = float(selected[-1] - selected[0]) if len(selected) > 1 else 0.0
    discard_ps = float(row["discard_ps"]) if str(row.get("discard_ps", "")).strip() else 0.0
    analyzed_ps = duration_ps - discard_ps
    if analyzed_ps <= 0:
        raise SystemExit("The hold discard removes the whole trajectory.")
    spacing = float(row["frame_ps"]) if str(row.get("frame_ps", "")).strip() else frame_spacing_ps(netcdf_time_ps(paths[0]))
    prefix = write_capwave_input(
        row,
        destination,
        args.stride,
        cpptraj,
        spacing * args.stride,
        analyzed_ps / 5.0,
    )
    print("Capillary outputs", prefix, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
