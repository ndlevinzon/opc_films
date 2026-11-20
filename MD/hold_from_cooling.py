#!/usr/bin/env python3
"""Write an NVT hold input that continues a cooling restart.

``prop_trajectory.mdin`` leaves ``temp0`` at 300 K and ``nstlim`` unlimited,
and ``prop_trajectory.gpu`` starts from ``min/step9.rst7`` when ``START_RST``
is unset. A 150–250 K hold started that way is an instant quench. This
driver writes ``temp0``, ``tempi``, and an explicit ``nstlim``, and names
the cooling restart at the requested temperature.

The length that was run is the manifest column ``duration_ps``, the span of
``TIME(PS)`` in the finished mdout. A hold at 78 K continues the final
80 K to 78 K window. Restart names use three-digit temperatures, matching
the cooling script.
"""

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from opcfilm.catalog import parse_rate
from opcfilm.config import CONFIG
from opcfilm.mdinput import render

# Total cooling time for the 222 K drop, used in the restart file name.
_PREFIX_NS = {4.44: 50, 11.1: 20, 22.2: 10, 44.4: 5, 111.0: 2}


def cooling_restart(temperature_K, rate):
    """Restart written at the end of the cooling window that arrives at ``temperature_K``.

    Parameters
    ----------
    temperature_K : float
        Hold temperature. It is 78 K or the end of a 10 K cooling window.
    rate : float
        Physical cooling rate in K/ns. The 150–250 K holds continue 4.44 K/ns.

    Returns
    -------
    str
        Path of the restart relative to the cooling run directory.
    """
    rate = parse_rate(rate)
    end = int(temperature_K)
    if end == 78:
        start, index = 80, 22
    elif end % 10 == 0 and 80 <= end <= 290:
        start = end + 10
        index = (300 - start) // 10
    else:
        raise ValueError(
            "The hold temperature must be 78 K or a 10 K cooling-window endpoint from 290 K to 80 K."
        )
    prefix = _PREFIX_NS[rate]
    return f"RST/cool_{prefix}ns_chunk_{index:02d}_{start:03d}_to_{end:03d}K.rst7"


def write_mdin(temperature_K, length_ns, destination):
    """Write a fixed-temperature NVT input with an explicit length.

    Parameters
    ----------
    temperature_K : float
        Thermostat target.
    length_ns : float
        Requested length. ``nstlim = length_ns / 2 fs``.
    destination : path-like
        Output ``.mdin`` path.

    Returns
    -------
    pathlib.Path
        The file that was written.
    """
    return render("hold", temperature_K, length_ns, destination)


def main(argv=None):
    """Write the hold input and print the restart it must start from.

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
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--from-rate", type=float, default=4.44)
    parser.add_argument("--length-ns", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        restart = cooling_restart(args.temperature, args.from_rate)
        path = write_mdin(args.temperature, args.length_ns, args.out)
        sidecar = path.with_name(f"hold_{int(args.temperature)}K.json")
        sidecar.write_text(json.dumps({
            "source_restart": restart,
            "source_rate_K_per_ns": f"{parse_rate(args.from_rate):g}",
            "length_ns": float(args.length_ns),
            "cut_A": CONFIG.trajectory.production_cut_A,
        }, indent=2), encoding="utf-8")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote {path}")
    print(f"Wrote {sidecar}")
    print(f"START_RST={restart}")
    print("MDIN=" + str(path))
    print("The hold length that was actually run is the mdout TIME(PS) span, not this request, until that log is checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
