#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
import numpy as np
import MDAnalysis as mda
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from opcfilm.config import CONFIG


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = Path(
    "./representative_frames"
).resolve()

OUTPUT_DIR = (
    INPUT_DIR
    / "renders"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


LX, LY = (float(value) for value in CONFIG.film.box_A[:2])

# Thickness of central y slice
SLICE_WIDTH_A = 20

# Show slightly beyond Gibbs dividing surface (~61 A)
Z_DISPLAY_MAX_A = 70.0

# Oxygen-site marker size
MARKER_SIZE = 5.0

DPI = 300


# ============================================================
# RENDER ONE FRAME
# ============================================================

def render_frame(pdb_file):

    print(
        f"\nLoading: {pdb_file}",
        flush=True,
    )

    u = mda.Universe(
        str(pdb_file)
    )
    lengths = np.asarray(u.dimensions[:3], dtype=float)
    if np.any(lengths <= 0):
        raise RuntimeError(f"{pdb_file} has no box.")
    global LX, LY
    LX, LY = float(lengths[0]), float(lengths[1])

    # One site per water
    O = u.select_atoms(
        "resname WAT and name O"
    )

    print(
        f"  water oxygens: {len(O):,}",
        flush=True,
    )

    pos = (
        O.positions
        .astype(float)
        .copy()
    )


    # ========================================================
    # STANDARDIZE COORDINATES
    # ========================================================

    # Lateral periodic coordinates
    x = np.mod(
        pos[:, 0],
        LX,
    )

    y = np.mod(
        pos[:, 1],
        LY,
    )

    # Center the slab independently in z
    z_center = np.median(
        pos[:, 2]
    )

    z = (
        pos[:, 2]
        - z_center
    )


    # ========================================================
    # CENTRAL Y SLICE
    # ========================================================

    y_center = (
        0.5 * LY
    )

    half_width = (
        0.5 * SLICE_WIDTH_A
    )


    # Minimum-image distance from center in y
    dy = (
        y
        - y_center
    )

    dy -= (
        LY
        * np.round(
            dy / LY
        )
    )


    keep = (
        np.abs(dy)
        <= half_width
    )


    # Also discard remote vacuum for plotting
    keep &= (
        np.abs(z)
        <= Z_DISPLAY_MAX_A
    )


    x_plot = x[
        keep
    ]

    z_plot = z[
        keep
    ]


    print(
        f"  oxygens in plotted slice: "
        f"{len(x_plot):,}",
        flush=True,
    )


    if len(x_plot) == 0:

        raise RuntimeError(
            f"No waters found in plotting slice for "
            f"{pdb_file}"
        )


    # ========================================================
    # FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            10,
            7,
        )
    )


    ax.scatter(
        x_plot,
        z_plot,
        s=MARKER_SIZE,
        alpha=0.90,
        linewidths=0,
        rasterized=True,
    )


    # Exact same spatial scale for every frame
    ax.set_xlim(
        0.0,
        LX,
    )

    ax.set_ylim(
        -Z_DISPLAY_MAX_A,
        Z_DISPLAY_MAX_A,
    )


    ax.set_aspect(
        "equal",
        adjustable="box",
    )


    ax.set_xlabel(
        r"$x$ ($\AA$)"
    )

    ax.set_ylabel(
        r"$z$ ($\AA$)"
    )


    # Extract temperature / replica from filename
    # Example:
    # T150_r1_frame2216.pdb

    parts = (
        pdb_file.stem
        .split("_")
    )

    temperature = (
        parts[0]
        .replace(
            "T",
            ""
        )
    )

    replica = (
        parts[1]
        .replace(
            "r",
            ""
        )
    )


    ax.set_title(
        f"{temperature} K — Replica {replica}",
        fontweight="bold",
    )


    # Gibbs dividing surfaces for orientation
    ax.axhline(
        61.0,
        linestyle="--",
        linewidth=0.80,
        alpha=0.45,
    )

    ax.axhline(
        -61.0,
        linestyle="--",
        linewidth=0.80,
        alpha=0.45,
    )


    ax.grid(
        False
    )


    plt.tight_layout()


    # ========================================================
    # SAVE
    # ========================================================

    outfile = (
        OUTPUT_DIR
        / (
            pdb_file.stem
            + ".png"
        )
    )


    print(
        f"  rendering -> {outfile}",
        flush=True,
    )


    fig.savefig(
        outfile,
        dpi=DPI,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print(
        "  done",
        flush=True,
    )


# ============================================================
# MAIN
# ============================================================

pdb_files = sorted(
    INPUT_DIR.glob(
        "*.pdb"
    )
)


print(
    f"Found {len(pdb_files)} PDB files.",
    flush=True,
)


if len(pdb_files) == 0:

    raise RuntimeError(
        f"No PDB files found in:\n"
        f"{INPUT_DIR}"
    )


# First test one frame.
# Once confirmed, change pdb_files[:1] -> pdb_files

for pdb_file in pdb_files:
    render_frame(
        pdb_file
    )


print(
    "\n"
    + "=" * 80,
    flush=True,
)

print(
    "RENDERING COMPLETE",
    flush=True,
)

print(
    "=" * 80,
    flush=True,
)