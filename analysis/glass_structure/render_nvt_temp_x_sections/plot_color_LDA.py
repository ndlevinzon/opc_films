#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MDAnalysis as mda

from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = Path(
    "./representative_frames"
).resolve()

OUTPUT_DIR = (
    INPUT_DIR
    / "phase_density_maps"
)

import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.classifier import (
    CLASSIFIER_ID, DISPLAY_250_150, fit_discriminant, score_discriminant,
    structure_features, write_panel_record,
)
from opcfilm.config import CONFIG


TEMPERATURES = [
    250,
    240,
    230,
    220,
    210,
    200,
    190,
    180,
    170,
    160,
    150,
]


REPLICAS = [
    int(str(name)[1:]) if str(name).startswith("r") and str(name)[1:].isdigit() else str(name)
    for name in CONFIG.film.replicas
]


# ============================================================
# BOX
# ============================================================

LX, LY, LZ = (float(value) for value in CONFIG.film.box_A)

BOX = np.array(
    [
        LX,
        LY,
        LZ,
    ],
    dtype=float,
)


# ============================================================
# LOCAL STRUCTURE
# ============================================================

NEAREST_K = 25
LSI_CUTOFF = 3.7


# ============================================================
# ENDPOINT CLASSIFIER
# ============================================================

# Central region used to learn 250 K liquid-like
# and 150 K LDA-like structural distributions.
REFERENCE_Z_MAX = 45.0

REFERENCE_SAMPLE = 20000


# ============================================================
# DISPLAY SLICE
# ============================================================

# Thickness along y.
SLICE_WIDTH_A = 20

# Display slab and a little vacuum.
Z_MAX_A = 70.0


# ============================================================
# SPATIAL DENSITY GRID
# ============================================================

# 1 A pixels give considerably sharper spatial structure
# than the original 2 A grid.
DX = 1.0
DZ = 1.0

# Only light smoothing of the density field.
#
# sigma is measured in grid cells:
# 0.60 x 1 A ~= 0.6 A smoothing length.
SMOOTH_SIGMA = 0.60


# ============================================================
# PHASE CLASSIFICATION
# ============================================================

P_LDA_THRESHOLD = 0.50


# ============================================================
# DISPLAY SETTINGS
# ============================================================

# Liquid density threshold defining the red continuous region.
#
# Increase -> thinner / more selective red domains
# Decrease -> more extended red domains
LIQUID_DENSITY_ISOVALUE = 0.20  # g cm^-3

# Suppress classification/rendering in nearly empty regions.
TOTAL_DENSITY_MIN = 0.10  # g cm^-3


# Size of explicit LDA-like oxygen sites.
LDA_MARKER_SIZE = 10

# Optional liquid oxygen points within the red phase.
SHOW_LIQUID_POINTS = False
LIQUID_MARKER_SIZE = 3


# Publication quality.
PNG_DPI = 600


# ============================================================
# COLORS
# ============================================================

LIQUID_COLOR = (
    0.82,
    0.08,
    0.08,
)

LIQUID_POINT_COLOR = (
    0.55,
    0.02,
    0.02,
)

LDA_COLOR = (
    0.05,
    0.16,
    0.82,
)

LDA_EDGE_COLOR = (
    0.02,
    0.05,
    0.30,
)


# ============================================================
# CONSTANTS
# ============================================================

MW_WATER = CONFIG.film.water_molar_mass_g_mol
NA = CONFIG.film.avogadro_per_mol


# ============================================================
# LOCAL STRUCTURE METRICS
# ============================================================

def calculate_local_metrics(
    O_pos,
    sample_idx=None,
):
    """q, Sk, and LSI from common.order, using the frame box in ``BOX``.

    Parameters
    ----------
    O_pos : array_like, shape (n, 3)
        Oxygen coordinates in angstroms.
    sample_idx : array_like of int, optional
        Rows to score. The default is every oxygen.

    Returns
    -------
    tuple of numpy.ndarray
        Orientational q, Chau-Hardwick Sk, and the local structure index.
    """
    O_pos = np.asarray(O_pos, dtype=float)
    if sample_idx is None:
        sample_idx = np.arange(len(O_pos))
    features = structure_features(O_pos, BOX, sample_idx)
    return features[:, 0], features[:, 1], features[:, 2]


def use_frame_box(universe):
    """Store this frame's orthorhombic box in ``BOX``.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe positioned on the frame being scored.

    Returns
    -------
    numpy.ndarray
        Box lengths in angstroms.

    Raises
    ------
    RuntimeError
        If the frame does not store a positive box.
    """
    global BOX
    BOX = np.asarray(universe.dimensions[:3], dtype=float)
    if np.any(~np.isfinite(BOX)) or np.any(BOX <= 0):
        raise RuntimeError("This frame has no box.")
    return BOX


def read_oxygen_positions(
    pdb_file,
):

    u = mda.Universe(
        str(
            pdb_file
        )
    )
    use_frame_box(u)


    O = u.select_atoms(
        "resname WAT and name O"
    )


    pos = (
        O.positions
        .astype(float)
        .copy()
    )


    print(
        f"  oxygen sites: {len(pos):,}",
        flush=True,
    )


    # Wrap lateral coordinates.
    pos[
        :,
        0
    ] %= LX

    pos[
        :,
        1
    ] %= LY


    # Center slab in z independently for each frame.
    z_center = np.median(
        pos[
            :,
            2
        ]
    )


    pos[
        :,
        2
    ] -= z_center


    return pos


# ============================================================
# FIND REPRESENTATIVE FRAME
# ============================================================

def find_frame(
    temperature,
    replica,
):

    matches = list(
        INPUT_DIR.glob(
            f"T{temperature}_r{replica}_frame*.pdb"
        )
    )


    if len(matches) != 1:

        raise RuntimeError(
            f"Expected exactly one frame for "
            f"T={temperature} K, replica=r{replica} "
            f"(directories {tuple(CONFIG.film.replicas)}); "
            f"found {len(matches)}."
        )


    return matches[
        0
    ]


# ============================================================
# BUILD ENDPOINT REFERENCE POPULATION
# ============================================================

def endpoint_population(
    temperature,
):

    rows = []


    rng = np.random.default_rng(
        12345
        + temperature
    )


    for replica in REPLICAS:

        pdb_file = find_frame(
            temperature,
            replica,
        )


        print(
            f"Reference T={temperature} K, "
            f"r{replica}: "
            f"{pdb_file.name}",
            flush=True,
        )


        pos = read_oxygen_positions(
            pdb_file
        )


        eligible = np.where(
            np.abs(
                pos[
                    :,
                    2
                ]
            )
            <= REFERENCE_Z_MAX
        )[0]


        if len(eligible) > REFERENCE_SAMPLE:

            sample_idx = rng.choice(
                eligible,
                size=REFERENCE_SAMPLE,
                replace=False,
            )

        else:

            sample_idx = eligible


        q, Sk, LSI = calculate_local_metrics(
            pos,
            sample_idx,
        )


        valid = (
            np.isfinite(q)
            &
            np.isfinite(Sk)
            &
            np.isfinite(LSI)
        )


        X = np.column_stack(
            [
                q[
                    valid
                ],
                Sk[
                    valid
                ],
                LSI[
                    valid
                ],
            ]
        )


        rows.append(
            X
        )


    return np.vstack(
        rows
    )


# ============================================================
# BUILD STRUCTURAL CLASSIFIER
# ============================================================

def build_classifier():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DISPLAY ONLY: 250 K versus 150 K is not the quantitative classifier"
    )

    print(
        "=" * 80
    )


    X_liquid = endpoint_population(
        250
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "BUILDING 150 K LDA-LIKE REFERENCE"
    )

    print(
        "=" * 80
    )


    X_lda = endpoint_population(
        150
    )


    # ========================================================
    # ROBUST STANDARDIZATION
    # ========================================================

    model = fit_discriminant(X_liquid, X_lda)
    print("DISPLAY ONLY. Scoring uses common.classifier.fit_discriminant.")
    return {
        "center": model["center"],
        "scale": model["scale"],
        "mu_liquid": model["mu_liquid"],
        "mu_lda": model["mu_glass"],
        "inverse_covariance": model["inv_cov"],
    }


def lda_probability(
    q,
    Sk,
    LSI,
    model,
):
    """Glass-like score from the shared discriminant.

    Parameters
    ----------
    q, Sk, LSI : array_like
        Structural features of the waters being colored.
    model : dict
        Display model. ``mu_lda`` is the glass center.

    Returns
    -------
    numpy.ndarray
        Score in (0, 1). This figure is not the quantitative classifier.
    """
    adapted = {
        "center": model["center"],
        "scale": model["scale"],
        "mu_liquid": model["mu_liquid"],
        "mu_glass": model["mu_lda"],
        "inv_cov": model["inverse_covariance"],
    }
    return score_discriminant(np.column_stack([q, Sk, LSI]), adapted)


def counts_to_mass_density(
    counts,
):

    voxel_volume_A3 = (
        DX
        * SLICE_WIDTH_A
        * DZ
    )


    number_density_A3 = (
        counts
        / voxel_volume_A3
    )


    mass_density = (
        number_density_A3
        * MW_WATER
        / NA
        * 1.0e24
    )


    return mass_density


# ============================================================
# ANALYZE / RENDER ONE FRAME
# ============================================================

def analyze_frame(
    temperature,
    replica,
    model,
):

    pdb_file = find_frame(
        temperature,
        replica,
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"T={temperature} K, "
        f"replica {replica}"
    )

    print(
        f"  {pdb_file.name}"
    )


    pos = read_oxygen_positions(
        pdb_file
    )


    # ========================================================
    # CENTRAL Y SLICE
    # ========================================================

    y_center = (
        0.5
        * LY
    )


    dy = (
        pos[
            :,
            1
        ]
        - y_center
    )


    # Minimum-image distance along y.
    dy -= (
        LY
        * np.round(
            dy / LY
        )
    )


    slice_idx = np.where(
        (
            np.abs(dy)
            <= (
                0.5
                * SLICE_WIDTH_A
            )
        )
        &
        (
            np.abs(
                pos[
                    :,
                    2
                ]
            )
            <= Z_MAX_A
        )
    )[0]


    print(
        f"  waters in slice: "
        f"{len(slice_idx):,}",
        flush=True,
    )


    # ========================================================
    # q / Sk / LSI
    # ========================================================

    q, Sk, LSI = calculate_local_metrics(
        pos,
        slice_idx,
    )


    valid = (
        np.isfinite(q)
        &
        np.isfinite(Sk)
        &
        np.isfinite(LSI)
    )


    indices = slice_idx[
        valid
    ]


    q = q[
        valid
    ]

    Sk = Sk[
        valid
    ]

    LSI = LSI[
        valid
    ]


    x = pos[
        indices,
        0
    ]


    z = pos[
        indices,
        2
    ]


    # ========================================================
    # STRUCTURAL LDA SCORE
    # ========================================================

    p_lda = lda_probability(
        q,
        Sk,
        LSI,
        model,
    )


    lda_like = (
        p_lda
        >= P_LDA_THRESHOLD
    )


    liquid_like = (
        ~lda_like
    )


    print(
        f"  LDA-like fraction: "
        f"{np.mean(lda_like):.4f}",
        flush=True,
    )


    # ========================================================
    # GRID
    # ========================================================

    x_edges = np.arange(
        0.0,
        LX + DX,
        DX,
    )


    z_edges = np.arange(
        -Z_MAX_A,
        Z_MAX_A + DZ,
        DZ,
    )


    x_centers = (
        0.5
        * (
            x_edges[
                :-1
            ]
            + x_edges[
                1:
            ]
        )
    )


    z_centers = (
        0.5
        * (
            z_edges[
                :-1
            ]
            + z_edges[
                1:
            ]
        )
    )


    # ========================================================
    # PROBABILITY-WEIGHTED PHASE DENSITIES
    #
    # A water with P_LDA=0.80 contributes:
    #
    #     0.20 to liquid density
    #     0.80 to LDA density
    #
    # This gives a continuous density field while the
    # explicit plotted particles remain sharply classified.
    # ========================================================

    H_liquid, _, _ = np.histogram2d(
        z,
        x,
        bins=[
            z_edges,
            x_edges,
        ],
        weights=(
            1.0
            - p_lda
        ),
    )


    H_lda, _, _ = np.histogram2d(
        z,
        x,
        bins=[
            z_edges,
            x_edges,
        ],
        weights=p_lda,
    )


    # ========================================================
    # LIGHT SPATIAL COARSE-GRAINING
    # ========================================================

    H_liquid_smooth = gaussian_filter(
        H_liquid,
        sigma=SMOOTH_SIGMA,
        mode="nearest",
    )


    H_lda_smooth = gaussian_filter(
        H_lda,
        sigma=SMOOTH_SIGMA,
        mode="nearest",
    )


    # ========================================================
    # DENSITY
    # ========================================================

    rho_liquid = counts_to_mass_density(
        H_liquid_smooth
    )


    rho_lda = counts_to_mass_density(
        H_lda_smooth
    )


    rho_total = (
        rho_liquid
        + rho_lda
    )


    fraction_lda = np.divide(
        rho_lda,
        rho_total,
        out=np.zeros_like(
            rho_total
        ),
        where=(
            rho_total
            > 1.0e-12
        ),
    )


    # ========================================================
    # SAVE NUMERICAL MAP
    # ========================================================

    np.savez_compressed(
        OUTPUT_DIR
        / (
            f"T{temperature}_"
            f"r{replica}_"
            f"phase_density.npz"
        ),

        x_edges=x_edges,
        z_edges=z_edges,

        rho_liquid=rho_liquid,
        rho_lda=rho_lda,
        rho_total=rho_total,

        fraction_lda=fraction_lda,

        p_lda=p_lda,

        oxygen_indices=indices,

        q=q,
        Sk=Sk,
        LSI=LSI,
    )


    # ========================================================
    # FIGURE
    #
    # REFERENCE-LIKE REPRESENTATION:
    #
    # red   = continuous liquid-like density
    # blue  = explicit LDA-like water oxygens
    #
    # No imshow.
    # No bilinear interpolation.
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            10,
            7.4,
        )
    )


    # --------------------------------------------------------
    # RED LIQUID-LIKE DENSITY REGION
    # --------------------------------------------------------

    max_liquid_density = np.nanmax(
        rho_liquid
    )


    if (
        np.isfinite(
            max_liquid_density
        )
        and
        max_liquid_density
        > LIQUID_DENSITY_ISOVALUE
    ):

        ax.contourf(
            x_centers,
            z_centers,
            rho_liquid,
            levels=[
                LIQUID_DENSITY_ISOVALUE,
                max_liquid_density
                + 1.0e-8,
            ],
            colors=[
                LIQUID_COLOR
            ],
            alpha=0.96,
            antialiased=False,
            zorder=1,
        )


        # Sharp outer boundary of liquid density.
        ax.contour(
            x_centers,
            z_centers,
            rho_liquid,
            levels=[
                LIQUID_DENSITY_ISOVALUE
            ],
            colors=[
                LIQUID_POINT_COLOR
            ],
            linewidths=0.35,
            antialiased=True,
            zorder=2,
        )


    # --------------------------------------------------------
    # OPTIONAL LIQUID WATER PARTICLES
    # --------------------------------------------------------

    if SHOW_LIQUID_POINTS:

        ax.scatter(
            x[
                liquid_like
            ],
            z[
                liquid_like
            ],
            s=LIQUID_MARKER_SIZE,
            c=[
                LIQUID_POINT_COLOR
            ],
            edgecolors="none",
            alpha=0.45,
            rasterized=True,
            zorder=3,
        )


    # --------------------------------------------------------
    # BLUE LDA-LIKE WATER PARTICLES
    # --------------------------------------------------------

    ax.scatter(
        x[
            lda_like
        ],
        z[
            lda_like
        ],
        s=LDA_MARKER_SIZE,
        c=[
            LDA_COLOR
        ],
        edgecolors=[
            LDA_EDGE_COLOR
        ],
        linewidths=0.18,
        alpha=1.0,
        rasterized=True,
        zorder=4,
    )


    # ========================================================
    # OPTIONAL LDA-LIQUID DOMAIN BOUNDARY
    # ========================================================

    phase_valid = (
        rho_total
        >= TOTAL_DENSITY_MIN
    )


    fraction_for_contour = np.where(
        phase_valid,
        fraction_lda,
        np.nan,
    )


    if (
        np.nanmin(
            fraction_for_contour
        )
        < P_LDA_THRESHOLD
        <
        np.nanmax(
            fraction_for_contour
        )
    ):

        ax.contour(
            x_centers,
            z_centers,
            fraction_for_contour,
            levels=[
                P_LDA_THRESHOLD
            ],
            colors="black",
            linewidths=0.30,
            alpha=0.60,
            zorder=5,
        )


    # ========================================================
    # GIBBS DIVIDING SURFACES
    # ========================================================

    ax.axhline(
        61.0,
        linestyle="--",
        linewidth=0.6,
        color="black",
        alpha=0.30,
        zorder=6,
    )


    ax.axhline(
        -61.0,
        linestyle="--",
        linewidth=0.6,
        color="black",
        alpha=0.30,
        zorder=6,
    )


    # ========================================================
    # AXES
    # ========================================================

    ax.set_xlim(
        0.0,
        LX,
    )


    ax.set_ylim(
        -Z_MAX_A,
        Z_MAX_A,
    )


    ax.set_aspect(
        "equal",
        adjustable="box",
    )


    ax.set_xlabel(
        r"$x$ ($\AA$)",
        fontsize=13,
    )


    ax.set_ylabel(
        r"$z$ ($\AA$)",
        fontsize=13,
    )


    ax.set_title(
        f"{temperature} K — Replica {replica}",
        fontsize=15,
        fontweight="bold",
    )


    ax.tick_params(
        labelsize=11
    )


    ax.grid(
        False
    )


    # White vacuum/background.
    ax.set_facecolor(
        "white"
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    outfile_png = (
        OUTPUT_DIR
        / (
            f"T{temperature}_"
            f"r{replica}_"
            f"liquid_LDA_sharp.png"
        )
    )


    outfile_pdf = (
        OUTPUT_DIR
        / (
            f"T{temperature}_"
            f"r{replica}_"
            f"liquid_LDA_sharp.pdf"
        )
    )


    plt.tight_layout()


    fig.savefig(
        outfile_png,
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )


    # Vector version for publication.
    fig.savefig(
        outfile_pdf,
        bbox_inches="tight",
        facecolor="white",
    )


    plt.close(
        fig
    )


    # ========================================================
    # SAVE PER-WATER CLASSIFICATION
    # ========================================================

    pd.DataFrame({

        "oxygen_index":
            indices,

        "x_A":
            x,

        "z_A":
            z,

        "q":
            q,

        "Sk":
            Sk,

        "LSI_A2":
            LSI,

        "P_LDA":
            p_lda,

        "LDA_like":
            lda_like.astype(
                int
            ),

        "Liquid_like":
            liquid_like.astype(
                int
            ),

    }).to_csv(
        OUTPUT_DIR
        / (
            f"T{temperature}_"
            f"r{replica}_"
            f"classified_waters.csv"
        ),
        index=False,
    )


    print(
        f"  wrote: {outfile_png}",
        flush=True,
    )


# ============================================================
# MAIN
# ============================================================

def main(argv=None):
    """Draw the display-only maps.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments, excluding the program name.

    Returns
    -------
    int
        Zero when the maps were written.
    """
    global INPUT_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="Display-only 250 K versus 150 K phase maps. Not a transition temperature."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    args = parser.parse_args(argv)
    output = args.input_dir / "phase_density_maps"
    output.mkdir(parents=True, exist_ok=True)
    write_panel_record(
        output / "classifier_panel.json",
        DISPLAY_250_150,
        "display_only",
        "one representative frame at 250 K versus one at 150 K; "
        "20000 waters with |z| <= 45 Å",
        "This coloring is an illustration. The transition falls between 250 K and "
        "150 K because those temperatures are the training classes. It must not be "
        f"used for a transition temperature. The quantitative classifier is {CLASSIFIER_ID}.",
    )
    print(f"DISPLAY ONLY. Quantitative classifier: {CLASSIFIER_ID}")
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = output
    model = build_classifier()
    for temperature in TEMPERATURES:
        for replica in REPLICAS:
            analyze_frame(temperature, replica, model)
    print(f"Output directory: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())