#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
import numpy as np
import pandas as pd
import MDAnalysis as mda

from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


# ============================================================
# PATHS
# ============================================================

import os
_DATA_ROOT = os.environ.get("OPC_DATA_ROOT")
if not _DATA_ROOT:
    raise SystemExit("Set OPC_DATA_ROOT to the directory that contains opc/ and build/.")
TOPOLOGY = Path(_DATA_ROOT) / "build" / "opc_slab.prmtop"
BASE = Path(_DATA_ROOT) / "opc"


# Existing 250-150 K representative PDBs
REPRESENTATIVE_DIR = Path(
    "./representative_frames"
).resolve()

# 300 K and 78 K concatenated trajectories
ENDPOINT_TRAJECTORIES = {

    300:
        BASE
        / "300K_prop_analysis"
        / "cool_prod_centered_slab.nc",

    78:
        BASE
        / "78K_prop_analysis"
        / "cool_prod_centered_slab.nc",
}


# Endpoint representative PDBs are written here
ENDPOINT_PDB_DIR = (
    REPRESENTATIVE_DIR
    / "endpoint_controls"
)

ENDPOINT_PDB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Phase maps
OUTPUT_DIR = (
    REPRESENTATIVE_DIR
    / "phase_density_maps_300_78"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.classifier import (
    CLASSIFIER_ID, DISPLAY_MEDOID_Z45, fit_discriminant, score_discriminant,
    structure_features, write_panel_record,
)
from opcfilm.config import CONFIG
from opcfilm.structure.classifier import robust_location_scale

write_panel_record(
    OUTPUT_DIR / "classifier_panel.json",
    DISPLAY_MEDOID_Z45,
    "display_only",
    "one medoid frame at 300 K versus one at 78 K from concatenated "
    "endpoint trajectories; all waters with |z| <= 45 Å",
    "This coloring has no replica label and the 45 Å core overlaps the "
    "interface. It must not be used for a bulk-first claim. The quantitative "
    f"classifier is {CLASSIFIER_ID}.",
)
print(f"DISPLAY ONLY. Quantitative classifier: {CLASSIFIER_ID}")


# ============================================================
# TEMPERATURES
# ============================================================

INTERMEDIATE_TEMPERATURES = [
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
    1,
    2,
    3,
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
# ENDPOINT REPRESENTATIVE-FRAME SELECTION
# ============================================================

# Number of frames sampled uniformly from each concatenated
# 300/78 K trajectory.
N_ENDPOINT_CANDIDATE_FRAMES = 60

# Waters used per candidate frame to determine q/Sk/LSI
# descriptor medians.
FRAME_SELECTION_WATERS = 10000

# Select representative behavior from slab interior.
FRAME_SELECTION_Z_MAX = 45.0


# ============================================================
# ENDPOINT REFERENCE POPULATIONS
# ============================================================

# Use the interior of the representative 300 K and 78 K
# structures to train the phase classifier.
REFERENCE_Z_MAX = 45.0

# None = use every eligible central-slab oxygen.
REFERENCE_SAMPLE = None


# ============================================================
# PHASE CLASSIFICATION
# ============================================================

P_LDA_THRESHOLD = 0.50


# ============================================================
# DISPLAY SLICE
# ============================================================

SLICE_WIDTH_A = 20

Z_MAX_A = 70.0


# ============================================================
# SPATIAL DENSITY
# ============================================================

DX = 1.0
DZ = 1.0

SMOOTH_SIGMA = 0.60

LIQUID_DENSITY_ISOVALUE = 0.20  # g/cm^3
TOTAL_DENSITY_MIN = 0.10         # g/cm^3


# ============================================================
# DISPLAY
# ============================================================

LDA_MARKER_SIZE = 10

SHOW_LIQUID_POINTS = False
LIQUID_MARKER_SIZE = 3

PNG_DPI = 600


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


RNG = np.random.default_rng(
    12345
)


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


def normalize_oxygen_positions(
    O_pos,
):

    pos = np.asarray(
        O_pos,
        dtype=float,
    ).copy()


    pos[:, 0] %= LX
    pos[:, 1] %= LY


    z_center = np.median(
        pos[:, 2]
    )


    pos[:, 2] -= z_center


    return pos


# ============================================================
# CANDIDATE FRAME INDICES FOR CONCATENATED CONTROLS
# ============================================================

def candidate_frame_indices(
    universe,
):

    n_frames = len(
        universe.trajectory
    )


    if n_frames <= N_ENDPOINT_CANDIDATE_FRAMES:

        return np.arange(
            n_frames,
            dtype=int,
        )


    locations = np.linspace(
        0,
        n_frames - 1,
        N_ENDPOINT_CANDIDATE_FRAMES,
    )


    locations = np.unique(
        np.round(
            locations
        ).astype(int)
    )


    return locations


# ============================================================
# DESCRIPTORS FOR ONE ENDPOINT CANDIDATE FRAME
# ============================================================

def candidate_descriptor(
    universe,
    oxygen_atoms,
    frame_index,
):

    universe.trajectory[
        frame_index
    ]
    use_frame_box(universe)


    pos = normalize_oxygen_positions(
        oxygen_atoms.positions
    )


    eligible = np.where(
        np.abs(
            pos[:, 2]
        )
        <= FRAME_SELECTION_Z_MAX
    )[0]


    if len(eligible) > FRAME_SELECTION_WATERS:

        sample_idx = RNG.choice(
            eligible,
            size=FRAME_SELECTION_WATERS,
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


    return {

        "median_q":
            np.median(
                q[
                    valid
                ]
            ),

        "median_Sk":
            np.median(
                Sk[
                    valid
                ]
            ),

        "median_LSI":
            np.median(
                LSI[
                    valid
                ]
            ),
    }


# ============================================================
# EXTRACT ONE REPRESENTATIVE FRAME FROM CONCATENATED CONTROL
# ============================================================

def extract_endpoint_representative_frame(
    temperature,
):

    trajectory = ENDPOINT_TRAJECTORIES[
        temperature
    ]


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"SELECTING REPRESENTATIVE "
        f"{temperature} K CONTROL FRAME"
    )

    print(
        "=" * 80
    )

    print(
        trajectory
    )


    if not trajectory.exists():

        raise FileNotFoundError(
            trajectory
        )


    universe = mda.Universe(
        str(TOPOLOGY),
        str(trajectory),
    )
    use_frame_box(universe)


    oxygen_atoms = universe.select_atoms(
        "resname WAT and name O"
    )


    candidates = candidate_frame_indices(
        universe
    )


    rows = []


    for frame_index in candidates:

        metrics = candidate_descriptor(
            universe=universe,
            oxygen_atoms=oxygen_atoms,
            frame_index=frame_index,
        )


        universe.trajectory[
            frame_index
        ]
        use_frame_box(universe)


        try:

            time_ps = float(
                universe.trajectory.ts.time
            )

        except Exception:

            time_ps = np.nan


        rows.append({

            "frame":
                int(
                    frame_index
                ),

            "time_ps":
                time_ps,

            **metrics,
        })


    candidates_df = pd.DataFrame(
        rows
    )


    # ========================================================
    # ROBUST MEDOID IN q / Sk / LSI SPACE
    # ========================================================

    feature_columns = [
        "median_q",
        "median_Sk",
        "median_LSI",
    ]


    X = candidates_df[
        feature_columns
    ].to_numpy(
        dtype=float
    )


    center, scale = robust_location_scale(X)


    distance = np.sqrt(
        np.sum(
            (
                (
                    X
                    - center[
                        None,
                        :,
                    ]
                )
                / scale[
                    None,
                    :,
                ]
            )**2,
            axis=1,
        )
    )


    candidates_df[
        "medoid_distance"
    ] = distance


    selected = candidates_df.loc[
        candidates_df[
            "medoid_distance"
        ].idxmin()
    ]


    selected_frame = int(
        selected[
            "frame"
        ]
    )


    print(
        "\nSelected frame:"
    )

    print(
        selected.to_string()
    )


    # Save candidate-frame diagnostics.
    candidates_df.to_csv(
        ENDPOINT_PDB_DIR
        / (
            f"T{temperature}_"
            f"concat_candidate_frames.csv"
        ),
        index=False,
    )


    # ========================================================
    # WRITE ONE FULL-SYSTEM PDB
    # ========================================================

    universe.trajectory[
        selected_frame
    ]
    use_frame_box(universe)


    pdb_file = (
        ENDPOINT_PDB_DIR
        / (
            f"T{temperature}_"
            f"concat_frame"
            f"{selected_frame}.pdb"
        )
    )


    with mda.Writer(
        str(pdb_file),
        multiframe=False,
    ) as writer:

        writer.write(
            universe.atoms
        )


    print(
        f"\nWrote representative PDB:"
    )

    print(
        pdb_file
    )


    return (
        pdb_file,
        selected,
    )


# ============================================================
# READ OXYGENS FROM PDB
# ============================================================

def read_oxygen_positions(
    pdb_file,
):

    universe = mda.Universe(
        str(
            pdb_file
        )
    )
    use_frame_box(universe)


    oxygen_atoms = universe.select_atoms(
        "resname WAT and name O"
    )


    pos = normalize_oxygen_positions(
        oxygen_atoms.positions
    )


    return pos


# ============================================================
# REFERENCE POPULATION FROM ONE ENDPOINT PDB
# ============================================================

def endpoint_population_from_pdb(
    pdb_file,
    temperature,
):

    print(
        f"\nBuilding {temperature} K "
        f"reference population:"
    )

    print(
        f"  {pdb_file}"
    )


    pos = read_oxygen_positions(
        pdb_file
    )


    eligible = np.where(
        np.abs(
            pos[:, 2]
        )
        <= REFERENCE_Z_MAX
    )[0]


    if (
        REFERENCE_SAMPLE is not None
        and
        len(eligible) > REFERENCE_SAMPLE
    ):

        rng = np.random.default_rng(
            20000
            + temperature
        )


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


    print(
        f"  N waters = {len(X):,}"
    )

    print(
        f"  median q   = "
        f"{np.median(X[:, 0]):.6f}"
    )

    print(
        f"  median Sk  = "
        f"{np.median(X[:, 1]):.6f}"
    )

    print(
        f"  median LSI = "
        f"{np.median(X[:, 2]):.6f}"
    )


    return X


# ============================================================
# BUILD 300 K LIQUID vs 78 K LDA CLASSIFIER
# ============================================================

def build_classifier(
    pdb_300,
    pdb_78,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DISPLAY ONLY: one medoid and |z| <= 45 Å is not the quantitative classifier"
    )

    print(
        "=" * 80
    )


    X_liquid = endpoint_population_from_pdb(
        pdb_file=pdb_300,
        temperature=300,
    )


    X_lda = endpoint_population_from_pdb(
        pdb_file=pdb_78,
        temperature=78,
    )


    # ========================================================
    # SAVE ENDPOINT STRUCTURAL STATISTICS
    # ========================================================

    endpoint_stats = pd.DataFrame(
        [
            {
                "temperature_K":
                    300,

                "class":
                    "liquid_reference",

                "n_waters":
                    len(
                        X_liquid
                    ),

                "median_q":
                    np.median(
                        X_liquid[:, 0]
                    ),

                "median_Sk":
                    np.median(
                        X_liquid[:, 1]
                    ),

                "median_LSI":
                    np.median(
                        X_liquid[:, 2]
                    ),

                "mean_q":
                    np.mean(
                        X_liquid[:, 0]
                    ),

                "mean_Sk":
                    np.mean(
                        X_liquid[:, 1]
                    ),

                "mean_LSI":
                    np.mean(
                        X_liquid[:, 2]
                    ),
            },

            {
                "temperature_K":
                    78,

                "class":
                    "LDA_reference",

                "n_waters":
                    len(
                        X_lda
                    ),

                "median_q":
                    np.median(
                        X_lda[:, 0]
                    ),

                "median_Sk":
                    np.median(
                        X_lda[:, 1]
                    ),

                "median_LSI":
                    np.median(
                        X_lda[:, 2]
                    ),

                "mean_q":
                    np.mean(
                        X_lda[:, 0]
                    ),

                "mean_Sk":
                    np.mean(
                        X_lda[:, 1]
                    ),

                "mean_LSI":
                    np.mean(
                        X_lda[:, 2]
                    ),
            },
        ]
    )


    endpoint_stats.to_csv(
        OUTPUT_DIR
        / "endpoint_reference_metrics.csv",
        index=False,
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
# FIND EXISTING 250-150 K REPRESENTATIVE PDB
# ============================================================

def find_intermediate_frame(
    temperature,
    replica,
):

    matches = list(
        REPRESENTATIVE_DIR.glob(
            f"T{temperature}_r{replica}_frame*.pdb"
        )
    )


    if len(matches) != 1:

        raise RuntimeError(
            f"Expected one representative PDB for "
            f"T={temperature} K, r={replica}; "
            f"found {len(matches)}."
        )


    return matches[
        0
    ]


# ============================================================
# ANALYZE / PLOT ONE STRUCTURE
# ============================================================

def analyze_frame(
    pdb_file,
    temperature,
    model,
    replica=None,
    endpoint=False,
):

    if endpoint:

        label = (
            f"{temperature} K — "
            f"concatenated control"
        )

        stem = (
            f"T{temperature}_concat"
        )

    else:

        label = (
            f"{temperature} K — "
            f"Replica {replica}"
        )

        stem = (
            f"T{temperature}_r{replica}"
        )


    print(
        "\n"
        + "=" * 80
    )

    print(
        label
    )

    print(
        f"  {pdb_file}"
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
        pos[:, 1]
        - y_center
    )


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
                pos[:, 2]
            )
            <= Z_MAX_A
        )
    )[0]


    print(
        f"  waters in slice: "
        f"{len(slice_idx):,}"
    )


    # ========================================================
    # LOCAL STRUCTURE
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
    # LIQUID / LDA CLASSIFICATION
    # ========================================================
    
    # Raw q / Sk / LSI classifier result.
    # Preserve this independently of endpoint anchoring.
    p_lda_model = lda_probability(
        q=q,
        Sk=Sk,
        LSI=LSI,
        model=model,
    )
    
    
    if endpoint and temperature == 300:
    
        # 300 K DEFINES the liquid reference.
        p_lda = np.zeros(
            len(q),
            dtype=float,
        )
    
        lda_like = np.zeros(
            len(q),
            dtype=bool,
        )
    
        liquid_like = np.ones(
            len(q),
            dtype=bool,
        )
    
    
    elif endpoint and temperature == 78:
    
        # 78 K DEFINES the LDA reference.
        p_lda = np.ones(
            len(q),
            dtype=float,
        )
    
        lda_like = np.ones(
            len(q),
            dtype=bool,
        )
    
        liquid_like = np.zeros(
            len(q),
            dtype=bool,
        )
    
    
    else:
    
        # Intermediate temperatures are classified relative
        # to the 300 K liquid and 78 K LDA references.
        p_lda = p_lda_model.copy()
    
        lda_like = (
            p_lda
            >= P_LDA_THRESHOLD
        )
    
        liquid_like = (
            ~lda_like
        )
    
    
    # ========================================================
    # PHASE FRACTIONS
    # ========================================================
    
    lda_fraction = np.mean(
        lda_like
    )
    
    liquid_fraction = np.mean(
        liquid_like
    )
    
    
    print(
        f"  liquid-like fraction = "
        f"{liquid_fraction:.4f}"
    )
    
    print(
        f"  LDA-like fraction    = "
        f"{lda_fraction:.4f}"
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
            x_edges[:-1]
            + x_edges[1:]
        )
    )


    z_centers = (
        0.5
        * (
            z_edges[:-1]
            + z_edges[1:]
        )
    )


    # ========================================================
    # PROBABILITY-WEIGHTED PHASE DENSITIES
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


    H_liquid = gaussian_filter(
        H_liquid,
        sigma=SMOOTH_SIGMA,
        mode="nearest",
    )


    H_lda = gaussian_filter(
        H_lda,
        sigma=SMOOTH_SIGMA,
        mode="nearest",
    )


    rho_liquid = counts_to_mass_density(
        H_liquid
    )


    rho_lda = counts_to_mass_density(
        H_lda
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
            f"{stem}_phase_density.npz"
        ),

        x_edges=x_edges,
        z_edges=z_edges,

        rho_liquid=rho_liquid,
        rho_lda=rho_lda,
        rho_total=rho_total,

        fraction_lda=fraction_lda,

        p_lda_model=p_lda_model,
        p_lda_plot=p_lda,

        oxygen_indices=indices,

        q=q,
        Sk=Sk,
        LSI=LSI,
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

    # Raw classifier result based strictly on q/Sk/LSI.
    "P_LDA_model":
        p_lda_model,

    # Probability actually used for the phase map.
    #
    # 300 K = 0
    #  78 K = 1
    # intermediate temperatures = model probability
    "P_LDA_plot":
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
            f"{stem}_"
            f"classified_waters.csv"
        ),
        index=False,
    )


    # ========================================================
    # FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            10,
            7.4,
        )
    )


    # --------------------------------------------------------
    # RED LIQUID-LIKE DENSITY
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
            zorder=2,
        )


    # --------------------------------------------------------
    # OPTIONAL LIQUID OXYGEN SITES
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
    # BLUE LDA-LIKE OXYGEN SITES
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


    # --------------------------------------------------------
    # LIQUID/LDA DOMAIN BOUNDARY
    # --------------------------------------------------------

    phase_valid = (
        rho_total
        >= TOTAL_DENSITY_MIN
    )


    fraction_for_contour = np.where(
        phase_valid,
        fraction_lda,
        np.nan,
    )


    finite_fraction = fraction_for_contour[
        np.isfinite(
            fraction_for_contour
        )
    ]


    if (
        len(finite_fraction) > 0
        and
        np.min(
            finite_fraction
        )
        < P_LDA_THRESHOLD
        <
        np.max(
            finite_fraction
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


    # --------------------------------------------------------
    # GIBBS DIVIDING SURFACES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

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
        label,
        fontsize=15,
        fontweight="bold",
    )


    ax.tick_params(
        labelsize=11
    )


    ax.grid(
        False
    )


    ax.set_facecolor(
        "white"
    )


    # ========================================================
    # OUTPUT FIGURES
    # ========================================================

    outfile_png = (
        OUTPUT_DIR
        / (
            f"{stem}_"
            f"liquid_LDA_sharp.png"
        )
    )


    outfile_pdf = (
        OUTPUT_DIR
        / (
            f"{stem}_"
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


    fig.savefig(
        outfile_pdf,
        bbox_inches="tight",
        facecolor="white",
    )


    plt.close(
        fig
    )


    print(
        f"  wrote: {outfile_png}"
    )


    return {

        "temperature_K":
            temperature,

        "replica":
            (
                "concat"
                if endpoint
                else replica
            ),

        "source":
            (
                "concatenated_control"
                if endpoint
                else "quench_window"
            ),

        "pdb_file":
            str(
                pdb_file
            ),

        "n_slice_waters":
            len(
                indices
            ),

        "liquid_fraction":
            liquid_fraction,

        "LDA_fraction":
            lda_fraction,

        "median_q":
            np.median(
                q
            ),

        "median_Sk":
            np.median(
                Sk
            ),

        "median_LSI":
            np.median(
                LSI
            ),
    }


# ============================================================
# MAIN
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "EXTRACTING 300 K AND 78 K "
    "REPRESENTATIVE CONTROL FRAMES"
)

print(
    "=" * 80
)


pdb_300, selected_300 = (
    extract_endpoint_representative_frame(
        300
    )
)


pdb_78, selected_78 = (
    extract_endpoint_representative_frame(
        78
    )
)


# ============================================================
# BUILD CLASSIFIER USING ACTUAL 300 K AND 78 K CONTROLS
# ============================================================

model = build_classifier(
    pdb_300=pdb_300,
    pdb_78=pdb_78,
)


# ============================================================
# ANALYZE FULL TEMPERATURE SERIES
# ============================================================

summary_rows = []


# ------------------------------------------------------------
# 300 K
# ------------------------------------------------------------

summary_rows.append(
    analyze_frame(
        pdb_file=pdb_300,
        temperature=300,
        model=model,
        replica=None,
        endpoint=True,
    )
)


# ------------------------------------------------------------
# 250 -> 150 K, three independent replicas
# ------------------------------------------------------------

for temperature in INTERMEDIATE_TEMPERATURES:

    for replica in REPLICAS:

        pdb_file = find_intermediate_frame(
            temperature,
            replica,
        )


        summary_rows.append(
            analyze_frame(
                pdb_file=pdb_file,
                temperature=temperature,
                model=model,
                replica=replica,
                endpoint=False,
            )
        )


# ------------------------------------------------------------
# 78 K
# ------------------------------------------------------------

summary_rows.append(
    analyze_frame(
        pdb_file=pdb_78,
        temperature=78,
        model=model,
        replica=None,
        endpoint=True,
    )
)


# ============================================================
# SAVE TEMPERATURE-SERIES SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    OUTPUT_DIR
    / "liquid_LDA_temperature_series.csv",
    index=False,
)


print(
    "\n"
    + "=" * 80
)

print(
    "COMPLETE"
)

print(
    "=" * 80
)


print(
    summary_df[
        [
            "temperature_K",
            "replica",
            "liquid_fraction",
            "LDA_fraction",
            "median_q",
            "median_Sk",
            "median_LSI",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


print(
    "\nEndpoint representative structures:"
)

print(
    f"  300 K: {pdb_300}"
)

print(
    f"   78 K: {pdb_78}"
)


print(
    "\nPlots and data:"
)

print(
    OUTPUT_DIR.resolve()
)