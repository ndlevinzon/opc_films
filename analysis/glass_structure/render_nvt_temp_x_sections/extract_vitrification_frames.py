#!/usr/bin/env python3

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.classifier import structure_features
from opcfilm.config import CONFIG
from opcfilm.structure.classifier import robust_location_scale

import numpy as np
import pandas as pd
import MDAnalysis as mda

from scipy.spatial import cKDTree


# ============================================================
# USER SETTINGS
# ============================================================

import os
_DATA_ROOT = os.environ.get("OPC_DATA_ROOT")
if not _DATA_ROOT:
    raise SystemExit("Set OPC_DATA_ROOT to the directory that contains opc/ and build/.")
TOPOLOGY = Path(_DATA_ROOT) / "build" / "opc_slab.prmtop"
BASE = Path(_DATA_ROOT) / "opc"



REPLICAS = [
    1,
    2,
    3,
]


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


# ============================================================
# TRAJECTORY PATH
# ============================================================

def trajectory_path(
    replica,
    temperature,
):

    return (
        BASE
        / f"r{replica}"
        / "ramp_nvt"
        / "4K_ns"
        / "prop"
        / f"{temperature}K"
        / "ANALYSIS"
        / f"{temperature}K_centered_slab.nc"
    )


# ============================================================
# SYSTEM
# ============================================================

WATER_RESNAME = "WAT"
OXYGEN_NAME = "O"

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
# REPRESENTATIVE-FRAME SELECTION
# ============================================================

# Use central slab only for deciding which frame is
# representative. This avoids allowing interface fluctuations
# to dominate frame selection.

FRAME_SELECTION_ABS_Z_MAX = 45.0

# Number of candidate frames sampled uniformly through each
# trajectory.
N_CANDIDATE_FRAMES = 40

# Number of waters used to estimate q/Sk/LSI for each
# candidate frame.
FRAME_SELECTION_WATERS = 10000


# Do not automatically discard a large amount of trajectory
# unless you know the fixed-T trajectories require it.
#
# Set this to the amount of equilibration you genuinely want
# excluded.
DISCARD_PS = 0.0


# ============================================================
# OUTPUT
# ============================================================

OUTDIR = Path(
    "./representative_frames"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


RNG = np.random.default_rng(
    12345
)


# ============================================================
# LOCAL STRUCTURE CALCULATION
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


def usable_frame_indices(
    universe,
    n_requested,
):

    n_frames = len(
        universe.trajectory
    )


    try:

        dt_ps = float(
            universe.trajectory.dt
        )

    except Exception:

        dt_ps = np.nan


    if (
        DISCARD_PS > 0.0
        and
        np.isfinite(dt_ps)
        and
        dt_ps > 0.0
    ):

        discard_frames = int(
            round(
                DISCARD_PS
                / dt_ps
            )
        )

    else:

        discard_frames = 0


    first = min(
        discard_frames,
        n_frames - 1,
    )


    available = np.arange(
        first,
        n_frames,
        dtype=int,
    )


    if len(available) <= n_requested:

        return available


    locations = np.linspace(
        0,
        len(available) - 1,
        n_requested,
    )


    locations = np.unique(
        np.round(
            locations
        ).astype(int)
    )


    return available[
        locations
    ]


# ============================================================
# METRICS FOR ONE CANDIDATE FRAME
# ============================================================

def candidate_frame_metrics(
    universe,
    O_atoms,
    frame_index,
):

    universe.trajectory[
        frame_index
    ]
    use_frame_box(universe)


    O_pos = (
        O_atoms.positions
        .astype(float)
        .copy()
    )


    # Residual slab recentering
    z_center = np.median(
        O_pos[
            :,
            2
        ]
    )


    z_rel = (
        O_pos[
            :,
            2
        ]
        - z_center
    )


    eligible = np.where(
        np.abs(
            z_rel
        )
        <= FRAME_SELECTION_ABS_Z_MAX
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
        O_pos,
        sample_idx=sample_idx,
    )


    valid = (
        np.isfinite(q)
        &
        np.isfinite(Sk)
        &
        np.isfinite(LSI)
    )


    q = q[
        valid
    ]

    Sk = Sk[
        valid
    ]

    LSI = LSI[
        valid
    ]


    return {

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

        # Also save distribution widths as diagnostics,
        # but don't use them initially for medoid selection.

        "iqr_q":
            np.percentile(q, 75)
            - np.percentile(q, 25),

        "iqr_Sk":
            np.percentile(Sk, 75)
            - np.percentile(Sk, 25),

        "iqr_LSI":
            np.percentile(LSI, 75)
            - np.percentile(LSI, 25),
    }


# ============================================================
# REPRESENTATIVE FRAME = ROBUST MEDOID
# ============================================================

def choose_representative_frame(
    replica,
    temperature,
):

    traj = trajectory_path(
        replica,
        temperature,
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
        traj
    )


    if not traj.exists():

        raise FileNotFoundError(
            traj
        )


    u = mda.Universe(
        str(TOPOLOGY),
        str(traj),
    )
    use_frame_box(u)


    O_atoms = u.select_atoms(
        f"resname {WATER_RESNAME} "
        f"and name {OXYGEN_NAME}"
    )


    candidate_frames = usable_frame_indices(
        u,
        N_CANDIDATE_FRAMES,
    )


    rows = []


    for frame_index in candidate_frames:

        metrics = candidate_frame_metrics(
            universe=u,
            O_atoms=O_atoms,
            frame_index=frame_index,
        )


        u.trajectory[
            frame_index
        ]
        use_frame_box(u)


        rows.append({

            "frame":
                int(
                    frame_index
                ),

            "time_ps":
                float(
                    u.trajectory.ts.time
                ),

            **metrics,
        })


    candidates = pd.DataFrame(
        rows
    )


    # ========================================================
    # ROBUST MEDOID
    #
    # Descriptor vector:
    #
    #   median q
    #   median Sk
    #   median LSI
    #
    # Each dimension is robustly standardized using MAD.
    # ========================================================

    feature_columns = [
        "median_q",
        "median_Sk",
        "median_LSI",
    ]


    X = candidates[
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


    candidates[
        "medoid_distance"
    ] = distance


    selected_idx = candidates[
        "medoid_distance"
    ].idxmin()


    selected = candidates.loc[
        selected_idx
    ]


    selected_frame = int(
        selected[
            "frame"
        ]
    )


    # Save candidate diagnostics
    candidates.to_csv(
        OUTDIR
        / (
            f"T{temperature}_"
            f"r{replica}_"
            f"candidate_frames.csv"
        ),
        index=False,
    )


    print(
        "\nSelected representative frame:"
    )

    print(
        selected.to_string()
    )


    return (
        traj,
        selected_frame,
        selected,
    )


# ============================================================
# EXTRACT SELECTED FRAME
# ============================================================

def extract_frame(
    trajectory,
    replica,
    temperature,
    frame_index,
):

    u = mda.Universe(
        str(TOPOLOGY),
        str(trajectory),
    )
    use_frame_box(u)


    u.trajectory[
        frame_index
    ]
    use_frame_box(u)


    # ========================================================
    # WRITE COMPLETE WATER SYSTEM
    # ========================================================

    outfile = (
        OUTDIR
        / (
            f"T{temperature}_"
            f"r{replica}_"
            f"frame{frame_index}.pdb"
        )
    )


    with mda.Writer(
        str(outfile),
        multiframe=False,
    ) as writer:

        writer.write(
            u.atoms
        )


    print(
        f"Extracted: {outfile}"
    )


    return outfile


# ============================================================
# MAIN
# ============================================================

manifest_rows = []


for temperature in TEMPERATURES:

    for replica in REPLICAS:

        (
            trajectory,
            selected_frame,
            selected,
        ) = choose_representative_frame(
            replica=replica,
            temperature=temperature,
        )


        output_file = extract_frame(
            trajectory=trajectory,
            replica=replica,
            temperature=temperature,
            frame_index=selected_frame,
        )


        manifest_rows.append({

            "temperature_K":
                temperature,

            "replica":
                replica,

            "frame":
                selected_frame,

            "time_ps":
                selected[
                    "time_ps"
                ],

            "median_q":
                selected[
                    "median_q"
                ],

            "median_Sk":
                selected[
                    "median_Sk"
                ],

            "median_LSI":
                selected[
                    "median_LSI"
                ],

            "medoid_distance":
                selected[
                    "medoid_distance"
                ],

            "trajectory":
                str(
                    trajectory
                ),

            "extracted_frame":
                str(
                    output_file
                ),
        })


# ============================================================
# MANIFEST
# ============================================================

manifest = pd.DataFrame(
    manifest_rows
)


manifest.to_csv(
    OUTDIR
    / "representative_frames_manifest.csv",
    index=False,
)


print(
    "\n"
    + "=" * 80
)

print(
    "FRAME EXTRACTION COMPLETE"
)

print(
    "=" * 80
)


print(
    manifest[
        [
            "temperature_K",
            "replica",
            "frame",
            "time_ps",
            "median_q",
            "median_Sk",
            "median_LSI",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nOutput directory:"
)

print(
    OUTDIR.resolve()
)