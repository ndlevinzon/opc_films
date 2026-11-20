#!/bin/bash

set -e
set -o pipefail

# ============================================================
# USER SETTINGS
# ============================================================

# Topology, trajectory directory, and output directory. Override any of them
# in the environment. There is no default cluster path.
PRMTOP="${PRMTOP:-}"
TRAJ_DIR="${TRAJ_DIR:-TRAJ}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis}"

# Keep every stored frame. Analysis scripts apply their own stride and read
# the spacing from the NetCDF time variable, so this file must not thin the
# trajectory first.
FIRST=1
STRIDE="${STRIDE:-1}"

CPPTRAJ="cpptraj"

# ============================================================
# CHECKS
# ============================================================

if [ -z "$PRMTOP" ] || [ ! -s "$PRMTOP" ]; then
    echo "ERROR: set PRMTOP to the Amber topology (build/opc_slab.prmtop)."
    exit 1
fi

if [ ! -d "$TRAJ_DIR" ]; then
    echo "ERROR: trajectory directory not found: $TRAJ_DIR"
    exit 1
fi

if ! command -v "$CPPTRAJ" >/dev/null 2>&1; then
    echo "ERROR: cpptraj not found in PATH"
    exit 1
fi

mkdir -p "$ANALYSIS_DIR"

PRMTOP_ABS=$(realpath "$PRMTOP")

echo "============================================================"
echo "Per-replica cooling-window analysis"
echo "Working directory: $(pwd)"
echo "Topology:          $PRMTOP_ABS"
echo "Trajectory dir:    $TRAJ_DIR"
echo "Analysis dir:      $ANALYSIS_DIR"
echo "Frame selection:   first=${FIRST}, stride=${STRIDE}"
echo "============================================================"

# ============================================================
# FIND TRAJECTORIES
# ============================================================

mapfile -t TRAJS < <(
    find "$TRAJ_DIR" \
        -maxdepth 1 \
        -type f \
        \( -name "*.nc" -o -name "traj.*" \) \
        | sort
)

if [ "${#TRAJS[@]}" -eq 0 ]; then
    echo "ERROR: no trajectory files found in $TRAJ_DIR"
    echo "Expected files like:"
    echo "  TRAJ/*.nc"
    echo "  TRAJ/traj.*"
    exit 1
fi

echo "Found ${#TRAJS[@]} trajectory files:"
printf '  %s\n' "${TRAJS[@]}"
echo "============================================================"

# ============================================================
# LOOP OVER TRAJECTORIES / TEMPERATURE WINDOWS
# ============================================================

for traj in "${TRAJS[@]}"; do

    traj_base=$(basename "$traj")

    # Remove common trajectory suffixes for clean stage name
    stage="$traj_base"
    stage="${stage%.nc}"
    stage="${stage%.mdcrd}"
    stage="${stage%.crd}"

    STAGE_DIR="${ANALYSIS_DIR}/${stage}"
    mkdir -p "$STAGE_DIR"

    CPPTRAJ_IN="${STAGE_DIR}/${stage}.cpptraj.in"
    CPPTRAJ_LOG="${STAGE_DIR}/${stage}.cpptraj.log"

    CENTERED_TRAJ="${STAGE_DIR}/${stage}_centered_slab.nc"
    O_DENSITY="${STAGE_DIR}/${stage}_water_O_number_density_z.dat"
    MASS_DENSITY="${STAGE_DIR}/${stage}_water_mass_density_z.dat"
    CHARGE_DENSITY="${STAGE_DIR}/${stage}_charge_density_z.dat"

    traj_abs=$(realpath "$traj")

    echo ""
    echo "Processing:"
    echo "  Trajectory: $traj_abs"
    echo "  Stage:      $stage"
    echo "  Output dir: $STAGE_DIR"

    # ========================================================
    # WRITE CPPTRAJ INPUT
    # ========================================================

    cat > "$CPPTRAJ_IN" << EOF
parm ${PRMTOP_ABS}
trajin ${traj_abs} ${FIRST} last ${STRIDE}

autoimage
center :WAT mass origin
image origin center

box getbox ucell name BoxVectors out ${STAGE_DIR}/${stage}_box_vectors.dat

trajout ${CENTERED_TRAJ} netcdf

density out ${O_DENSITY} name WAT_O z number delta 0.25 :WAT@O
density out ${MASS_DENSITY} name WAT_mass z mass delta 0.25 :WAT
density out ${CHARGE_DENSITY} name charge_z z charge delta 0.25 :WAT

run
quit
EOF

    # ========================================================
    # RUN CPPTRAJ
    # ========================================================

    "$CPPTRAJ" -i "$CPPTRAJ_IN" > "$CPPTRAJ_LOG" 2>&1

    echo "  Done."
    echo "  Outputs:"
    echo "    $CENTERED_TRAJ"
    echo "    $O_DENSITY"
    echo "    $MASS_DENSITY"
    echo "    $CHARGE_DENSITY"
    echo "    $CPPTRAJ_LOG"

done

echo ""
echo "============================================================"
echo "All per-replica trajectory analyses complete."
echo "Results written to: $ANALYSIS_DIR"
echo "============================================================"
