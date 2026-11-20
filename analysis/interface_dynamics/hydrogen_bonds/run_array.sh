#!/usr/bin/env bash
#SBATCH --job-name=opc-hbonds
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/slurm-%A_%a.out
#SBATCH --error=slurm/slurm-%A_%a.err
# Account and partition are site-specific. From this directory:
#   sbatch --account=ACCOUNT --partition=PARTITION --array=1-N run_array.sh
# 03_analyze_free_oh.py writes both awi_0-6 and awi_3-12. The bulk is |z| <= 15 A.
# Set CONDA_ENV when the default python is not the analysis environment.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm
if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi
MANIFEST="${MANIFEST:-$PWD/config/manifest.tsv}"
OUTROOT="${OUTROOT:-$PWD/outputs/per_chunk}"
: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch --array=1-N}"
python "$PWD/02_run_row.py" \
    --manifest "$MANIFEST" \
    --row "$SLURM_ARRAY_TASK_ID" \
    --outroot "$OUTROOT" \
    --frames "${FRAMES:-24}" \
    --nref "${NREF:-2048}" \
    --outward-cos "${OUTWARD_COS:-0.5}" \
    --core-halfwidth 15.0
