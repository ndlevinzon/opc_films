#!/usr/bin/env bash
#SBATCH --job-name=opc-density
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/density-%A_%a.out
#SBATCH --error=logs/density-%A_%a.err
# Account and partition are site-specific. From this directory:
#   sbatch --account=ACCOUNT --partition=PARTITION --array=1-N run_workflow.slurm
# Set CONDA_ENV to an environment prefix when the default python is not the analysis environment.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
MANIFEST="${MANIFEST:-$PWD/config/manifest.tsv}"
: "${SLURM_ARRAY_TASK_ID:?Submit using --array=1-N}"
cmd=(python "$PWD/03_run_row.py" --manifest "$MANIFEST" --row "$SLURM_ARRAY_TASK_ID"
    --stride "${STRIDE:-10}" --nref "${NREF:-4000}")
if [[ "${CAPWAVE:-0}" == 1 ]]; then cmd+=(--capwave); fi
if [[ "${OVERWRITE:-0}" == 1 ]]; then cmd+=(--overwrite); fi
printf 'Command: '
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
