#!/bin/bash
# Account and partition are passed on the sbatch command line.
set -euo pipefail
cd "$(dirname "$0")"
exec bash cool_300K_to_78K_nvt.gpu "$@"
