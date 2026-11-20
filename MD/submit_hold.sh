#!/bin/bash
# Account and partition are passed on the sbatch command line.
set -euo pipefail
cd "$(dirname "$0")"
exec python hold_from_cooling.py "$@"
