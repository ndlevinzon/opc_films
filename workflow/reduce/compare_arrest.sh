#!/bin/bash
# Dynamic T_g versus energetic arrest. Both reductions must already exist.
set -euo pipefail
cd "$(dirname "$0")/../.."
python analysis/glass_transition/06d_compare_dynamic_energetic.py \
  -d results/tg/tg_dyn_vs_rate.csv \
  -e results/energy/arrest_by_replica.csv \
  --out-prefix results/energy/compare
