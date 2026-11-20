#!/bin/bash
# Dynamic T_g versus energetic arrest. Both reductions must already exist.
set -euo pipefail
cd "$(dirname "$0")/../.."
python analysis/glass_transition/06d_compare_dynamic_energetic.py \
  -d results/tg/tg_dyn_vs_rate.csv \
  -e results/energy/ramp_energetic_arrest_summary.csv \
  --dynamic-bootstrap-input results/tg/tau_all_whole.csv \
  --out-prefix results/energy/compare
