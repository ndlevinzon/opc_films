#!/bin/bash
# T_g^dyn from the F_s tables: collect, fit each physical rate, then plot.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results/tg
shopt -s nullglob
files=(results/fs_tau/*/tau_summary.csv)
if (( ${#files[@]} == 0 )); then
  echo "No tau tables under results/. fs_tau has not written them." >&2
  exit 1
fi
python analysis/glass_transition/04_collect_tau.py -o results/tg/tau_all.csv "${files[@]}"
mapfile -t rates < <(python -c "import pandas as pd; m=pd.read_csv('config/manifest.tsv', sep='\t', dtype=str, keep_default_na=False); print('\n'.join(sorted(set(m.loc[m.kind=='hold','source_rate_K_per_ns']) - {''})))")
for rate in "${rates[@]}"; do
  python analysis/glass_transition/05_fit_relaxation_tg.py \
    -i results/tg/tau_all.csv --cooling-rate "$rate" --out-prefix "results/tg/${rate}"
done
python analysis/glass_transition/05b_plot_tg_vs_cooling_rate.py \
  results/tg/*_fit_summary.csv -o results/tg/tg_dyn_vs_rate
